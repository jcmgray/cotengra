import collections.abc
import functools
import itertools
import math
import time
from numbers import Integral

from ..parallel import (
    can_scatter,
    maybe_leave_pool,
    maybe_rejoin_pool,
    parse_parallel_arg,
    scatter,
    submit,
)
from ..utils import get_rng


def compute_contracted_info(legsa, legsb, appearances, size_dict):
    """Compute the contracted legs, cost and size of a pair of legs.

    Parameters
    ----------
    legsa : dict[str, int]
        The legs of the first tensor.
    legsb : dict[str, int]
        The legs of the second tensor.
    appearances : dict[str, int]
        The total number of appearances of each index in the contraction.
    size_dict : dict[str, int]
        The size of each index.

    Returns
    -------
    legsab : dict[str, int]
        The contracted legs.
    cost : int
        The cost of the contraction.
    size : int
        The size of the resulting tensor.
    """
    legsab = {}
    cost = 1
    size = 1

    # handle all left indices
    for ix, ix_count in legsa.items():
        d = size_dict[ix]
        # all involved indices contribute to cost
        cost *= d
        if ix in legsb:
            ix_count += legsb[ix]
        if ix_count < appearances[ix]:
            # index appears on output
            legsab[ix] = ix_count
            # and so contributes to size
            size *= d

    # now handle right indices that we haven't seen yet
    for ix, ix_count in legsb.items():
        if ix not in legsa:
            d = size_dict[ix]
            cost *= d
            if ix_count < appearances[ix]:
                legsab[ix] = ix_count
                size *= d

    return legsab, cost, size


def linspace_generator(start, stop, num, log=False):
    """Generate a sequence of ``num`` evenly spaced floats between ``start``
    and ``stop``.

    Parameters
    ----------
    start : float
        The starting value.
    stop : float
        The stopping value.
    num : int
        The number of values to generate.
    log : bool, optional
        Whether to generate the sequence in log space.

    Yields
    ------
    float
    """
    if not log:
        if num == 1:
            yield (start + stop) / 2
        else:
            step = (stop - start) / (num - 1)
            for i in range(num):
                yield start + i * step

    else:
        log_start = math.log2(start)
        log_stop = math.log2(stop)

        if num == 1:
            yield 2 ** ((log_start + log_stop) / 2)
        else:
            step = (log_stop - log_start) / (num - 1)
            for i in range(num):
                yield 2 ** (log_start + i * step)


def _describe_tree(tree, info="concise"):
    return tree.describe(info=info)


def _score_tree(scorer, tree, target_size=None, coeff_size_penalty=1.0):
    trial = {"tree": tree}
    x = scorer(trial)
    if target_size is not None:
        # penalize oversize
        x += coeff_size_penalty * math.log2(
            max(trial["size"] / target_size, 1)
        )
    return x


def _slice_tree_basic(tree, current_target_size, rng, unslice=1):
    # always unslice one or more random indices
    for _ in range(unslice):
        if not tree.sliced_inds:
            break
        tree.unslice_rand_(seed=rng)
    tree.slice_(target_size=current_target_size, seed=rng)


def _slice_tree_reslice(tree, current_target_size, rng):
    tree.slice_(target_size=current_target_size, reslice=True, seed=rng)


def _slice_tree_drift(tree, current_target_size, rng):
    current_size = tree.contraction_width(log=None)
    if current_size > current_target_size:
        prob_slice = 3 / 4
    else:
        # below or reached target - can unslice
        prob_slice = 0.0

    if rng.random() < prob_slice:
        tree.slice_(target_slices=2, seed=rng)
    elif tree.sliced_inds:
        tree.unslice_rand_(seed=rng)


def simulated_anneal_tree(
    tree,
    tfinal=0.05,
    tstart=2,
    tsteps=50,
    numiter=50,
    minimize=None,
    target_size=None,
    target_size_initial=None,
    slice_mode="basic",
    seed=None,
    progbar=False,
    inplace=False,
):
    """Perform a simulated annealing optimization of this contraction
    tree, based on "Multi-Tensor Contraction for XEB Verification of
    Quantum Circuits" by Gleb Kalachev, Pavel Panteleev, Man-Hong Yung
    (arXiv:2108.05665), and the "treesa" implementation in
    OMEinsumContractionOrders.jl by Jin-Guo Liu and Pan Zhang.

    Parameters
    ----------
    tfinal : float, optional
        The final temperature.
    tstart : float, optional
        The starting temperature.
    tsteps : int, optional
        The number of temperature steps.
    numiter : int, optional
        The number of sweeps at each temperature step.
    minimize : {'flops', 'combo', 'write', 'size', ...}, optional
        The objective function to minimize.
    target_size : int, optional
        The target size to slice the contraction to. A schedule is used to
        reach this only at the final temperature step.
    target_size_initial : int, optional
        The initial target size to use in the slicing schedule. If None, then
        the current size is used.
    slice_mode : {'basic', 'reslice', 'drift', int}, optional
        The mode for slicing the contraction tree within each annealing
        iteration. 'basic' always unslices a random index and then slices to
        the target size. 'reslice' unslices all indices and then slices to the
        target size. 'drift' unslices a random index with probability 1/4 and
        slices to the target size with probability 3/4. It is therefore not
        guaranteed to reach the target size, but may be more explorative for
        long annealing schedules.
    seed : int, optional
        A random seed.
    progbar : bool, optional
        Whether to show live progress.
    inplace : bool, optional
        Whether to perform the optimization inplace.

    Returns
    -------
    ContractionTree
    """
    from ..scoring import get_score_fn

    tree = tree if inplace else tree.copy()
    # ensure stats tracking is on
    tree.contract_stats()

    if minimize is None:
        minimize = tree.get_default_objective()
    scorer = get_score_fn(minimize)
    rng = get_rng(seed)

    # create a schedule for annealing temperatures
    temps = linspace_generator(tstart, tfinal, tsteps, log=True)

    if target_size is not None:
        # create a schedule for slicing target sizes
        if target_size_initial is None:
            # start with the current size
            current_size = max(tree.contraction_width(log=None), target_size)
        else:
            current_size = max(target_size_initial, target_size)

        target_sizes = linspace_generator(
            current_size,
            target_size,
            tsteps,
            log=True,
        )

        if isinstance(slice_mode, Integral):
            # unslice this many random indices at each step
            _slice_tree = functools.partial(
                _slice_tree_basic, unslice=slice_mode
            )
        else:
            _slice_tree = {
                "basic": _slice_tree_basic,
                "reslice": _slice_tree_reslice,
                "drift": _slice_tree_drift,
            }[slice_mode]
    else:
        target_sizes = itertools.repeat(None)

        def _slice_tree(tree, current_target_size, rng):
            pass

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=tsteps)
        pbar.set_description(_describe_tree(tree))

    for temp in temps:
        # handle slicing
        _slice_tree(tree, next(target_sizes), rng)

        for _ in range(numiter):
            candidates = [tree.root]

            while candidates:
                p = candidates.pop(0)
                l, r = tree.children[p]

                # check which local moves are possible
                if len(l) == 1:
                    if len(r) == 1:
                        # both are leaves
                        continue
                    else:
                        # left is leaf
                        rule = rng.randint(2, 3)
                elif len(r) == 1:
                    # right is leaf
                    rule = rng.randint(0, 1)
                else:
                    # neither are leaves
                    rule = rng.randint(0, 3)

                if rule < 2:
                    # ((AB)C)
                    x, c = l, r
                    a, b = tree.children[x]
                    if rule == 0:
                        # -> ((AC)B)
                        new_order = [a, c, b]
                    else:
                        # -> (A(BC))
                        new_order = [b, c, a]
                else:
                    # (A(BC))
                    a, x = l, r
                    b, c = tree.children[x]
                    if rule == 2:
                        # -> (B(AC))
                        new_order = [a, c, b]
                    else:
                        # -> (C(AB))
                        new_order = [a, b, c]

                current_score = scorer.score_local(
                    flops=[tree.get_flops(p), tree.get_flops(x)],
                    size=[tree.get_size(p), tree.get_size(x)],
                )

                # legs0 = tree.get_legs(new_order[0])
                # legs1 = tree.get_legs(new_order[1])
                # if any(ix0 in legs1 for ix0 in legs0):

                # compute new intermediate
                new_legs0, new_cost0, new_size0 = compute_contracted_info(
                    tree.get_legs(new_order[0]),
                    tree.get_legs(new_order[1]),
                    tree.appearances,
                    tree.size_dict,
                )
                # compute new parent costs
                new_legs1, new_cost1, new_size1 = compute_contracted_info(
                    new_legs0,
                    tree.get_legs(new_order[2]),
                    tree.appearances,
                    tree.size_dict,
                )
                proposed_score = scorer.score_local(
                    flops=[new_cost0, new_cost1],
                    size=[new_size0, new_size1],
                )

                dE = proposed_score - current_score
                accept = (dE <= 0) or (math.log(rng.random()) < -dE / temp)

                if accept:
                    tree._remove_node(p)
                    tree._remove_node(x)

                    tree.contract_nodes_pair(
                        tree.contract_nodes_pair(
                            new_order[0],
                            new_order[1],
                            legs=new_legs0,
                            cost=new_cost0,
                            size=new_size0,
                        ),
                        new_order[2],
                        legs=new_legs1,
                        cost=new_cost1,
                        size=new_size1,
                    )

                    if progbar:
                        pbar.set_description(
                            f"T: {temp:.2e} " + _describe_tree(tree),
                            refresh=False,
                        )

                # check which children to recurse into
                l, r = tree.children[p]
                if len(l) > 2:
                    candidates.append(l)
                if len(r) > 2:
                    candidates.append(r)

        if progbar:
            pbar.update()

    return tree


def _do_anneal(tree, *args, **kwargs):
    return tree.simulated_anneal(*args, **kwargs)


def parallel_temper_tree(
    tree_or_trees,
    tfinal=0.01,
    tstart=1,
    tsteps=50,
    num_trees=8,
    numiter=50,
    minimize=None,
    target_size=None,
    target_size_initial=None,
    slice_mode="drift",
    parallel_slice_mode="temperature",
    swappiness=1.0,
    coeff_size_penalty=1.0,
    max_time=None,
    seed=None,
    parallel="auto",
    info=None,
    progbar=False,
    inplace=False,
):
    """Perform parallel tempering optimization of a contraction tree. This
    anneals ``num_trees`` different trees at a range of temperatures between
    ``tfinal`` and ``tstart``. After each step, trees are exchanged between
    neighboring temperatures according to the Metropolis-Hastings criterion.

    Parameters
    ----------
    tree_or_trees : ContractionTree or sequence of ContractionTree
        The tree or trees to optimize. If less than ``num_trees`` are given,
        then they will be cycled. If more than ``num_trees`` are given, then
        the length will override ``num_trees``.
    tfinal : float, optional
        The final temperature.
    tstart : float, optional
        The starting temperature.
    tsteps : int, optional
        The number of temperature steps, each with ``numiter`` iterations.
        After each step, trees are exchanged between neighboring temperatures.
    num_trees : int, optional
        The number of trees and thus temperatures to optimize in parallel.
    numiter : int, optional
        The number of iterations to perform at each step. The total number of
        sweeps (per parallel temperature) is ``numiter * tsteps``.
    minimize : {'flops', 'combo', 'write', 'size', ...}, optional
        The objective function to minimize.
    target_size : int, optional
        The target size of the contraction.
    slice_mode : {'basic', 'reslice', 'drift'}, optional
        The mode for slicing the contraction tree within each annealing
        iteration.
    parallel_slice_mode : {'temperature', 'time', 'constant'}, optional
        The parallel mode for slicing the contraction tree. If 'temperature',
        then the target size decreases with temperature. If 'time', then the
        target size decreases with time. If 'constant', then the target size is
        constant.
    seed : int, optional
        A random seed.
    parallel : 'auto', False, True, int, or distributed.Client
        Whether to parallelize the search.
    progbar : bool, optional
        Whether to show live progress.
    inplace : bool, optional
        Whether to perform the optimization inplace.
    """
    from ..scoring import get_score_fn

    # allow a single or sequence of trees
    sequence_supplied = isinstance(tree_or_trees, collections.abc.Iterable)

    if sequence_supplied:
        tree_or_trees = tuple(tree_or_trees)
        num_trees = max(num_trees, len(tree_or_trees))
        initial_trees = itertools.cycle(tree_or_trees)
    else:
        initial_trees = itertools.repeat(tree_or_trees)
        tree_or_trees = (tree_or_trees,)

    if minimize is None:
        scorer = tree_or_trees[0].get_default_objective()
    else:
        scorer = get_score_fn(minimize)

    rng = get_rng(seed)

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=tsteps)
        pbar.set_description(_describe_tree(tree_or_trees[0]))

    temps = tuple(linspace_generator(tfinal, tstart, num_trees, log=True))

    if target_size is None:
        # target_sizes = itertools.repeat(None)
        target_sizes = (None,) * num_trees
    else:
        if target_size_initial is None:
            target_size_initial = max(
                t.contraction_width(log=None) for t in tree_or_trees
            )

        if parallel_slice_mode == "temperature":
            # target size decreases with temperature
            target_sizes = tuple(
                linspace_generator(
                    target_size,
                    target_size_initial,
                    num_trees,
                    log=True,
                )
            )
        elif parallel_slice_mode == "time":
            # target size decreases with time
            target_sizes = linspace_generator(
                target_size_initial,
                target_size,
                tsteps,
                log=True,
            )
        elif parallel_slice_mode == "constant":
            target_sizes = (target_size,) * num_trees
        else:
            raise ValueError(
                f"Unrecognized parallel_slice_mode: {parallel_slice_mode}"
            )

    # setup the parallel machinery
    pool = parse_parallel_arg(parallel)
    is_scatter_pool = can_scatter(pool)
    if is_scatter_pool:
        is_worker = maybe_leave_pool(pool)
        # store the trees as futures for the entire process
        trees = [scatter(pool, next(initial_trees)) for _ in range(num_trees)]
    else:
        trees = [next(initial_trees) for _ in range(num_trees)]

    best_score = float("inf")
    best_tree = None

    if max_time is not None:
        # convert to absolute time
        max_time = time.time() + max_time

    for _ in range(tsteps):
        # perform annealing moves

        if parallel_slice_mode == "time":
            # get the next target size and repeat for all temperatues
            target_sizes_t = [next(target_sizes)] * num_trees
        else:
            # constant in time
            target_sizes_t = target_sizes

        args = zip(trees, temps, target_sizes_t)
        sa_opts = dict(
            tsteps=1,
            numiter=numiter,
            minimize=minimize,
            slice_mode=slice_mode,
            seed=rng.randrange(0, 2**32),
        )

        if pool is None:
            trees = [
                _do_anneal(
                    t, tfinal=temp, tstart=temp, target_size=tsz, **sa_opts
                )
                for t, temp, tsz in args
            ]
            scores = [
                _score_tree(
                    scorer,
                    t,
                    target_size,
                    coeff_size_penalty=coeff_size_penalty,
                )
                for t in trees
            ]

        else:
            # submit in smaller steps for scatter pools?
            trees = [
                submit(
                    pool,
                    _do_anneal,
                    t,
                    tfinal=temp,
                    tstart=temp,
                    target_size=tsz,
                    **sa_opts,
                )
                for t, temp, tsz in args
            ]

            if not is_scatter_pool:
                # gather trees and compute scores locally
                trees = [f.result() for f in trees]
                scores = [
                    _score_tree(
                        scorer,
                        t,
                        target_size,
                        coeff_size_penalty=coeff_size_penalty,
                    )
                    for t in trees
                ]
            else:
                # compute scores remotely also
                scores = [
                    submit(pool, _score_tree, scorer, t, target_size)
                    for t in trees
                ]
                scores = [f.result() for f in scores]

        # perform exchange moves
        nswap = 0
        for i in range(num_trees - 1):
            dbeta = (1 / swappiness) * (1 / temps[i] - 1 / temps[i + 1])
            dE = scores[i + 1] - scores[i]
            accept = (dE < 0) or (math.log(rng.random()) < -dbeta * dE)
            if accept:
                # swap trees
                nswap += 1
                trees[i], trees[i + 1] = trees[i + 1], trees[i]
                scores[i], scores[i + 1] = scores[i + 1], scores[i]

        # record the best tree (might not be the lowest temperature)
        idxmin = min(range(num_trees), key=scores.__getitem__)

        if scores[idxmin] < best_score:
            # new best tree
            best_score = scores[idxmin]
            best_tree = trees[idxmin]

            if progbar:
                # update the progress bar
                if is_scatter_pool:
                    desc = submit(pool, _describe_tree, best_tree).result()
                else:
                    desc = _describe_tree(best_tree)
                pbar.set_description(f"nswap: {nswap} " + desc, refresh=False)

        if progbar:
            pbar.update()

        if info is not None:
            info.setdefault("scores", []).append(scores)

        if (max_time is not None) and (time.time() > max_time):
            break

    if is_scatter_pool:
        best_tree = best_tree.result()
        maybe_rejoin_pool(is_worker, pool)

    if not inplace:
        return best_tree

    if inplace:
        if sequence_supplied:
            for tree0, tree1 in zip(tree_or_trees, trees):
                if is_scatter_pool:
                    tree1 = tree1.result()
                tree0.set_state_from(tree1)
            return best_tree
        else:
            tree_or_trees[0].set_state_from(best_tree)
            return tree_or_trees[0]
