import heapq
import functools
import itertools


def is_simplifiable(legs, appearances):
    """Check if ``legs`` contains any diag (repeated) or reduced (appears
    nowhere else) indices.
    """
    prev_ix = None
    for ix, ix_cnt in legs:
        if (ix == prev_ix) or (ix_cnt == appearances[ix]):
            # found a diag or reduced index
            return True
        prev_ix = ix
    return False


def compute_simplified(legs, appearances):
    """Compute the diag and reduced legs of a term. This function assumes that
    the legs are already sorted. It handles the case where a index is both
    diag and reduced (i.e. traced).
    """
    if not legs:
        return []

    new_legs = []
    cur_ix, cur_cnt = legs[0]
    for ix, ix_cnt in legs[1:]:
        if ix == cur_ix:
            # diag index-> accumulate count and continue
            cur_cnt += ix_cnt
        else:
            # index changed, flush
            if cur_cnt != appearances[cur_ix]:
                # index is not reduced -> keep
                new_legs.append((cur_ix, cur_cnt))
            cur_ix, cur_cnt = ix, ix_cnt

    if cur_cnt != appearances[cur_ix]:
        new_legs.append((cur_ix, cur_cnt))

    return new_legs


def compute_contracted(ilegs, jlegs, appearances):
    """Compute the contracted legs of two terms."""
    # do sorted simultaneous iteration over ilegs and jlegs
    ip = 0
    jp = 0
    ni = len(ilegs)
    nj = len(jlegs)
    new_legs = []
    while True:
        if ip == ni:
            # all remaining legs are from j
            new_legs.extend(jlegs[jp:])
            break
        if jp == nj:
            # all remaining legs are from i
            new_legs.extend(ilegs[ip:])
            break

        iix, ic = ilegs[ip]
        jix, jc = jlegs[jp]
        if iix < jix:
            # index only appears on i
            new_legs.append((iix, ic))
            ip += 1
        elif iix > jix:
            # index only appears on j
            new_legs.append((jix, jc))
            jp += 1
        else:  # iix == jix
            # shared index
            ijc = ic + jc
            if ijc != appearances[iix]:
                new_legs.append((iix, ijc))
            ip += 1
            jp += 1

    return new_legs


def compute_size(legs, sizes):
    """Compute the size of a term."""
    size = 1
    for ix, _ in legs:
        size *= sizes[ix]
    return size


def compute_con_cost_flops(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the total flops cost of a contraction given by temporary legs,
    also removing any contracted indices from the temporary legs.
    """
    cost = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]

    return iscore + jscore + cost


def compute_con_cost_size(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the max size of a contraction given by temporary legs, also
    removing any contracted indices from the temporary legs.
    """
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            size *= sizes[ix]

    return max((iscore, jscore, size))


def compute_con_cost_write(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the total write cost of a contraction given by temporary legs,
    also removing any contracted indices from the temporary legs.
    """
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            # kept index, contributes to new size
            size *= sizes[ix]

    return iscore + jscore + size


def compute_con_cost_combo(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
    factor,
):
    """Compute the combined total flops and write cost of a contraction given
    by temporary legs, also removing any contracted indices from the temporary
    legs. The combined cost is given by:

        cost = flops + factor * size
    """
    cost = 1
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            # kept index, contributes to new size
            size *= d

    return iscore + jscore + (cost + factor * size)


def compute_con_cost_limit(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
    factor,
):
    """Compute the combined total flops and write cost of a contraction given
    by temporary legs, also removing any contracted indices from the temporary
    legs. The combined cost is given by:

        cost = max(flops, factor * size)

    I.e. assuming one or another to be the limiting factor.
    """
    cost = 1
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            # kept index, contributes to new size
            size *= d

    new_local_score = max(cost, factor * size)
    return iscore + jscore + new_local_score


@functools.lru_cache(128)
def parse_minimize_for_optimal(minimize):
    """Given a string, parse it into a function that computes the cost of a
    contraction. The string can be one of the following:

        - "flops": compute_con_cost_flops
        - "size": compute_con_cost_size
        - "write": compute_con_cost_write
        - "combo": compute_con_cost_combo
        - "combo-{factor}": compute_con_cost_combo with specified factor
        - "limit": compute_con_cost_limit
        - "limit-{factor}": compute_con_cost_limit with specified factor

    This function is cached for speed.
    """
    import re

    if minimize == "flops":
        return compute_con_cost_flops
    elif minimize == "size":
        return compute_con_cost_size
    elif minimize == "write":
        return compute_con_cost_write

    minimize_finder = re.compile(r"(flops|size|write|combo|limit)-*(\d*)")

    # parse out a customized value for the combination factor
    match = minimize_finder.fullmatch(minimize)
    if match is None:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")

    minimize, custom_factor = match.groups()
    factor = float(custom_factor) if custom_factor else 64
    if minimize == "combo":
        return functools.partial(compute_con_cost_combo, factor=factor)
    elif minimize == "limit":
        return functools.partial(compute_con_cost_limit, factor=factor)
    else:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")


class ContractionProcessor:
    """A helper class for combining bottom up simplifications, greedy, and
    optimal contraction path optimization.
    """

    def __init__(self, inputs, output, size_dict):
        self.nodes = {}
        self.edges = {}
        self.indmap = {}
        self.appearances = []
        self.sizes = []
        c = 0

        for i, term in enumerate(inputs):
            legs = []
            for ind in term:
                ix = self.indmap.get(ind, None)
                if ix is None:
                    # index not processed yet
                    ix = self.indmap[ind] = c
                    self.edges[ix] = {i}
                    self.appearances.append(1)
                    self.sizes.append(size_dict[ind])
                    c += 1
                else:
                    # seen index already
                    self.appearances[ix] += 1
                    self.edges[ix].add(i)
                legs.append((ix, 1))

            legs.sort()
            self.nodes[i] = tuple(legs)

            # if len(legs) >= 2:
            #     # if legs has any repeated indices,
            #     # we combine and sum their counts
            #     legs.sort()
            #     ixl, cl = legs[0]
            #     ulegs = [(ixl, cl)]
            #     for ik in range(1, len(legs)):
            #         ixr, cr = legs[ik]
            #         if ixl == ixr:
            #             cl += cr
            #             ulegs[-1] = (ixr, cl)
            #         else:
            #             ixl, cl = ixr, cr
            #             ulegs.append((ixl, cl))
            #     self.nodes[i] = ulegs
            # else:
            #     self.nodes[i] = legs

        for ind in output:
            self.appearances[self.indmap[ind]] += 1

        self.ssa = len(self.nodes)
        self.ssa_path = []

    def neighbors(self, i):
        """Get all neighbors of node ``i``."""
        # only want to yield each neighbor once and not i itself
        seen = {i}
        for ix, _ in self.nodes[i]:
            for j in self.edges[ix]:
                if j not in seen:
                    yield j
                    seen.add(j)

    def print_current_terms(self):
        return ",".join(
            "".join(str(ix) for ix, c in term) for term in self.nodes.values()
        )

    def remove_ix(self, ix):
        """Drop the index ``ix``, removing it from all nodes and the edgemap."""
        for node in self.edges.pop(ix):
            self.nodes[node] = tuple(
                (jx, jx_count) for jx, jx_count in self.nodes[node] if jx != ix
            )

    def pop_node(self, i):
        """Remove node ``i`` from the graph, updating the edgemap and returning
        the legs of the node.
        """
        legs = self.nodes.pop(i)
        for j, _ in legs:
            es = self.edges[j]
            if len(es) == 1:
                del self.edges[j]
            else:
                self.edges[j].discard(i)
        return legs

    def add_node(self, legs):
        """Add a new node to the graph, updating the edgemap and returning the
        node index of the new node.
        """
        i = self.ssa
        self.ssa += 1
        self.nodes[i] = legs
        for j, _ in legs:
            self.edges.setdefault(j, set()).add(i)
        return i

    def contract_nodes(self, i, j):
        """Contract the nodes ``i`` and ``j``, adding a new node to the graph
        and returning its index.
        """
        ilegs = self.pop_node(i)
        jlegs = self.pop_node(j)
        new_legs = compute_contracted(ilegs, jlegs, self.appearances)
        k = self.add_node(new_legs)
        self.ssa_path.append((i, j))
        return k

    def simplify_batch(self):
        """Find any indices that appear in all terms and remove them, since
        they simply add an constant factor to the cost of the contraction, but
        create a fully connected graph if left.
        """
        ix_to_remove = []
        for ix, ix_nodes in self.edges.items():
            if len(ix_nodes) >= len(self.nodes):
                ix_to_remove.append(ix)
        for ix in ix_to_remove:
            # print("removing batch", ix)
            self.remove_ix(ix)

    def simplify_single_terms(self):
        """Take any diags, reductions and traces of single terms."""
        for i, legs in tuple(self.nodes.items()):
            if is_simplifiable(legs, self.appearances):
                new_legs = compute_simplified(
                    self.pop_node(i), self.appearances
                )
                self.add_node(new_legs)
                self.ssa_path.append((i,))

    def simplify_scalars(self):
        """Remove all scalars, contracting them into the smallest remaining
        node, if there is one.
        """
        scalars = []
        for i, legs in self.nodes.items():
            if len(legs) == 0:
                # scalar
                scalars.append(i)

        if scalars:
            for i in scalars:
                self.pop_node(i)

            try:
                j, res = min(
                    ((i, legs) for i, legs in self.nodes.items()),
                    key=lambda x: len(x[1]),
                )
                con = (*scalars, j)
                self.pop_node(j)
            except ValueError:
                # all scalars!
                con = tuple(scalars)
                res = ()

            # print("contracting scalars", con, "->", res)
            self.add_node(res)
            self.ssa_path.append(con)

    def simplify_hadamard(self):
        groups = {}
        hadamards = set()
        for i, legs in self.nodes.items():
            key = frozenset(ix for ix, _ in legs)
            if key in groups:
                groups[key].append(i)
                hadamards.add(key)
            else:
                groups[key] = [i]

        for key in hadamards:
            group = groups[key]
            # print("contracting hadamard", key, group)
            while len(group) > 1:
                i = group.pop()
                j = group.pop()
                group.append(self.contract_nodes(i, j))

    def simplify(self):
        self.simplify_batch()
        should_run = True
        while should_run:
            self.simplify_single_terms()
            self.simplify_scalars()
            ssa_before = self.ssa
            self.simplify_hadamard()
            # only rerun if we did hadamard deduplication
            should_run = ssa_before != self.ssa

    def subgraphs(self):
        remaining = set(self.nodes)
        groups = []
        while remaining:
            i = remaining.pop()
            queue = [i]
            group = {i}
            while queue:
                i = queue.pop()
                for j in self.neighbors(i):
                    if j not in group:
                        group.add(j)
                        queue.append(j)

            remaining -= group
            groups.append(sorted(group))

        groups.sort()
        return groups

    def optimize_greedy(self, costmod=1.0, temperature=0.0):
        """ """

        if temperature == 0.0:

            def local_score(sa, sb, sab):
                return sab - costmod * (sa + sb)

        else:
            from ..utils import GumbelBatchedGenerator
            import numpy as np
            import math

            gmblgen = GumbelBatchedGenerator(np.random.default_rng())

            def local_score(sa, sb, sab):
                score = sab - costmod * (sa + sb)
                if score < 0:
                    return -math.log(-score) - temperature * gmblgen()
                else:
                    return math.log(score) - temperature * gmblgen()

                # return sab - costmod * (sa + sb) - temperature * gmblgen()

        node_sizes = {}
        for i, ilegs in self.nodes.items():
            node_sizes[i] = compute_size(ilegs, self.sizes)

        queue = []
        contractions = {}
        c = 0
        for ix_nodes in self.edges.values():
            for i, j in itertools.combinations(ix_nodes, 2):
                isize = node_sizes[i]
                jsize = node_sizes[j]
                klegs = compute_contracted(
                    self.nodes[i], self.nodes[j], self.appearances
                )
                ksize = compute_size(klegs, self.sizes)
                score = local_score(isize, jsize, ksize)
                heapq.heappush(queue, (score, c))
                contractions[c] = (i, j, ksize, klegs)
                c += 1

        while queue:
            _, c0 = heapq.heappop(queue)
            i, j, ksize, klegs = contractions.pop(c0)
            if (i not in self.nodes) or (j not in self.nodes):
                # one of nodes already contracted
                continue

            self.pop_node(i)
            self.pop_node(j)
            k = self.add_node(klegs)
            self.ssa_path.append((i, j))
            node_sizes[k] = ksize

            for l in self.neighbors(k):
                lsize = node_sizes[l]
                mlegs = compute_contracted(
                    klegs, self.nodes[l], self.appearances
                )
                msize = compute_size(mlegs, self.sizes)
                score = local_score(ksize, lsize, msize)
                heapq.heappush(queue, (score, c))
                contractions[c] = (k, l, msize, mlegs)
                c += 1

    def optimize_optimal_connected(
        self,
        where,
        minimize="flops",
        cost_cap=2,
        allow_outer=False,
    ):
        compute_con_cost = parse_minimize_for_optimal(minimize)

        nterms = len(where)
        contractions = [{} for _ in range(nterms + 1)]
        # we use linear index within terms given during optimization, this maps
        # back to the original node index
        termmap = {}

        for i, node in enumerate(where):
            ilegs = self.nodes[node]
            isubgraph = 1 << i
            termmap[isubgraph] = node
            iscore = 0
            ipath = ()
            contractions[1][isubgraph] = (ilegs, iscore, ipath)

        while not contractions[nterms]:
            for m in range(2, nterms + 1):
                # try and make subgraphs of size m
                contractions_m = contractions[m]
                for k in range(1, m // 2 + 1):
                    # made up of bipartitions of size k, m - k
                    if k != m - k:
                        # need to check all combinations
                        pairs = itertools.product(
                            contractions[k].items(),
                            contractions[m - k].items(),
                        )
                    else:
                        # only want unique combinations
                        pairs = itertools.combinations(
                            contractions[k].items(), 2
                        )

                    for (subgraph_i, (ilegs, iscore, ipath)), (
                        subgraph_j,
                        (jlegs, jscore, jpath),
                    ) in pairs:
                        if subgraph_i & subgraph_j:
                            # subgraphs overlap -> invalid
                            continue

                        # do sorted simultaneous iteration over ilegs and jlegs
                        ip = 0
                        jp = 0
                        ni = len(ilegs)
                        nj = len(jlegs)
                        new_legs = []
                        # if allow_outer -> we will never skip
                        skip_because_outer = not allow_outer
                        while (ip < ni) and (jp < nj):
                            iix, ic = ilegs[ip]
                            jix, jc = jlegs[jp]
                            if iix < jix:
                                new_legs.append((iix, ic))
                                ip += 1
                            elif iix > jix:
                                new_legs.append((jix, jc))
                                jp += 1
                            else:  # iix == jix:
                                # shared index
                                new_legs.append((iix, ic + jc))
                                ip += 1
                                jp += 1
                                skip_because_outer = False

                        if skip_because_outer:
                            # no shared indices found
                            continue

                        # add any remaining non-shared indices
                        new_legs.extend(ilegs[ip:])
                        new_legs.extend(jlegs[jp:])

                        new_score = compute_con_cost(
                            new_legs,
                            self.appearances,
                            self.sizes,
                            iscore,
                            jscore,
                        )

                        if new_score > cost_cap:
                            # sieve contraction
                            continue

                        new_subgraph = subgraph_i | subgraph_j
                        current = contractions_m.get(new_subgraph, None)
                        if (current is None) or (new_score < current[1]):
                            new_path = (
                                *ipath,
                                *jpath,
                                (subgraph_i, subgraph_j),
                            )
                            contractions_m[new_subgraph] = (
                                new_legs,
                                new_score,
                                new_path,
                            )

            # make the holes of our 'sieve' wider
            cost_cap *= 2

        ((_, _, bitpath),) = contractions[nterms].values()
        for subgraph_i, subgraph_j in bitpath:
            i = termmap[subgraph_i]
            j = termmap[subgraph_j]
            k = self.contract_nodes(i, j)
            termmap[subgraph_i | subgraph_j] = k

    def optimize_optimal(
        self, minimize="flops", cost_cap=2, allow_outer=False
    ):
        # we need to optimize each disconnected subgraph separately
        for where in self.subgraphs():
            self.optimize_optimal_connected(
                where,
                minimize=minimize,
                cost_cap=cost_cap,
                allow_outer=allow_outer,
            )

    def optimize_remaining_by_size(self):
        """This function simply contracts remaining terms in order of size, and
        is meant to handle the disconnected terms left after greedy or optimal
        optimization.
        """
        if len(self.nodes) == 1:
            # nothing to do
            return

        if len(self.nodes) == 2:
            self.contract_nodes(*self.nodes)
            return

        nodes_sizes = [
            (compute_size(legs, self.sizes), i)
            for i, legs in self.nodes.items()
        ]
        heapq.heapify(nodes_sizes)

        while len(nodes_sizes) > 1:
            # contract the smallest two nodes until only one remains
            _, i = heapq.heappop(nodes_sizes)
            _, j = heapq.heappop(nodes_sizes)
            k = self.contract_nodes(i, j)
            ksize = compute_size(self.nodes[k], self.sizes)
            heapq.heappush(nodes_sizes, (ksize, k))


def optimize_simplify(inputs, output, size_dict):
    """Find the (likely only partial) contraction path corresponding to
    simplifications only. Those simplifiactions are:

    - ignore any indices that appear in all terms
    - combine any repeated indices within a single term
    - reduce any non-output indices that only appear on a single term
    - combine any scalar terms
    - combine any tensors with matching indices (hadamard products)

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.

    Returns
    -------
    ssa_path : list[list[int]]
        The contraction path, given as a sequence of pairs of node indices in
        'SSA' format (i.e. as if each intermediate is appended to the list of
        inputs, without removals).
    """
    cp = ContractionProcessor(inputs, output, size_dict)
    cp.simplify()
    return cp.ssa_path


def optimize_greedy(
    inputs,
    output,
    size_dict,
    costmod=1.0,
    temperature=0.0,
    simplify=True,
):
    """Find a contraction path using a greedy algorithm.

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.
    costmod : float, optional
        When assessing local greedy scores how much to weight the size of the
        tensors removed compared to the size of the tensor added::

            score = size_ab - costmod * (size_a + size_b)

        This can be a useful hyper-parameter to tune.
    temperature : float, optional
        When asessing local greedy scores, how much to randomly perturb the
        score. This is implemented as::

            score -> sign(score) * log(|score|) - temperature * gumbel()

        which implements boltzmann sampling.
    simplify : bool, optional
        Whether to perform simplifications before optimizing. These are:

            - ignore any indices that appear in all terms
            - combine any repeated indices within a single term
            - reduce any non-output indices that only appear on a single term
            - combine any scalar terms
            - combine any tensors with matching indices (hadamard products)

        Such simpifications may be required in the general case for the proper
        functioning of the core optimization, but may be skipped if the input
        indices are already in a simplified form.

    Returns
    -------
    ssa_path : list[list[int]]
        The contraction path, given as a sequence of pairs of node indices in
        'SSA' format (i.e. as if each intermediate is appended to the list of
        inputs, without removals).
    """
    cp = ContractionProcessor(inputs, output, size_dict)
    if simplify:
        cp.simplify()
    cp.optimize_greedy(costmod=costmod, temperature=temperature)
    # handle disconnected subgraphs
    cp.optimize_remaining_by_size()
    return cp.ssa_path


def optimize_optimal(
    inputs,
    output,
    size_dict,
    minimize="flops",
    cost_cap=2,
    allow_outer=False,
    simplify=True,
):
    """Find the optimal contraction path using a dynamic programming
    algorithm (by default excluding outer products).

    The algorithm is an optimized version of Phys. Rev. E 90, 033315 (2014)
    (preprint: https://arxiv.org/abs/1304.6112), adapted from the
    ``opt_einsum`` implementation.

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.
    minimize : str, optional
        How to compute the cost of a contraction. The default is "flops".
        Can be one of:

            - "flops": minimize with respect to total operation count only
              (also known as contraction cost)
            - "size": minimize with respect to maximum intermediate size only
              (also known as contraction width)
            - "write": minimize with respect to total write cost only
            - "combo" or "combo(-{factor}": minimize with respect sum of flops
              and write weighted by specified factor. If the factor is not
              given a default value is used.
            - "limit" or "limit-{factor}": minimize with respect to max (at
              each contraction) of flops or write weighted by specified
              factor. If the factor is not given a default value is used.

    cost_cap : float, optional
        The maximum cost of a contraction to initially consider. This acts like
        a sieve and is doubled at each iteration until the optimal path can
        be found, but supplying an accurate guess can speed up the algorithm.
    allow_outer : bool, optional
        Whether to allow outer products in the contraction path. The default is
        False, especially when considering write costs, the fastest path is
        very unlikely to include outer products.
    simplify : bool, optional
        Whether to perform simplifications before optimizing. These are:

            - ignore any indices that appear in all terms
            - combine any repeated indices within a single term
            - reduce any non-output indices that only appear on a single term
            - combine any scalar terms
            - combine any tensors with matching indices (hadamard products)

        Such simpifications may be required in the general case for the proper
        functioning of the core optimization, but may be skipped if the input
        indices are already in a simplified form.

    Returns
    -------
    ssa_path : list[list[int]]
        The contraction path, given as a sequence of pairs of node indices in
        'SSA' format (i.e. as if each intermediate is appended to the list of
        inputs, without removals).
    """
    cp = ContractionProcessor(inputs, output, size_dict)
    if simplify:
        cp.simplify()
    cp.optimize_optimal(
        minimize=minimize, cost_cap=cost_cap, allow_outer=allow_outer
    )
    # handle disconnected subgraphs
    cp.optimize_remaining_by_size()
    return cp.ssa_path
