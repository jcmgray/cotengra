"""Gather all external opt_einsum functions into one module.

Various adapted opt_einsum functions. Original license below.

The MIT License (MIT)

Copyright (c) 2014 Daniel Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import re
import math
import heapq
import random
import functools
import itertools
import collections


# # external imports we are replacing:

# from opt_einsum import get_symbol
# from opt_einsum.helpers import (
#     compute_size_by_dict,
#     flop_count,
# )
# from opt_einsum.parser import (
#     find_output_str,
# )
# from opt_einsum.paths import (
#     DynamicProgramming,
#     get_path_fn,
#     linear_to_ssa,
#     PathOptimizer,
#     register_path_fn,
#     ssa_to_linear,
#     ssa_greedy_optimize,
# )
# from opt_einsum.path_random import thermal_chooser

# try:
#     from opt_einsum.paths import DEFAULT_COMBO_FACTOR
# except ImportError:
#     DEFAULT_COMBO_FACTOR = 64


try:
    from opt_einsum.paths import PathOptimizer, get_path_fn, register_path_fn
except ImportError:
    PathOptimizer = object

    def get_path_fn(*_, **__):
        raise NotImplementedError("opt_einsum not installed")

    def register_path_fn(*_, **__):
        raise NotImplementedError("opt_einsum not installed")


DEFAULT_COMBO_FACTOR = 64

_einsum_symbols_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_symbol(i) -> str:
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)``
    and skipping surrogates.

    **Examples:**

    ```python
    get_symbol(2)
    #> 'c'

    get_symbol(200)
    #> 'Ŕ'

    get_symbol(20000)
    #> '京'
    ```
    """
    if i < 52:
        return _einsum_symbols_base[i]
    elif i >= 55296:
        # Skip chr(57343) - chr(55296) as surrogates
        return chr(i + 2048)
    else:
        return chr(i + 140)


def compute_size_by_dict(indices, idx_dict) -> int:
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index _sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def find_output_str(subscripts) -> str:
    """
    Find the output string for the inputs ``subscripts`` under canonical
    einstein summation rules. That is, repeated indices are summed over by
    default, and the output is also sorted.

    Examples
    --------
    >>> oe.parser.find_output_str("ab,bc")
    'ac'

    >>> oe.parser.find_output_str("a,b")
    'ab'

    >>> oe.parser.find_output_str("a,a,b,b")
    ''
    """
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(
        s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1
    )


def flop_count(
    idx_contraction,
    inner,
    num_terms,
    size_dictionary,
) -> int:
    """
    Computes the number of FLOPS in the contraction, *assuming real dtype*.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """
    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


def linear_to_ssa(path):
    """
    Convert a path with recycled linear ids to a path with static single
    assignment ids. For example::

    ```python
    linear_to_ssa([(0, 3), (1, 2), (0, 1)])
    #> [(0, 3), (2, 4), (1, 5)]
    ```
    """
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def ssa_to_linear(ssa_path):
    """
    Convert a path with static single assignment ids to a path with recycled
    linear ids. For example:

    ```python
    ssa_to_linear([(0, 3), (2, 4), (1, 5)])
    #> [(0, 3), (1, 2), (0, 1)]
    ```
    """
    import numpy as np

    ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)
    path = []
    for ssa_ids in ssa_path:
        path.append(tuple(int(ids[ssa_id]) for ssa_id in ssa_ids))
        for ssa_id in ssa_ids:
            ids[ssa_id:] -= 1
    return path


def _get_candidate(
    output,
    sizes,
    remaining,
    footprints,
    dim_ref_counts,
    k1,
    k2,
    cost_fn,
):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (
        (either & output)
        | (two & dim_ref_counts[3])
        | (one & dim_ref_counts[2])
    )
    cost = cost_fn(
        compute_size_by_dict(k12, sizes),
        footprints[k1],
        footprints[k2],
        k12,
        k1,
        k2,
    )
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(
    output,
    sizes,
    remaining,
    footprints,
    dim_ref_counts,
    k1,
    k2s,
    queue,
    push_all,
    cost_fn,
):
    candidates = (
        _get_candidate(
            output,
            sizes,
            remaining,
            footprints,
            dim_ref_counts,
            k1,
            k2,
            cost_fn,
        )
        for k2 in k2s
    )
    if push_all:
        # want to do this if we e.g. are using a custom 'choose_fn'
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))


def _update_ref_counts(
    dim_to_keys,
    dim_ref_counts,
    dims,
):
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _simple_chooser(queue, remaining):
    """Default contraction chooser that simply takes the minimum cost option,
    with no jitter."""
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None  # candidate is obsolete
    return cost, k1, k2, k12


def thermal_chooser(
    queue, remaining, nbranch=8, temperature=1, rel_temperature=True
):
    """A contraction 'chooser' that weights possible contractions using a
    Boltzmann distribution. Explicitly, given costs `c_i` (with `c_0` the
    smallest), the relative weights, `w_i`, are computed as:

        $$w_i = exp( -(c_i - c_0) / temperature)$$

    Additionally, if `rel_temperature` is set, scale `temperature` by
    `abs(c_0)` to account for likely fluctuating cost magnitudes during the
    course of a contraction.

    **Parameters:**

    - **queue** - *(list)* The heapified list of candidate contractions.
    - **remaining** - *(dict[str, int])* Mapping of remaining inputs' indices
      to the ssa id.
    - **temperature** - *(float, optional)* When choosing a possible
      contraction, its relative probability will be
      proportional to `exp(-cost / temperature)`. Thus the larger
      `temperature` is, the further random paths will stray from the normal
      'greedy' path. Conversely, if set to zero, only paths with exactly the
      same cost as the best at each step will be explored.
    - **rel_temperature** - *(bool, optional)* Whether to normalize the
      `temperature` at each step to the scale of
      the best cost. This is generally beneficial as the magnitude of costs
      can vary significantly throughout a contraction.
    - **nbranch** - *(int, optional)* How many potential paths to calculate
      probability for and choose from at each step.

    **Returns:**

    - **cost**
    - **k1**
    - **k2**
    - **k3**
    """
    n = 0
    choices = []
    while queue and n < nbranch:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete
        choices.append((cost, k1, k2, k12))
        n += 1

    if n == 0:
        return None
    if n == 1:
        return choices[0]

    costs = [choice[0][0] for choice in choices]
    cmin = costs[0]

    # adjust by the overall scale to account for fluctuating absolute costs
    if rel_temperature:
        temperature *= max(1, abs(cmin))

    # compute relative probability for each potential contraction
    if temperature == 0.0:
        energies = [1 if c == cmin else 0 for c in costs]
    else:
        # shift by cmin for numerical reasons
        energies = [math.exp(-(c - cmin) / temperature) for c in costs]

    # randomly choose a contraction based on energies
    (chosen,) = random.choices(range(n), weights=energies)
    cost, k1, k2, k12 = choices.pop(chosen)

    # put the other choice back in the heap
    for other in choices:
        heapq.heappush(queue, other)

    return cost, k1, k2, k12


def cost_memory_removed(
    size12: int, size1: int, size2: int, k12: int, k1: int, k2: int
) -> float:
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - size1 - size2


def cost_memory_removed_jitter(
    size12: int, size1: int, size2: int, k12: int, k1: int, k2: int
) -> float:
    """Like memory-removed, but with a slight amount of noise that breaks ties
    and thus jumbles the contractions a bit.
    """
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)


_COST_FNS = {
    "memory-removed": cost_memory_removed,
    "memory-removed-jitter": cost_memory_removed_jitter,
}


def ssa_greedy_optimize(
    inputs,
    output,
    sizes,
    choose_fn=None,
    cost_fn="memory-removed",
):
    """
    This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        # Perform a single contraction to match output shape.
        return [(0,)]

    # set the function that assigns a heuristic cost to a possible contraction
    cost_fn = _COST_FNS.get(cost_fn, cost_fn)

    # set the function that chooses which contraction to take
    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        # assume chooser wants access to all possible contractions
        push_all = True

    # A dim that is common to all tensors might as well be an output dim, since
    # it cannot be contracted until the final step. This avoids an expensive
    # all-pairs comparison to search for possible contractions at each step,
    # leading to speedup in many practical problems where all tensors share a
    # common batch dimension.
    fs_inputs = [frozenset(x) for x in inputs]
    output = frozenset(output) | frozenset.intersection(*fs_inputs)

    # Deduplicate shapes by eagerly computing Hadamard products.
    remaining = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(fs_inputs))
    ssa_path = []
    for ssa_id, key in enumerate(fs_inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    # Keep track of possible contraction dims.
    dim_to_keys = collections.defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    # Keep track of the number of tensors using each dim; when the dim is no
    # longer used it can be contracted. Since we specialize to binary ops, we
    # only care about ref counts of >=2 or >=3.
    dim_ref_counts = {
        count: set(
            dim for dim, keys in dim_to_keys.items() if len(keys) >= count
        )
        - output
        for count in [2, 3]
    }

    # Compute separable part of the objective function for contractions.
    footprints = {key: compute_size_by_dict(key, sizes) for key in remaining}

    # Find initial candidate contractions.
    queue = []
    for dim, dim_keys in dim_to_keys.items():
        dim_keys_list = sorted(dim_keys, key=remaining.__getitem__)
        for i, k1 in enumerate(dim_keys_list[:-1]):
            k2s_guess = dim_keys_list[1 + i :]
            _push_candidate(
                output,
                sizes,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                k2s_guess,
                queue,
                push_all,
                cost_fn,
            )

    # Greedily contract pairs of tensors.
    while queue:
        con = choose_fn(queue, remaining)
        if con is None:
            continue  # allow choose_fn to flag all candidates obsolete
        cost, k1, k2, k12 = con

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)
        footprints[k12] = compute_size_by_dict(k12, sizes)

        # Find new candidate contractions.
        k1 = k12
        k2s = set(k2 for dim in k1 for k2 in dim_to_keys[dim])
        k2s.discard(k1)
        if k2s:
            _push_candidate(
                output,
                sizes,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                list(k2s),
                queue,
                push_all,
                cost_fn,
            )

    # Greedily compute pairwise outer products.
    final_queue = [
        (compute_size_by_dict(key & output, sizes), ssa_id, key)
        for key, ssa_id in remaining.items()
    ]
    heapq.heapify(final_queue)
    _, ssa_id1, k1 = heapq.heappop(final_queue)
    while final_queue:
        _, ssa_id2, k2 = heapq.heappop(final_queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(final_queue, (cost, ssa_id12, k12))

    return ssa_path


def _tree_to_sequence(tree):
    """
    Converts a contraction tree to a contraction path as it has to be
    returned by path optimizers. A contraction tree can either be an int
    (=no contraction) or a tuple containing the terms to be contracted. An
    arbitrary number (>= 1) of terms can be contracted at once. Note that
    contractions are commutative, e.g. (j, k, l) = (k, l, j). Note that in
    general, solutions are not unique.

    **Parameters:**

    - **c** - *(tuple or int)* Contraction tree

    **Returns:**

    - **path** - *(list[set[int]])* Contraction path

    **Examples:**

    ```python
    _tree_to_sequence(((1,2),(0,(4,5,3))))
    #> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    ```
    """

    # ((1,2),(0,(4,5,3))) --> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    #
    # 0     0         0           (1,2)       --> ((1,2),(0,(3,4,5)))
    # 1     3         (1,2)   --> (0,(3,4,5))
    # 2 --> 4     --> (3,4,5)
    # 3     5
    # 4     (1,2)
    # 5
    #
    # this function iterates through the table shown above from right to left;

    if type(tree) == int:
        return []

    c = [
        tree
    ]  # list of remaining contractions (lower part of columns shown above)
    t = []  # list of elementary tensors (upper part of columns)
    s = []  # resulting contraction sequence

    while len(c) > 0:
        j = c.pop(-1)
        s.insert(0, tuple())

        for i in sorted([i for i in j if type(i) == int]):
            s[0] += (sum(1 for q in t if q < i),)
            t.insert(s[0][-1], i)

        for i_tup in [i_tup for i_tup in j if type(i_tup) != int]:
            s[0] += (len(t) + len(c),)
            c.append(i_tup)

    return s


def _find_disconnected_subgraphs(inputs, output):
    """
    Finds disconnected subgraphs in the given list of inputs. Inputs are
    connected if they share summation indices. Note: Disconnected subgraphs
    can be contracted independently before forming outer products.

    **Parameters:**
    - **inputs** - *(list[set])* List of sets that represent the lhs side of
      the einsum subscript
    - **output** - *(set)* Set that represents the rhs side of the overall
      einsum subscript

    **Returns:**

    - **subgraphs** - *(list[set[int]])* List containing sets of indices for
      each subgraph

    **Examples:**

    ```python
    _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("bd"))
    #> [{0, 2}, {1}]

    _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("abd"))
    #> [{0}, {1}, {2}]
    ```
    """

    subgraphs = []
    unused_inputs = set(range(len(inputs)))

    i_sum = frozenset.union(*inputs) - output  # all summation indices

    while len(unused_inputs) > 0:
        g = set()
        q = [unused_inputs.pop()]
        while len(q) > 0:
            j = q.pop()
            g.add(j)
            i_tmp = i_sum & inputs[j]
            n = {k for k in unused_inputs if len(i_tmp & inputs[k]) > 0}
            q.extend(n)
            unused_inputs.difference_update(n)

        subgraphs.append(g)

    return [frozenset(x) for x in subgraphs]


def _bitmap_select(s: int, seq):
    """Select elements of ``seq`` which are marked by the bitmap set ``s``.

    E.g.:

        >>> list(_bitmap_select(0b11010, ['A', 'B', 'C', 'D', 'E']))
        ['B', 'D', 'E']
    """
    return (x for x, b in zip(seq, bin(s)[:1:-1]) if b == "1")


def _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2):
    """Calculates the effective outer indices of the intermediate tensor
    corresponding to the subgraph ``s``.
    """
    # set of remaining tensors (=g-s)
    r = g & (all_tensors ^ s)
    # indices of remaining indices:
    if r:
        i_r = frozenset.union(*_bitmap_select(r, inputs))
    else:
        i_r = frozenset()
    # contraction indices:
    i_contract = i1_cut_i2_wo_output - i_r
    return i1_union_i2 - i_contract


def _dp_compare_flops(
    cost1,
    cost2,
    i1_union_i2,
    size_dict,
    cost_cap,
    s1,
    s2,
    xn,
    g,
    all_tensors,
    inputs,
    i1_cut_i2_wo_output,
    contract1,
    contract2,
):
    """Performs the inner comparison of whether the two subgraphs (the bitmaps
    `s1` and `s2`) should be merged and added to the dynamic programming
    search. Will skip for a number of reasons:

    1. If the number of operations to form `s = s1 | s2` including previous
       contractions is above the cost-cap.
    2. If we've already found a better way of making `s`.
    3. If the intermediate tensor corresponding to `s` is going to break the
       memory limit.
    """

    # TODO: Odd usage with an Iterable[int] to map a dict of type List[int]
    cost = cost1 + cost2 + compute_size_by_dict(i1_union_i2, size_dict)
    if cost <= cost_cap:
        s = s1 | s2
        if s not in xn or cost < xn[s][1]:
            i = _dp_calc_legs(
                g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2
            )
            xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_size(
    cost1,
    cost2,
    i1_union_i2,
    size_dict,
    cost_cap,
    s1,
    s2,
    xn,
    g,
    all_tensors: int,
    inputs,
    i1_cut_i2_wo_output,
    contract1,
    contract2,
):
    """Like `_dp_compare_flops` but sieves the potential contraction based
    on the size of the intermediate tensor created, rather than the number of
    operations, and so calculates that first.
    """

    s = s1 | s2
    i = _dp_calc_legs(
        g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2
    )
    mem = compute_size_by_dict(i, size_dict)
    cost = max(cost1, cost2, mem)
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_write(
    cost1,
    cost2,
    i1_union_i2,
    size_dict,
    cost_cap,
    s1,
    s2,
    xn,
    g,
    all_tensors,
    inputs,
    i1_cut_i2_wo_output,
    contract1,
    contract2,
):
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on the total size of memory created, rather than the number of
    operations, and so calculates that first.
    """
    s = s1 | s2
    i = _dp_calc_legs(
        g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2
    )
    mem = compute_size_by_dict(i, size_dict)
    cost = cost1 + cost2 + mem
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_combo(
    cost1,
    cost2,
    i1_union_i2,
    size_dict,
    cost_cap,
    s1,
    s2,
    xn,
    g,
    all_tensors,
    inputs,
    i1_cut_i2_wo_output,
    contract1,
    contract2,
    factor=DEFAULT_COMBO_FACTOR,
    combine=sum,
):
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on some combination of both the flops and size,
    """
    s = s1 | s2
    i = _dp_calc_legs(
        g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2
    )
    mem = compute_size_by_dict(i, size_dict)
    f = compute_size_by_dict(i1_union_i2, size_dict)
    cost = cost1 + cost2 + combine((f, factor * mem))
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            xn[s] = (i, cost, (contract1, contract2))


minimize_finder = re.compile(r"(flops|size|write|combo|limit)-*(\d*)")


@functools.lru_cache(128)
def _parse_minimize(minimize):
    """This works out what local scoring function to use for the dp algorithm,
    based on a str."""
    if minimize == "flops":
        return _dp_compare_flops
    elif minimize == "size":
        return _dp_compare_size
    elif minimize == "write":
        return _dp_compare_write
    elif callable(minimize):
        # default to naive_scale=inf for this and remaining options
        # as otherwise memory_limit check can cause problems
        return minimize

    # parse out a customized value for the combination factor
    match = minimize_finder.fullmatch(minimize)
    if match is None:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")

    minimize, custom_factor = match.groups()
    factor = float(custom_factor) if custom_factor else DEFAULT_COMBO_FACTOR
    if minimize == "combo":
        return functools.partial(_dp_compare_combo, factor=factor, combine=sum)
    elif minimize == "limit":
        return functools.partial(_dp_compare_combo, factor=factor, combine=max)
    else:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")


def simple_tree_tuple(seq):
    """Make a simple left to right binary tree out of iterable `seq`.

    ```python
    tuple_nest([1, 2, 3, 4])
    #> (((1, 2), 3), 4)
    ```

    """
    return functools.reduce(lambda x, y: (x, y), seq)


def _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts):
    """Take `inputs` and parse for single term index operations, i.e. where
    an index appears on one tensor and nowhere else.

    If a term is completely reduced to a scalar in this way it can be removed
    to `inputs_done`. If only some indices can be summed then add a 'single
    term contraction' that will perform this summation.
    """
    i_single = frozenset(
        i for i, c in enumerate(all_inds) if ind_counts[c] == 1
    )
    inputs_parsed = []
    inputs_done = []
    inputs_contractions = []
    for j, i in enumerate(inputs):
        i_reduced = i - i_single
        if (not i_reduced) and (len(i) > 0):
            # input reduced to scalar already - remove
            inputs_done.append((j,))
        else:
            # if the input has any index reductions, add single contraction
            inputs_parsed.append(i_reduced)
            inputs_contractions.append((j,) if i_reduced != i else j)

    return inputs_parsed, inputs_done, inputs_contractions


class DynamicProgramming(PathOptimizer):
    """
    Finds the optimal path of pairwise contractions without intermediate outer
    products based a dynamic programming approach presented in
    Phys. Rev. E 90, 033315 (2014) (the corresponding preprint is publicly
    available at https://arxiv.org/abs/1304.6112). This method is especially
    well-suited in the area of tensor network states, where it usually
    outperforms all the other optimization strategies.

    This algorithm shows exponential scaling with the number of inputs
    in the worst case scenario (see example below). If the graph to be
    contracted consists of disconnected subgraphs, the algorithm scales
    linearly in the number of disconnected subgraphs and only exponentially
    with the number of inputs per subgraph.

    **Parameters:**

    - **minimize** - *({'flops', 'size', 'write', 'combo', 'limit', callable},
      optional)* What to minimize:

        - 'flops' - minimize the number of flops
        - 'size' - minimize the size of the largest intermediate
        - 'write' - minimize the size of all intermediate tensors
        - 'combo' - minimize `flops + alpha * write` summed over intermediates,
          a default ratio of alpha=64 is used, or it can be customized with
          `f'combo-{alpha}'`
        - 'limit' - minimize `max(flops, alpha * write)` summed over
          intermediates, a default ratio of alpha=64 is used, or it can be
          customized with `f'limit-{alpha}'`
        - callable - a custom local cost function

    - **cost_cap** - *({True, False, int}, optional)* How to implement
      cost-capping:

        - True - iteratively increase the cost-cap
        - False - implement no cost-cap at all
        - int - use explicit cost cap

    - **search_outer** - *(bool, optional)* In rare circumstances the optimal
      contraction may involve an outer product, this option allows searching
      such contractions but may well slow down the path finding considerably on
      all but very small graphs.
    """

    def __init__(
        self,
        minimize: str = "flops",
        cost_cap: bool = True,
        search_outer: bool = False,
    ):
        self.minimize = minimize
        self.search_outer = search_outer
        self.cost_cap = cost_cap

    def __call__(
        self,
        inputs_,
        output_,
        size_dict_,
        memory_limit=None,
    ):
        """
        **Parameters:**

        - **inputs** - *(list)* List of sets that represent the lhs side of the
          einsum subscript
        - **output** - *(set)* Set that represents the rhs side of the overall
          einsum subscript
        - **size_dict** - *(dictionary)* Dictionary of index sizes

        **Returns:**

        - **path** - *(list)* The contraction order (a list of tuples of ints).

        **Examples:**

        ```python
        n_in = 3  # exponential scaling
        n_out = 2 # linear scaling
        s = dict()
        i_all = []
        for _ in range(n_out):
            i = [set() for _ in range(n_in)]
            for j in range(n_in):
                for k in range(j+1, n_in):
                    c = oe.get_symbol(len(s))
                    i[j].add(c)
                    i[k].add(c)
                    s[c] = 2
            i_all.extend(i)
        o = DynamicProgramming()
        o(i_all, set(), s)
        #> [(1, 2), (0, 4), (1, 2), (0, 2), (0, 1)]
        ```
        """
        _check_contraction = _parse_minimize(self.minimize)
        _check_outer = (lambda x: True) if self.search_outer else (lambda x: x)

        ind_counts = collections.Counter(itertools.chain(*inputs_, output_))
        all_inds = tuple(ind_counts)

        # convert all indices to integers (makes set operations ~10 % faster)
        symbol2int = {c: j for j, c in enumerate(all_inds)}
        inputs = [frozenset(symbol2int[c] for c in i) for i in inputs_]
        output = frozenset(symbol2int[c] for c in output_)
        size_dict_canonical = {
            symbol2int[c]: v for c, v in size_dict_.items() if c in symbol2int
        }
        size_dict = [
            size_dict_canonical[j] for j in range(len(size_dict_canonical))
        ]

        (
            inputs,
            inputs_done,
            inputs_contractions,
        ) = _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts)

        if not inputs:
            # nothing left to do after single axis reductions!
            return _tree_to_sequence(simple_tree_tuple(inputs_done))

        # a list of all necessary contraction expressions for each of the
        # disconnected subgraphs and their size
        subgraph_contractions = inputs_done
        subgraph_contractions_size = [1] * len(inputs_done)

        if self.search_outer:
            # optimize everything together if we are considering outer products
            subgraphs = [frozenset(range(len(inputs)))]
        else:
            subgraphs = _find_disconnected_subgraphs(inputs, output)

        # the bitmap set of all tensors is computed as it is needed to
        # compute set differences: s1 - s2 transforms into
        # s1 & (all_tensors ^ s2)
        all_tensors = (1 << len(inputs)) - 1

        for g in subgraphs:
            # dynamic programming approach to compute x[n] for subgraph g;
            # x[n][set of n tensors] = (indices, cost, contraction)
            # the set of n tensors is represented by a bitmap: if bit j is 1,
            # tensor j is in the set, e.g. 0b100101 = {0,2,5}; set unions
            # (intersections) can then be computed by bitwise or (and);
            x = [None] * 2 + [dict() for j in range(len(g) - 1)]
            x[1] = {1 << j: (inputs[j], 0, inputs_contractions[j]) for j in g}

            # convert set of tensors g to a bitmap set:
            bitmap_g = functools.reduce(
                lambda x, y: x | y, (1 << j for j in g)
            )

            # try to find contraction with cost <= cost_cap and increase
            # cost_cap successively if no such contraction is found;
            # this is a major performance improvement; start with product of
            # output index dimensions as initial cost_cap
            subgraph_inds = frozenset.union(*_bitmap_select(bitmap_g, inputs))
            if self.cost_cap is True:
                cost_cap = compute_size_by_dict(
                    subgraph_inds & output, size_dict
                )
            elif self.cost_cap is False:
                cost_cap = float("inf")  # type: ignore
            else:
                cost_cap = self.cost_cap
            # set the factor to increase the cost
            # by each iteration (ensure > 1)
            if len(subgraph_inds) == 0:
                cost_increment = 2
            else:
                cost_increment = max(
                    min(map(size_dict.__getitem__, subgraph_inds)), 2
                )

            while len(x[-1]) == 0:
                for n in range(2, len(x[1]) + 1):
                    xn = x[n]

                    # try to combine solutions from x[m] and x[n-m]
                    for m in range(1, n // 2 + 1):
                        for s1, (i1, cost1, contract1) in x[m].items():
                            for s2, (i2, cost2, contract2) in x[n - m].items():
                                # can only merge if s1 and s2 are disjoint
                                # and avoid e.g:
                                # s1={0}, s2={1} and s1={1}, s2={0}
                                if (not s1 & s2) and (m != n - m or s1 < s2):
                                    i1_cut_i2_wo_output = (i1 & i2) - output

                                    # maybe ignore outer products:
                                    if _check_outer(i1_cut_i2_wo_output):
                                        i1_union_i2 = i1 | i2
                                        _check_contraction(
                                            cost1,
                                            cost2,
                                            i1_union_i2,
                                            size_dict,
                                            cost_cap,
                                            s1,
                                            s2,
                                            xn,
                                            bitmap_g,
                                            all_tensors,
                                            inputs,
                                            i1_cut_i2_wo_output,
                                            contract1,
                                            contract2,
                                        )

                # increase cost cap for next iteration:
                cost_cap = cost_increment * cost_cap

            i, _, contraction = list(x[-1].values())[0]
            subgraph_contractions.append(contraction)
            subgraph_contractions_size.append(
                compute_size_by_dict(i, size_dict)
            )

        # sort the subgraph contractions by the size of the subgraphs in
        # ascending order (will give the cheapest contractions); note that
        # outer products should be performed pairwise (to use BLAS functions)
        subgraph_contractions = [
            subgraph_contractions[j]
            for j in sorted(
                range(len(subgraph_contractions_size)),
                key=subgraph_contractions_size.__getitem__,
            )
        ]

        # build the final contraction tree
        tree = simple_tree_tuple(subgraph_contractions)
        return _tree_to_sequence(tree)


__all__ = (
    "compute_size_by_dict",
    "DEFAULT_COMBO_FACTOR",
    "DynamicProgramming",
    "find_output_str",
    "flop_count",
    "get_path_fn",
    "get_symbol",
    "linear_to_ssa",
    "PathOptimizer",
    "register_path_fn",
    "ssa_to_linear",
    "thermal_chooser",
    "ssa_greedy_optimize",
)
