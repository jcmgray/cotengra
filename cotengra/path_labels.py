import math
import random
import collections


from .core import HyperGraph, PartitionTreeBuilder
from .hyper import register_hyper_function


def pop_fact(p, parts, N, pop_small_bias, pop_big_bias):
    M = N / parts
    if p <= M:
        return pop_small_bias * N * math.sin(math.pi * p / M)
    else:
        return - pop_big_bias * N * math.sin(math.pi / 2 * (p - M) / (N - M))


def labels_partition(
    inputs,
    output,
    size_dict,
    weight_nodes='linear',
    weight_edges='log',
    fuse_output_inds=False,
    parts=2,
    maxiter=None,
    memory=0,
    pop_small_bias=1,
    pop_big_bias=1,
    pop_decay=1,
    con_pow=1,
    final_sweep=True,
):
    """
    Parameters
    ----------
    inputs
    output
    size_dict
    weight_nodes
    weight_edges
    fuse_output_inds
    parts
    maxiter
    memory
    pop_small_bias
    pop_big_bias
    pop_decay
    con_pow
    final_sweep
    """

    H = HyperGraph(
        inputs, output, size_dict,
        weight_nodes=weight_nodes, weight_edges=weight_edges,
        fuse_output_inds=fuse_output_inds
    )
    N = H.num_nodes
    H._compute_weights()

    sites = list(range(len(H.nodes)))
    neighbs = collections.defaultdict(set)
    max_edge_weight = max(H.edge_weights)
    weights = {}

    # populate neighbor list and weights by edge weight
    for i in sites:
        for e in H.nodes[i]:
            for j in H.indmap[e]:
                if j != i:
                    neighbs[i].add(j)
                    weights[i, j] = H.edge_weight_map[e] / max_edge_weight

    # weight by mutual connectivity
    for i, j in weights:
        weights[i, j] *= (1 + len(neighbs[i] & neighbs[j]))**con_pow

    labels = sites.copy()
    pops = collections.Counter(labels)

    if maxiter is None:
        maxiter = N

    for r in range(maxiter):

        random.shuffle(sites)
        all_static = True

        for i in sites:
            old_label = labels[i]
            scores = collections.Counter()

            # possibly bias towards or against current value
            scores[old_label] = memory

            # add the main scores based on neighboring labels
            for j in neighbs[i]:
                lbl = labels[j]
                scores[lbl] += weights[i, j]

            # augment the scores based on global label populations
            for lbl in scores:
                p = pops[lbl]
                scores[lbl] += (
                    pop_fact(p, parts, N, pop_small_bias, pop_big_bias)
                ) / (r + 1)**pop_decay

            new_label = scores.most_common(1)[0][0]

            labels[i] = new_label
            pops[old_label] -= 1
            pops[new_label] += 1

            all_static &= (new_label == old_label)

        if all_static:
            break

    if final_sweep:
        random.shuffle(sites)
        for i in sites:
            old_label = labels[i]
            scores = collections.Counter()
            scores[old_label] = 0
            for j in neighbs[i]:
                lbl = labels[j]
                scores[lbl] += weights[i, j]
            new_label = scores.most_common(1)[0][0]
            labels[i] = new_label

    return labels


labels_to_tree = PartitionTreeBuilder(labels_partition)


register_hyper_function(
    name='labels',
    ssa_func=labels_to_tree.trial_fn,
    space={
        'random_strength': {'type': 'FLOAT_EXP', 'min': 0.01, 'max': 1.},
        'weight_edges': {'type': 'STRING', 'options': ['const', 'log']},
        'cutoff': {'type': 'INT', 'min': 10, 'max': 40},
        'parts': {'type': 'INT', 'min': 1, 'max': 16},
        'memory': {'type': 'INT', 'min': -2, 'max': 1},
        'pop_small_bias': {'type': 'FLOAT', 'min': 0.0, 'max': 2.0},
        'pop_big_bias': {'type': 'FLOAT', 'min': 0.0, 'max': 2.0},
        'pop_decay': {'type': 'FLOAT', 'min': 0.0, 'max': 10.0},
        'con_pow': {'type': 'FLOAT', 'min': 0.0, 'max': 10.0},
        'final_sweep': {'type': 'BOOL'},
    },
)


register_hyper_function(
    name='labels-agglom',
    ssa_func=labels_to_tree.trial_fn_agglom,
    space={
        'weight_edges': {'type': 'STRING', 'options': ['const', 'log']},
        'memory': {'type': 'INT', 'min': -2, 'max': 1},
        'pop_small_bias': {'type': 'FLOAT', 'min': 0.0, 'max': 2.0},
        'pop_big_bias': {'type': 'FLOAT', 'min': 0.0, 'max': 2.0},
        'pop_decay': {'type': 'FLOAT', 'min': 0.0, 'max': 10.0},
        'con_pow': {'type': 'FLOAT', 'min': 0.0, 'max': 10.0},
        'final_sweep': {'type': 'BOOL'},
        'fuse_output_inds': {'type': 'BOOL'},
    },
    constants={
        'random_strength': 0.0,
    }
)
