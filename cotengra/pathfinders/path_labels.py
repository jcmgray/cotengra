"""Contraction tree finders using pure python 'labels' hypergraph partitioning."""

import collections
import math

from ..core import PartitionTreeBuilder
from ..hypergraph import HyperGraph
from ..hyperoptimizers.hyper import register_hyper_function
from ..utils import get_rng


def pop_fact(p, parts, n, pop_small_bias, pop_big_bias):
    m = n / parts
    if p <= m:
        return pop_small_bias * n * math.sin(math.pi * p / m)
    else:
        return -pop_big_bias * n * math.sin(math.pi / 2 * (p - m) / (n - m))


def labels_partition(
    inputs,
    output,
    size_dict,
    weight_nodes="linear",
    weight_edges="log",
    parts=2,
    maxiter=None,
    memory=0,
    pop_small_bias=1,
    pop_big_bias=1,
    pop_decay=1,
    con_pow=1,
    final_sweep=True,
    seed=None,
):
    """
    Parameters
    ----------
    inputs
    output
    size_dict
    weight_nodes
    weight_edges
    parts
    maxiter
    memory
    pop_small_bias
    pop_big_bias
    pop_decay
    con_pow
    final_sweep
    """

    hg = HyperGraph(inputs, output, size_dict)
    n = hg.get_num_nodes()
    winfo = hg.compute_weights(
        weight_nodes=weight_nodes, weight_edges=weight_edges
    )

    sites = list(hg.nodes)
    neighbs = collections.defaultdict(set)
    max_edge_weight = max(winfo["edge_weights"])
    weights = {}

    # populate neighbor list and weights by edge weight
    for i in sites:
        for e in hg.get_node(i):
            for j in hg.get_edge(e):
                if j != i:
                    neighbs[i].add(j)
                    weights[i, j] = (
                        winfo["edge_weight_map"][e] / max_edge_weight
                    )

    # weight by mutual connectivity
    for i, j in weights:
        weights[i, j] *= (1 + len(neighbs[i] & neighbs[j])) ** con_pow

    labels = sites.copy()
    pops = collections.Counter(labels)

    rng = get_rng(seed)

    if maxiter is None:
        maxiter = n

    for r in range(maxiter):
        rng.shuffle(sites)
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
                    pop_fact(p, parts, n, pop_small_bias, pop_big_bias)
                ) / (r + 1) ** pop_decay

            new_label = scores.most_common(1)[0][0]

            labels[i] = new_label
            pops[old_label] -= 1
            pops[new_label] += 1

            all_static &= new_label == old_label

        if all_static:
            break

    if final_sweep:
        rng.shuffle(sites)
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
    name="labels",
    ssa_func=labels_to_tree.trial_fn,
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.01, "max": 1.0},
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "cutoff": {"type": "INT", "min": 10, "max": 40},
        "parts": {"type": "INT", "min": 1, "max": 16},
        "memory": {"type": "INT", "min": -2, "max": 1},
        "pop_small_bias": {"type": "FLOAT", "min": 0.0, "max": 2.0},
        "pop_big_bias": {"type": "FLOAT", "min": 0.0, "max": 2.0},
        "pop_decay": {"type": "FLOAT", "min": 0.0, "max": 10.0},
        "con_pow": {"type": "FLOAT", "min": 0.0, "max": 10.0},
        "final_sweep": {"type": "BOOL"},
    },
)


register_hyper_function(
    name="labels-agglom",
    ssa_func=labels_to_tree.trial_fn_agglom,
    space={
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "memory": {"type": "INT", "min": -2, "max": 1},
        "pop_small_bias": {"type": "FLOAT", "min": 0.0, "max": 2.0},
        "pop_big_bias": {"type": "FLOAT", "min": 0.0, "max": 2.0},
        "pop_decay": {"type": "FLOAT", "min": 0.0, "max": 10.0},
        "con_pow": {"type": "FLOAT", "min": 0.0, "max": 10.0},
        "final_sweep": {"type": "BOOL"},
    },
    constants={
        "random_strength": 0.0,
    },
)
