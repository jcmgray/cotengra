"""Contraction tree finders using kahypar hypergraph partitioning."""

import functools
import itertools
from os.path import abspath, dirname, join

from ..core import PartitionTreeBuilder, get_hypergraph
from ..hyperoptimizers.hyper import register_hyper_function
from ..utils import get_rng


@functools.lru_cache(1)
def get_kahypar_profile_dir():
    # needed to supply kahypar profile files
    # if kahypar is built from source, the version number may not match the
    # <major>.<minor>.<patch> format; rather than assuming the format, add
    # a fallback option for unrecognized versions
    import re

    import kahypar

    m = re.compile(r"(\d+)\.(\d+)\.(\d+)").match(kahypar.__version__)
    path_components = [abspath(dirname(__file__)), "kahypar_profiles"]

    if m is not None:
        version = tuple(map(int, m.groups()))
        if version <= (1, 1, 6):
            path_components.append("old")

    return join(*path_components)


def to_sparse(hg, weight_nodes="const", weight_edges="log"):
    winfo = hg.compute_weights(
        weight_nodes=weight_nodes, weight_edges=weight_edges
    )

    hyperedge_indices = []
    hyperedges = []
    for e in winfo["edge_list"]:
        hyperedge_indices.append(len(hyperedges))
        hyperedges.extend(hg.edges[e])
    hyperedge_indices.append(len(hyperedges))

    winfo["hyperedge_indices"] = hyperedge_indices
    winfo["hyperedges"] = hyperedges
    return winfo


def kahypar_subgraph_find_membership(
    inputs,
    output,
    size_dict,
    weight_nodes="const",
    weight_edges="log",
    fix_output_nodes=False,
    parts=2,
    imbalance=0.01,
    compress=0,
    seed=None,
    profile=None,
    mode="direct",
    objective="cut",
    quiet=True,
):
    import kahypar as kahypar

    rng = get_rng(seed)
    seed = rng.randint(0, 2**31 - 1)

    nv = len(inputs)
    if parts >= nv:
        return list(range(nv))

    hg = get_hypergraph(inputs, output, size_dict, accel=False)

    if fix_output_nodes:
        # make sure all the output nodes (those with output indices) are in
        # the same partition. Need to do this before removing danglers
        onodes = tuple(hg.output_nodes())

        if parts >= nv - len(onodes) + 1:
            # too many partitions, simply group all outputs and return
            groups = itertools.count(1)
            return [0 if i in onodes else next(groups) for i in range(nv)]

    for e, nodes in tuple(hg.edges.items()):
        if len(nodes) == 1:
            hg.remove_edge(e)

    if hg.num_edges == 0:
        # completely disconnected graph -> kahypar will crash
        # round robin partition instead
        return [
            i
            for k in range(parts)
            for i in
            (k,) * ((nv // parts) + (k < (nv % parts)))
        ]

    if compress:
        hg.compress(compress)

    winfo = to_sparse(hg, weight_nodes=weight_nodes, weight_edges=weight_edges)

    hypergraph_kwargs = {
        "num_nodes": hg.get_num_nodes(),
        "num_edges": hg.get_num_edges(),
        "index_vector": winfo["hyperedge_indices"],
        "edge_vector": winfo["hyperedges"],
        "k": parts,
    }

    edge_weights, node_weights = {
        (False, False): (None, None),
        (False, True): ([], winfo["node_weights"]),
        (True, False): (winfo["edge_weights"], []),
        (True, True): (winfo["edge_weights"], winfo["node_weights"]),
    }[winfo["has_edge_weights"], winfo["has_node_weights"]]

    if edge_weights or node_weights:
        hypergraph_kwargs["edge_weights"] = edge_weights
        hypergraph_kwargs["node_weights"] = node_weights

    hypergraph = kahypar.Hypergraph(**hypergraph_kwargs)

    if fix_output_nodes:
        for i in onodes:
            hypergraph.fixNodeToBlock(i, 0)

        # silences various warnings
        mode = "recursive"

    if profile is None:
        profile_mode = {"direct": "k", "recursive": "r"}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"

    context = kahypar.Context()
    context.loadINIconfiguration(join(get_kahypar_profile_dir(), profile))
    context.setK(parts)
    context.setSeed(seed)
    context.suppressOutput(quiet)
    context.setEpsilon(imbalance * parts)

    kahypar.partition(hypergraph, context)

    return [hypergraph.blockID(i) for i in hypergraph.nodes()]


kahypar_to_tree = PartitionTreeBuilder(kahypar_subgraph_find_membership)

register_hyper_function(
    name="kahypar",
    ssa_func=kahypar_to_tree.trial_fn,
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.01, "max": 10.0},
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "cutoff": {"type": "INT", "min": 10, "max": 40},
        "imbalance": {"type": "FLOAT", "min": 0.01, "max": 1.0},
        "imbalance_decay": {"type": "FLOAT", "min": -5, "max": 5},
        "parts": {"type": "INT", "min": 2, "max": 16},
        "parts_decay": {"type": "FLOAT", "min": 0.0, "max": 1.0},
        "mode": {"type": "STRING", "options": ["direct", "recursive"]},
        "objective": {"type": "STRING", "options": ["cut", "km1"]},
        "fix_output_nodes": {"type": "STRING", "options": ["auto", ""]},
    },
)

register_hyper_function(
    name="kahypar-balanced",
    ssa_func=kahypar_to_tree.trial_fn,
    space={
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "cutoff": {"type": "INT", "min": 2, "max": 4},
        "imbalance": {"type": "FLOAT", "min": 0.001, "max": 0.01},
        "mode": {"type": "STRING", "options": ["direct", "recursive"]},
        "objective": {"type": "STRING", "options": ["cut", "km1"]},
        "fix_output_nodes": {"type": "STRING", "options": ["auto", ""]},
    },
    constants={
        "random_strength": 0.0,
        "imbalance_decay": 0.0,
        "parts": 2,
    },
)


register_hyper_function(
    name="kahypar-agglom",
    ssa_func=kahypar_to_tree.trial_fn_agglom,
    space={
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "imbalance": {"type": "FLOAT", "min": 0.001, "max": 0.05},
        "mode": {"type": "STRING", "options": ["direct", "recursive"]},
        "objective": {"type": "STRING", "options": ["cut", "km1"]},
        "groupsize": {"type": "INT", "min": 2, "max": 64},
        "fix_output_nodes": {"type": "STRING", "options": ["auto", ""]},
        "compress": {"type": "STRING", "options": [0, 3, 10, 30, 100]},
        "sub_optimize": {
            "type": "STRING",
            "options": ["greedy", "greedy-compressed"],
        },
    },
    constants={
        "random_strength": 0.0,
    },
)
