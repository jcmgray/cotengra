"""igraph based pathfinders."""

import functools
from collections import defaultdict

from ..core import (
    ContractionTree,
    PartitionTreeBuilder,
    jitter_dict,
)
from ..hypergraph import (
    calc_edge_weight_float,
    calc_node_weight_float,
)
from ..hyperoptimizers.hyper import register_hyper_function
from ..utils import get_rng


def oe_to_igraph(
    inputs, output, size_dict, weight_nodes="const", weight_edges="log"
):
    """Convert opt_einsum format to igraph graph incl. weights."""
    import igraph as ig

    G = ig.Graph()

    # which positions each edge links
    ind2terms = defaultdict(list)
    for i, term in enumerate(inputs):
        nweight = calc_node_weight_float(term, size_dict, weight_nodes)
        G.add_vertex(str(i), weight=nweight)
        for ix in term:
            if ix not in output:
                ind2terms[ix].append(str(i))

    for ix, enodes in ind2terms.items():
        if len(enodes) != 2:
            continue

        eweight = calc_edge_weight_float(ix, size_dict, weight_edges)
        G.add_edge(*enodes, ind=ix, weight=eweight)

    return G


def igraph_subgraph_find_membership(
    inputs,
    output,
    size_dict,
    weight_nodes="const",
    weight_edges="log",
    method="spinglass",
    parts=2,
    seed=None,
    **igraph_opts,
):
    G = oe_to_igraph(inputs, output, size_dict, weight_nodes, weight_edges)

    # first check for disconnected components
    components = G.components()
    if len(components) > 1:
        return components.membership

    weight_lbl = "weight" if weight_edges != "const" else None
    nd_weight_lbl = "weight" if weight_nodes != "const" else None

    if method == "spinglass":
        igraph_opts.setdefault("spins", parts)
        clustering = G.community_spinglass(weight_lbl, **igraph_opts)

    elif method == "infomap":
        clustering = G.community_infomap(
            weight_lbl, nd_weight_lbl, **igraph_opts
        )

    elif method == "label_propagation":
        rng = get_rng(seed)

        initial = [rng.choice(range(parts)) for _ in range(len(G.vs))]
        clustering = G.community_label_propagation(
            weight_lbl, initial=initial, **igraph_opts
        )

    elif method == "multilevel":
        clustering = G.community_multilevel(weight_lbl, **igraph_opts)

    elif method == "leading_eigenvector":
        igraph_opts.setdefault("clusters", parts)
        clustering = G.community_leading_eigenvector(
            weights=weight_lbl, **igraph_opts
        )

    return clustering.membership


igraph_to_tree = PartitionTreeBuilder(igraph_subgraph_find_membership)
trial_igraph_partition = igraph_to_tree.trial_fn


def trial_igraph_dendrogram(
    inputs,
    output,
    size_dict,
    weight_nodes="const",
    weight_edges="log",
    random_strength=0.1,
    method="betweenness",
    **kwargs,
):
    """A single, repeatable, igraph trial run. This is for igraph methods that
    naturally produce a dendrogram (== ssa_path).
    """
    G = oe_to_igraph(
        inputs,
        output,
        size_dict=jitter_dict(size_dict, random_strength),
        weight_nodes=weight_nodes,
        weight_edges=weight_edges,
    )

    if weight_edges != "const":
        kwargs.setdefault("weights", "weight")

    if method == "betweenness":
        kwargs.setdefault("clusters", 2)
        kwargs.setdefault("directed", False)
        dendrogram = G.community_edge_betweenness(**kwargs)

    elif method == "walktrap":
        kwargs.setdefault("steps", 100)
        dendrogram = G.community_walktrap(**kwargs)

    elif method == "fastgreedy":
        dendrogram = G.community_fastgreedy(**kwargs)

    else:
        raise ValueError("Invalid method: '{}'.".format(method))

    ssa_path = dendrogram.merges
    return ContractionTree.from_path(
        inputs, output, size_dict, ssa_path=ssa_path, autocomplete=True
    )


# ----------------------------- HYPER REGISTERS ----------------------------- #


register_hyper_function(
    name="walktrap",
    ssa_func=functools.partial(trial_igraph_dendrogram, method="walktrap"),
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.01, "max": 10.0},
        "steps": {"type": "INT", "min": 4, "max": 200},
    },
)

register_hyper_function(
    name="betweenness",
    ssa_func=functools.partial(trial_igraph_dendrogram, method="betweenness"),
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.01, "max": 10.0},
    },
)


def trial_spinglass(
    inputs, output, size_dict, icool_fact=0.01, igamma=0.01, **kwargs
):
    return trial_igraph_partition(
        inputs,
        output,
        size_dict,
        method="spinglass",
        gamma=(1 - igamma),
        cool_fact=(1 - icool_fact),
        **kwargs,
    )


register_hyper_function(
    name="spinglass",
    ssa_func=trial_spinglass,
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.001, "max": 1.0},
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "start_temp": {"type": "FLOAT_EXP", "min": 0.5, "max": 5.0},
        "stop_temp": {"type": "FLOAT_EXP", "min": 0.001, "max": 0.2},
        "icool_fact": {"type": "FLOAT_EXP", "min": 0.001, "max": 0.05},
        "update_rule": {"type": "STRING", "options": ["config", "simple"]},
        "igamma": {"type": "FLOAT_EXP", "min": 0.001, "max": 0.1},
        "cutoff": {"type": "INT", "min": 10, "max": 40},
        "parts": {"type": "INT", "min": 2, "max": 16},
        "parts_decay": {"type": "FLOAT", "min": 0.0, "max": 1.0},
    },
)


register_hyper_function(
    name="labelprop",
    ssa_func=functools.partial(
        trial_igraph_partition, method="label_propagation"
    ),
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.01, "max": 10.0},
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "cutoff": {"type": "INT", "min": 10, "max": 40},
        "parts": {"type": "INT", "min": 2, "max": 16},
        "parts_decay": {"type": "FLOAT", "min": 0.0, "max": 1.0},
    },
)
