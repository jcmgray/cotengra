import random
from os.path import join, abspath, dirname

from .core import HyperGraph, PartitionTreeBuilder
from .hyper import register_hyper_function


# needed to supply kahypar profile files
KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'kahypar_profiles')


def kahypar_subgraph_find_membership(
    inputs,
    output,
    size_dict,
    weight_nodes='const',
    weight_edges='log',
    fuse_output_inds=False,
    parts=2,
    imbalance=0.01,
    seed=None,
    profile=None,
    mode='direct',
    objective='cut',
    quiet=True,
):
    import kahypar as kahypar

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    nv = len(inputs)
    if parts >= nv:
        return list(range(nv))

    HG = HyperGraph(inputs, output, size_dict,
                    weight_edges=weight_edges,
                    weight_nodes=weight_nodes,
                    fuse_output_inds=fuse_output_inds)
    hyperedge_indices, hyperedges = HG.to_sparse()

    hypergraph_kwargs = {
        'num_nodes': HG.num_nodes,
        'num_edges': HG.num_edges,
        'index_vector': hyperedge_indices,
        'edge_vector': hyperedges,
        'k': parts,
    }

    edge_weights, node_weights = {
        (False, False): (None, None),
        (False, True): ([], HG.node_weights),
        (True, False): (HG.edge_weights, []),
        (True, True): (HG.edge_weights, HG.node_weights),
    }[HG.has_edge_weights, HG.has_node_weights]

    if edge_weights or node_weights:
        hypergraph_kwargs['edge_weights'] = edge_weights
        hypergraph_kwargs['node_weights'] = node_weights

    hypergraph = kahypar.Hypergraph(**hypergraph_kwargs)

    if profile is None:
        profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"

    context = kahypar.Context()
    context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
    context.setK(parts)
    context.setSeed(seed)
    context.suppressOutput(quiet)
    context.setEpsilon(imbalance * parts)

    kahypar.partition(hypergraph, context)

    return [hypergraph.blockID(i) for i in hypergraph.nodes()]


kahypar_to_tree = PartitionTreeBuilder(kahypar_subgraph_find_membership)
trial_kahypar = kahypar_to_tree.trial_fn

register_hyper_function(
    name='kahypar',
    ssa_func=trial_kahypar,
    space={
        'random_strength': {'type': 'FLOAT_EXP', 'min': 0.01, 'max': 10.},
        'weight_edges': {'type': 'STRING', 'options': ['const', 'log']},
        'cutoff': {'type': 'INT', 'min': 10, 'max': 40},
        'imbalance': {'type': 'FLOAT', 'min': 0.01, 'max': 1.0},
        'imbalance_decay': {'type': 'FLOAT', 'min': -5, 'max': 5},
        'parts': {'type': 'INT', 'min': 2, 'max': 16},
        'parts_decay': {'type': 'FLOAT', 'min': 0.0, 'max': 1.0},
        'mode': {'type': 'STRING', 'options': ['direct', 'recursive']},
        'objective': {'type': 'STRING', 'options': ['cut', 'km1']},
    },
)

register_hyper_function(
    name='kahypar-balanced',
    ssa_func=trial_kahypar,
    space={
        'weight_edges': {'type': 'STRING', 'options': ['const', 'log']},
        'cutoff': {'type': 'INT', 'min': 2, 'max': 4},
        'imbalance': {'type': 'FLOAT', 'min': 0.001, 'max': 0.01},
        'mode': {'type': 'STRING', 'options': ['direct', 'recursive']},
        'objective': {'type': 'STRING', 'options': ['cut', 'km1']},
        'fuse_output_inds': {'type': 'BOOL'},
    },
    constants={
        'random_strength': 0.0,
        'imbalance_decay': 0.0,
        'parts': 2,
    }
)


register_hyper_function(
    name='kahypar-agglom',
    ssa_func=kahypar_to_tree.trial_fn_agglom,
    space={
        'weight_edges': {'type': 'STRING', 'options': ['const', 'log']},
        'imbalance': {'type': 'FLOAT', 'min': 0.001, 'max': 0.01},
        'mode': {'type': 'STRING', 'options': ['direct', 'recursive']},
        'objective': {'type': 'STRING', 'options': ['cut', 'km1']},
        'groupsize': {'type': 'INT', 'min': 2, 'max': 8},
        'fuse_output_inds': {'type': 'BOOL'},
    },
    constants={
        'random_strength': 0.0,
    }
)
