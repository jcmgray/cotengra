import functools

from opt_einsum.paths import ssa_greedy_optimize, register_path_fn
from opt_einsum.path_random import thermal_chooser

from .core import (
    jitter_dict,
    ContractionTree,
)

from .hyper import register_hyper_function

# ------------------------------ GREEDY HYPER ------------------------------- #


def cost_memory_removed_mod(size12, size1, size2, k12, k1, k2, costmod=1):
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - costmod * (size1 + size2)


def trial_greedy(inputs, output, size_dict,
                 random_strength=0.1,
                 temperature=1.0,
                 rel_temperature=True,
                 costmod=1):

    rand_size_dict = jitter_dict(size_dict, random_strength)

    cost_fn = functools.partial(cost_memory_removed_mod, costmod=costmod)
    choose_fn = functools.partial(thermal_chooser, temperature=temperature,
                                  rel_temperature=rel_temperature)

    ssa_path = ssa_greedy_optimize(inputs, output, rand_size_dict,
                                   choose_fn=choose_fn, cost_fn=cost_fn)

    return ContractionTree.from_path(inputs, output, size_dict,
                                     ssa_path=ssa_path)


register_hyper_function(
    name='greedy',
    ssa_func=trial_greedy,
    space={
        'random_strength': {'type': 'FLOAT_EXP', 'min': 0.01, 'max': 10.},
        'temperature': {'type': 'FLOAT_EXP', 'min': 0.01, 'max': 10.},
        'rel_temperature': {'type': 'BOOL'},
        'costmod': {'type': 'FLOAT', 'min': 0.0, 'max': 2.0},
    },
)


# --------------------------------------------------------------------------- #

def greconf_rf(inputs, output, size_dict, memory_limit=None):
    """Greedy-reconf path -- find a single greedy path then perform a round of
    cheap subtree reconfigurations to optimize it.
    """
    ssa_path = ssa_greedy_optimize(inputs, output, size_dict)
    tree = ContractionTree.from_path(
        inputs, output, size_dict, ssa_path=ssa_path)
    tree.subtree_reconfigure_(subtree_size=6, minimize='flops')
    return tree.path()


register_path_fn('greedy-rf', greconf_rf)


def greconf_rw(inputs, output, size_dict, memory_limit=None):
    """Greedy-reconf path -- find a single greedy path then perform a round of
    cheap subtree reconfigurations to optimize it.
    """
    ssa_path = ssa_greedy_optimize(inputs, output, size_dict)
    tree = ContractionTree.from_path(
        inputs, output, size_dict, ssa_path=ssa_path)
    tree.subtree_reconfigure_(subtree_size=6, minimize='write')
    return tree.path()


register_path_fn('greedy-rw', greconf_rw)


def greconf_rc(inputs, output, size_dict, memory_limit=None):
    """Greedy-reconf path -- find a single greedy path then perform a round of
    cheap subtree reconfigurations to optimize it.
    """
    ssa_path = ssa_greedy_optimize(inputs, output, size_dict)
    tree = ContractionTree.from_path(
        inputs, output, size_dict, ssa_path=ssa_path)
    tree.subtree_reconfigure_(subtree_size=6, minimize='combo')
    return tree.path()


register_path_fn('greedy-rc', greconf_rc)
