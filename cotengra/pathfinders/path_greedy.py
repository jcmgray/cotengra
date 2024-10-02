import functools

from ..core import ContractionTree, jitter_dict
from ..hyperoptimizers.hyper import register_hyper_function
from .path_basic import get_optimize_greedy

ssa_greedy_optimize = functools.partial(get_optimize_greedy(), use_ssa=True)

# ------------------------------ GREEDY HYPER ------------------------------- #


def trial_greedy(
    inputs,
    output,
    size_dict,
    random_strength=0.0,
    temperature=0.0,
    costmod=1.0,
):
    if random_strength != 0.0:
        # don't supply randomized sizes to actual contraction tree
        greedy_size_dict = jitter_dict(size_dict, random_strength)
    else:
        greedy_size_dict = size_dict

    ssa_path = ssa_greedy_optimize(
        inputs,
        output,
        greedy_size_dict,
        temperature=temperature,
        costmod=costmod,
    )

    return ContractionTree.from_path(
        inputs, output, size_dict, ssa_path=ssa_path
    )


register_hyper_function(
    name="greedy",
    ssa_func=trial_greedy,
    space={
        "random_strength": {"type": "FLOAT_EXP", "min": 0.001, "max": 1.0},
        "temperature": {"type": "FLOAT_EXP", "min": 0.001, "max": 1.0},
        "costmod": {"type": "FLOAT", "min": 0.1, "max": 4.0},
    },
)

# greedy but less exploratative -> better for a small number of runs
# note this hyper driver is slightly different from overall preset
# "random-greedy" which doesn't use the hyper framework
register_hyper_function(
    name="random-greedy",
    ssa_func=trial_greedy,
    space={
        "temperature": {"type": "FLOAT_EXP", "min": 0.001, "max": 0.1},
        "costmod": {"type": "FLOAT", "min": 0.5, "max": 3.0},
    },
    constants={
        "random_strength": 0.0,
    },
)
