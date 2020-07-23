import functools
import itertools

from opt_einsum.paths import register_path_fn

from .core import ContractionTree
from .slicer import SliceFinder, SlicedContractor

from . import path_greedy
from . import path_igraph
from . import path_kahypar
from . import path_labels

from .path_quickbb import QuickBBOptimizer, optimize_quickbb
from .path_flowcutter import FlowCutterOptimizer, optimize_flowcutter

from . import hyper_baytune
from . import hyper_choco
from . import hyper_nevergrad
from . import hyper_skopt
from . import hyper_random

from .hyper import list_hyper_functions, HyperOptimizer, get_hyper_space


from .plot import (
    plot_contractions,
    plot_contractions_alt,
    plot_scatter,
    plot_scatter_alt,
    plot_slicings,
    plot_slicings_alt,
    plot_tree,
    plot_tree_ring,
    plot_tree_tent,
    plot_trials,
    plot_trials_alt,
)


UniformOptimizer = functools.partial(HyperOptimizer, optlib='random')
"""Does no gaussian process tuning by default, just randomly samples - requires
no optimization library.
"""

QuasiRandOptimizer = functools.partial(
    HyperOptimizer, optlib='chocolate', sampler='QuasiRandom')
"""Does no gaussian process tuning by default, just randomly samples but in a
more 'even' way than purely random - requires ``chocolate``.
"""


__all__ = (
    "ContractionTree",
    "FlowCutterOptimizer",
    "get_hyper_space",
    "hyper_baytune",
    "hyper_choco",
    "hyper_nevergrad",
    "hyper_optimize",
    "hyper_random",
    "hyper_skopt",
    "HyperOptimizer",
    "list_hyper_functions",
    "optimize_flowcutter",
    "optimize_quickbb",
    "path_greedy",
    "path_igraph",
    "path_kahypar",
    "path_labels",
    "plot_contractions",
    "plot_contractions_alt",
    "plot_scatter",
    "plot_scatter_alt",
    "plot_slicings",
    "plot_slicings_alt",
    "plot_tree",
    "plot_tree_ring",
    "plot_tree_tent",
    "plot_trials",
    "plot_trials_alt",
    "QuasiRandOptimizer",
    "QuickBBOptimizer",
    "SlicedContractor",
    "SliceFinder",
    "UniformOptimizer",
)


# add some defaults to opt_einsum

def hyper_optimize(inputs, output, size_dict, memory_limit=None, **opts):
    optimizer = HyperOptimizer(**opts)
    return optimizer(inputs, output, size_dict, memory_limit)


named_pars = ('', '-par', '-seq')
named_meths = ('g', 'k', 'l', 'gk', 'gl')
named_reps = (32, 64, 128, 256)
named_reconfs = ('', '-rf', '-rw', '-rc')
named_progbars = ('', '-v')

for par, meths, rep, rec, progbar in itertools.product(
    named_pars, named_meths, named_reps, named_reconfs, named_progbars
):
    methods = ()
    if 'g' in meths:
        methods += ('greedy',)
    if 'k' in meths:
        methods += ('kahypar',)
    if 'l' in meths:
        methods += ('labels',)

    parallel = {'': 'auto', '-seq': False, '-par': True}[par]
    reconf_opts = {
        '': None,
        '-rf': dict(minimize='flops'),
        '-rw': dict(minimize='write'),
        '-rc': dict(minimize='combo'),
    }[rec]

    register_path_fn(
        f'hyp-{meths}{rec}{par}-{rep}{progbar}',
        functools.partial(
            hyper_optimize,
            methods=methods,
            parallel=parallel,
            reconf_opts=reconf_opts,
            max_repeats=rep,
            progbar=bool(progbar),
        )
    )


register_path_fn(
    'hyper',
    hyper_optimize,
)
register_path_fn(
    'hyper-256',
    functools.partial(hyper_optimize, max_repeats=256),
)
register_path_fn(
    'hyper-greedy',
    functools.partial(hyper_optimize, methods=['greedy']),
)
register_path_fn(
    'hyper-labels',
    functools.partial(hyper_optimize, methods=['labels']),
)
register_path_fn(
    'hyper-kahypar',
    functools.partial(hyper_optimize, methods=['kahypar']),
)
register_path_fn(
    'hyper-spinglass',
    functools.partial(hyper_optimize, methods=['spinglass']),
)
register_path_fn(
    'hyper-betweenness',
    functools.partial(hyper_optimize, methods=['betweenness']),
)
register_path_fn(
    'flowcutter-2',
    functools.partial(optimize_flowcutter, max_time=2),
)
register_path_fn(
    'flowcutter-10',
    functools.partial(optimize_flowcutter, max_time=10),
)
register_path_fn(
    'flowcutter-60',
    functools.partial(optimize_flowcutter, max_time=60),
)
register_path_fn(
    'quickbb-2',
    functools.partial(optimize_quickbb, max_time=2),
)
register_path_fn(
    'quickbb-10',
    functools.partial(optimize_quickbb, max_time=10),
)
register_path_fn(
    'quickbb-60',
    functools.partial(optimize_quickbb, max_time=60),
)
