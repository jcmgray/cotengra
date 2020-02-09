import functools

from opt_einsum.paths import register_path_fn

from . import path_kahypar
from . import path_igraph

from .core import ContractionTree
from .slicer import SliceFinder, SlicedContractor
from .path_quickbb import QuickBBOptimizer, optimize_quickbb
from .path_flowcutter import FlowCutterOptimizer, optimize_flowcutter
from .hyper import HyperOptimizer, hyper_optimize, list_hyper_functions
from .plot import (
    plot_trials,
    plot_trials_alt,
    plot_scatter,
    plot_scatter_alt,
    plot_tree,
    plot_tree_ring,
    plot_tree_tent,
    plot_contractions,
    plot_contractions_alt,
    plot_slicings,
    plot_slicings_alt,
)

UniformOptimizer = functools.partial(HyperOptimizer, tuner='Uniform')
"""Does no gaussian process tuning by default, just randomly samples.
"""

UniformGreedy = functools.partial(UniformOptimizer, methods=['greedy'])
UniformKaHyPar = functools.partial(UniformOptimizer, methods=['kahypar'])
UniformBetweenness = functools.partial(UniformOptimizer,
                                       methods=['betweenness'])
UniformSpinglass = functools.partial(UniformOptimizer, methods=['spinglass'])


__all__ = (
    "path_kahypar",
    "path_igraph",
    "ContractionTree",
    "SliceFinder",
    "SlicedContractor",
    "QuickBBOptimizer",
    "optimize_quickbb",
    "FlowCutterOptimizer",
    "optimize_flowcutter",
    "HyperOptimizer",
    "hyper_optimize",
    "list_hyper_functions",
    "UniformOptimizer",
    "UniformGreedy",
    "UniformKaHyPar",
    "UniformBetweenness",
    "UniformSpinglass",
    "plot_trials",
    "plot_trials_alt",
    "plot_scatter",
    "plot_scatter_alt",
    "plot_tree",
    "plot_tree_ring",
    "plot_tree_tent",
    "plot_contractions",
    "plot_contractions_alt",
    "plot_slicings",
    "plot_slicings_alt",
)


# add some defaults to opt_einsum

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


# monkey patch plot methods

HyperOptimizer.plot_trials = plot_trials
HyperOptimizer.plot_trials_alt = plot_trials_alt
HyperOptimizer.plot_scatter = plot_scatter
HyperOptimizer.plot_scatter_alt = plot_scatter_alt
ContractionTree.plot_ring = plot_tree_ring
ContractionTree.plot_tent = plot_tree_tent
ContractionTree.plot_contractions = plot_contractions
ContractionTree.plot_contractions_alt = plot_contractions_alt
SliceFinder.plot_slicings = plot_slicings
SliceFinder.plot_slicings_alt = plot_slicings_alt
