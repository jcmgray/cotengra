"""Preset configured optimizers."""

import threading

from .core import ContractionTree
from .hyperoptimizers.hyper import (
    HyperOptimizer,
    ReusableHyperOptimizer,
    get_default_hq_methods,
    get_default_optlib,
    get_default_optlib_eco,
)
from .interface import register_preset
from .oe import (
    PathOptimizer,
)
from .pathfinders.path_basic import (
    GreedyOptimizer,
    OptimalOptimizer,
    get_optimize_optimal,
)
from .pathfinders.path_edgesort import EdgeSortOptimizer


def estimate_optimal_hardness(inputs):
    r"""Provides a very rough estimate of how long it would take to find the
    optimal contraction order for a given set of inputs. The runtime is
    *very* approximately exponential in this number:

    .. math::

        T \propto \exp {n^2 * k^0.5}

    Where :math:`n` is the number of tensors and :math:`k` is the average
    degree of the hypergraph.
    """
    n = len(inputs)
    # average degree
    k = sum(map(len, inputs)) / n
    return n**2 * k**0.5


class AutoOptimizer(PathOptimizer):
    """An optimizer that automatically chooses between optimal and
    hyper-optimization, designed for everyday use.
    """

    def __init__(
        self,
        optimal_cutoff=250,
        minimize="combo",
        cache=True,
        **hyperoptimizer_kwargs,
    ):
        self.minimize = minimize
        self.optimal_cutoff = optimal_cutoff
        self._optimize_optimal_fn = get_optimize_optimal()

        self.kwargs = hyperoptimizer_kwargs
        self.kwargs.setdefault("methods", ("random-greedy",))
        self.kwargs.setdefault("max_repeats", 128)
        self.kwargs.setdefault("max_time", "rate:1e9")
        self.kwargs.setdefault("optlib", get_default_optlib_eco())
        self.kwargs.setdefault("parallel", False)
        self.kwargs.setdefault("reconf_opts", {})
        self.kwargs["reconf_opts"].setdefault("subtree_size", 4)
        self.kwargs["reconf_opts"].setdefault("maxiter", 100)

        self._hyperoptimizers_by_thread = {}
        if cache:
            self._optimizer_hyper_cls = ReusableHyperOptimizer
        else:
            self._optimizer_hyper_cls = HyperOptimizer

    def _get_optimizer_hyper_threadsafe(self):
        # since the hyperoptimizer is stateful while running,
        # we need to instantiate a separate one for each thread
        tid = threading.get_ident()
        try:
            return self._hyperoptimizers_by_thread[tid]
        except KeyError:
            opt = self._optimizer_hyper_cls(
                minimize=self.minimize, **self.kwargs
            )
            self._hyperoptimizers_by_thread[tid] = opt
            return opt

    def search(self, inputs, output, size_dict, **kwargs):
        if estimate_optimal_hardness(inputs) < self.optimal_cutoff:
            # easy to solve exactly
            ssa_path = self._optimize_optimal_fn(
                inputs,
                output,
                size_dict,
                use_ssa=True,
                minimize=self.minimize,
                **kwargs,
            )
            return ContractionTree.from_path(
                inputs,
                output,
                size_dict,
                ssa_path=ssa_path,
            )
        else:
            # use hyperoptimizer
            return self._get_optimizer_hyper_threadsafe().search(
                inputs,
                output,
                size_dict,
                **kwargs,
            )

    def __call__(self, inputs, output, size_dict, **kwargs):
        if estimate_optimal_hardness(inputs) < self.optimal_cutoff:
            # easy to solve exactly
            return self._optimize_optimal_fn(
                inputs,
                output,
                size_dict,
                use_ssa=False,
                minimize=self.minimize,
                **kwargs,
            )
        else:
            # use hyperoptimizer
            return self._get_optimizer_hyper_threadsafe()(
                inputs, output, size_dict, **kwargs
            )


class AutoHQOptimizer(AutoOptimizer):
    """An optimizer that automatically chooses between optimal and
    hyper-optimization, designed for everyday use on harder contractions or
    those that will be repeated many times, and thus warrant a more extensive
    search.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("optimal_cutoff", 650)
        kwargs.setdefault("methods", get_default_hq_methods())
        kwargs.setdefault("max_repeats", 128)
        kwargs.setdefault("max_time", "rate:1e8")
        kwargs.setdefault("optlib", get_default_optlib())
        kwargs.setdefault("parallel", False)
        kwargs.setdefault("reconf_opts", {})
        kwargs["reconf_opts"].setdefault("subtree_size", 8)
        kwargs["reconf_opts"].setdefault("maxiter", 500)
        super().__init__(**kwargs)


auto_optimize = AutoOptimizer()
auto_hq_optimize = AutoHQOptimizer()

# these names overlap with opt_einsum, but won't override presets there
register_preset(
    "auto",
    auto_optimize,
    auto_optimize.search,
)
register_preset(
    "auto-hq",
    auto_hq_optimize,
    auto_optimize.search,
)

greedy_optimize = GreedyOptimizer()

register_preset(
    ["greedy", "eager", "opportunistic"],
    greedy_optimize,
    greedy_optimize.search,
)

optimal_optimize = OptimalOptimizer()

register_preset(
    ["optimal", "dp", "dynamic-programming"],
    optimal_optimize, optimal_optimize.search
)

optimal_outer_optimize = OptimalOptimizer(search_outer=True)

register_preset(
    "optimal-outer", optimal_outer_optimize, optimal_outer_optimize.search
)

edgesort_optimize = EdgeSortOptimizer()

register_preset(
    ["edgesort", "ncon"],
    edgesort_optimize,
    edgesort_optimize.search,
)
