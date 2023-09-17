"""Preset configured optimizers.
"""
from .core import ContractionTree
from .hyperoptimizers.hyper import HyperOptimizer, ReusableHyperOptimizer
from .oe import (
    PathOptimizer,
)
from .pathfinders.path_basic import (
    GreedyOptimizer,
    OptimalOptimizer,
    get_optimize_optimal,
)
from .interface import register_preset
from .hyperoptimizers.hyper import get_default_hq_methods


def estimate_optimal_hardness(inputs):
    """Provides a very rough estimate of how long it would take to find the
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

        hyperoptimizer_kwargs.setdefault("methods", ("rgreedy",))
        hyperoptimizer_kwargs.setdefault("max_repeats", 128)
        hyperoptimizer_kwargs.setdefault("max_time", "rate:1e9")
        hyperoptimizer_kwargs.setdefault("parallel", False)
        hyperoptimizer_kwargs.setdefault("reconf_opts", {})
        hyperoptimizer_kwargs["reconf_opts"].setdefault("subtree_size", 4)
        hyperoptimizer_kwargs["reconf_opts"].setdefault("maxiter", 100)

        if cache:
            self._optimizer_hyper = ReusableHyperOptimizer(
                minimize=minimize, **hyperoptimizer_kwargs
            )
        else:
            self._optimizer_hyper = HyperOptimizer(
                minimize=minimize, **hyperoptimizer_kwargs
            )

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
            return self._optimizer_hyper.search(
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
            return self._optimizer_hyper(inputs, output, size_dict, **kwargs)


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
        kwargs.setdefault("parallel", False)
        kwargs.setdefault("reconf_opts", {})
        kwargs["reconf_opts"].setdefault("subtree_size", 8)
        kwargs["reconf_opts"].setdefault("maxiter", 500)
        super().__init__(**kwargs)


auto_optimize = AutoOptimizer()
auto_hq_optimize = AutoHQOptimizer()
greedy_optimize = GreedyOptimizer()
optimal_optimize = OptimalOptimizer()
optimal_outer_optimize = OptimalOptimizer(search_outer=True)


# these names overlap with opt_einsum, but won't override presets there
register_preset("auto", auto_optimize)
register_preset("auto-hq", auto_hq_optimize)
register_preset("greedy", greedy_optimize)
register_preset("eager", greedy_optimize)
register_preset("opportunistic", greedy_optimize)
register_preset("optimal", optimal_optimize)
register_preset("dp", optimal_optimize)
register_preset("dynamic-programming", optimal_optimize)
register_preset("optimal-outer", optimal_outer_optimize)
