"""Purely random pathfinder, for initialization and testing purposes."""

from ..core import ContractionTree
from ..hyperoptimizers.hyper import register_hyper_function
from ..interface import register_preset
from ..oe import PathOptimizer
from ..utils import get_rng


class RandomOptimizer(PathOptimizer):
    """A fully random pathfinder, that randomly selects pairs of tensors to
    contract (even if they are not connected). This is useful for testing
    purposes, and as a baseline for comparison with other pathfinders.

    Parameters
    ----------
    seed : None, int or np.random.Generator, optional
        Random seed. If None, a random seed is selected. Default is None.
    """

    def __init__(self, seed=None):
        self.rng = get_rng(seed)

    def __call__(self, inputs, outputs, size_dict):
        N = len(inputs)
        path = []
        for Nrem in range(N - 1, 0, -1):
            i = j = self.rng.randint(0, Nrem)
            while j == i:
                j = self.rng.randint(0, Nrem)
            if i > i:
                i, j = j, i
            path.append((i, j))
        return path

    def search(self, inputs, outputs, size_dict):
        return ContractionTree.from_path(
            inputs,
            outputs,
            size_dict,
            path=self(inputs, outputs, size_dict),
        )


register_preset("random", RandomOptimizer())
register_hyper_function("random", RandomOptimizer().search, {})
