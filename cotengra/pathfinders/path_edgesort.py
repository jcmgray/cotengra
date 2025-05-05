from ..core import ContractionTree
from ..oe import PathOptimizer


class EdgeSortOptimizer(PathOptimizer):
    """A path optimizer that proceeds by contacting edges in sorted order, as
    is the default in the original `ncon` function. The path is thus specified
    entirely by the naming of the edges.

    Parameters
    ----------
    reverse : bool, optional
        If True, the edges are sorted in reverse order. The default is False.
    """

    def __init__(self, reverse=False):
        self.reverse = reverse

    def search(self, inputs, output, size_dict, **kwargs):
        edge_path = sorted(
            (ix for ix in size_dict if ix not in output), reverse=self.reverse
        )

        return ContractionTree.from_path(
            inputs,
            output,
            size_dict,
            edge_path=edge_path,
            **kwargs,
        )

    def __call__(self, inputs, output, size_dict, **kwargs):
        tree = self.search(
            inputs,
            output,
            size_dict,
            **kwargs,
        )
        return tree.get_path()
