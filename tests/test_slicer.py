import pytest

import cotengra as ctg


def test_slicer():
    tree = ctg.utils.rand_tree(30, 5, seed=42, d_max=3)
    sf = ctg.SliceFinder(tree, target_size=100_000, target_overhead=None)
    inds, ccost = sf.search()
    assert tree.max_size() > 500_000
    assert ccost.size <= 100_000
    assert ccost.total_flops > tree.contraction_cost()
    assert len(inds) > 1


def test_plot():
    pytest.importorskip("matplotlib")

    import matplotlib

    matplotlib.use("Template")
    tree = ctg.utils.rand_tree(30, 5, seed=42, d_max=3)
    sf = ctg.SliceFinder(tree, target_size=100_000, target_overhead=None)
    sf.search()
    sf.plot_slicings()


def test_plot_alt():
    pytest.importorskip("altair")
    tree = ctg.utils.rand_tree(30, 5, seed=42, d_max=3)
    sf = ctg.SliceFinder(tree, target_size=100_000, target_overhead=None)
    sf.search()
    sf.plot_slicings_alt()
