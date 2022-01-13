import math
import pytest
import opt_einsum as oe
from cotengra.core import ContractionTree


def test_contraction_tree_equivalency():
    eq = "a,ab,bc,c->"
    shapes = [(4,), (4, 2), (2, 5), (5,)]
    # optimal contraction is like:
    #    o
    #   / \
    #  o   o
    # / \ / \
    _, info1 = oe.contract_path(eq, *shapes, shapes=True,
                                optimize=[(0, 1), (0, 1), (0, 1)])
    _, info2 = oe.contract_path(eq, *shapes, shapes=True,
                                optimize=[(2, 3), (0, 1), (0, 1)])
    assert info1.contraction_list != info2.contraction_list
    ct1 = ContractionTree.from_info(info1, check=True)
    ct2 = ContractionTree.from_info(info2, check=True)
    assert ct1.total_flops() == ct2.total_flops() == 40
    assert ct1.children == ct2.children
    assert ct1.is_complete()
    assert ct2.is_complete()


@pytest.mark.parametrize(('forested', 'parallel', 'requires'), [
    (False, False, ''),
    (True, False, ''),
    (True, 'dask', 'distributed'),
    (True, 'ray', 'ray'),
])
def test_reconfigure(forested, parallel, requires):
    if requires:
        pytest.importorskip(requires)

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=3)

    info_gr = oe.contract_path(eq, *shapes, shapes=True, optimize='greedy')[1]
    tree_gr = ContractionTree.from_info(info_gr)

    assert tree_gr.total_flops() == info_gr.opt_cost

    if forested:
        tree_gr.subtree_reconfigure_forest_(
            num_trees=2, subtree_size=6, progbar=True, parallel=parallel)
    else:
        tree_gr.subtree_reconfigure_(progbar=True)

    assert tree_gr.total_flops() < info_gr.opt_cost

    info_tsr = oe.contract_path(
        eq, *shapes, shapes=True, optimize=tree_gr.get_path())[1]

    assert tree_gr.total_flops() == info_tsr.opt_cost


def test_reconfigure_with_n_smaller_than_subtree_size():
    eq, shapes = oe.helpers.rand_equation(10, 3)
    path, info = oe.contract_path(eq, *shapes, shapes=True)
    tree = ContractionTree.from_info(info)
    tree.subtree_reconfigure(12)


@pytest.mark.parametrize(('forested', 'parallel', 'requires'), [
    (False, False, ''),
    (True, False, ''),
    (True, True, 'distributed'),
])
def test_slice_and_reconfigure(forested, parallel, requires):
    if requires:
        pytest.importorskip(requires)

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info_gr = oe.contract_path(eq, *shapes, shapes=True, optimize='greedy')[1]
    tree_gr = ContractionTree.from_info(info_gr)

    target_size = tree_gr.max_size() // 32

    if forested:
        tree_gr.slice_and_reconfigure_forest_(
            target_size, num_trees=2, progbar=True, parallel=parallel)
    else:
        tree_gr.slice_and_reconfigure_(target_size, progbar=True)

    assert tree_gr.max_size() <= target_size


def test_plot():
    pytest.importorskip('matplotlib')

    import matplotlib
    matplotlib.use('Template')

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize='greedy')[1]
    tree = ContractionTree.from_info(info)

    tree.plot_ring()
    tree.plot_tent()
    tree.plot_contractions()


def test_plot_alt():
    pytest.importorskip('altair')

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize='greedy')[1]
    tree = ContractionTree.from_info(info)

    tree.plot_contractions_alt()


@pytest.mark.parametrize("optimize", ["greedy-compressed", "greedy-span"])
def test_compressed_rank(optimize):
    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize=optimize)[1]
    tree = ContractionTree.from_info(info)
    assert tree.max_size_compressed(1) < tree.max_size()
