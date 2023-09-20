import pytest
import cotengra as ctg


def test_contraction_tree_equivalency():
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    eq = "a,ab,bc,c->"
    shapes = [(4,), (4, 2), (2, 5), (5,)]
    # optimal contraction is like:
    #    o
    #   / \
    #  o   o
    # / \ / \
    _, info1 = oe.contract_path(
        eq, *shapes, shapes=True, optimize=[(0, 1), (0, 1), (0, 1)]
    )
    _, info2 = oe.contract_path(
        eq, *shapes, shapes=True, optimize=[(2, 3), (0, 1), (0, 1)]
    )
    assert info1.contraction_list != info2.contraction_list
    ct1 = ctg.ContractionTree.from_info(info1, check=True)
    ct2 = ctg.ContractionTree.from_info(info2, check=True)
    assert ct1.total_flops() == ct2.total_flops() == 20
    assert ct1.children == ct2.children
    assert ct1.is_complete()
    assert ct2.is_complete()


@pytest.mark.parametrize(
    ("forested", "parallel", "requires"),
    [
        (False, False, ""),
        (True, False, ""),
        (True, "dask", "distributed"),
        (True, "ray", "ray"),
    ],
)
def test_reconfigure(forested, parallel, requires):
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    if requires:
        pytest.importorskip(requires)

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=3)

    info_gr = oe.contract_path(eq, *shapes, shapes=True, optimize="greedy")[1]
    tree_gr = ctg.ContractionTree.from_info(info_gr)

    assert tree_gr.total_flops() == info_gr.opt_cost // 2

    if forested:
        tree_gr.subtree_reconfigure_forest_(
            num_trees=2, subtree_size=6, progbar=True, parallel=parallel
        )
    else:
        tree_gr.subtree_reconfigure_(progbar=True)

    assert tree_gr.total_flops() < info_gr.opt_cost // 2

    info_tsr = oe.contract_path(
        eq, *shapes, shapes=True, optimize=tree_gr.get_path()
    )[1]

    assert tree_gr.total_flops() == info_tsr.opt_cost // 2


def test_reconfigure_with_n_smaller_than_subtree_size():
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    eq, shapes = oe.helpers.rand_equation(10, 3)
    _, info = oe.contract_path(eq, *shapes, shapes=True)
    tree = ctg.ContractionTree.from_info(info)
    tree.subtree_reconfigure(12)


@pytest.mark.parametrize(
    ("forested", "parallel", "requires"),
    [
        (False, False, ""),
        (True, False, ""),
        (True, True, "distributed"),
    ],
)
def test_slice_and_reconfigure(forested, parallel, requires):
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    if requires:
        pytest.importorskip(requires)

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info_gr = oe.contract_path(eq, *shapes, shapes=True, optimize="greedy")[1]
    tree_gr = ctg.ContractionTree.from_info(info_gr)

    target_size = tree_gr.max_size() // 32

    if forested:
        tree_gr.slice_and_reconfigure_forest_(
            target_size, num_trees=2, progbar=True, parallel=parallel
        )
    else:
        tree_gr.slice_and_reconfigure_(target_size, progbar=True)

    assert tree_gr.max_size() <= target_size


def test_plot():
    pytest.importorskip("opt_einsum")
    pytest.importorskip("matplotlib")

    import opt_einsum as oe
    import matplotlib

    matplotlib.use("Template")

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize="greedy")[1]
    tree = ctg.ContractionTree.from_info(info)

    tree.plot_ring()
    tree.plot_tent()
    tree.plot_contractions()


def test_plot_alt():
    pytest.importorskip("opt_einsum")
    pytest.importorskip("altair")

    import opt_einsum as oe

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize="greedy")[1]
    tree = ctg.ContractionTree.from_info(info)

    tree.plot_contractions_alt()


@pytest.mark.parametrize("optimize", ["greedy-compressed", "greedy-span"])
def test_compressed_rank(optimize):
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=2)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize=optimize)[1]
    tree = ctg.ContractionTree.from_info(info)
    assert tree.max_size_compressed(1) < tree.max_size()


@pytest.mark.parametrize("seed", range(10))
def test_print_contractions(seed):
    ctg.utils.rand_tree(10, 3, 2, 2, 2).print_contractions()


def test_remove_ind():
    import copy

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        10, 3, n_out=0, n_hyper_in=4, n_hyper_out=1, seed=42
    )
    tree = ctg.array_contract_tree(
        inputs, output, size_dict, optimize="greedy"
    )
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)

    x = tree.contract(arrays)

    sf = ctg.SliceFinder(tree, target_slices=2)
    ix_sl, _ = sf.search()
    (ix,) = ix_sl

    orig_stats = tree.contract_stats()
    orig_info = copy.deepcopy(tree.info)

    tree_sliced = tree.remove_ind(ix, project=None)
    sliced_stats = tree_sliced.contract_stats()
    assert sliced_stats['flops'] > orig_stats['flops']
    # make sure we haven't mutated original tree
    assert tree.info == orig_info
    assert tree_sliced.info != orig_info
    assert tree_sliced.contract(arrays) == pytest.approx(x)

    tree_rem0 = tree.remove_ind(ix, project=0)
    rem_stats = tree_rem0.contract_stats()
    assert rem_stats['flops'] < orig_stats['flops']
    # make sure we haven't mutated original tree
    assert tree.info == orig_info
    assert tree_rem0.info != orig_info
    y0 = tree_rem0.contract(arrays)
    assert y0 != pytest.approx(x)

    for j in range(1, tree.size_dict[ix]):

        tree_remj = tree.remove_ind(ix, project=j)
        rem_stats = tree_remj.contract_stats()
        assert rem_stats['flops'] < orig_stats['flops']
        # make sure we haven't mutated original tree
        assert tree.info == orig_info
        assert tree_remj.info != orig_info
        yj = tree_remj.contract(arrays)
        assert yj != pytest.approx(x)
        y0 += yj

    assert y0 == pytest.approx(x)
