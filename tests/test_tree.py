import pytest

import cotengra as ctg


def test_contraction_tree_equivalency():
    eq = "a,ab,bc,c->"
    shapes = [(4,), (4, 2), (2, 5), (5,)]
    # optimal contraction is like:
    #    o
    #   / \
    #  o   o
    # / \ / \
    ct1 = ctg.einsum_tree(eq, *shapes, optimize=[(0, 1), (0, 1), (0, 1)])
    ct2 = ctg.einsum_tree(eq, *shapes, optimize=[(2, 3), (0, 1), (0, 1)])
    assert ct1.total_flops() == ct2.total_flops() == 20
    assert ct1.children == ct2.children
    assert ct1.is_complete()
    assert ct2.is_complete()


@pytest.mark.parametrize("ssa", [False, True])
@pytest.mark.parametrize("autocomplete", [False, True, "auto"])
def test_contraction_tree_from_path_incomplete(ssa, autocomplete):
    inputs = ["a", "ab", "bc", "c"]
    output = ""
    size_dict = {"a": 4, "b": 2, "c": 5}
    if ssa:
        ssa_path = [
            (0, 1),
            (2, 3),
        ]
        tree = ctg.ContractionTree.from_path(
            inputs,
            output,
            size_dict,
            ssa_path=ssa_path,
            autocomplete=autocomplete,
        )
    else:
        path = [(0, 1), (0, 1)]
        tree = ctg.ContractionTree.from_path(
            inputs, output, size_dict, path=path, autocomplete=autocomplete
        )

    if not autocomplete:
        assert not tree.is_complete()
        assert tree.get_incomplete_nodes() == {
            frozenset([0, 1, 2, 3]): [
                frozenset([0, 1]),
                frozenset([2, 3]),
            ],
        }
    else:
        assert tree.is_complete()
        assert tree.get_incomplete_nodes() == {}


def test_tree_incomplete():
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=10,
        reg=3,
        n_out=1,
        n_hyper_in=1,
        n_hyper_out=1,
        seed=42,
    )
    tree = ctg.ContractionTree(inputs, output, size_dict)
    assert len(tree.info) == 11
    tree.contract_nodes(
        [
            frozenset([3, 6, 8]),
            frozenset([4, 7]),
        ]
    )
    assert len(tree.info) == 14
    assert not tree.is_complete()
    groups = tree.get_incomplete_nodes()
    assert len(groups) == 3
    tree.autocomplete()
    assert tree.is_complete()
    assert tree.get_incomplete_nodes() == {}
    assert len(tree.info) == 19


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
    if requires:
        pytest.importorskip(requires)

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        30, reg=5, seed=42, d_max=3
    )

    path_gr = ctg.array_contract_path(
        inputs, output, size_dict, optimize="greedy"
    )

    tree_gr = ctg.array_contract_tree(
        inputs,
        output,
        size_dict,
        optimize=path_gr,
    )

    initial_cost = tree_gr.total_flops()

    if forested:
        tree_gr.subtree_reconfigure_forest_(
            num_trees=2, subtree_size=6, progbar=True, parallel=parallel
        )
    else:
        tree_gr.subtree_reconfigure_(progbar=True)

    assert tree_gr.total_flops() < initial_cost


def test_reconfigure_with_n_smaller_than_subtree_size():
    inputs, output, _, size_dict = ctg.utils.rand_equation(10, 3)
    tree = ctg.array_contract_tree(
        inputs, output, size_dict, optimize="greedy"
    )
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
    if requires:
        pytest.importorskip(requires)

    tree_gr = ctg.utils.rand_tree(
        30,
        reg=5,
        seed=42,
        d_max=2,
        optimize="greedy",
    )

    target_size = tree_gr.max_size() // 32

    if forested:
        tree_gr.slice_and_reconfigure_forest_(
            target_size, num_trees=2, progbar=True, parallel=parallel
        )
    else:
        tree_gr.slice_and_reconfigure_(target_size, progbar=True)

    assert tree_gr.max_size() <= target_size


def test_plot():
    pytest.importorskip("matplotlib")

    import matplotlib

    matplotlib.use("Template")
    tree = ctg.utils.rand_tree(
        30,
        reg=5,
        seed=42,
        d_max=2,
        optimize="greedy",
    )
    tree.plot_flat()
    tree.plot_ring()
    tree.plot_tent()
    tree.plot_circuit()
    tree.plot_contractions()


def test_plot_alt():
    pytest.importorskip("altair")
    tree = ctg.utils.rand_tree(
        30,
        reg=5,
        seed=42,
        d_max=2,
        optimize="greedy",
    )
    tree.plot_contractions_alt()


@pytest.mark.parametrize("optimize", ["greedy-compressed", "greedy-span"])
def test_compressed_rank(optimize):
    tree = ctg.utils.rand_tree(
        30,
        reg=5,
        seed=42,
        d_max=2,
        optimize=optimize,
    )
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
    assert sliced_stats["flops"] > orig_stats["flops"]
    # make sure we haven't mutated original tree
    assert tree.info == orig_info
    assert tree_sliced.info != orig_info
    assert tree_sliced.contract(arrays) == pytest.approx(x)

    tree_rem0 = tree.remove_ind(ix, project=0)
    rem_stats = tree_rem0.contract_stats()
    assert rem_stats["flops"] < orig_stats["flops"]
    # make sure we haven't mutated original tree
    assert tree.info == orig_info
    assert tree_rem0.info != orig_info
    y0 = tree_rem0.contract(arrays)
    assert y0 != pytest.approx(x)

    for j in range(1, tree.size_dict[ix]):
        tree_remj = tree.remove_ind(ix, project=j)
        rem_stats = tree_remj.contract_stats()
        assert rem_stats["flops"] < orig_stats["flops"]
        # make sure we haven't mutated original tree
        assert tree.info == orig_info
        assert tree_remj.info != orig_info
        yj = tree_remj.contract(arrays)
        assert yj != pytest.approx(x)
        y0 += yj

    assert y0 == pytest.approx(x)


@pytest.mark.parametrize("ind", ["a", "b", "d", "g"])
def test_restore_ind(ind):
    from numpy.testing import assert_allclose

    tree = ctg.utils.rand_tree(10, 3, 2, 1, 1, seed=42)
    arrays = ctg.utils.make_arrays_from_inputs(tree.inputs, tree.size_dict)
    stree = tree.remove_ind(ind)
    assert stree.contract_stats() != tree.contract_stats()
    utree = stree.restore_ind(ind)
    assert not utree.sliced_inds
    assert utree.multiplicity == 1
    assert_allclose(tree.contract(arrays), utree.contract(arrays))
    assert utree.contract_stats() == tree.contract_stats()


def test_unslice_rand():
    tree = ctg.utils.rand_tree(10, 3, 2, 1, 1, seed=42)
    arrays = ctg.utils.make_arrays_from_inputs(tree.inputs, tree.size_dict)
    x = tree.contract(arrays)
    tree.remove_ind_("a")
    tree.remove_ind_("b")
    tree.unslice_rand_()
    assert len(tree.sliced_inds) == 1
    assert tree.contract(arrays) == pytest.approx(x)
    tree.unslice_rand_()
    assert len(tree.sliced_inds) == 0
    assert tree.contract(arrays) == pytest.approx(x)


def test_unslice_all():
    tree = ctg.utils.rand_tree(10, 3, 2, 1, 1, seed=42)
    arrays = ctg.utils.make_arrays_from_inputs(tree.inputs, tree.size_dict)
    x = tree.contract(arrays)
    tree.remove_ind_("a")
    tree.remove_ind_("b")
    tree.unslice_all_()
    assert len(tree.sliced_inds) == 0
    assert tree.contract(arrays) == pytest.approx(x)


def test_reslice_and_reconfigure():
    tree = ctg.utils.rand_tree(10, 3, 2, 1, 1, seed=42)
    arrays = ctg.utils.make_arrays_from_inputs(tree.inputs, tree.size_dict)
    x = tree.contract(arrays)
    W = tree.max_size()
    target_size = W // 10
    tree.slice_and_reconfigure_(target_size, reslice=True)
    assert tree.max_size() <= target_size
    assert tree.contract(arrays) == pytest.approx(x)


def test_tree_with_one_node():
    eq = "abc->abc"
    size_dict = {"a": 2, "b": 3, "c": 4}
    tree = ctg.ContractionTree.from_eq(eq, size_dict)
    assert tree.get_path() == ()
    assert tree.contraction_cost() == 0
    assert tree.contraction_width(None) == 2 * 3 * 4


@pytest.mark.parametrize("seed", range(4))
def test_slice_and_restore_preprocessed_inds(seed):
    import numpy as np

    eq = "abc,bde,dfg,fah->"
    inputs, output = ctg.utils.eq_to_inputs_output(eq)
    size_dict = ctg.utils.make_rand_size_dict_from_inputs(inputs, seed=seed)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)
    tree = ctg.ContractionTree(inputs, output, size_dict)
    tree.autocomplete()
    stats0 = tree.contract_stats()
    xe = np.einsum(eq, *arrays)
    assert tree.contract(arrays) == pytest.approx(xe)
    assert len(tree.preprocessing) == 4
    tree.remove_ind_("a")
    assert tree.has_preprocessing()
    assert len(tree.preprocessing) == 4
    assert tree.contract(arrays) == pytest.approx(xe)
    # preprocessed ind, slicing it prevents the preprocessing
    tree.remove_ind_("c")
    assert tree.has_preprocessing()
    assert len(tree.preprocessing) == 3
    assert tree.contract(arrays) == pytest.approx(xe)
    tree.restore_ind_("a")
    assert tree.has_preprocessing()
    assert len(tree.preprocessing) == 3
    assert tree.contract(arrays) == pytest.approx(xe)
    tree.restore_ind_("c")
    assert tree.has_preprocessing()
    assert len(tree.preprocessing) == 4
    assert tree.contract(arrays) == pytest.approx(xe)
    assert tree.contract_stats() == stats0


@pytest.mark.parametrize("n", [3, 10, 30])
@pytest.mark.parametrize("seed", range(4))
def test_tree_from_edge_path(n, seed):
    import random

    con = ctg.utils.rand_equation(n, 3, 2, 2, 2, seed=seed)
    indices = list(con.size_dict)
    rng = random.Random(seed)
    rng.shuffle(indices)

    tree = ctg.ContractionTree.from_path(
        con.inputs,
        con.output,
        con.size_dict,
        edge_path=indices,
        check=True,
    )

    assert tree.is_complete()
