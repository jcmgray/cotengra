import pytest

import cotengra as ctg


def test_compressed_greedy():
    chi = 4
    inputs, output, _, size_dict = ctg.utils.lattice_equation([10, 10])
    opt = ctg.HyperCompressedOptimizer(
        chi=chi,
        methods=[
            "greedy-compressed",
        ],
    )
    tree = opt.search(inputs, output, size_dict)
    assert isinstance(tree, ctg.ContractionTreeCompressed)
    assert tree.total_flops(chi) < tree.total_flops_exact()
    assert tree.total_write(chi) < tree.total_write_exact()
    assert tree.max_size(chi) < tree.max_size_exact()
    assert tree.peak_size(chi) < tree.peak_size_exact()
    assert tree.contraction_width(chi) < 20


def test_compressed_span():
    chi = 4
    inputs, output, _, size_dict = ctg.utils.lattice_equation([10, 10])
    opt = ctg.HyperCompressedOptimizer(
        chi=chi,
        methods=[
            "greedy-span",
        ],
    )
    tree = opt.search(inputs, output, size_dict)
    assert isinstance(tree, ctg.ContractionTreeCompressed)
    assert tree.total_flops(chi) < tree.total_flops_exact()
    assert tree.total_write(chi) < tree.total_write_exact()
    assert tree.max_size(chi) < tree.max_size_exact()
    assert tree.peak_size(chi) < tree.peak_size_exact()
    assert tree.contraction_width(chi) < 20


def test_compressed_agglom():
    pytest.importorskip("kahypar")
    chi = 4
    inputs, output, _, size_dict = ctg.utils.lattice_equation([16, 16])
    opt = ctg.HyperCompressedOptimizer(
        chi=chi,
        methods=[
            "kahypar-agglom",
        ],
    )
    tree = opt.search(inputs, output, size_dict)
    assert isinstance(tree, ctg.ContractionTreeCompressed)
    assert tree.total_flops(chi) < tree.total_flops_exact()
    assert tree.total_write(chi) < tree.total_write_exact()
    assert tree.max_size(chi) < tree.max_size_exact()
    assert tree.peak_size(chi) < tree.peak_size_exact()
    assert tree.contraction_width(chi) < 20


@pytest.mark.parametrize("order_only", (False, True))
def test_compressed_reconfigure(order_only):
    chi = 16
    minimize = ctg.scoring.CompressedPeakObjective(chi)
    inputs, output, _, size_dict = ctg.utils.lattice_equation([16, 16])
    tree = ctg.array_contract_tree(
        inputs,
        output,
        size_dict,
        optimize=ctg.path_compressed_greedy.GreedyCompressed(chi),
    )
    tree_wr = tree.windowed_reconfigure(minimize, order_only=order_only)
    assert tree_wr.peak_size(chi) < tree.peak_size(chi)


def test_compressed_windowed_reconfigure():
    chi = 4
    minimize = ctg.scoring.CompressedPeakObjective(chi)
    inputs, output, _, size_dict = ctg.utils.lattice_equation([8, 8])
    tree = ctg.array_contract_tree(
        inputs,
        output,
        size_dict,
        optimize=ctg.path_compressed_greedy.GreedyCompressed(chi),
    )
    tree_wr = tree.windowed_reconfigure(minimize)
    assert tree_wr.get_default_objective().chi == chi
    assert tree_wr.peak_size(chi) < tree.peak_size(chi)
