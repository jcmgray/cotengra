import cotengra as ctg


def test_compressed_greedy():
    chi = 4
    inputs, output, shapes, size_dict = ctg.utils.lattice_equation([10, 10])
    opt = ctg.HyperCompressedOptimizer(
        chi=chi,
        methods=['greedy-compressed',]
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
    inputs, output, shapes, size_dict = ctg.utils.lattice_equation([10, 10])
    opt = ctg.HyperCompressedOptimizer(
        chi=chi,
        methods=['greedy-span',]
    )
    tree = opt.search(inputs, output, size_dict)
    assert isinstance(tree, ctg.ContractionTreeCompressed)
    assert tree.total_flops(chi) < tree.total_flops_exact()
    assert tree.total_write(chi) < tree.total_write_exact()
    assert tree.max_size(chi) < tree.max_size_exact()
    assert tree.peak_size(chi) < tree.peak_size_exact()
    assert tree.contraction_width(chi) < 20


def test_compressed_agglom():
    chi = 4
    inputs, output, shapes, size_dict = ctg.utils.lattice_equation([10, 10])
    opt = ctg.HyperCompressedOptimizer(
        chi=chi,
        methods=['kahypar-agglom',]
    )
    tree = opt.search(inputs, output, size_dict)
    assert isinstance(tree, ctg.ContractionTreeCompressed)
    assert tree.total_flops(chi) < tree.total_flops_exact()
    assert tree.total_write(chi) < tree.total_write_exact()
    assert tree.max_size(chi) < tree.max_size_exact()
    assert tree.peak_size(chi) < tree.peak_size_exact()
    assert tree.contraction_width(chi) < 20
