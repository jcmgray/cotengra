def test_contraction_tree_equivalency():
    import opt_einsum as oe
    from cotengra.core import ContractionTree
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
    ct1 = ContractionTree.from_pathinfo(info1)
    ct2 = ContractionTree.from_pathinfo(info2)
    assert ct1.total_flops() == ct2.total_flops() == 40
    assert ct1.children == ct2.children
    # assert ct1.path() == ct2.path()
    # assert ct1.ssa_path() == ct2.ssa_path()
