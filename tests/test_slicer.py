import pytest
import numpy as np
import opt_einsum as oe
import cotengra as ctg


def test_slicer():

    eq, shapes = oe.helpers.rand_equation(30, reg=5, seed=42, d_max=3)
    arrays = [np.random.uniform(size=s) for s in shapes]
    path, info = oe.contract_path(eq, *shapes, shapes=True)
    expected = oe.contract(eq, *arrays, optimize=path)

    sf = ctg.SliceFinder(info, target_size=1_000_000, target_flops=None)
    inds, ccost = sf.search()

    assert info.largest_intermediate > 1_000_000
    assert ccost.size <= 1_000_000
    assert ccost.total_flops > info.opt_cost
    assert len(inds) > 1

    sc = sf.SlicedContractor(arrays)
    assert sc.total_flops == ccost.total_flops
    assert sc.contract_all() == pytest.approx(expected)
