import pytest

import cotengra as ctg

import numpy as np
from numpy.testing import assert_allclose
import opt_einsum as oe


@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("d_min", [2])
@pytest.mark.parametrize("d_max", [4])
@pytest.mark.parametrize("reg", [2, 3])
@pytest.mark.parametrize("n_out", [0, 1, 2])
@pytest.mark.parametrize("n_hyper_in", [0, 1, 2])
@pytest.mark.parametrize("n_hyper_out", [0, 1, 2])
@pytest.mark.parametrize("seed", [42, 666])
@pytest.mark.parametrize("indices_sort", [None, 'root', 'flops'])
def test_rand_equation(
    n,
    reg,
    n_out,
    n_hyper_in,
    n_hyper_out,
    d_min,
    d_max,
    seed,
    indices_sort,
):
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=n, reg=reg, n_out=n_out, n_hyper_in=n_hyper_in,
        n_hyper_out=n_hyper_out, d_min=d_min, d_max=d_max, seed=seed,
    )
    arrays = [np.random.normal(size=s) for s in shapes]
    eq = ",".join(map("".join, inputs)) + "->" + "".join(output)

    path, info = oe.contract_path(eq, *arrays, optimize='greedy')
    if info.largest_intermediate > 2**20:
        raise RuntimeError("Contraction too big.")

    x = oe.contract(eq, *arrays, optimize=path)

    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)

    if indices_sort:
        tree.sort_contraction_indices(indices_sort)

    # base contract
    y1 = tree.contract(arrays, check=True)
    assert_allclose(x, y1)

    # contract after modifying tree
    tree.subtree_reconfigure_()
    y2 = tree.contract(arrays, check=True)
    assert_allclose(x, y2)

    size = tree.max_size()
    if size < 600:
        return

    # contract after slicing and modifying
    tree.slice_and_reconfigure_(target_size=size // 6)
    y3 = tree.contract(arrays, check=True)
    assert_allclose(x, y3)

    # contract after slicing some output indices
    remaining_out = list(tree.output_legs)
    nsout = np.random.randint(low=0, high=len(remaining_out) + 1)
    so_ix = np.random.choice(remaining_out, replace=False, size=nsout)
    for ind in so_ix:
        tree.remove_ind_(ind)

        if indices_sort:
            tree.sort_contraction_indices(indices_sort)

        y4 = tree.contract(arrays, check=True)
        assert_allclose(x, y4)
