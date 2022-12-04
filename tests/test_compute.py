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
    y1 = tree.contract(arrays)
    assert_allclose(x, y1)

    # contract after modifying tree
    tree.subtree_reconfigure_()
    y2 = tree.contract(arrays)
    assert_allclose(x, y2)

    size = tree.max_size()
    if size < 600:
        return

    # contract after slicing and modifying
    tree.slice_and_reconfigure_(target_size=size // 6)
    y3 = tree.contract(arrays)
    assert_allclose(x, y3)

    # contract after slicing some output indices
    remaining_out = list(tree.output_legs)
    nsout = np.random.randint(low=0, high=len(remaining_out) + 1)
    so_ix = np.random.choice(remaining_out, replace=False, size=nsout)
    for ind in so_ix:
        tree.remove_ind_(ind)

        if indices_sort:
            tree.sort_contraction_indices(indices_sort)

        y4 = tree.contract(arrays)
        assert_allclose(x, y4)


def test_lazy_sliced_output_reduce():
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=10, reg=5, n_out=3, d_max=2,
        seed=666,
    )
    arrays = [np.random.rand(*s) for s in shapes]
    opt = ctg.HyperOptimizer(max_repeats=32, methods=['greedy'])
    tree = opt.search(inputs, output, size_dict)

    # slice both inner and outer indices
    tree.remove_ind_('g')
    tree.remove_ind_('b')
    tree.remove_ind_('y')
    tree.remove_ind_('a')

    # for such a quantity, sum(f(x)), the inner slice sum must be performed 1st
    x = (tree.contract(arrays)**2).sum()

    # ... so that the outer sum can be lazily generated correctly
    y = 0.0
    for chunk in tree.gen_output_chunks(arrays):
        y += (chunk**2).sum()

    assert y == pytest.approx(x)


@pytest.mark.parametrize("autojit", [False, True])
def test_exponent_stripping(autojit):
    inputs, output, shapes, size_dict = ctg.utils.lattice_equation([8, 8])
    rng = np.random.default_rng(42)
    arrays = [rng.uniform(size=s) for s in shapes]

    operands = []
    for term, array in zip(inputs, arrays):
        operands.append(array)
        operands.append(term)
    operands.append(output)
    ex = oe.contract(*operands)
    path, _ = oe.contract_path(*operands)

    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)

    x1 = tree.contract(arrays, autojit=autojit)
    assert x1 == pytest.approx(ex)

    m, p = tree.contract(arrays, autojit=autojit, strip_exponent=True)
    x2 = m * 10**p
    assert x2 == pytest.approx(ex)

    tree.slice_(target_size=64)
    assert tree.nslices >= 4

    x3 = tree.contract(arrays, autojit=autojit)
    assert x3 == pytest.approx(ex)

    m, p = tree.contract(arrays, autojit=autojit, strip_exponent=True)
    x4 = m * 10**p
    assert x4 == pytest.approx(ex)
