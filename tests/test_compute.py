import pytest

import cotengra as ctg

import numpy as np
from numpy.testing import assert_allclose
import opt_einsum as oe


def rand_equation(
    n, reg,
    n_out=0,
    n_hyper_in=0,
    n_hyper_out=0,
    d_min=2, d_max=3,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    num_inds = max((n * reg) // 2, n_hyper_out + n_hyper_in + n_out)
    size_dict = {oe.get_symbol(i): np.random.randint(d_min, d_max + 1)
                 for i in range(num_inds)}

    inds = iter(size_dict)
    inputs = [[] for _ in range(n)]
    output = []

    for _ in range(n_hyper_out):
        ind = next(inds)
        output.append(ind)
        s = np.random.randint(3, n + 1)
        where = np.random.choice(np.arange(n), size=s, replace=False)
        for i in where:
            inputs[i].append(ind)

    for _ in range(n_hyper_in):
        ind = next(inds)
        s = np.random.randint(3, n + 1)
        where = np.random.choice(np.arange(n), size=s, replace=False)
        for i in where:
            inputs[i].append(ind)

    for _ in range(n_out):
        ind = next(inds)
        output.append(ind)
        where = np.random.choice(np.arange(n), size=2, replace=False)
        for i in where:
            inputs[i].append(ind)

    for ind in inds:
        where = np.random.choice(np.arange(n), size=2, replace=False)
        for i in where:
            inputs[i].append(ind)

    shapes = [tuple(size_dict[ix] for ix in term) for term in inputs]

    output = list(np.random.permutation(output))

    return inputs, output, shapes, size_dict


@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("d_min", [2])
@pytest.mark.parametrize("d_max", [4])
@pytest.mark.parametrize("reg", [2, 3])
@pytest.mark.parametrize("n_out", [0, 1, 2])
@pytest.mark.parametrize("n_hyper_in", [0, 1, 2])
@pytest.mark.parametrize("n_hyper_out", [0, 1, 2])
@pytest.mark.parametrize("seed", [42, 666])
def test_rand_equation(
    n,
    reg,
    n_out,
    n_hyper_in,
    n_hyper_out,
    d_min,
    d_max,
    seed,
):
    inputs, output, shapes, size_dict = rand_equation(
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
        y4 = tree.contract(arrays, check=True)
        assert_allclose(x, y4)
