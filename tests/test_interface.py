import numpy as np
import pytest

import cotengra as ctg


@pytest.mark.parametrize("optimize_type", ["preset", "list", "tuple"])
def test_array_contract_path_cache(optimize_type):
    if optimize_type == "preset":
        optimize = "auto"
    elif optimize_type == "list":
        optimize = [(0, 1)] * 9
    elif optimize_type == "tuple":
        optimize = tuple([(0, 1)] * 9)

    inputs, output, shapes, size_dict = ctg.utils.rand_equation(10, 3)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)
    pa = ctg.array_contract_path(
        inputs, output, shapes=shapes, cache=True, optimize=optimize
    )
    pb = ctg.array_contract_path(
        inputs, output, shapes=shapes, cache=True, optimize=optimize
    )
    pc = ctg.array_contract_path(
        inputs, output, shapes=shapes, cache=False, optimize=optimize
    )
    assert pa is pb
    assert (pb is not pc) or (optimize_type == "tuple")
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    xa = np.einsum(eq, *arrays)
    xb = ctg.einsum(eq, *arrays, optimize=pa)
    assert np.allclose(xa, xb)


@pytest.mark.parametrize("optimize_type", ["preset", "list", "tuple"])
@pytest.mark.parametrize("strip_exponent", [False, True])
def test_array_contract_expression_cache(optimize_type, strip_exponent):
    if optimize_type == "preset":
        optimize = "auto"
    elif optimize_type == "list":
        optimize = [(0, 1)] * 9
    elif optimize_type == "tuple":
        optimize = tuple([(0, 1)] * 9)

    inputs, output, shapes, size_dict = ctg.utils.rand_equation(10, 3)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)
    expra = ctg.array_contract_expression(
        inputs,
        output,
        shapes=shapes,
        cache=True,
        optimize=optimize,
        strip_exponent=strip_exponent,
    )
    exprb = ctg.array_contract_expression(
        inputs,
        output,
        shapes=shapes,
        cache=True,
        optimize=optimize,
        strip_exponent=strip_exponent,
    )
    exprc = ctg.array_contract_expression(
        inputs,
        output,
        shapes=shapes,
        cache=False,
        optimize=optimize,
        strip_exponent=strip_exponent,
    )
    assert expra is exprb
    assert exprb is not exprc
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    xa = np.einsum(eq, *arrays)
    xb = expra(*arrays)
    if strip_exponent:
        xb = xb[0] * 10 ** xb[1]
    assert np.allclose(xa, xb)
    xc = expra(*arrays)
    if strip_exponent:
        xc = xc[0] * 10 ** xc[1]
    assert np.allclose(xa, xc)


def test_einsum_formats_interleaved():
    args = (
        np.random.rand(2, 3, 4),
        [2, 3, 4],
        np.random.rand(4, 5, 6),
        [4, 5, 6],
        np.random.rand(2, 7),
        [2, 7],
        [7, 3],
    )
    x = np.einsum(*args)
    y = ctg.einsum(*args)
    assert np.allclose(x, y)


@pytest.mark.parametrize(
    "eq,shapes",
    [
        ("c...a,b...c->b...a", [(2, 5, 6, 3), (4, 6, 2)]),
        ("a...a->...", [(3, 3)]),
        ("a...a->...a", [(3, 4, 5, 3)]),
        ("...,...ab->ba...", [(), (2, 3, 4, 5)]),
        ("a,b,ab...c->b...a", [(2,), (3,), (2, 3, 4, 5, 6)]),
    ],
)
def test_einsum_ellipses(eq, shapes):
    arrays = [np.random.rand(*shape) for shape in shapes]
    x = np.einsum(eq, *arrays)
    y = ctg.einsum(eq, *arrays)
    assert np.allclose(x, y)


def test_slice_and_strip_exponent():
    inputs, output, _, size_dict = ctg.utils.rand_equation(
        n=10,
        reg=3,
        n_out=2,
        n_hyper_in=2,
        n_hyper_out=2,
        seed=42,
    )
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict, seed=42)

    x0 = ctg.array_contract(arrays, inputs, output, size_dict=size_dict)
    x1 = ctg.array_contract(
        arrays,
        inputs,
        output,
        size_dict=size_dict,
        optimize=ctg.HyperOptimizer(
            methods=["greedy"],
            slicing_opts=dict(target_slices=2),
            max_repeats=4,
        ),
    )
    np.testing.assert_allclose(x0, x1)
    x2 = ctg.array_contract(
        arrays,
        inputs,
        output,
        size_dict=size_dict,
        strip_exponent=True,
    )
    x2 = x2[0] * 10 ** x2[1]
    np.testing.assert_allclose(x0, x2)
    x3 = ctg.array_contract(
        arrays,
        inputs,
        output,
        size_dict=size_dict,
        optimize=ctg.HyperOptimizer(
            methods=["greedy"],
            slicing_opts=dict(target_slices=2),
            max_repeats=4,
        ),
        strip_exponent=True,
    )
    x3 = x3[0] * 10 ** x3[1]
    np.testing.assert_allclose(x0, x3)
