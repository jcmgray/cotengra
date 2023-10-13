import pytest

import cotengra as ctg
import numpy as np


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
def test_array_contract_expression_cache(optimize_type):

    if optimize_type == "preset":
        optimize = "auto"
    elif optimize_type == "list":
        optimize = [(0, 1)] * 9
    elif optimize_type == "tuple":
        optimize = tuple([(0, 1)] * 9)

    inputs, output, shapes, size_dict = ctg.utils.rand_equation(10, 3)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)
    expra = ctg.array_contract_expression(
        inputs, output, shapes=shapes, cache=True, optimize=optimize,
    )
    exprb = ctg.array_contract_expression(
        inputs, output, shapes=shapes, cache=True, optimize=optimize,
    )
    exprc = ctg.array_contract_expression(
        inputs, output, shapes=shapes, cache=False, optimize=optimize,
    )
    assert expra is exprb
    assert exprb is not exprc
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    xa = np.einsum(eq, *arrays)
    xb = expra(*arrays)
    assert np.allclose(xa, xb)
    xc = expra(*arrays)
    assert np.allclose(xa, xc)
