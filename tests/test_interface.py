import cotengra as ctg
import numpy as np


def test_array_contract_path_cache():
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(10, 3)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)
    pa = ctg.array_contract_path(inputs, output, shapes=shapes, cache=True)
    pb = ctg.array_contract_path(inputs, output, shapes=shapes, cache=True)
    pc = ctg.array_contract_path(inputs, output, shapes=shapes, cache=False)
    assert pa is pb
    assert pb is not pc
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    xa = np.einsum(eq, *arrays)
    xb = ctg.einsum(eq, *arrays, optimize=pa)
    assert np.allclose(xa, xb)
