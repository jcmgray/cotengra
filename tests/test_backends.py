"""Cross-backend correctness tests for ``ContractionTree.contract``.

Each backend is exercised across the cartesian product of:
- real / complex dtype,
- ``strip_exponent`` on / off,
- sliced / unsliced contraction.

Backends that are not installed are skipped, so this file is intended to
be the single place that exercises non-numpy backends — CI would then
provision a per-backend pixi environment and run only this file.
"""

import importlib

import pytest
from autoray import to_numpy
from numpy.testing import assert_allclose

import cotengra as ctg


def _has(name):
    return importlib.util.find_spec(name) is not None


BACKENDS = [
    pytest.param("numpy", id="numpy"),
    pytest.param(
        "torch",
        marks=pytest.mark.skipif(
            not _has("torch"), reason="torch not installed"
        ),
        id="torch",
    ),
    pytest.param(
        "jax",
        marks=pytest.mark.skipif(not _has("jax"), reason="jax not installed"),
        id="jax",
    ),
    pytest.param(
        "tensorflow",
        marks=pytest.mark.skipif(
            not _has("tensorflow"), reason="tensorflow not installed"
        ),
        id="tensorflow",
    ),
    pytest.param(
        "cupy",
        marks=pytest.mark.skipif(
            not _has("cupy"), reason="cupy not installed"
        ),
        id="cupy",
    ),
    pytest.param(
        "autograd",
        marks=pytest.mark.skipif(
            not _has("autograd"), reason="autograd not installed"
        ),
        id="autograd",
    ),
]


def _setup_backend(backend):
    if backend == "jax":
        # jax defaults to float32; opt-in to x64 for float64 / complex128.
        from jax import config

        config.update("jax_enable_x64", True)


def _to_backend(x, backend):
    if backend == "numpy":
        return x
    if backend == "torch":
        import torch

        return torch.from_numpy(x)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(x)
    if backend == "tensorflow":
        import tensorflow as tf

        return tf.convert_to_tensor(x)
    if backend == "cupy":
        import cupy

        return cupy.asarray(x)
    if backend == "autograd":
        import autograd.numpy as anp

        return anp.asarray(x)
    raise ValueError(f"unknown backend {backend}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("strip_exponent", [False, True])
@pytest.mark.parametrize("slicing", [False, True])
def test_contract_backend(backend, dtype, strip_exponent, slicing):
    _setup_backend(backend)

    c = ctg.utils.lattice_equation([4, 4])
    arrays_np = ctg.utils.make_arrays_from_inputs(
        c.inputs, c.size_dict, seed=42, dtype=dtype
    )
    eq = ctg.utils.inputs_output_to_eq(c.inputs, c.output)
    expected = ctg.einsum(eq, *arrays_np)

    arrays = [_to_backend(a, backend) for a in arrays_np]

    tree = ctg.array_contract_tree(
        c.inputs, c.output, c.size_dict, optimize="greedy"
    )
    if slicing:
        tree.slice_(target_slices=4)
        assert tree.nslices >= 4

    result = tree.contract(arrays, strip_exponent=strip_exponent)

    if strip_exponent:
        m, p = result
        result = to_numpy(m) * 10.0 ** float(to_numpy(p))
    else:
        result = to_numpy(result)

    assert_allclose(result, expected, rtol=5e-6, atol=1e-8)
