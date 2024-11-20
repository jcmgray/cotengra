"""This script shows how to manually use jax to jit-compile the core
contraction.
"""

import jax
import numpy as np
import cotengra as ctg
from concurrent.futures import ThreadPoolExecutor

# generate a random contraction
inputs, output, shapes, size_dict = ctg.utils.rand_equation(
    140,
    3,
    n_out=2,
    seed=666,
)
arrays = [np.random.randn(*s) for s in shapes]

# ------------------------ Find the contraction tree ------------------------ #

print("Finding tree...")

# find a contraction tree
opt = ctg.HyperOptimizer(
    parallel=True,
    # make sure contractions fit onto GPU
    slicing_reconf_opts={"target_size": 2**28},
    max_repeats=32,
    progbar=True,
)

# run the optimizer and extract the contraction tree
tree = opt.search(inputs, output, size_dict)

# ------------------------- Perform the contraction ------------------------- #

print("1: Contracting slices with jax...")

# we'll run the GPU contraction on a separate single thread, which mostly
# serves as an example of how one might distribute contractions to multi-GPUs
pool = ThreadPoolExecutor(1)

# we'll compile the core contraction algorithm (other options here are
# tensorflow and torch) since we call it possibly many times
contract_core_jit = jax.jit(tree.contract_core)

# eagerly submit all the contractions to the thread pool
fs = [
    pool.submit(contract_core_jit, tree.slice_arrays(arrays, i))
    for i in range(tree.nslices)
]

# lazily gather all the slices in the main process with progress bar
slices = (np.array(f.result()) for f in fs)

x = tree.gather_slices(slices, progbar=True)
print(x)

# ------------------------ Perform the contraction 2 ------------------------ #

print("2: Contracting slices with jax and constants...")

# this time we'll treat the input arrays as constant, so they ideally be
# folded in, possibly with some extra memory overhead
jax_arrays = [jax.numpy.asarray(x) for x in arrays]
contract_core_jit = jax.jit(lambda i: tree.contract_slice(jax_arrays, i))

# eagerly submit all the contractions to the thread pool
fs = [pool.submit(contract_core_jit, np.array(i)) for i in range(tree.nslices)]

# lazily gather all the slices in the main process with progress bar
slices = (np.array(f.result()) for f in fs)

x = tree.gather_slices(slices, progbar=True)
print(x)


# ------------------------ Perform the contraction 3 ------------------------ #

print("3: Contracting slices with jax but no jit...")

# we can also contract the raw jax arrays with no jit - though jax is
# significantly slower in this mode
x = tree.contract(jax_arrays, progbar=True)
print(x)
