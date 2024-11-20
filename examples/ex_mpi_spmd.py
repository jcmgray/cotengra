"""This script illustrates how to parallelize both the contraction path
finding and sliced contraction computation using SPMD style MPI.
"""

import numpy as np
import cotengra as ctg
from mpi4py import MPI

comm = MPI.COMM_WORLD

# generate a random contraction
inputs, output, shapes, size_dict = ctg.utils.rand_equation(
    100,
    3,
    n_out=2,
    seed=666,
)
# numpy seeded by    ^^^^ so these are synced
arrays = [np.random.randn(*s) for s in shapes]


# ------- STAGE 1: find contract tree with many independant searches -------- #

print(f"{comm.rank}:: Finding tree SPMD style ...")

# each worker will be running a *separate* contraction optimizer
opt = ctg.HyperOptimizer(
    # make sure we generate at least 1 slice per process
    slicing_opts={"target_slices": comm.size},
    # each worker optimizes its own trials so we don't need a fast sampler
    optlib="optuna",
    # since the load is not balanced, makes more sense to limit by time
    max_repeats=1_000_000,
    max_time=5,
)
# perform the search
tree = opt.search(inputs, output, size_dict)
score = opt.best["score"]

# need to get the best tree from across all processes
print(f"{comm.rank}:: Sharing best tree ...")
_, tree = comm.allreduce((score, tree), op=MPI.MIN)


# -------------- STAGE 2: use SPMD mode to perform contraction -------------- #

print(f"{comm.rank}:: Contracting tree SPMD style ...")

# perform the contraction distributed over mpi
# (set root=i to reduce output tensor to a specific worker only)
x = tree.contract_mpi(arrays, comm=comm)

if comm.rank == 0:
    print(x)
