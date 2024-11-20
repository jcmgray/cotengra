"""This script illustrates how to parallelize both the contraction path
finding and sliced contraction computation using 'executor' style MPI.
"""

import numpy as np
import cotengra as ctg
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor


comm = MPI.COMM_WORLD

with MPICommExecutor() as pool:
    # only need to make calls from the root process
    if pool is not None:
        # generate a random contraction
        inputs, output, shapes, size_dict = ctg.utils.rand_equation(
            100,
            3,
            n_out=2,
            seed=666,
        )
        arrays = [np.random.randn(*s) for s in shapes]

        # -- STAGE 1: find the contraction tree with a single optimization -- #

        print(f"{comm.rank}:: Finding tree executor style ...")

        # find a contraction tree
        opt = ctg.HyperOptimizer(
            parallel=pool,
            # make sure we generate at least 1 slice per process
            slicing_opts={"target_slices": comm.size},
            # cmaes is suited to generating many trials quickly
            optlib="cmaes",
            max_repeats=512,
            progbar=True,
        )
        # run the optimizer and extract the contraction tree
        tree = opt.search(inputs, output, size_dict)

        # ------------- STAGE 2: perform contraction on workers ------------- #

        # root process just submits and gather results - workers contract
        print(f"{comm.rank}:: Contracting tree executor style ...")

        # submit contractions eagerly
        fs = [
            pool.submit(tree.contract_slice, arrays, i)
            for i in range(tree.nslices)
        ]

        # gather results lazily (i.e. using generator)
        x = tree.gather_slices((f.result() for f in fs))
        print(x)
