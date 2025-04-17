<p align="left"><img src="https://imgur.com/OM5XyaD.png" alt="cotengra" width="400px"></p>

[![tests](https://github.com/jcmgray/cotengra/actions/workflows/test.yml/badge.svg)](https://github.com/jcmgray/cotengra/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jcmgray/cotengra/branch/main/graph/badge.svg?token=Q5evNiuT9S)](https://codecov.io/gh/jcmgray/cotengra)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/84f825f5a7044762be62600c0650473d)](https://app.codacy.com/gh/jcmgray/cotengra/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Docs](https://readthedocs.org/projects/cotengra/badge/?version=latest)](https://cotengra.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/cotengra?color=teal)](https://pypi.org/project/cotengra/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/cotengra/badges/version.svg)](https://anaconda.org/conda-forge/cotengra)

`cotengra` is a python library for contracting tensor networks or einsum
expressions involving large numbers of tensors - the main docs can be found
at [cotengra.readthedocs.io](https://cotengra.readthedocs.io/).
Some of the key feautures of `cotengra` include:

* drop-in ``einsum`` and ``ncon`` replacement
* an explicit **contraction tree** object that can be flexibly built, modified and visualized
* a **'hyper optimizer'** that samples trees while tuning the generating meta-paremeters
* **dynamic slicing** for massive memory savings and parallelism
* **simulated annealing** as an alternative optimizing and slicing strategy
* support for **hyper** edge tensor networks and thus arbitrary einsum equations
* **paths** that can be supplied to [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`opt_einsum`](https://dgasmith.github.io/opt_einsum/), [`quimb`](https://quimb.readthedocs.io/en/latest/) among others
* **performing contractions** with tensors from many libraries via [`autoray`](https://github.com/jcmgray/autoray),
  even if they don't provide `einsum` or `tensordot` but do have (batch) matrix
  multiplication

<p align="center"><img src="https://imgur.com/jMO138y.png" alt="cotengra" width="500px"></p>
