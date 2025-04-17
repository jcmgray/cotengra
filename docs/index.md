<p align="center"><img src="https://imgur.com/jMO138y.png" alt="cotengra" width="60%" height="60%"></p>

[![tests](https://github.com/jcmgray/cotengra/actions/workflows/test.yml/badge.svg)](https://github.com/jcmgray/cotengra/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jcmgray/cotengra/branch/main/graph/badge.svg?token=Q5evNiuT9S)](https://codecov.io/gh/jcmgray/cotengra)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/84f825f5a7044762be62600c0650473d)](https://app.codacy.com/gh/jcmgray/cotengra/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Docs](https://readthedocs.org/projects/cotengra/badge/?version=latest)](https://cotengra.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/cotengra?color=teal)](https://pypi.org/project/cotengra/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/cotengra/badges/version.svg)](https://anaconda.org/conda-forge/cotengra)

`cotengra` is a python library for contracting tensor networks or einsum expressions involving large numbers of tensors.
The original accompanying paper is [***'Hyper-optimized tensor network contraction'***](https://quantum-journal.org/papers/q-2021-03-15-410/) - if this project is useful please consider citing this to encourage further development!
Some of the key feautures of `cotengra` include:

* drop-in ``einsum`` and ``ncon`` replacement
* an explicit **contraction tree** object that can be flexibly built, modified and visualized
* a **'hyper optimizer'** that samples trees while tuning the generating meta-paremeters
* **dynamic slicing** for massive memory savings and parallelism
* **simulated annealing** as an alternative optimizing and slicing strategy
* support for **hyper** edge tensor networks and thus arbitrary einsum equations
* **paths** that can be supplied to [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`opt_einsum`](https://dgasmith.github.io/opt_einsum/), [`quimb`](https://quimb.readthedocs.io/en/latest/) among others
* **performing contractions** with tensors from many libraries via [`autoray`](https://github.com/jcmgray/autoray), even if they don't provide einsum or tensordot but do have (batch) matrix multiplication

---

Commits since [906f838](https://github.com/jcmgray/cotengra/commit/906f838d21840dc41c652efb63faf1f88c6113ee) added functionality first demonstrated in ['Classical Simulation of Quantum Supremacy Circuits'](https://arxiv.org/abs/2005.06787) - namely, contraction subtree reconfiguration and the interleaving of this with slicing - *dynamic slicing*.
In the ``examples`` folder you can find notebooks reproducing (in terms of sliced contraction complexity) the results of that paper as well as ['Simulating the Sycamore quantum supremacy circuits'](https://arxiv.org/abs/2103.03074).
Indeed, in collaboration with NVidia, ``cotengra`` & ``quimb`` have now been used for a state-of-the-art simulation of the Sycamore chip with ``cutensor`` on the Selene supercomputer, [producing a sample from a circuit of depth 20 in less than 10 minutes](https://blogs.nvidia.com/blog/2021/04/12/what-is-quantum-computing/).

---

```{toctree}
:caption: Guides
:maxdepth: 2

installation.md
high-level-interface.ipynb
basics.ipynb
advanced.ipynb
visualization.ipynb
contraction.ipynb
trees.ipynb
index_examples.md
```

```{toctree}
:caption: Development
:hidden:

changelog.md
GitHub Repository <https://github.com/jcmgray/cotengra>
```
