# Installation

`cotengra` is available on both [pypi](https://pypi.org/project/cotengra/) and [conda-forge](https://anaconda.org/conda-forge/cotengra). While `cotengra` is itself pure python, the recommended distribution would be [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) for installing the various optional dependencies.

**Installing with `pip`:**
```bash
pip install cotengra
```

**Installing with `conda`:**
```bash
conda install -c conda-forge cotengra
```

**Installing with `mambaforge`:**
```bash
mamba install cotengra
```

```{hint}
Mamba is a faster version of `conda`, and the -forge distritbution comes pre-configured with only the `conda-forge` channel, which further simplifies and speeds up installing dependencies.
```

**Installing the latest version directly from github:**

If you want to checkout the latest version of features and fixes, you can install directly from the github repository:
```bash
pip install -U git+https://github.com/jcmgray/cotengra.git
```

**Installing a local, editable development version:**

If you want to make changes to the source code and test them out, you can install a local editable version of the package:
```bash
git clone https://github.com/jcmgray/cotengra.git
pip install --no-deps -U -e cotengra/
```

Other than that, the optional dependencies are detailed below.

````{hint}
The recommended selection of optional dependencies from below covering most use-cases is:
```
autoray cytoolz kahypar loky networkx opt_einsum optuna tqdm
```
````

## Optimization

* [`kahypar`](https://github.com/SebastianSchlag/kahypar) - **Karlsruhe Hypergraph Partitioning** for high quality divisive tree building ([available via pip](https://pypi.org/project/kahypar/), unfortunately not yet via `conda` or for windows)
* [`tqdm`](https://github.com/tqdm/tqdm) - for showing live progress (available via [pip](https://pypi.org/project/tqdm/) or `conda`)
* [`cytoolz`](https://github.com/pytoolz/cytoolz/) - a couple of slightly faster utility functions

To perform the hyper-optimization (and not just randomly sample) one of the following libraries is needed:

* [`optuna`](https://github.com/optuna/optuna) - **Tree of Parzen Estimators** used by default
* [`baytune`](https://github.com/HDI-Project/BTB) - *Bayesian Tuning and Bandits* - **Gaussian Processes** used by default
* [`chocolate`](https://chocolate.readthedocs.io/en/latest/) - the **CMAES** optimization algorithm is used by default (`sampler='QuasiRandom'` also useful)
* [`skopt`](https://scikit-optimize.github.io/stable/) - random forest as well as Gaussian process regressors (high quality but slow)
* [`nevergrad`](https://facebookresearch.github.io/nevergrad/) - various population and evolutionary algorithms (v fast & suitable for highly parallel path findings)

If you want to experiment with other algorithms then the following can be used:

* [`python-igraph`](https://igraph.org/python/) - various other community detection algorithms (though no hyperedge support and usually worse performance than `kahypar`).
* [`QuickBB`](https://www.hlt.utdallas.edu/~vgogate/quickbb.html)
* [`FlowCutter`](https://github.com/kit-algo/flow-cutter-pace17)

The latter two are both accessed simply using their command line interface and so the following executables should be placed on the path somewhere:
[`quickbb_64`, `flow_cutter_pace17`].


## Parallelization

The parallel functionality can requires any of the following:

* [`loky`](https://github.com/joblib/loky) either directly or via [`joblib`](https://joblib.readthedocs.io/)
* [`ray`](https://www.ray.io/) - distributed computing
* [`dask distributed`](http://distributed.dask.org) - distributed computing


## Visualization

The following packages enable visualization:

* [`networkx`](https://networkx.org/) for most visualizations
* [`quimb`](https://github.com/jcmgray/quimb) for the 'rubberband' visualization


## Contraction

If you want to perform the contractions using ``cotengra`` itself you'll need:

* [`autoray`](https://github.com/jcmgray/autoray)

which supports at least `numpy`, `cupy`, `torch`, `tensorflow`, `jax`, and `autograd` among others.
