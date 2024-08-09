# Installation

`cotengra` is available on both [pypi](https://pypi.org/project/cotengra/) and [conda-forge](https://anaconda.org/conda-forge/cotengra). While `cotengra` is itself pure python, the recommended distribution would be [miniforge](https://github.com/conda-forge/miniforge) for installing the various optional dependencies.

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
Mamba is a faster version of `conda`, and the -forge distribution comes pre-configured with only the `conda-forge` channel, which further simplifies and speeds up installing dependencies.
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
autoray cmaes cotengrust cytoolz kahypar loky networkx opt_einsum optuna tqdm
```
````

## Contraction

If you want to perform the contractions using ``cotengra`` itself you'll need:

* [`autoray`](https://github.com/jcmgray/autoray)

which supports at least `numpy`, `cupy`, `torch`, `tensorflow`, `jax`, and `autograd` among others.

## Optimization

* [`kahypar`](https://github.com/SebastianSchlag/kahypar) - **Karlsruhe Hypergraph Partitioning** for high quality divisive tree building ([available via pip](https://pypi.org/project/kahypar/), unfortunately not yet via `conda` or for windows)
* [`cotengrust`](https://github.com/jcmgray/cotengrust/tree/main) - rust accelerated pathfinding primitives
* [`tqdm`](https://github.com/tqdm/tqdm) - for showing live progress (available via [pip](https://pypi.org/project/tqdm/) or `conda`)
* [`cytoolz`](https://github.com/pytoolz/cytoolz/) - a couple of slightly faster utility functions

To perform the hyper-optimization (and not just randomly sample) one of the following libraries is needed:

* [`optuna`](https://github.com/optuna/optuna) - **Tree of Parzen Estimators** used by default, high quality but only medium fast
* [`cmaes`](https://github.com/CyberAgentAILab/cmaes) - **Covariance Matrix Adaptation Evolution Strategy**, medium quality but very fast
* [`nevergrad`](https://facebookresearch.github.io/nevergrad/) - various population and evolutionary algorithms (also fast & suitable for highly parallel path findings)
* [`skopt`](https://scikit-optimize.github.io/stable/) - random forest as well as Gaussian process regressors (very high quality but very slow)

If you want to experiment with other algorithms then the following can be used:

* [`python-igraph`](https://igraph.org/python/) - various other community detection algorithms (though no hyperedge support and usually worse performance than `kahypar`).
* [`QuickBB`](https://www.hlt.utdallas.edu/~vgogate/quickbb.html)
* [`FlowCutter`](https://github.com/kit-algo/flow-cutter-pace17)

The latter two are both accessed simply using their command line interface and so the following executables should be placed on the path somewhere:
[`quickbb_64`, `flow_cutter_pace17`].


## Parallelization

The parallel functionality can requires any of the following:

* [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) - python standard library
* [`loky`](https://github.com/joblib/loky) either directly or via [`joblib`](https://joblib.readthedocs.io/)
* [`ray`](https://www.ray.io/) - distributed computing
* [`dask distributed`](http://distributed.dask.org) - distributed computing


## Visualization

The following packages enable visualization:

* [`networkx`](https://networkx.org/) for most visualizations
* [`pygraphviz`](https://pygraphviz.github.io/) for fast and nice graph layouts
