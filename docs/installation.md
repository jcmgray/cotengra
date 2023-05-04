# Installation

Basic requirements are
[`opt_einsum`](https://optimized-einsum.readthedocs.io/) and either
[`cytoolz`](https://github.com/pytoolz/cytoolz) or
[`toolz`](https://github.com/pytoolz/toolz).
To install this package from source, you can clone it
locally, navigate into the source directory and then call:

```
pip install -U .
```
or should you want to edit the source:
```
pip install --no-deps -U -e .
```
To install it directly from github (e.g. in a [colab notebook](https://colab.research.google.com/)):
```
pip install -U git+https://github.com/jcmgray/cotengra.git
```

Other than that, the optional dependencies are detailed below.

````{hint}
The recommended selection of optional dependencies from below covering most use-cases is:
```
kahypar tqdm optuna loky networkx autoray
```
````

## Optimization

* [`kahypar`](https://github.com/SebastianSchlag/kahypar) - **Karlsruhe Hypergraph Partitioning** for high quality divisive tree building ([available via pip](https://pypi.org/project/kahypar/), unfortunately not yet via `conda` or for windows)
* [`tqdm`](https://github.com/tqdm/tqdm) - for showing live progress (available via [pip](https://pypi.org/project/tqdm/) or `conda`)

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
