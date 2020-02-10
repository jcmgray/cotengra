<p align="center"><img src="https://i.imgur.com/3C5EAmr.png" alt="cotengra" width="60%" height="60%"></p>

# `CoTenGra`

Contraction paths for large tensor networks using various graph based methods - compatible with [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/latest/) and so [`quimb`](https://quimb.readthedocs.io/en/latest/). For a full explanation see the paper this project accompanies: [***'Hyper-optimized tensor network contraction'***](https://arxiv.org/abs/2002.01935) - and if this project is useful please consider citing to encourage further developement!

The key methods here repeatedly build a **contraction tree**, using some combination of agglomerative and divisive heuristics, then sample these stochastically whilst having their algorithmic parameters tuned by a Bayesian optimizer so as to target the lowest space or time complexity.

This repository also contains a **tensor network slicing** implementation (for splitting contractions into many independent chunks each with lower memory) that will hopefully be added directly into `opt_einsum` at some point. The slicing can be performed *within* the Bayesian optimization loop to explicitly target contraction paths that slice well to low memory.

## Installation :hammer:

Basic requirements are ``numpy``, ``opt_einsum`` and ``psutil``, other than which the following python packages are used by default:

* [kahypar](https://github.com/SebastianSchlag/kahypar) - **Karlsruhe Hypergraph Partitioning** for high quality divisive tree building
* [baytune](https://github.com/HDI-Project/BTB) - **Baysian Tuning and Bandits** for the hyper optimization

If you want to experiment with other algorithms then the following can be used:

* [python-igraph](https://igraph.org/python/) - various other community detection algorithms (though no hyperedge support and usually worse performance than `kahypar`).
* [QuickBB](https://www.hlt.utdallas.edu/~vgogate/quickbb.html)
* [FlowCutter](https://github.com/kit-algo/flow-cutter-pace17)

The latter two are both accessed simply using their command line interface and so the following executables should be placed on the path somewhere:
[`quickbb_64`, `flow_cutter_pace17`].
## Basic usage :zap:

All the [optimizers are `opt_einsum.PathOptimizer` instances](https://optimized-einsum.readthedocs.io/en/stable/custom_paths.html) which can be supplied as the `optimize=` kwarg to either `opt_einsum` or `quimb`:

### With [`opt_einsum`](https://github.com/dgasmith/opt_einsum)

```python
import opt_einsum as oe
import cotengra as ctg

eq, shapes = oe.helpers.rand_equation(30, 4)
opt = ctg.HyperOptimizer()
path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
```

Various useful information regarding the trials is retained in the optimizer instance `opt`. Once imported, `cotengra` also registers the following named optimizers which can just be specified directly:

```python
[
    'hyper', 'hyper-256', 'hyper-greedy',
    'hyper-kahypar', 'hyper-spinglass', 'hyper-betweenness'
    'flowcutter-2', 'flowcutter-10', 'flowcutter-60'
    'quickbb-2', 'quickbb-10', 'quickbb-60'
]
```

By default, the hyper optimization only trials `'greedy'` and `'kahypar'` (since generally one of these always offers the best performance and they both support hyperedges - indices repeated 3+ times), but you can supply any of the path finders from `list_hyper_functions` via `HyperOptimizer`(`methods=[...])` or register your own.

### With [`quimb`](https://github.com/jcmgray/quimb)

In `quimb` the `optimize` kwarg to contraction functions is passed on to `opt_einsum`.

```python
import quimb.tensor as qtn

mera = qtn.MERA.rand(32)
norm = mera.H & mera

info = norm.contract(all, get='path-info', optimize='hyper')
```

Additionally, if the optimizer is specified like such as a string, ``quimb`` will cache the contraction path found for that particular tensor network shape to be automatically reused. A fuller example using random quantum circuits can be found in the notebook [`examples/Quantum Circuit Example.ipynb`](https://github.com/jcmgray/cotengra/blob/master/examples/Quantum%20Circuit%20Example.ipynb).

> :warning: By convention `opt_einsum` assumes real floating point arithmetic when calculating `info.opt_cost`, if every contraction is an inner product and there are no output indices then this is generally 2x the cost (or number of operations), C, defined in [the paper](https://arxiv.org/abs/2002.01935). The contraction width, W, is just `math.log2(info.largest_intermediate)`.

### Advanced Settings

Various settings can be speicifed when intializing the `HyperOptimizer` object, here are some key ones:

```python
opt = ctg.HyperOptimizer(
    minimize='size',    # {'size', 'flops', 'combo'}, what to target
    parallel=True,      # {bool, int, pool-like} see opt_einsum
    max_time=60,        # maximum seconds to run for (None for no limit)
    max_repeats=1024,   # maximum number of repeats
    progbar=True,       # whether to show live progress
)
```

The Gaussian process based hyper optimization can be turned off by supplying `tuner='Uniform'` or indeed [any `baytune` tuner](https://hdi-project.github.io/BTB/readme.html#tuners).


> :watch: It's worth noting that after **a few thousand trials** fitting and sampling the Gaussian Processes becomes a significant computational task.

## List of optimizer methods

The following are valid initial methods to `HyperOptimizer`(`methods=[...])`, which will tune each one and ultimately select the best performer:

* `'kahypar'` *(default)*
* `'greedy'` *(default)*
* [igraph](https://igraph.org/python/) **partition** based
  * `'spinglass'`
  * `'labelprop'`
* [igraph](https://igraph.org/python/) **dendrogram** based (yield entire communitry strcuture tree at once)
  * `'betweenness'`
  * `'walktrap'`
* **linegraph tree decomposition** based:
  * `'quickbb'`
  * `'flowcutter'`

The linegraph methods are not really hyper-optimized, just repeated with random seeds for different lengths of time (and they also can't take into account bond dimensions). Similarly, some of the igraph methods have no or few parameters to tune.

The following uniform sampling optimizers aliases are also available:

```python
ctg.UniformOptimizer
ctg.UniformGreedy
ctg.UniformKaHyPar
ctg.UniformBetweenness
ctg.UniformSpinglass
```

### Adding your own :evergreen_tree:

`cotengra` has a few features that may be useful for developing new contraction path finders.

1. The `ContractionTree` object - the core data structure that allows one to build the contraction tree bottom-up, top-down, middle-out etc, whilst keeping track of all indices, sizes, costs. A tree is a more abstract and fundamental object than a path.
2. The `PartitionTreeBuilder` object - allows one to define a function that just partitions a subgraph, and then handles all the other details of turning this into a contraction tree builder
3. The `register_hyper_function` function - allows any function that returns a single contraction path to be run in the Bayesian optimization loop

The file ``path_kahypar.py`` is a good example of usage of these second two features.

## Slicing :knife:

Slicing is the technique of choosing some indices to explicitly sum over rather than include as tensor dimensions - thereby taking indexed *slices* of those tensors. It is also known as **variable projection** and **bond cutting**.
It is done for two main reasons:

1. To reduce the amount of memory required
2. To introduce embarassing parallelism


Generally it makes sense to find good indices to slice with respect to some existing contraction path, so the recommended starting point is a ``PathInfo`` or completed ``ContractionTree`` object.

First we set up a contraction:

```python
import math
import numpy as np
import cotengra as ctg
import opt_einsum as oe

eq, shapes = oe.helpers.rand_equation(n=50, reg=5, seed=42, d_max=2)
arrays = [np.random.uniform(size=s) for s in shapes]
print(eq)
# 'ÓøÅÁ,dÛuÃD,WáYÔÎ,EĄÏZ,ÄPåÍ,æuëX,îĆxò,ègMÑza,eNS,À,ðÙÖ,øĂTê,ĂÏĈ,Ąąô,HóÌ,üÂÿØãÈ,íËòPh,hÐÓMç,äCFÁU,mGùbB,ćÐnë,jQãrkñÜ,ÅÚvóćS,IÒöõĀú,äþözbKC,wâTiZ,BfĆd,fàÉìúå,pVwç÷,ÈQXÙGñJ,ðÆôA,qËÜÝyR,ÒüÊnÞ×cÔVþā,ùïà,mßclYÄÌÉx÷,ÃrÕÿ,jolÚ,îosE,æÇD×,ÛăHvýûõ,ÇÎRNØ,WÊÀáÍéê,Âes,èJAÖ,ûÝFÆ,iïíÞtìă,ÕqOL,IāLéUaĈÑg,âKpýOą,tĀyßk->'

path, info = oe.contract_path(eq, *arrays, optimize='greedy')
print(math.log2(info.largest_intermediate))
# 25.0
```

With that we can instantiate the `SliceFinder` object that searches for good indices to slice, here specifying we want the new sliced contraction to have tensors of maximum size `2**20`:

```python
sf = ctg.SliceFinder(info, target_size=2**20)
```

You can specifiy some combination of at least one of the following:

* `target_size`: slice until largest tensor is at maximum this size
* `target_flops`: slice until the overhead reaches this factor
* `target_slices`: slice until this number of slices is generated

Now we can search for indices:

```python
ix_sl, cost_sl = sf.search()
print(ix_sl)
# frozenset({'G', 'd', 'i', 'y', 'Ë', 'ô'})

print(cost_sl)
# <ContractionCosts(flops=2.323e+09, size=1.049e+06, nslices=6.400e+01)>
```

The `SliceFinder` now contains various combinations of indices and the associated `ContractionCosts`, the best of which has been returned. You can also search multiple times, and adjust `temperature` to make it more explorative, before returning `ix_sl, cost_sl = sf.best()` (possibly with different target criteria).

The contraction should now require ~32x less memory, so whats the drawback? We can check the **slicing overhead**:

```python
>>> cost_sl.overhead
1.0464983401972021
```

So less than 5% more floating point operations overall, (arising from redundantly repeated contractions).

To actually perform the sliced contraction we need the `SlicedContractor` object. This is most easily instantiated directly from the `SliceFinder`, from which it will automatically pick up the `.best()` indices and original path, so we just need to give it the arrays:

```python
sc = sf.SlicedContractor(arrays)
```

The different combinations of indices to sum over are enumerated in a deterministic fashion:

```python
results = [
    sc.contract_slice(i)
    for i in range(sc.nslices)
]
```

Which can obviously be embarassingly parallelized as well.
Moreover, a single [`ContractionExpression`](https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.ContractExpression.html#opt_einsum.contract.ContractExpression) is generated to perform each sliced contraction, and supplying a [backend](https://optimized-einsum.readthedocs.io/en/stable/backends.html) kwarg like ``sc.contract_slice(i, backend='jax')`` will result in the same compiled expression being used for every slice.

The sum of all of these is the result of the full contraction:

```python
>>> sum(results)
1.069624485044523e+23
```

(And since the memory of the original contraction is actually manageable we can verify this directly:)

```python
>>> oe.contract(eq, *arrays, optimize=path)
1.069624485044523e+23
```

### Bayesian Sliced Searching :mag:

If one wants to search for contraction paths that slice well than this can be performed *within the Bayesian optimization loop*, with paths being judged on their **sliced** cost and width.

For example, if we are only interested in paths that might fit in GPU memory then we can specify our contraction optimizer like so:

```python
opt = ctg.HyperOptimizer(
    minimze='flops',
    slicing_opts={'target_size': 2**27},
)
```

The slicing performed within the loop is a cheap single shot version, once you have the path you'll need to slice it again with a more exhaustive search yourself, see above.

## Visualization

``cotengra`` has the following functions to help visualize what the hyperoptimizer and contraction trees are doing:

* `HyperOptimizer.plot_trials` - progress of the Bayesian optimizer
* `HyperOptimizer.plot_scatter` - relationship between FLOPs and SIZE of trial paths
* `ContractionTree.plot_ring` - tree plotted as sorted ring with chords for original TN
* `ContractionTree.plot_tent` - tree plotted above original TN
* `ContractionTree.plot_contractions` - relationship between SIZE and FLOPs for each contraction in a tree
* `SliceFinder.plot_slicings` - explore relation between saved memory and increased cost of sliced indices. You can also supply the sequence of sliced indices to the tree plotters as the ``highlight=`` kwarg to directly visualize where they are.

<p align="left"><img src="https://imgur.com/idOvZSy.png" alt="cotengra" width="100%" height="100%"></p>

These are all illustrated in the example notebook [`examples/Quantum Circuit Example.ipynb`](https://github.com/jcmgray/cotengra/blob/master/examples/Quantum%20Circuit%20Example.ipynb).

## Possible ToDos

* [ ] tree fusing
* [ ] subtree reconfiguration
* [ ] improve slicing performance (`numpy` or `cython`?)
* [ ] early pruning with the `'Uniform'` optimizers
* [ ] investigate different KaHyPar profiles and partitioners
* [ ] compute more relevant peak memory of contraction, not just 'width', W
* [ ] further improve `ContractionTree` efficiency (move to `opt_einsum`?) with index counting

Contributions welcome! Consider opening an issue to discuss first.
