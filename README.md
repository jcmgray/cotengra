<p align="center"><img src="https://i.imgur.com/3C5EAmr.png" alt="cotengra" width="60%" height="60%"></p>

# `CoTenGra`

Contraction paths for large tensor networks using various graph based methods - compatible with [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/latest/) and so [`quimb`](https://quimb.readthedocs.io/en/latest/). For a full explanation see the paper this project accompanies: [***'Hyper-optimized tensor network contraction'***](https://arxiv.org/abs/2002.01935) - and if this project is useful please consider citing to encourage further development!

The key methods here repeatedly build a **contraction tree**, using some combination of agglomerative and divisive heuristics, then sample these stochastically whilst having their algorithmic parameters tuned by a Bayesian optimizer so as to target the lowest space or time complexity.

This repository also contains a **tensor network slicing** implementation (for splitting contractions into many independent chunks each with lower memory) that will hopefully be added directly into `opt_einsum` at some point. The slicing can be performed *within* the Bayesian optimization loop to explicitly target contraction paths that slice well to low memory.

>  :warning: Commits since [906f838](https://github.com/jcmgray/cotengra/commit/906f838d21840dc41c652efb63faf1f88c6113ee) *additionally* add functionality first demonstrated in ['Classical Simulation of Quantum Supremacy Circuits'](https://arxiv.org/abs/2005.06787) - namely, contraction subtree reconfiguration and the interleaving of this with slicing.



* [Installation :hammer:](#installation-hammer)

* [Basic usage :zap:](#basic-usage-zap)
  
  + [With `opt_einsum`](#with-opt-einsum-https-githubcom-dgasmith-opt-einsum)
  + [With `quimb`](#with-quimb-https-githubcom-jcmgray-quimb)
  + [Advanced Settings](#advanced-settings)
  
* [List of optimizer methods](#list-of-optimizer-methods)
  
  + [Adding your own :evergreen_tree:](#adding-your-own-evergreen_tree)
  
* [Tree Modifications](#tree-modifications)
  
  + [Basic Slicing :knife:](#basic-slicing-knife)
  + [Subtree Reconfiguration :wrench:](#subtree-reconfiguration-wrench)
  + [Sliced Subtree Reconfiguration](#sliced-subtree-reconfiguration)
  + [Bayesian Sliced'n'Reconfed Searching :mag:](#bayesian-sliced-n-reconfed-searching-mag)
  
* [Parallelization](#parallelization)

* [Reusing and Caching Paths :card_file_box:](#reusing-and-caching-paths-card_file_box)

* [Visualization](#visualization)

* [Possible ToDos](#possible-todos)

    

## Installation :hammer:

Basic requirements are ``opt_einsum`` and either `cytoolz` or `toolz`. Other than that the following python packages are recommended:

* [kahypar](https://github.com/SebastianSchlag/kahypar) - **Karlsruhe Hypergraph Partitioning** for high quality divisive tree building
* [tqdm]() - for showing live progress

To perform the hyper-optimization (and not just randomly sample) one of the following libraries is needed:

* [baytune](https://github.com/HDI-Project/BTB) - *Bayesian Tuning and Bandits* - **Gaussian Processes** used by default
* [chocolate](https://chocolate.readthedocs.io/en/latest/) - the **CMAES** optimization algorithm is used by default (`sampler='QuasiRandom'` also useful)
* [skopt](https://scikit-optimize.github.io/stable/) - random forest as well as Gaussian process regressors (high quality but slow)
* [nevergrad](https://facebookresearch.github.io/nevergrad/) - various population and evolutionary algorithms (v fast & suitable for highly parallel path findings)

If you want to experiment with other algorithms then the following can be used:

* [python-igraph](https://igraph.org/python/) - various other community detection algorithms (though no hyperedge support and usually worse performance than `kahypar`).
* [QuickBB](https://www.hlt.utdallas.edu/~vgogate/quickbb.html)
* [FlowCutter](https://github.com/kit-algo/flow-cutter-pace17)

The latter two are both accessed simply using their command line interface and so the following executables should be placed on the path somewhere:
[`quickbb_64`, `flow_cutter_pace17`].

If you want to automatically cache paths to disk, you'll need:

* [diskcache](http://www.grantjenks.com/docs/diskcache/index.html)

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
    'hyper', 'hyper-256', 'hyper-greedy', 'hyper-labels',
    'hyper-kahypar', 'hyper-spinglass', 'hyper-betweenness'
    'flowcutter-2', 'flowcutter-10', 'flowcutter-60'
    'quickbb-2', 'quickbb-10', 'quickbb-60'
]
```

By default, the hyper optimization only trials `'greedy'` and `'kahypar'` (since generally one of these always offers the best performance and they both support hyper-edges - indices repeated 3+ times), but you can supply any of the path finders from `list_hyper_functions` via `HyperOptimizer`(`methods=[...])` or register your own.

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

Various settings can be specified when initializing the `HyperOptimizer` object, here are some key ones:

```python
opt = ctg.HyperOptimizer(
    minimize='size',    # {'size', 'flops', 'combo'}, what to target
    parallel=True,      # {'auto', bool, int, distributed.Client}
    max_time=60,        # maximum seconds to run for (None for no limit)
    max_repeats=1024,   # maximum number of repeats
    progbar=True,       # whether to show live progress
    optlib='nevergrad'  # optimization library to use
    sampler='de'        # which e.g. nevergrad algo to use
)
```

Options for `optlib` are (if you have them installed):

* `'random'` - a pure python random sampler (aliased to `ctg.UniformOptimizer`)

* `'baytune'` - by default configured to use **Gaussian Processes**
* `'chocolate'` - by default configured to use **CMAES**
* `'skopt'` - by default configured to use **Extra Trees**
* `'nevergrad'` - by default configured to use **Test-Based Population Size Adaptation**


> :watch: It's worth noting that after **a few thousand trials** fitting and sampling e.g. a Gaussian Process model becomes a significant computational task - in this case you may want to try the `'nevergrad'` optimizer or a random or quasi random approach.

## List of optimizer methods

The following are valid initial methods to `HyperOptimizer`(`methods=[...])`, which will tune each one and ultimately select the best performer:

* `'kahypar'` *(default)*
* `'greedy'` *(default)*
* `'labels'` (*default if `kahypar` not installed* ) - a pure python implementation of hypergraph community detection
* [igraph](https://igraph.org/python/) **partition** based
  * `'spinglass'`
  * `'labelprop'`
* [igraph](https://igraph.org/python/) **dendrogram** based (yield entire community structure tree at once)
  * `'betweenness'`
  * `'walktrap'`
* **linegraph tree decomposition** based:
  * `'quickbb'`
  * `'flowcutter'`

The linegraph methods are not really hyper-optimized, just repeated with random seeds for different lengths of time (and they also can't take into account bond dimensions). Similarly, some of the igraph methods have no or few parameters to tune.

### Adding your own :evergreen_tree:

`cotengra` has a few features that may be useful for developing new contraction path finders.

1. The `ContractionTree` object - the core data structure that allows one to build the contraction tree bottom-up, top-down, middle-out etc, whilst keeping track of all indices, sizes, costs. A tree is a more abstract and fundamental object than a path.
2. The `PartitionTreeBuilder` object - allows one to define a function that just partitions a subgraph, and then handles all the other details of turning this into a contraction tree builder
3. The `register_hyper_function` function - allows any function that returns a single contraction path to be run in the Bayesian optimization loop

The file ``path_kahypar.py`` is a good example of usage of these second two features.

4. The `register_hyper_optlib` function is used to register hyper-optimizers.



## Tree Modifications

### Basic Slicing :knife:

Slicing is the technique of choosing some indices to explicitly sum over rather than include as tensor dimensions - thereby taking indexed *slices* of those tensors. It is also known as **variable projection** and **bond cutting**.



<p align="center"><img src="https://imgur.com/402ivDV.png" alt="cotengra" width="80%" height="80%"></p>





It is done for two main reasons:

1. **To reduce the amount of memory required**
2. **To introduce embarrassing parallelism**


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

You can specify some combination of at least one of the following:

* `target_size`: slice until largest tensor is at maximum this size
* `target_overhead`: slice until the overhead reaches this factor
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

> :evergreen_tree: You can directly slice a `ContractionTree` with the `slice` method (and in-place `slice_` version). This instantiates the `SliceFinder` object above, performs the search, and calls `ContractionTree.remove_ind` for each index sliced. The tree object keeps track of the sliced indices (`tree.sliced_inds`) and size of the outer sum (`tree.multiplicity`) so that `tree.total_flops()` gives the cost of contracting all slices.

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

Which can obviously be embarrassingly parallelized as well.
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

### Subtree Reconfiguration :wrench:

Any subtree of a contraction tree itself describes a smaller contraction, with the subtree leaves being the effective inputs (generally intermediate tensors) and the subtree root being the effective output (also generally an intermediate). One advantage of `cotengra` keeping an explicit representation of the contraction tree is that such subtrees can be easily selected and re-optimized as illustrated in the following schematic:

<p align="center"><img src="https://imgur.com/vi60oh7.png" alt="cotengra" width="100%" height="100%"></p>

If we do this and improve the contraction cost of a **subtree** (e.g. by using an optimal contraction path), then the contraction cost of the **whole tree** is improved. Moreover we can iterate across many or all subtrees in a `ContractionTree`, reconfiguring them and thus potentially updating the entire tree in incremental 'local' steps.

Here's an example:

```python
import opt_einsum as oe
import cotengra as ctg

# generate a tree
eq, shapes = oe.helpers.rand_equation(100, 3, seed=42)
path, info = oe.contract_path(eq, *shapes, shapes=True)
tree = ctg.ContractionTree.from_info(info)

# reconfigure it (call tree.subtree_reconfigure? to see many options)
tree_r = tree.subtree_reconfigure(progbar=True)
# log2[SIZE]: 41.33 log10[FLOPs]: 14.68: : 208it [00:00, 631.81it/s]

# check the speedup
tree.total_flops() / tree_r.total_flops()
# 12803.1423636353
```

Since it is a local optimization it is possible to get stuck. `ContractionTree.subtree_reconfigure_forest` offers a basic stochastic search of multiple reconfigurations that can avoid this and also be easily parallelized:

```python
tree_f = tree.subtree_reconfigure_forest(progbar=True)
# log2[SIZE]: 39.24 log10[FLOPs]: 14.54: 100%|██████████| 10/10 [00:22<00:00,  2.22s/it]

# check the speedup
tree.total_flops() / tree_f.total_flops()
# 17928.323601154407
```

So indeed a little better. 

Subtree reconfiguration is often powerful enough to allow even 'bad' initial paths (like those generated by `'greedy'` ) to become very high quality.



### Sliced Subtree Reconfiguration

A powerful application for reconfiguration (first implemented in ['Classical Simulation of Quantum Supremacy Circuits'](https://arxiv.org/abs/2005.06787)) is to interleave it with *slicing*. Namely:

1. Choose an index to slice
2. Reconfigure subtrees to account for the slightly different TN structure without this index
3. Check if the tree has reached a certain size, if not return to 1.

In this way, the contraction tree is slowly updated to account for potentially many indices being sliced.

For example imagine we wanted to slice the ``tree_f`` from above to achieve a maximum size of `2**28` (approx suitable for 8GB of memory). We could directly slice it without changing the tree structure at all:

```python
tree_s = tree_f.slice(target_size=2**28)
tree_s.sliced_inds
# ('k', 'ì', 'W', 'O', 'Ñ', 'o')
```

Or we could simultaneously interleave subtree reconfiguration:

```python
tree_sr = tree_f.slice_and_reconfigure(target_size=2**28, progbar=True)
# log2[SIZE]: 27.51 log10[FLOPs]: 14.76: : 5it [00:00, 12.43it/s]

tree_sr.sliced_inds
# ('o', 'W', 'O', 'Ñ', 'å')

tree_s.total_flops() / tree_sr.total_flops()
# 2.29912454
```

We can see it has achieved the target size with 1 less index sliced, and 2.3x better cost. There is also a 'forested' version of this algorithm  which again performs a stochastic search of multiple possible slicing+reconfiguring options:

```python
tree_fsr = tree_f.slice_and_reconfigure_forest(target_size=2**28, progbar=True)
# log2[SIZE]: 26.87 log10[FLOPs]: 14.72: : 11it [00:01,  7.25it/s]

tree_s.total_flops() / tree_fsr.total_flops()
# 2.530230094281716
```

We can see here it has done a little better. The foresting looks roughly like the following:

<p align="center"><img src="https://imgur.com/s1iSS1u.png" alt="cotengra" width="100%" height="100%"></p>

The subtree reconfiguration within the slicing can *itself be forested* for a doubly forested algorithm. This will give the highest quality (but also slowest) search.

```python
tree_fsfr = tree_f.slice_and_reconfigure_forest(
    target_size=2**28,
    num_trees=4,
    progbar=True, 
    reconf_opts={
        'subtree_size': 12,
        'forested': True,
        'num_trees': 4,
    }
)
# log2[SIZE]: 26.87 log10[FLOPs]: 14.71: : 11it [01:44,  9.52s/it]

tree_s.total_flops() / tree_fsfr.total_flops()
# 2.5980093432674374
```

We've set the `subtree_size` here to `12` for higher quality reconfiguration, but reduced the `num_trees` in the forests (from default `8`) to `4` which will still lead to 4 x 4 = 16 trees being generated at each step. Again we see a slight improvement. This level of effort might only be required for very heavily slicing contraction trees, and in this case it might be best simply to trial many initial paths with a basic `slice_and_reconfigure` (see below).



### Bayesian Sliced'n'Reconfed Searching :mag:

If one wants to search for contraction paths that slice well than this can be performed *within the Bayesian optimization loop*, with paths being judged on their **sliced** cost and width. Similarly, subtree reconfiguration and sliced reconfiguration can be performed within the loop.

---

For example, if we are only interested in paths that might fit in GPU memory then we can specify our contraction optimizer to *slice* each trial contraction tree like so:

```python
opt = ctg.HyperOptimizer(slicing_opts={'target_size': 2**27})
```

Or if we want paths with at least 1024 independent contractions we could use:

```python
opt = ctg.HyperOptimizer(slicing_opts={'target_slices': 2**10})
```

The sliced tree can be retrieved from `tree = opt.best['tree']` which will have `tree.sliced_inds`, or you can re-slice the final returned path yourself.

---

One can also perform *subtree reconfiguration* on each trial tree before it is returned to the Bayesian optimizer by specifying `reconf_opts`:

```python
opt = ctg.HyperOptimizer(reconf_opts={})                  # default reconf
opt = ctg.HyperOptimizer(reconf_opts={'forested': True})  # forested reconf
```

There are generally not any drawbacks to doing this apart from extra path-finding run-time.

---

Similarly, one can perform *sliced reconfiguration* on each trial tree and train on the sliced cost of this:

```python
opt = ctg.HyperOptimizer(slicing_reconf_opts={'target_size': 2**27})
```

A pretty extensive hour long search might look like:

```python
opt = ctg.HyperOptimizer(
    max_repeats=1_000_000,
    max_time=3600,
    slicing_reconf_opts={
        'target_size': 2**27,
        'forested': True,
        'num_trees': 2,
        # these are the reconf opts applied *in-between* slice
        'reconf_opts': {
            'forested': True,
            'num_trees': 2,
        }
    }
)
```

---

Finally all three of these can be applied successively:

```python
opt = ctg.HyperOptimizer(
    slicing_opts={'target_size': 2**40},         # first do basic slicing
    slicing_reconf_opts={'target_size': 2**28},  # then advanced slicing with reconfiguring
    reconf_opts={'subtree_size': 14},            # then finally just higher quality reconfiguring
)
```

## Parallelization

The trials of the hyper-optimizer can be trivially parallelized over (as long as the `optlib` backend can fit and suggest trials quickly enough - for many more than 8 processes you might try `(optlib='random')`, `(optlib='chocolate', sampler='QuasiRandom')` or `(optlib='nevergrad')`. 

Both the 'forested' algorithms above can also be parallelized and nested within the hyper-optimizer - i.e. there can be parallel calls from the hyper-optimizer, calling the parallel the version of the slice and reconfiguring, each calling the parallel version of subtree reconfiguring.

This nested parallelism is enabled by using [`dask.distributed`](https://distributed.dask.org/en/latest/) which should also hopefully allow scaling up the whole process to large clusters.

`HyperOptimizer`,  `subtree_reconfigure_forest`, and `slice_and_reconfigure_forest` all take a `parallel` kwarg with the following options:

* `'auto'` : look for a globally registered `distributed.Client` and use it if found (the default)
* `True` : like `'auto'`, but create a `distributed.Client` if not found (which will then be picked up by subsequent calls)
* an `int` : like `True` but create the client with this many worker processes (ignored if it already exists)
* `False` : don't use any parallelism

As such, simply calling

```python
from distributed import Client
client = Client()  # creates a dask-scheduler and dask-workers automatically
```

should enable parallelism automatically where possible.

## Reusing and Caching Paths :card_file_box:

If you've found `HyperOptimizer` settings you would like to re-use for many contractions then you can create a `ReusableHyperOptimizer` with the settings, e.g.

```python
opt = ReusableHyperOptimizer(
	max_repeats=16, parallel=True, reconf_opts={},
    directory='ctg_path_cache',  # None for non-persistent caching
)
```

This creates a `HyperOptimizer` for each contraction it sees, finds a path, then caches the result so nothing needs to be done the next time the contraction is encountered. If you supply the `directory` option (and have `diskcache` installed), the cache can persist on-disk between sessions. 



## Visualization

``cotengra`` has the following functions to help visualize what the hyper-optimizer and contraction trees are doing:

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
* [ ] early pruning with the `'Uniform'` optimizers
* [ ] investigate different KaHyPar profiles and partitioners
* [ ] compute more relevant peak memory of contraction, not just 'width', W
* [x] ~~further improve `ContractionTree` efficiency (move to `opt_einsum`?) with index counting~~
* [x] ~~subtree reconfiguration~~
* [x] ~~improve slicing performance (`numpy` or `cython`?)~~

Contributions welcome! Consider opening an issue to discuss first.
