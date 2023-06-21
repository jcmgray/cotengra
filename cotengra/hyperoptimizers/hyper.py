import re
import time
import pickle
import hashlib
import warnings
import importlib
import collections
from math import log2, log10

import numpy as np

from ..oe import PathOptimizer
from ..scoring import get_score_fn
from ..core import (
    ContractionTree,
    ContractionTreeCompressed,
    ContractionTreeMulti,
)
from ..utils import DiskDict, BadTrial
from ..parallel import parse_parallel_arg, get_n_workers, submit, should_nest
from ..plot import plot_trials, plot_trials_alt, plot_scatter, plot_scatter_alt


DEFAULT_METHODS = ["greedy"]
if importlib.util.find_spec("kahypar"):
    DEFAULT_METHODS += ["kahypar"]
else:
    DEFAULT_METHODS += ["labels"]
    warnings.warn(
        "Couldn't import `kahypar` - skipping from default "
        "hyper optimizer and using basic `labels` method instead."
    )


if importlib.util.find_spec("optuna"):
    DEFAULT_OPTLIB = "optuna"
elif importlib.util.find_spec("btb"):
    DEFAULT_OPTLIB = "baytune"
elif importlib.util.find_spec("chocolate"):
    DEFAULT_OPTLIB = "chocolate"
elif importlib.util.find_spec("nevergrad"):
    DEFAULT_OPTLIB = "nevergrad"
elif importlib.util.find_spec("skopt"):
    DEFAULT_OPTLIB = "skopt"
else:
    DEFAULT_OPTLIB = "random"
    warnings.warn(
        "Couldn't find `optuna`, `baytune (btb)`, `chocolate`, "
        "`nevergrad` or `skopt` so will use completely random "
        "sampling in place of hyper-optimization."
    )


_PATH_FNS = {}
_OPTLIB_FNS = {}
_HYPER_SEARCH_SPACE = {}
_HYPER_CONSTANTS = {}


def get_hyper_space():
    global _HYPER_SEARCH_SPACE
    return _HYPER_SEARCH_SPACE


def get_hyper_constants():
    global _HYPER_CONSTANTS
    return _HYPER_CONSTANTS


def register_hyper_optlib(name, init_optimizers, get_setting, report_result):
    global _OPTLIB_FNS
    _OPTLIB_FNS[name] = (init_optimizers, get_setting, report_result)


def register_hyper_function(name, ssa_func, space, constants=None):
    """Register a contraction path finder to be used by the hyper-optimizer.

    Parameters
    ----------
    name : str
        The name to call the method.
    ssa_func : callable
        The raw ``opt_einsum`` style function that returns a 'ContractionTree'.
    space : dict[str, dict]
        The space of hyper-parameters to search.
    """
    global _PATH_FNS
    global _HYPER_SEARCH_SPACE
    global _HYPER_CONSTANTS

    if constants is None:
        constants = {}

    _PATH_FNS[name] = ssa_func
    _HYPER_SEARCH_SPACE[name] = space
    _HYPER_CONSTANTS[name] = constants


def list_hyper_functions():
    """Return a list of currently registered hyper contraction finders."""
    global _PATH_FNS
    return sorted(_PATH_FNS)


def find_tree(*args, **kwargs):
    method = kwargs.pop("method")
    tree = _PATH_FNS[method](*args, **kwargs)
    return {"tree": tree}


class TrialConvertTree:
    def __init__(self, trial_fn, cls):
        self.trial_fn = trial_fn
        self.cls = cls

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)

        tree = trial["tree"]
        if not isinstance(tree, self.cls):
            tree.__class__ = self.cls

        return trial


class TrialTreeMulti:
    def __init__(self, trial_fn, varmults, numconfigs):
        self.trial_fn = trial_fn
        self.varmults = varmults
        self.numconfigs = numconfigs

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)

        tree = trial["tree"]
        if not isinstance(tree, ContractionTreeMulti):
            tree.__class__ = ContractionTreeMulti

        tree.set_varmults(self.varmults)
        tree.set_numconfigs(self.numconfigs)

        return trial


class SlicedTrialFn:
    def __init__(self, trial_fn, **opts):
        self.trial_fn = trial_fn
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        tree = trial["tree"]

        trial.setdefault("original_flops", tree.total_flops())
        trial.setdefault("original_write", tree.total_write())
        trial.setdefault("original_size", tree.max_size())

        tree.slice_(**self.opts)

        trial["flops"] = tree.total_flops()
        trial["write"] = tree.total_write()
        trial["size"] = tree.max_size()

        return trial


class ReconfTrialFn:
    def __init__(self, trial_fn, forested=False, parallel=False, **opts):
        self.trial_fn = trial_fn
        self.forested = forested
        self.parallel = parallel
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        tree = trial["tree"]

        trial.setdefault("original_flops", tree.total_flops())
        trial.setdefault("original_write", tree.total_write())
        trial.setdefault("original_size", tree.max_size())

        if self.forested:
            tree.subtree_reconfigure_forest_(
                parallel=self.parallel, **self.opts
            )
        else:
            tree.subtree_reconfigure_(**self.opts)

        tree.already_optimized.clear()
        trial["flops"] = tree.total_flops()
        trial["write"] = tree.total_write()
        trial["size"] = tree.max_size()

        return trial


class SlicedReconfTrialFn:
    def __init__(self, trial_fn, forested=False, parallel=False, **opts):
        self.trial_fn = trial_fn
        self.forested = forested
        self.parallel = parallel
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        tree = trial["tree"]

        trial.setdefault("original_flops", tree.total_flops())
        trial.setdefault("original_write", tree.total_write())
        trial.setdefault("original_size", tree.max_size())

        if self.forested:
            tree.slice_and_reconfigure_forest_(
                parallel=self.parallel, **self.opts
            )
        else:
            tree.slice_and_reconfigure_(**self.opts)

        tree.already_optimized.clear()
        trial["flops"] = tree.total_flops()
        trial["write"] = tree.total_write()
        trial["size"] = tree.max_size()

        return trial


class CompressedReconfTrial:
    def __init__(self, trial_fn, chi, **opts):
        self.trial_fn = trial_fn
        self.chi = chi
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        tree = trial["tree"]
        tree.compressed_reconfigure_(chi=self.chi, **self.opts)
        return trial


class ComputeScore:
    """The final score wrapper, that performs some simple arithmetic on the
    trial score to make it more suitable for hyper-optimization.
    """

    def __init__(
        self,
        fn,
        score_fn,
        score_compression=0.75,
        score_smudge=1e-6,
        seed=0,
    ):
        self.fn = fn
        self.score_fn = score_fn
        self.score_compression = score_compression
        self.score_smudge = score_smudge
        self.rng = np.random.default_rng(seed)

    def __call__(self, *args, **kwargs):
        ti = time.time()
        try:
            trial = self.fn(*args, **kwargs)
            trial["score"] = self.score_fn(trial) ** self.score_compression
            # random smudge is for baytune/scikit-learn nan/inf bug
            trial["score"] += self.rng.normal(scale=self.score_smudge)
        except BadTrial:
            trial = {
                "score": float("inf"),
                "flops": float("inf"),
                "write": float("inf"),
                "size": float("inf"),
            }
        tf = time.time()
        trial["time"] = tf - ti
        return trial


def progress_description(best):
    return (
        f"log2[SIZE]: {log2(best['size']):.2f} "
        f"log10[FLOPs]: {log10(best['flops']):.2f}"
    )


class HyperOptimizer(PathOptimizer):
    """A path optimizer that samples a series of contraction trees
    while optimizing the hyper parameters used to generate them.

    Parameters
    ----------
    methods : None or sequence[str] or str, optional
        Which method(s) to use from ``list_hyper_functions()``.
    minimize : str, Objective or callable, optional
        How to score each trial, used to train the optimizer and rank the
        results. If a custom callable, it should take a ``trial`` dict as its
        argument and return a single float.
    max_repeats : int, optional
        The maximum number of trial contraction trees to generate.
        Default: 128.
    max_time : None or float, optional
        The maximum amount of time to run for. Use ``None`` for no limit. You
        can also set an estimated execution 'rate' here like ``'rate:1e9'``
        that will terminate the search when the estimated FLOPs of the best
        contraction found divided by the rate is greater than the time spent
        searching, allowing quick termination on easy contractions.
    parallel : 'auto', False, True, int, or distributed.Client
        Whether to parallelize the search.
    slicing_opts : dict, optional
        If supplied, once a trial contraction path is found, try slicing with
        the given options, and then update the flops and size of the trial with
        the sliced versions.
    slicing_reconf_opts : dict, optional
        If supplied, once a trial contraction path is found, try slicing
        interleaved with subtree reconfiguation with the given options, and
        then update the flops and size of the trial with the sliced and
        reconfigured versions.
    reconf_opts : dict, optional
        If supplied, once a trial contraction path is found, try subtree
        reconfiguation with the given options, and then update the flops and
        size of the trial with the reconfigured versions.
    optlib : {'baytune', 'nevergrad', 'chocolate', 'skopt'}, optional
        Which optimizer to sample and train with.
    space : dict, optional
        The hyper space to search, see ``get_hyper_space`` for the default.
    score_compression : float, optional
        Raise scores to this power in order to compress or accentuate the
        differences. The lower this is, the more the selector will sample from
        various optimizers rather than quickly specializing.
    max_training_steps : int, optional
        The maximum number of trials to train the optimizer with. Setting this
        can be helpful when the optimizer itself becomes costly to train (e.g.
        for Gaussian Processes).
    progbar : bool, optional
        Show live progress of the best contraction found so far.
    optlib_opts
        Supplied to the hyper-optimizer library initialization.
    """

    def __init__(
        self,
        methods=None,
        minimize="flops",
        max_repeats=128,
        max_time=None,
        parallel="auto",
        slicing_opts=None,
        slicing_reconf_opts=None,
        reconf_opts=None,
        optlib=DEFAULT_OPTLIB,
        space=None,
        score_compression=0.75,
        max_training_steps=None,
        compressed=False,
        multicontraction=False,
        progbar=False,
        **optlib_opts,
    ):
        self.max_repeats = max_repeats
        self._repeats_start = 0
        self.max_time = max_time
        self.parallel = parallel

        self.method_choices = []
        self.param_choices = []
        self.scores = []
        self.times = []
        self.costs_flops = []
        self.costs_write = []
        self.costs_size = []

        if methods is None:
            self._methods = DEFAULT_METHODS
        elif isinstance(methods, str):
            self._methods = [methods]
        else:
            self._methods = list(methods)

        # which score to feed to the hyper optimizer (setter below handles)
        self.minimize = minimize
        self.compressed = compressed
        self.multicontraction = multicontraction
        self.score_compression = score_compression
        self.best_score = float("inf")
        self.max_training_steps = max_training_steps

        inf = float("inf")
        self.best = {"score": inf, "size": inf, "flops": inf}
        self.trials_since_best = 0

        self.slicing_opts = (
            None if slicing_opts is None else dict(slicing_opts)
        )
        self.reconf_opts = None if reconf_opts is None else dict(reconf_opts)
        self.slicing_reconf_opts = (
            None if slicing_reconf_opts is None else dict(slicing_reconf_opts)
        )
        self.progbar = progbar

        if space is None:
            space = get_hyper_space()

        self._optimizer = dict(
            zip(["init", "get_setting", "report_result"], _OPTLIB_FNS[optlib])
        )

        self._optimizer["init"](self, self._methods, space, **optlib_opts)

    @property
    def minimize(self):
        return self._minimize

    @minimize.setter
    def minimize(self, minimize):
        self._minimize = minimize
        if callable(minimize):
            self._minimize_score_fn = minimize
        else:
            self._minimize_score_fn = get_score_fn(minimize)

    @property
    def parallel(self):
        return self._parallel

    @parallel.setter
    def parallel(self, parallel):
        self._parallel = parallel
        self._pool = parse_parallel_arg(parallel)
        if self._pool is not None:
            self._num_workers = get_n_workers(self._pool)
            self.pre_dispatch = max(
                self._num_workers + 4, int(1.2 * self._num_workers)
            )

    @property
    def tree(self):
        return self.best["tree"]

    @property
    def path(self):
        return self.tree.get_path()

    def setup(self, inputs, output, size_dict):
        trial_fn = find_tree

        if self.compressed:
            assert not self.multicontraction
            trial_fn = TrialConvertTree(trial_fn, ContractionTreeCompressed)

        if self.multicontraction:
            assert not self.compressed
            trial_fn = TrialTreeMulti(trial_fn, self.varmults, self.numconfigs)

        nested_parallel = should_nest(self._pool)

        if self.slicing_opts is not None:
            self.slicing_opts.setdefault("minimize", self.minimize)
            trial_fn = SlicedTrialFn(trial_fn, **self.slicing_opts)

        if self.slicing_reconf_opts is not None:
            self.slicing_reconf_opts.setdefault("minimize", self.minimize)
            self.slicing_reconf_opts.setdefault("parallel", nested_parallel)
            trial_fn = SlicedReconfTrialFn(
                trial_fn, **self.slicing_reconf_opts
            )

        if self.reconf_opts is not None:
            if self.compressed:
                self.reconf_opts.setdefault("minimize", self.minimize)
                trial_fn = CompressedReconfTrial(trial_fn, **self.reconf_opts)
            else:
                self.reconf_opts.setdefault("minimize", self.minimize)
                self.reconf_opts.setdefault("parallel", nested_parallel)
                trial_fn = ReconfTrialFn(trial_fn, **self.reconf_opts)

        # make sure score computation is performed worker side
        trial_fn = ComputeScore(
            trial_fn,
            score_fn=self._minimize_score_fn,
            score_compression=self.score_compression,
        )

        return trial_fn, (inputs, output, size_dict)

    def _maybe_cancel_futures(self):
        if self._pool is not None:
            while self._futures:
                f = self._futures.pop()[-1]
                f.cancel()

    def _maybe_report_result(self, setting, trial):
        score = trial["score"]

        new_best = score < self.best_score
        if new_best:
            self.best_score = score

        # only fit optimizers after the training epoch if the score is best
        if (
            (self.max_training_steps is None)
            or (len(self.scores) < self.max_training_steps)
            or new_best
        ):
            self._optimizer["report_result"](self, setting, trial, score)

        self.method_choices.append(setting["method"])
        self.param_choices.append(setting["params"])
        # keep track of all costs and sizes
        self.costs_flops.append(trial["flops"])
        self.costs_write.append(trial["write"])
        self.costs_size.append(trial["size"])
        self.scores.append(trial["score"])
        self.times.append(trial["time"])

    def _gen_results(self, repeats, trial_fn, trial_args):
        constants = get_hyper_constants()

        for _ in repeats:
            setting = self._optimizer["get_setting"](self)
            method = setting["method"]

            trial = trial_fn(
                *trial_args,
                method=method,
                **setting["params"],
                **constants[method],
            )

            self._maybe_report_result(setting, trial)

            yield trial

    def _get_and_report_next_future(self):
        # scan over the futures, yield whichever completes first
        while True:
            for i in range(len(self._futures)):
                setting, future = self._futures[i]
                if future.done():
                    del self._futures[i]
                    trial = future.result()
                    self._maybe_report_result(setting, trial)
                    return trial
            time.sleep(1e-6)

    def _gen_results_parallel(self, repeats, trial_fn, trial_args):
        constants = get_hyper_constants()
        self._futures = []

        for _ in repeats:
            setting = self._optimizer["get_setting"](self)
            method = setting["method"]

            future = submit(
                self._pool,
                trial_fn,
                *trial_args,
                method=method,
                **setting["params"],
                **constants[method],
            )
            self._futures.append((setting, future))

            if len(self._futures) >= self.pre_dispatch:
                yield self._get_and_report_next_future()

        while self._futures:
            yield self._get_and_report_next_future()

    def _search(self, inputs, output, size_dict):
        self._check_args_against_first_call(inputs, output, size_dict)

        # start a timer?
        if self.max_time is not None:
            t0 = time.time()
            if isinstance(self.max_time, str):
                which, amount = re.match(
                    r"(rate|equil):(.+)", self.max_time
                ).groups()

                if which == "rate":
                    rate = float(amount)

                    def should_stop():
                        return (time.time() - t0) > (self.best["flops"] / rate)

                elif which == "equil":
                    amount = int(amount)

                    def should_stop():
                        return self.trials_since_best > amount

            else:

                def should_stop():
                    return (time.time() - t0) > self.max_time

        else:

            def should_stop():
                return False

        trial_fn, trial_args = self.setup(inputs, output, size_dict)

        r_start = self._repeats_start + len(self.scores)
        r_stop = r_start + self.max_repeats
        repeats = range(r_start, r_stop)

        # create the trials lazily
        if self._pool is not None:
            trials = self._gen_results_parallel(repeats, trial_fn, trial_args)
        else:
            trials = self._gen_results(repeats, trial_fn, trial_args)

        if self.progbar:
            import tqdm

            pbar = tqdm.tqdm(trials, total=self.max_repeats)
            pbar.set_description(
                progress_description(self.best), refresh=False
            )
            trials = pbar

        # assess the trials
        for trial in trials:
            # check if we have found a new best
            if trial["score"] < self.best["score"]:
                self.trials_since_best = 0
                self.best = trial
                self.best["params"] = dict(self.param_choices[-1])
                self.best["params"]["method"] = self.method_choices[-1]

                if self.progbar:
                    pbar.set_description(
                        progress_description(self.best), refresh=False
                    )

            else:
                self.trials_since_best += 1

            # check if we have run out of time
            if should_stop():
                break

        if self.progbar:
            pbar.close()

        self._maybe_cancel_futures()

    def search(self, inputs, output, size_dict):
        """Run this optimizer and return the ``ContractionTree`` for the best
        path it finds.
        """
        self._search(
            inputs,
            output,
            size_dict,
        )
        return self.tree

    def get_tree(self):
        """Return the ``ContractionTree`` for the best path found."""
        return self.tree

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """``opt_einsum`` interface, returns direct ``path``."""
        self._search(
            inputs,
            output,
            size_dict,
        )
        return tuple(self.path)

    def get_trials(self, sort=None):
        trials = list(
            zip(
                self.method_choices,
                self.costs_size,
                self.costs_flops,
                self.costs_write,
                self.param_choices,
            )
        )

        if sort == "method":
            trials.sort(key=lambda t: t[0])
        if sort == "combo":
            trials.sort(
                key=lambda t: log2(t[1]) / 1e3 + log2(t[2] + 256 * t[3])
            )
        if sort == "size":
            trials.sort(
                key=lambda t: log2(t[1]) + log2(t[2]) / 1e3 + log2(t[3]) / 1e3
            )
        if sort == "flops":
            trials.sort(
                key=lambda t: log2(t[1]) / 1e3 + log2(t[2]) + log2(t[3]) / 1e3
            )
        if sort == "write":
            trials.sort(
                key=lambda t: log2(t[1]) / 1e3 + log2(t[2]) / 1e3 + log2(t[3])
            )

        return trials

    def print_trials(self, sort=None):
        header = "{:>11} {:>11} {:>11}     {}"
        print(
            header.format(
                "METHOD",
                "log2[SIZE]",
                "log10[FLOPS]",
                "log10[WRITE]",
                "PARAMS",
            )
        )
        row = "{:>11} {:>11.2f} {:>11.2f} {:>11.2f}    {}"
        for choice, size, flops, write, params in self.get_trials(sort):
            print(
                row.format(
                    choice, log2(size), log10(flops), log10(write), params
                )
            )

    def to_df(self):
        """Create a single ``pandas.DataFrame`` with all trials and scores.
        """
        import pandas

        return pandas.DataFrame(
            data={
                "run": list(range(len(self.costs_size))),
                "time": self.times,
                "method": self.method_choices,
                "size": list(map(log2, self.costs_size)),
                "flops": list(map(log10, self.costs_flops)),
                "write": list(map(log10, self.costs_write)),
                "random_strength": [
                    p.get("random_strength", 1e-6) for p in self.param_choices
                ],
                "score": self.scores,
            }
        ).sort_values(by="method")

    def to_dfs_parametrized(self):
        """Create a ``pandas.DataFrame`` for each method, with all parameters
        and scores for each trial.
        """
        import pandas as pd

        rows = {}
        for i in range(len(self.scores)):
            row = {
                'run': i,
                'time': self.times[i],
                **self.param_choices[i],
                'flops': log10(self.costs_flops[i]),
                'write': log2(self.costs_write[i]),
                'size': log2(self.costs_size[i]),
                'score': self.scores[i],
            }
            method = self.method_choices[i]
            rows.setdefault(method, []).append(row)

        return {
            method: pd.DataFrame(rows[method]).sort_values(by='score')
            for method in rows
        }

    plot_trials = plot_trials
    plot_trials_alt = plot_trials_alt
    plot_scatter = plot_scatter
    plot_scatter_alt = plot_scatter_alt


def sortedtuple(x):
    return tuple(sorted(x))


def make_hashable(x):
    """Make ``x`` hashable by recursively turning list into tuples and dicts
    into sorted tuples of key-value pairs.
    """
    if isinstance(x, list):
        return tuple(map(make_hashable, x))
    if isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    return x


def hash_contraction_a(inputs, output, size_dict):
    if not isinstance(next(iter(size_dict.values()), 1), int):
        # hashing e.g. numpy int won't match!
        size_dict = {k: int(v) for k, v in size_dict.items()}

    return hashlib.sha1(
        pickle.dumps(
            (
                tuple(map(sortedtuple, inputs)),
                sortedtuple(output),
                sortedtuple(size_dict.items()),
            )
        )
    ).hexdigest()


def hash_contraction_b(inputs, output, size_dict):
    # label each index as the sorted tuple of nodes it is incident to
    edges = collections.defaultdict(list)
    for ix in output:
        edges[ix].append(-1)
    for i, term in enumerate(inputs):
        for ix in term:
            edges[ix].append(i)

    # then sort edges by each's incidence nodes
    canonical_edges = sortedtuple(map(sortedtuple, edges.values()))

    return hashlib.sha1(
        pickle.dumps((canonical_edges, sortedtuple(size_dict.items())))
    ).hexdigest()


def hash_contraction(inputs, output, size_dict, method="a"):
    """Compute a hash for a particular contraction geometry."""
    if method == "a":
        return hash_contraction_a(inputs, output, size_dict)
    elif method == "b":
        return hash_contraction_b(inputs, output, size_dict)
    else:
        raise ValueError("Unknown hash method: {}".format(method))


class ReusableHyperOptimizer(PathOptimizer):
    """Like ``HyperOptimizer`` but it will re-instantiate the optimizer
    whenever a new contraction is detected, and also cache the paths found.

    Parameters
    ----------
    directory : None, True, or str, optional
        If specified use this directory as a persistent cache. If ``True`` auto
        generate a directory in the current working directory based on the
        options which are most likely to affect the path (see
        `ReusableHyperOptimizer.get_path_relevant_opts`).
    overwrite : bool, optional
        If ``True``, the optimizer will always run, overwriting old results in
        the cache. This can be used to update paths with deleting the whole
        cache.
    set_surface_order : bool, optional
        If ``True``, when reloading a path to turn into a ``ContractionTree``,
        the 'surface order' of the path (used for compressed paths), will be
        set manually to the order the disk path is.
    hash_method : {'a', ...}, optional
        The method used to hash the contraction tree. The default, ``'a'``, is
        faster but doesn't recognize when indices are permuted.
    cache_only : bool, optional
        If ``True``, the optimizer will only use the cache, and will raise
        ``KeyError`` if a contraction is not found.
    opt_kwargs
        Supplied to ``HyperOptimizer``.
    """

    def __init__(
        self,
        *,
        directory=None,
        overwrite=False,
        set_surface_order=False,
        hash_method="a",
        cache_only=False,
        **opt_kwargs,
    ):
        self._opt = None
        self._opt_kwargs = opt_kwargs
        if directory is True:
            # automatically generate the directory
            directory = f"ctg_cache/opts{self.auto_hash_path_relevant_opts()}"
        self._cache = DiskDict(directory)
        self.overwrite = overwrite
        self._set_surface_order = set_surface_order
        self._hash_method = hash_method
        self.cache_only = cache_only

    @property
    def last_opt(self):
        return self._opt

    def get_path_relevant_opts(self):
        """Get a frozenset of the options that are most likely to affect the
        path. These are the options that we use when the directory name is not
        manually specified.
        """
        return tuple(
            (key, make_hashable(self._opt_kwargs.get(key, default)))
            for key, default in [
                ("methods", None),
                ("minimize", "flops"),
                ("max_repeats", 128),
                ("max_time", None),
                ("slicing_opts", None),
                ("slicing_reconf_opts", None),
                ("reconf_opts", None),
                ("compressed", False),
                ("multicontraction", False),
            ]
        )

    def auto_hash_path_relevant_opts(self):
        """Automatically hash the path relevant options used to create the
        optimizer.
        """
        return hashlib.sha1(
            pickle.dumps(self.get_path_relevant_opts())
        ).hexdigest()

    def hash_query(self, inputs, output, size_dict):
        """Hash the contraction specification, returning this and whether the
        contraction is already present as a tuple.
        """
        h = hash_contraction(inputs, output, size_dict, self._hash_method)
        missing = self.overwrite or (h not in self._cache)
        return h, missing

    def _compute_path(self, inputs, output, size_dict):
        self._opt = HyperOptimizer(**self._opt_kwargs)
        self._opt._search(inputs, output, size_dict)
        return {
            "path": self._opt.path,
            "sliced_inds": self._opt.tree.sliced_inds,
        }

    def update_from_tree(self, tree, overwrite=True):
        """Explicitly add the contraction that ``tree`` represents into the
        cache. For example, if you have manually improved it via reconfing.
        If ``overwrite=False`` and the contracton is present already then do
        nothing.
        """
        h, missing = self.hash_query(tree.inputs, tree.output, tree.size_dict)
        if overwrite or missing:
            self._cache[h] = {
                "path": tree.get_path(),
                "sliced_inds": tree.sliced_inds,
            }

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        h, missing = self.hash_query(inputs, output, size_dict)
        if missing:
            if self.cache_only:
                raise KeyError("Contraction missing from cache.")
            self._cache[h] = self._compute_path(inputs, output, size_dict)
        return self._cache[h]["path"]

    def search(self, inputs, output, size_dict):
        h, missing = self.hash_query(inputs, output, size_dict)

        if missing:
            if self.cache_only:
                raise KeyError("Contraction missing from cache.")
            # run and immediately retrieve tree directly
            self._cache[h] = self._compute_path(inputs, output, size_dict)
            return self._opt.tree

        # reconstruct tree
        con = self._cache[h]

        if self._set_surface_order:
            # need ssa_path to set order
            tree = ContractionTreeCompressed.from_path(
                inputs, output, size_dict, path=con["path"]
            )
        else:
            tree = ContractionTree.from_path(
                inputs, output, size_dict, path=con["path"]
            )

        for ix in con["sliced_inds"]:
            tree.remove_ind_(ix)

        return tree

    def cleanup(self):
        self._cache.cleanup()
