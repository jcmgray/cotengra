"""Base hyper optimization functionality."""

import functools
import importlib
import re
import time
import warnings
from math import log2, log10

from ..core import (
    ContractionTree,
    ContractionTreeCompressed,
)
from ..core_multi import ContractionTreeMulti
from ..oe import PathOptimizer
from ..parallel import get_n_workers, parse_parallel_arg, should_nest, submit
from ..plot import (
    plot_parameters_parallel,
    plot_scatter,
    plot_scatter_alt,
    plot_trials,
    plot_trials_alt,
)
from ..reusable import ReusableOptimizer
from ..scoring import get_score_fn
from ..utils import BadTrial, get_rng


@functools.lru_cache(maxsize=None)
def get_default_hq_methods():
    methods = ["greedy"]
    if importlib.util.find_spec("kahypar"):
        methods.append("kahypar")
    else:
        methods.append("labels")
        warnings.warn(
            "Couldn't import `kahypar` - skipping from default "
            "hyper optimizer and using basic `labels` method instead."
        )
    return tuple(methods)


@functools.lru_cache(maxsize=None)
def get_default_optlib_eco():
    """Get the default optimizer favoring speed."""
    if importlib.util.find_spec("cmaes"):
        optlib = "cmaes"
    elif importlib.util.find_spec("nevergrad"):
        optlib = "nevergrad"
    elif importlib.util.find_spec("optuna"):
        optlib = "optuna"
    else:
        optlib = "random"
        warnings.warn(
            "Couldn't find `optuna`, `cmaes`, "
            "or `nevergrad` so will use completely random "
            "sampling in place of hyper-optimization."
        )
    return optlib


@functools.lru_cache(maxsize=None)
def get_default_optlib():
    """Get the default optimizer balancing quality and speed."""
    if importlib.util.find_spec("optuna"):
        optlib = "optuna"
    elif importlib.util.find_spec("cmaes"):
        optlib = "cmaes"
    elif importlib.util.find_spec("nevergrad"):
        optlib = "nevergrad"
    else:
        optlib = "random"
        warnings.warn(
            "Couldn't find `optuna`, `cmaes`, "
            "or `nevergrad` so will use completely random "
            "sampling in place of hyper-optimization."
        )
    return optlib


_PATH_FNS = {}
_OPTLIB_FNS = {}
_HYPER_SEARCH_SPACE = {}
_HYPER_CONSTANTS = {}


def get_hyper_space():
    return _HYPER_SEARCH_SPACE


def get_hyper_constants():
    return _HYPER_CONSTANTS


def register_hyper_optlib(name, init_optimizers, get_setting, report_result):
    _OPTLIB_FNS[name] = (init_optimizers, get_setting, report_result)


def register_hyper_function(name, ssa_func, space, constants=None):
    """Register a contraction path finder to be used by the hyper-optimizer.

    Parameters
    ----------
    name : str
        The name to call the method.
    ssa_func : callable
        The raw function that returns a 'ContractionTree', with signature
        ``(inputs, output, size_dict, **kwargs)``.
    space : dict[str, dict]
        The space of hyper-parameters to search.
    """
    if constants is None:
        constants = {}

    _PATH_FNS[name] = ssa_func
    _HYPER_SEARCH_SPACE[name] = space
    _HYPER_CONSTANTS[name] = constants


def list_hyper_functions():
    """Return a list of currently registered hyper contraction finders."""
    return sorted(_PATH_FNS)


def base_trial_fn(inputs, output, size_dict, method, **kwargs):
    tree = _PATH_FNS[method](inputs, output, size_dict, **kwargs)
    return {"tree": tree}


class TrialSetObjective:
    def __init__(self, trial_fn, objective):
        self.trial_fn = trial_fn
        self.objective = objective

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        trial["tree"].set_default_objective(self.objective)
        return trial


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

        stats = tree.contract_stats()
        trial.setdefault("original_flops", stats["flops"])
        trial.setdefault("original_write", stats["write"])
        trial.setdefault("original_size", stats["size"])

        tree.slice_(**self.opts)
        trial.update(tree.contract_stats())

        return trial


class SimulatedAnnealingTrialFn:
    def __init__(self, trial_fn, **opts):
        self.trial_fn = trial_fn
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        tree = trial["tree"]
        stats = tree.contract_stats()
        trial.setdefault("original_flops", stats["flops"])
        trial.setdefault("original_write", stats["write"])
        trial.setdefault("original_size", stats["size"])
        tree.simulated_anneal_(**self.opts)
        trial.update(tree.contract_stats())
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

        stats = tree.contract_stats()
        trial.setdefault("original_flops", stats["flops"])
        trial.setdefault("original_write", stats["write"])
        trial.setdefault("original_size", stats["size"])

        if self.forested:
            tree.subtree_reconfigure_forest_(
                parallel=self.parallel, **self.opts
            )
        else:
            tree.subtree_reconfigure_(**self.opts)

        tree.already_optimized.clear()
        trial.update(tree.contract_stats())

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

        stats = tree.contract_stats()
        trial.setdefault("original_flops", stats["flops"])
        trial.setdefault("original_write", stats["write"])
        trial.setdefault("original_size", stats["size"])

        if self.forested:
            tree.slice_and_reconfigure_forest_(
                parallel=self.parallel, **self.opts
            )
        else:
            tree.slice_and_reconfigure_(**self.opts)

        tree.already_optimized.clear()
        trial.update(tree.contract_stats())

        return trial


class CompressedReconfTrial:
    def __init__(self, trial_fn, minimize=None, **opts):
        self.trial_fn = trial_fn
        self.minimize = minimize
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)
        tree = trial["tree"]
        tree.windowed_reconfigure_(minimize=self.minimize, **self.opts)
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
        on_trial_error="warn",
        seed=0,
    ):
        self.fn = fn
        self.score_fn = score_fn
        self.score_compression = score_compression
        self.score_smudge = score_smudge
        self.on_trial_error = {
            "raise": "raise",
            "warn": "warn",
            "ignore": "ignore",
        }[on_trial_error]
        self.rng = get_rng(seed)

    def __call__(self, *args, **kwargs):
        ti = time.time()
        try:
            trial = self.fn(*args, **kwargs)
            trial["score"] = self.score_fn(trial) ** self.score_compression
            # random smudge is for baytune/scikit-learn nan/inf bug
            trial["score"] += self.rng.gauss(0.0, self.score_smudge)
        except BadTrial:
            trial = {
                "score": float("inf"),
                "flops": float("inf"),
                "write": float("inf"),
                "size": float("inf"),
            }
        except Exception as e:
            if self.on_trial_error == "raise":
                raise e
            elif self.on_trial_error == "warn":
                warnings.warn(
                    f"Trial error: {e}. Set `HyperOptimizer` kwarg "
                    "`on_trial_error='raise'` to raise this error, or "
                    "`on_trial_error='ignore'` to silence."
                )
            trial = {
                "score": float("inf"),
                "flops": float("inf"),
                "write": float("inf"),
                "size": float("inf"),
            }

        tf = time.time()
        trial["time"] = tf - ti
        return trial


def progress_description(best, info="concise"):
    try:
        return best["tree"].describe(info=info)
    except KeyError:
        return (
            f"log10[FLOPs]={log10(best['flops']):.2f} "
            f"log2[SIZE]={log2(best['size']):.2f}"
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
    optlib : {'optuna', 'cmaes', 'nevergrad', 'skopt', ...}, optional
        Which optimizer to sample and train with.
    space : dict, optional
        The hyper space to search, see ``get_hyper_space`` for the default.
    score_compression : float, optional
        Raise scores to this power in order to compress or accentuate the
        differences. The lower this is, the more the selector will sample from
        various optimizers rather than quickly specializing.
    on_trial_error : {'warn', 'raise', 'ignore'}, optional
        What to do if a trial fails. If ``'warn'`` (default), a warning will be
        printed and the trial will be given a score of ``inf``. If ``'raise'``
        the error will be raised. If ``'ignore'`` the trial will be given a
        score of ``inf`` silently.
    max_training_steps : int, optional
        The maximum number of trials to train the optimizer with. Setting this
        can be helpful when the optimizer itself becomes costly to train (e.g.
        for Gaussian Processes).
    progbar : bool, optional
        Show live progress of the best contraction found so far.
    optlib_opts
        Supplied to the hyper-optimizer library initialization.
    """

    compressed = False
    multicontraction = False

    def __init__(
        self,
        methods=None,
        minimize="flops",
        max_repeats=128,
        max_time=None,
        parallel="auto",
        simulated_annealing_opts=None,
        slicing_opts=None,
        slicing_reconf_opts=None,
        reconf_opts=None,
        optlib=None,
        space=None,
        score_compression=0.75,
        on_trial_error="warn",
        max_training_steps=None,
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
            self._methods = get_default_hq_methods()
        elif isinstance(methods, str):
            self._methods = [methods]
        else:
            self._methods = list(methods)

        if optlib is None:
            optlib = get_default_optlib()

        # which score to feed to the hyper optimizer (setter below handles)
        self.minimize = minimize
        self.score_compression = score_compression
        self.on_trial_error = on_trial_error
        self.best_score = float("inf")
        self.max_training_steps = max_training_steps

        inf = float("inf")
        self.best = {"score": inf, "size": inf, "flops": inf}
        self.trials_since_best = 0

        self.simulated_annealing_opts = (
            None
            if simulated_annealing_opts is None
            else dict(simulated_annealing_opts)
        )
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
            self.objective = minimize
        else:
            self.objective = get_score_fn(minimize)

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
        trial_fn = TrialSetObjective(base_trial_fn, self.objective)

        if self.compressed:
            assert not self.multicontraction
            trial_fn = TrialConvertTree(trial_fn, ContractionTreeCompressed)

        if self.multicontraction:
            assert not self.compressed
            trial_fn = TrialTreeMulti(trial_fn, self.varmults, self.numconfigs)

        nested_parallel = should_nest(self._pool)

        if self.simulated_annealing_opts is not None:
            trial_fn = SimulatedAnnealingTrialFn(
                trial_fn, **self.simulated_annealing_opts
            )

        if self.slicing_opts is not None:
            trial_fn = SlicedTrialFn(trial_fn, **self.slicing_opts)

        if self.slicing_reconf_opts is not None:
            self.slicing_reconf_opts.setdefault("parallel", nested_parallel)
            trial_fn = SlicedReconfTrialFn(
                trial_fn, **self.slicing_reconf_opts
            )

        if self.reconf_opts is not None:
            if self.compressed:
                trial_fn = CompressedReconfTrial(trial_fn, **self.reconf_opts)
            else:
                self.reconf_opts.setdefault("parallel", nested_parallel)
                trial_fn = ReconfTrialFn(trial_fn, **self.reconf_opts)

        # make sure score computation is performed worker side
        trial_fn = ComputeScore(
            trial_fn,
            score_fn=self.objective,
            score_compression=self.score_compression,
            on_trial_error=self.on_trial_error,
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
        should_report = (
            (self.max_training_steps is None)
            or (len(self.scores) < self.max_training_steps)
            or new_best
        ) and (
            # don't report bad trials
            # XXX: should we map to some high value?
            trial["score"] < float("inf")
        )

        if should_report:
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
        """Create a single ``pandas.DataFrame`` with all trials and scores."""
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
                "run": i,
                "time": self.times[i],
                **self.param_choices[i],
                "flops": log10(self.costs_flops[i]),
                "write": log2(self.costs_write[i]),
                "size": log2(self.costs_size[i]),
                "score": self.scores[i],
            }
            method = self.method_choices[i]
            rows.setdefault(method, []).append(row)

        return {
            method: pd.DataFrame(rows[method]).sort_values(by="score")
            for method in rows
        }

    plot_trials = plot_trials
    plot_trials_alt = plot_trials_alt
    plot_scatter = plot_scatter
    plot_scatter_alt = plot_scatter_alt
    plot_parameters_parallel = plot_parameters_parallel


class ReusableHyperOptimizer(ReusableOptimizer):
    """Like ``HyperOptimizer`` but it will re-instantiate the optimizer
    whenever a new contraction is detected, and also cache the paths (and
    sliced indices) found.

    Parameters
    ----------
    directory : None, True, or str, optional
        If specified use this directory as a persistent cache. If ``True`` auto
        generate a directory in the current working directory based on the
        options which are most likely to affect the path (see
        `ReusableHyperOptimizer._get_path_relevant_opts`).
    overwrite : bool or 'improved', optional
        If ``True``, the optimizer will always run, overwriting old results in
        the cache. This can be used to update paths without deleting the whole
        cache. If ``'improved'`` then only overwrite if the new path is better.
    hash_method : {'a', 'b', ...}, optional
        The method used to hash the contraction tree. The default, ``'a'``, is
        faster hashwise but doesn't recognize when indices are permuted.
    cache_only : bool, optional
        If ``True``, the optimizer will only use the cache, and will raise
        ``KeyError`` if a contraction is not found.
    directory_split : "auto" or bool, optional
        If specified, the hash will be split into two parts, the first part
        will be used as a subdirectory, and the second part will be used as the
        filename. This is useful for avoiding a very large flat diretory. If
        "auto" it will check the current cache if any and guess from that.
    opt_kwargs
        Supplied to ``HyperOptimizer``.
    """

    def _get_path_relevant_opts(self):
        """Get a frozenset of the options that are most likely to affect the
        path. These are the options that we use when the directory name is not
        manually specified.
        """
        return [
            ("methods", None),
            ("minimize", "flops"),
            ("max_repeats", 128),
            ("max_time", None),
            ("slicing_opts", None),
            ("slicing_reconf_opts", None),
            ("simulated_annealing_opts", None),
            ("reconf_opts", None),
            ("compressed", False),
            ("multicontraction", False),
        ]

    def _get_suboptimizer(self):
        return HyperOptimizer(**self._suboptimizer_kwargs)

    def _deconstruct_tree(self, opt, tree):
        return {
            "path": tree.get_path(),
            "score": tree.get_score(),
            # dont' need to store all slice info, just which indices
            "sliced_inds": tuple(tree.sliced_inds),
        }

    def _reconstruct_tree(self, inputs, output, size_dict, con):
        tree = ContractionTree.from_path(
            inputs,
            output,
            size_dict,
            path=con["path"],
            objective=self.minimize,
        )

        for ix in con["sliced_inds"]:
            tree.remove_ind_(ix)

        return tree


class HyperCompressedOptimizer(HyperOptimizer):
    """A compressed contraction path optimizer that samples a series of ordered
    contraction trees while optimizing the hyper parameters used to generate
    them.

    Parameters
    ----------
    chi : None or int, optional
        The maximum bond dimension to compress to. If ``None`` then use the
        square of the largest existing dimension. If ``minimize`` is specified
        as a full scoring function, this is ignored.
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

    compressed = True
    multicontraction = False

    def __init__(
        self,
        chi=None,
        methods=("greedy-compressed", "greedy-span", "kahypar-agglom"),
        minimize="peak-compressed",
        **kwargs,
    ):
        if (chi is not None) and not callable(minimize):
            minimize += f"-{chi}"

        kwargs["methods"] = methods
        kwargs["minimize"] = minimize

        if kwargs.pop("slicing_opts", None) is not None:
            raise ValueError(
                "Cannot use slicing_opts with compressed contraction."
            )
        if kwargs.pop("slicing_reconf_opts", None) is not None:
            raise ValueError(
                "Cannot use slicing_reconf_opts with compressed contraction."
            )

        super().__init__(**kwargs)


class ReusableHyperCompressedOptimizer(ReusableHyperOptimizer):
    """Like ``HyperCompressedOptimizer`` but it will re-instantiate the
    optimizer whenever a new contraction is detected, and also cache the paths
    found.

    Parameters
    ----------
    chi : None or int, optional
        The maximum bond dimension to compress to. If ``None`` then use the
        square of the largest existing dimension. If ``minimize`` is specified
        as a full scoring function, this is ignored.
    directory : None, True, or str, optional
        If specified use this directory as a persistent cache. If ``True`` auto
        generate a directory in the current working directory based on the
        options which are most likely to affect the path (see
        `ReusableHyperOptimizer._get_path_relevant_opts`).
    overwrite : bool, optional
        If ``True``, the optimizer will always run, overwriting old results in
        the cache. This can be used to update paths without deleting the whole
        cache.
    hash_method : {'a', 'b', ...}, optional
        The method used to hash the contraction tree. The default, ``'a'``, is
        faster hashwise but doesn't recognize when indices are permuted.
    cache_only : bool, optional
        If ``True``, the optimizer will only use the cache, and will raise
        ``KeyError`` if a contraction is not found.
    opt_kwargs
        Supplied to ``HyperCompressedOptimizer``.
    """

    def __init__(
        self,
        chi=None,
        methods=("greedy-compressed", "greedy-span", "kahypar-agglom"),
        minimize="peak-compressed",
        **kwargs,
    ):
        if (chi is not None) and not callable(minimize):
            minimize += f"-{chi}"

        kwargs["methods"] = methods
        kwargs["minimize"] = minimize

        if kwargs.pop("slicing_opts", None) is not None:
            raise ValueError(
                "Cannot use slicing_opts with compressed contraction."
            )
        if kwargs.pop("slicing_reconf_opts", None) is not None:
            raise ValueError(
                "Cannot use slicing_reconf_opts with compressed contraction."
            )
        if kwargs.pop("simulated_annealing_opts", None) is not None:
            raise ValueError(
                "Cannot use simulated_annealing_opts "
                "with compressed contraction."
            )

        super().__init__(**kwargs)

    def _get_suboptimizer(self):
        return HyperCompressedOptimizer(**self._suboptimizer_kwargs)

    def _deconstruct_tree(self, opt, tree):
        return {
            "path": tree.get_path(),
            "score": tree.get_score(),
            "sliced_inds": tuple(tree.sliced_inds),
        }

    def _reconstruct_tree(self, inputs, output, size_dict, con):
        return ContractionTreeCompressed.from_path(
            inputs,
            output,
            size_dict,
            path=con["path"],
            objective=self.minimize,
        )


class HyperMultiOptimizer(HyperOptimizer):
    compressed = False
    multicontraction = True
