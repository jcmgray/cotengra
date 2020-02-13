import time
import functools
import importlib
from math import log2, log10

from opt_einsum.paths import ssa_greedy_optimize
from opt_einsum.path_random import thermal_chooser, RandomOptimizer

from .core import jitter_dict, ContractionTree
from .slicer import SliceFinder


DEFAULT_METHODS = ['greedy']
if importlib.util.find_spec('kahypar'):
    DEFAULT_METHODS += ['kahypar']


_OPT_FNS = {}
_HYPER_SEARCH_SPACE = {}


def register_hyper_function(name, ssa_func, space):
    """Register a contraction path finder to be used by the hyper-optimizer.

    Parameters
    ----------
    name : str
        The name to call the method.
    ssa_func : callable
        The raw ``opt_einsum`` style function that returns a 'ssa_path'.
    space : dict[str, dict]
        The space of hyper-parameters to search.
    """
    global _OPT_FNS
    global _HYPER_SEARCH_SPACE
    _OPT_FNS[name] = ssa_func
    _HYPER_SEARCH_SPACE[name] = space


def list_hyper_functions():
    """Return a list of currently registered hyper contraction finders.
    """
    global _OPT_FNS
    return sorted(_OPT_FNS)


def find_path(method, *args, **kwargs):
    return _OPT_FNS[method](*args, **kwargs)


BTB_TYPE_TO_HYPERPARAM = {
    'BOOL': 'BooleanHyperParam',
    'INT': 'IntHyperParam',
    'INT_CAT': 'CategoricalHyperParam',
    'STRING': 'CategoricalHyperParam',
    'FLOAT': 'FloatHyperParam',
    'FLOAT_EXP': 'FloatHyperParam',  # no more EXP support in baytune?
}


def key_to_hyperparameter(method, name, k):
    from btb.tuning import hyperparams

    hp = getattr(hyperparams, BTB_TYPE_TO_HYPERPARAM[k['type']])

    if k['type'] in ('BOOL',):
        return hp()
    elif k['type'] in ('INT_CAT', 'STRING'):
        return hp(_HYPER_SEARCH_SPACE[method][name]['options'])
    else:
        return hp(_HYPER_SEARCH_SPACE[method][name]['min'],
                  _HYPER_SEARCH_SPACE[method][name]['max'],
                  include_min=True, include_max=True)


def get_default_tuner_spaces(tuner='GP'):
    import btb
    from btb.tuning import Tunable

    # for compatability
    if 'Tuner' not in tuner:
        tuner += 'Tuner'

    tuner_fn = getattr(btb.tuning.tuners, tuner)

    return {
        method: tuner_fn(Tunable({
            name: key_to_hyperparameter(method, name, k)
            for name, k in _HYPER_SEARCH_SPACE[method].items()
        }))
        for method in _HYPER_SEARCH_SPACE
    }


def score_flops(trial):
    return log2(trial['flops']) + log2(trial['size']) / 1000


def score_size(trial):
    return log2(trial['flops']) / 1000 + log2(trial['size'])


def score_combo(trial):
    return log2(trial['flops']) + log2(trial['size'])


_SCORE_FUNCTIONS = {
    'flops': score_flops,
    'size': score_size,
    'combo': score_combo,
}


class SlicedTrialFn:

    def __init__(self, trial_fn, **opts):
        self.trial_fn = trial_fn

        # perform very cheap, single, low temperature trial slicing
        opts.setdefault('max_repeats', 1)
        opts.setdefault('temperature', 0.0)

        self.max_repeats = opts.pop('max_repeats')
        self.opts = opts

    def __call__(self, *args, **kwargs):
        trial = self.trial_fn(*args, **kwargs)

        slicefinder = SliceFinder(trial['tree'], **self.opts)
        sliced, costs = slicefinder.search(max_repeats=self.max_repeats)

        trial['original_flops'] = trial['flops']
        trial['original_size'] = trial['size']
        trial['flops'] = costs.total_flops
        trial['size'] = costs.size
        trial['sliced'] = sliced
        trial['nslices'] = costs.nslices

        return trial


def progress_description(best):
    return(f"log2[SIZE]: {log2(best['size']):.2f} "
           f"log10[FLOPs]: {log10(best['flops']):.2f}")


class HyperOptimizer(RandomOptimizer):
    """Users Bayesian optimization to hyper-optimizer the settings used to
    optimize the path.

    Parameters
    ----------
    methods : None or sequence[str] or str, optional
        Which method(s) to use from ``list_hyper_functions()``.
    minimize : {'flops', 'size', 'combo' or callable}, optional
        How to score each trial, used to train the optimizer and rank the
        results. If a custom callable, it should take a ``trial`` dict as its
        argument and return a single float.
    score_compression : float, optional
        Raise scores to this power in order to compress or accentuate the
        differences. The lower this is, the more the selector will sample from
        various optimizers rather than quickly specializing.
    slicing_opts : dict, optional
        If supplied, once a trial contraction path is found, try slicing with
        the given options, and then update the flops and size of the trial with
        the sliced versions.
    tuner : str, optional
        Which ``btb`` parameter fitter to use - default ``'GP'`` means gaussian
        process. Other options include ``'Uniform'`` and ``'GPEi'``.
        See https://hdi-project.github.io/BTB/api/btb.tuning.tuners.html.
    selector : str, optional
        Which ``btb`` selector to use - default 'UCB1'.
        See https://hdi-project.github.io/BTB/api/btb.selection.html.
    selector_opts : dict, optional
        Options to supply to ``btb``.
    progbar : bool, optional
        Show live progress of the best contraction found so far.
    kwargs
        Supplied to :class:`opt_einsum.RandomOptimizer`, valid options include
        ``max_repeats``, ``max_time`` and ``parallel``.
    """

    def __init__(
        self,
        methods=None,
        minimize='flops',
        score_compression=0.75,
        slicing_opts=None,
        tuner='GP',
        selector='UCB1',
        selector_opts=None,
        progbar=False,
        **kwargs
    ):
        import btb.selection

        self.method_choices = []
        self.param_choices = []
        self.scores = []

        if methods is None:
            self._methods = DEFAULT_METHODS
        elif isinstance(methods, str):
            self._methods = [methods]
        else:
            self._methods = list(methods)

        selector_opts = {} if selector_opts is None else dict(selector_opts)
        self._selector = getattr(btb.selection, selector)(self._methods,
                                                          **selector_opts)
        self._tuners = get_default_tuner_spaces(tuner)

        kwargs.setdefault('max_repeats', 128)
        kwargs.setdefault('parallel', True)

        if isinstance(kwargs['parallel'], int):
            nworkers = kwargs['parallel']
        elif hasattr(kwargs['parallel'], '_max_workers'):
            nworkers = kwargs['parallel']._max_workers
        elif kwargs['parallel'] is not False:
            import psutil
            nworkers = psutil.cpu_count()
        kwargs.setdefault('pre_dispatch', 2 * nworkers)

        super().__init__(**kwargs)
        self.minimize = minimize
        # which score to feed to the hyper optimizer
        self.score_compression = score_compression
        inf = float('inf')
        self.best = {'score': inf, 'size': inf, 'flops': inf}
        self.best_size = {'size': inf, 'flops': inf}
        self.best_flops = {'size': inf, 'flops': inf}
        self.best_combo = {'size': inf, 'flops': inf}

        self.slicing_opts = {} if slicing_opts is None else dict(slicing_opts)
        self.progbar = progbar

    @property
    def minimize(self):
        return self._minimize

    @minimize.setter
    def minimize(self, minimize):
        self._minimize = minimize
        if callable(minimize):
            self._score_fn = minimize
        else:
            self._score_fn = _SCORE_FUNCTIONS[minimize]

    def setup(self, inputs, output, size_dict):
        if self.slicing_opts:
            trial_fn = SlicedTrialFn(find_path, **self.slicing_opts)
        else:
            trial_fn = find_path
        return trial_fn, (inputs, output, size_dict)

    def compute_score(self, trial):
        trial['score'] = self._score_fn(trial)**self.score_compression
        return -trial['score']

    def get_setting(self):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module='sklearn')
            possible_methods = {
                m: getattr(self._tuners[m], 'scores', ())
                for m in self._methods
            }
            method = self._selector.select(possible_methods)
            params = self._tuners[method].propose()
            return method, params

    def report_result(self, method, params, trial):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module='sklearn')
            compressed_score = self.compute_score(trial)
            self._tuners[method].record(params, compressed_score)
            self.method_choices.append(method)
            self.param_choices.append(params)

    def _gen_results(self, repeats, trial_fn, trial_args):
        for _ in repeats:
            method, params = self.get_setting()
            trial = trial_fn(method, *trial_args, **params)
            self.report_result(method, params, trial)
            yield trial

    def get_and_report_next_future(self):
        # scan over the futures, yield whichever completes first
        while True:
            for i in range(len(self._futures)):
                method, params, future = self._futures[i]
                if future.done():
                    del self._futures[i]
                    trial = future.result()
                    self.report_result(method, params, trial)
                    return trial
                time.sleep(0.01)

    def _gen_results_parallel(self, repeats, trial_fn, trial_args):
        self._futures = []

        for _ in repeats:
            method, params = self.get_setting()
            future = self._executor.submit(
                trial_fn, method, *trial_args, **params
            )
            self._futures.append((method, params, future))

            if len(self._futures) >= self.pre_dispatch:
                yield self.get_and_report_next_future()

        while self._futures:
            yield self.get_and_report_next_future()

    def _maybe_cancel_futures(self):
        if self._executor is not None:
            while self._futures:
                _, _, f = self._futures.pop()
                f.cancel()

    def __call__(self, inputs, output, size_dict, memory_limit):
        # start a timer?
        if self.max_time is not None:
            t0 = time.time()

        trial_fn, trial_args = self.setup(inputs, output, size_dict)

        r_start = self._repeats_start + len(self.costs)
        r_stop = r_start + self.max_repeats
        repeats = range(r_start, r_stop)

        # create the trials lazily
        if self._executor is not None:
            trials = self._gen_results_parallel(repeats, trial_fn, trial_args)
        else:
            trials = self._gen_results(repeats, trial_fn, trial_args)

        if self.progbar:
            import tqdm
            pbar = tqdm.tqdm(trials, total=self.max_repeats)
            pbar.set_description(progress_description(self.best))
            trials = pbar

        # assess the trials
        for trial in trials:

            # keep track of all costs and sizes
            self.costs.append(trial['flops'])
            self.sizes.append(trial['size'])
            self.scores.append(trial['score'])

            # check if we have found a new best
            if trial['score'] < self.best['score']:
                self.best = trial
                self.best['params'] = dict(self.param_choices[-1])
                self.best['params']['method'] = self.method_choices[-1]

                if self.progbar:
                    pbar.set_description(progress_description(self.best))

            # keep track of the best size, flops and combo score trials
            if (
                (trial['size'], trial['flops']) <
                (self.best_size['size'], self.best_size['flops'])
            ):
                self.best_size = trial
                self.best_size['params'] = dict(self.param_choices[-1])
                self.best_size['params']['method'] = self.method_choices[-1]

            if (
                (trial['flops'], trial['size']) <
                (self.best_flops['flops'], self.best_flops['size'])
            ):
                self.best_flops = trial
                self.best_flops['params'] = dict(self.param_choices[-1])
                self.best_flops['params']['method'] = self.method_choices[-1]

            if (
                log2(trial['size']) + log2(trial['flops']) <
                log2(self.best_combo['size']) + log2(self.best_combo['flops'])
            ):
                self.best_combo = trial
                self.best_combo['params'] = dict(self.param_choices[-1])
                self.best_combo['params']['method'] = self.method_choices[-1]

            # check if we have run out of time
            if (self.max_time is not None) and (time.time() > t0 +
                                                self.max_time):
                break

        if self.progbar:
            pbar.close()

        self._maybe_cancel_futures()
        return tuple(self.path)

    def get_trials(self, sort=None):
        trials = list(zip(self.method_choices, self.sizes,
                          self.costs, self.param_choices))
        if sort == 'method':
            trials.sort(key=lambda t: t[0])
        if sort == 'combo':
            trials.sort(key=lambda t: log2(t[1]) + log2(t[2]))
        if sort == 'size':
            trials.sort(key=lambda t: log2(t[1]) + log2(t[2]) / 1000)
        if sort == 'flops':
            trials.sort(key=lambda t: log2(t[1]) / 1000 + log2(t[2]))
        return trials

    def print_trials(self, sort=None):
        header = "{:>11} {:>7} {:>7}     {}"
        print(header.format('METHOD', 'SIZE', 'FLOPS', 'PARAMS'))
        row = "{:>11} {:>7.2f} {:>7.2f}    {}"
        for choice, size, cost, params in self.get_trials(sort):
            print(row.format(choice, log2(size), log2(cost), params))

    def to_df(self):
        import pandas

        return pandas.DataFrame(
            data={
                'run': list(range(len(self.sizes))),
                'method': self.method_choices,
                'size': list(map(log2, self.sizes)),
                'flops': list(map(log10, self.costs)),
                'random_strength': [p.get('random_strength', 1e-6)
                                    for p in self.param_choices],
                'score': self.scores,
            }
        ).sort_values(by='method')


def hyper_optimize(inputs, output, size_dict, memory_limit=None, **opts):
    optimizer = HyperOptimizer(**opts)
    return optimizer(inputs, output, size_dict, memory_limit)


# ------------------------------ GREEDY HYPER ------------------------------- #


def cost_memory_removed_mod(size12, size1, size2, k12, k1, k2, costmod=1):
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - costmod * (size1 + size2)


def trial_greedy(inputs, output, size_dict,
                 random_strength=0.1,
                 temperature=1.0,
                 rel_temperature=True,
                 costmod=1):

    rand_size_dict = jitter_dict(size_dict, random_strength)

    cost_fn = functools.partial(cost_memory_removed_mod, costmod=costmod)
    choose_fn = functools.partial(thermal_chooser, temperature=temperature,
                                  rel_temperature=rel_temperature)

    ssa_path = ssa_greedy_optimize(inputs, output, rand_size_dict,
                                   choose_fn=choose_fn, cost_fn=cost_fn)

    ctree = ContractionTree.from_path(inputs, output, size_dict,
                                      ssa_path=ssa_path)

    return {'tree': ctree, 'ssa_path': ssa_path,
            'flops': ctree.total_flops(), 'size': ctree.max_size()}


register_hyper_function(
    name='greedy',
    ssa_func=trial_greedy,
    space={
        'random_strength': {'type': 'FLOAT_EXP', 'min': 0.01, 'max': 10.},
        'temperature': {'type': 'FLOAT_EXP', 'min': 0.01, 'max': 10.},
        'rel_temperature': {'type': 'BOOL'},
        'costmod': {'type': 'FLOAT', 'min': 0.0, 'max': 2.0},
    },
)
