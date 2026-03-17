"""Hyper optimization using random or Latin Hypercube sampling."""

import functools
import math

from ..utils import get_rng
from .hyper import HyperOptLib, register_hyper_optlib


def sample_bool(rng):
    return rng.choice([False, True])


def sample_int(rng, low, high):
    return rng.randint(low, high)


def sample_option(rng, options):
    return rng.choice(options)


def sample_uniform(rng, low, high):
    return rng.uniform(low, high)


def sample_loguniform(rng, low, high):
    return 2 ** rng.uniform(math.log2(low), math.log2(high))


class RandomSpace:
    def __init__(self, space, seed=None):
        self.rng = get_rng(seed)
        self._samplers = {}

        for k, param in space.items():
            if param["type"] == "BOOL":
                self._samplers[k] = sample_bool

            elif param["type"] == "INT":
                self._samplers[k] = functools.partial(
                    sample_int, low=param["min"], high=param["max"]
                )

            elif param["type"] == "STRING":
                self._samplers[k] = functools.partial(
                    sample_option, options=param["options"]
                )

            elif param["type"] == "FLOAT":
                self._samplers[k] = functools.partial(
                    sample_uniform, low=param["min"], high=param["max"]
                )

            elif param["type"] == "FLOAT_EXP":
                self._samplers[k] = functools.partial(
                    sample_loguniform, low=param["min"], high=param["max"]
                )

            else:
                raise ValueError("Didn't understand space {}.".format(param))

    def sample(self):
        return {k: fn(self.rng) for k, fn in self._samplers.items()}


class LHSRandomSpace:
    """Latin Hypercube Sampled random space. Pre-generates ``n`` samples with
    stratified coverage over each parameter's native domain. For continuous
    parameters (FLOAT, FLOAT_EXP, INT) the range is divided into ``n`` equal
    strata with one sample per stratum. For categorical parameters (STRING,
    BOOL) the options are cycled through in balanced fashion and randomly
    permuted. Once all pre-generated samples are exhausted, falls back to pure
    random sampling.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single method.
    n : int
        Number of LHS samples to pre-generate.
    seed : None or int or random.Random, optional
        Random seed.
    """

    def __init__(self, space, n, seed=None):
        self.rng = get_rng(seed)
        self._n = n
        self._params = list(space.items())
        self._fallback = RandomSpace(space, seed=self.rng)

        # pre-generate n stratified values per parameter
        columns = {}
        for k, param in self._params:
            ptype = param["type"]

            if ptype == "FLOAT":
                columns[k] = self._stratify_float(
                    n, param["min"], param["max"]
                )

            elif ptype == "FLOAT_EXP":
                log_lo = math.log2(param["min"])
                log_hi = math.log2(param["max"])
                log_vals = self._stratify_float(n, log_lo, log_hi)
                columns[k] = [2**v for v in log_vals]

            elif ptype == "INT":
                columns[k] = self._stratify_int(n, param["min"], param["max"])

            elif ptype == "STRING":
                columns[k] = self._stratify_categorical(n, param["options"])

            elif ptype == "BOOL":
                columns[k] = self._stratify_categorical(n, [False, True])

            else:
                raise ValueError(f"Didn't understand space {param}.")

        # assemble into list of dicts
        self._samples = []
        for i in range(n):
            self._samples.append({k: columns[k][i] for k in columns})

        # shuffle so draw order is random
        self.rng.shuffle(self._samples)

    def _stratify_float(self, n, lo, hi):
        """Divide ``[lo, hi]`` into ``n`` equal strata, sample one
        uniform point per stratum, then randomly permute.
        """
        vals = []
        for i in range(n):
            s_lo = lo + (hi - lo) * i / n
            s_hi = lo + (hi - lo) * (i + 1) / n
            vals.append(self.rng.uniform(s_lo, s_hi))
        self.rng.shuffle(vals)
        return vals

    def _stratify_int(self, n, lo, hi):
        """Stratify an integer range ``[lo, hi]``.

        The continuous interval ``[lo, hi + 1)`` is divided into ``n``
        strata, a uniform float is drawn per stratum, and then floored
        to an integer (clamped to ``hi``).
        """
        vals = []
        flo = float(lo)
        fhi = float(hi + 1)
        for i in range(n):
            s_lo = flo + (fhi - flo) * i / n
            s_hi = flo + (fhi - flo) * (i + 1) / n
            v = int(self.rng.uniform(s_lo, s_hi))
            vals.append(min(v, hi))
        self.rng.shuffle(vals)
        return vals

    def _stratify_categorical(self, n, options):
        """Cycle through ``options`` in balanced fashion with random
        permutation. Each option appears either ``floor(n / k)`` or
        ``ceil(n / k)`` times, where ``k = len(options)``.
        """
        k = len(options)
        # repeat enough times then trim to n
        full_cycles = (n // k) + 1
        vals = list(options) * full_cycles
        vals = vals[:n]
        self.rng.shuffle(vals)
        return vals

    def sample(self):
        """Return the next sample. Uses pre-generated LHS points first, then
        falls back to pure random sampling.
        """
        if self._samples:
            return self._samples.pop()
        return self._fallback.sample()


class RandomSampler:
    """Random parameter sampler with optional LHS warm-up.

    Parameters
    ----------
    methods : list[str]
        The contraction methods to sample from.
    spaces : dict[str, dict[str, dict]]
        The per-method search spaces.
    n_samples : int or None, optional
        If given, use Latin Hypercube Sampling for the first ``n_samples``
        calls per method, then fall back to pure random. If ``None``, use pure
        random sampling throughout.
    seed : None or int or random.Random, optional
        Random seed.
    """

    def __init__(self, methods, spaces, n_samples=None, seed=None):
        self.rng = get_rng(seed)
        self._rmethods = tuple(methods)
        if n_samples is not None and n_samples > 0:
            self._rspaces = {
                m: LHSRandomSpace(spaces[m], n_samples, self.rng)
                for m in methods
            }
        else:
            self._rspaces = {
                m: RandomSpace(spaces[m], self.rng) for m in methods
            }

    def ask(self):
        method = self.rng.choice(self._rmethods)
        rspace = self._rspaces[method]
        params = rspace.sample()
        return method, params


class RandomOptLib(HyperOptLib):
    """Random sampling optimizer with optional Latin Hypercube warm-up.

    When ``lhs=True`` (the default), the first batch of samples per
    method uses Latin Hypercube Sampling for better coverage of the
    search space. The batch size is derived from the parent optimizer's
    ``max_repeats`` attribute divided by the number of methods. After
    the LHS batch is exhausted, sampling falls back to pure random.
    """

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        lhs=False,
        seed=None,
        **kwargs,
    ):
        """Initialize random sampling.

        Parameters
        ----------
        methods : list[str]
            The list of contraction methods to sample from.
        space : dict[str, dict[str, dict]]
            The search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer instance, used for ``max_repeats``.
        lhs : bool, optional
            Whether to use Latin Hypercube Sampling for the initial batch.
            Default ``True``.
        seed : None or int, optional
            Random seed.
        """
        n_samples = None
        if lhs:
            max_repeats = getattr(optimizer, "max_repeats", 128)
            n_methods = max(len(methods), 1)
            n_samples = max_repeats // n_methods

        self.sampler = RandomSampler(
            methods, space, n_samples=n_samples, seed=seed
        )

    def get_setting(self):
        method, params = self.sampler.ask()
        return {"method": method, "params": params}

    def report_result(self, setting, trial, score):
        pass


register_hyper_optlib("random", RandomOptLib, defaults={"lhs": True})
register_hyper_optlib("random-uniform", RandomOptLib, defaults={"lhs": False})
