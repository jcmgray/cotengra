"""Fake hyper optimization using random sampling."""

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


class RandomSampler:
    def __init__(self, methods, spaces, seed=None):
        self.rng = get_rng(seed)
        self._rmethods = tuple(methods)
        self._rspaces = {m: RandomSpace(spaces[m], self.rng) for m in methods}

    def ask(self):
        method = self.rng.choice(self._rmethods)
        rspace = self._rspaces[method]
        params = rspace.sample()
        return method, params


class RandomOptLib(HyperOptLib):
    """Completely random sampling optimizer."""

    def setup(self, methods, space, optimizer=None, seed=None, **kwargs):
        """Initialize random sampling.

        Parameters
        ----------
        methods : list[str]
            The list of contraction methods to sample from.
        space : dict[str, dict[str, dict]]
            The search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer instance.
        seed : None or int, optional
            Random seed.
        """
        self.sampler = RandomSampler(methods, space, seed=seed)

    def get_setting(self):
        method, params = self.sampler.ask()
        return {"method": method, "params": params}

    def report_result(self, setting, trial, score):
        pass


register_hyper_optlib("random", RandomOptLib)
