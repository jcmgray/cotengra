"""Shared parameter mapping utilities for hyper-optimization backends.

Provides classes for mapping heterogeneous parameter types (float, int,
string, bool) to and from a uniform [-1, 1] optimization space, plus an
LCB-based method selector.
"""

import math

from ..utils import get_rng


class LCBOptimizer:
    """Lower Confidence Bound Optimizer.

    This optimizer selects the option with the lowest lower confidence bound.
    """

    def __init__(self, options, exploration=1.0, temperature=1.0, seed=None):
        self.options = tuple(options)
        self.index = {o: i for i, o in enumerate(self.options)}
        self.nopt = len(self.options)
        self.counts = [0] * self.nopt
        self.values = [float("inf")] * self.nopt
        self.total = 0
        self.exploration = exploration
        self.temperature = temperature
        self.rng = get_rng(seed)

    def ask(self):
        """Suggest an option based on the lower confidence bound."""
        best_option = best_lcb = None
        for option, ci, vi in zip(self.options, self.counts, self.values):
            if ci < 1:
                # need to gather initial samples
                return option

            # modify lcb to include exploration term
            lcb = (
                vi
                # exploration term to favor less sampled options
                - math.sqrt(2 * self.exploration * math.log(self.total) / ci)
                # thermal (via gumbel trick) term to add noise
                - self.temperature * math.log(-math.log(self.rng.random()))
            )

            if best_lcb is None or lcb < best_lcb:
                best_lcb = lcb
                best_option = option

        return best_option

    def tell(self, option, score):
        i = self.index[option]
        self.counts[i] += 1
        self.values[i] = min(score, self.values[i])
        self.total += 1


class Param:
    """A basic parameter class for mapping various types of parameters to
    and from uniform optimization space of [-1, 1].
    """

    def __init__(self, name):
        self.name = name
        self.size = 1

    def get_raw_bounds(self):
        raise NotImplementedError()

    def convert_raw(self, vi):
        raise NotImplementedError()


class ParamFloat(Param):
    def __init__(self, min, max, **kwargs):
        self.min = min
        self.max = max
        super().__init__(**kwargs)

    def convert_raw(self, x):
        x += 1.0
        x /= 2.0
        x *= self.max - self.min
        x += self.min
        return x


class ParamFloatExp(ParamFloat):
    """An exponentially distributed (i.e. uniform in logspace) parameter."""

    def __init__(self, min, max, power=0.5, **kwargs):
        self.power = power

        if self.power is None:
            mn = math.log(min)
            mx = math.log(max)
        else:
            mn = min**self.power
            mx = max**self.power

        super().__init__(min=mn, max=mx, **kwargs)

    def convert_raw(self, x):
        if self.power is None:
            return math.exp(super().convert_raw(x))
        else:
            return super().convert_raw(x) ** (1 / self.power)


class ParamInt(Param):
    def __init__(self, min, max, **kwargs):
        self.min = min
        self.max = max
        super().__init__(**kwargs)

    def convert_raw(self, x):
        x += 1.0
        x /= 2.0
        x *= self.max - self.min + 1
        x += self.min
        return min(int(x), self.max)


class ParamString(Param):
    def __init__(self, options, name):
        self.options = tuple(options)
        self.size = len(self.options)
        self.name = name

    def convert_raw(self, x):
        i = max(range(len(self.options)), key=lambda i: x[i])
        return self.options[i]


class ParamBool(Param):
    def __init__(self, name):
        super().__init__(name)
        self.size = 2
        self.name = name

    def convert_raw(self, x):
        return x[0] > x[1]


def build_params(space, exponential_param_power=None):
    """Build a list of ``Param`` objects from a search space dict.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single method.
    exponential_param_power : float, optional
        Power parameter for ``ParamFloatExp``.

    Returns
    -------
    params : list[Param]
    """
    params = []
    for name, p in space.items():
        if p["type"] == "FLOAT":
            params.append(ParamFloat(p["min"], p["max"], name=name))
        elif p["type"] == "FLOAT_EXP":
            params.append(
                ParamFloatExp(
                    p["min"],
                    p["max"],
                    power=exponential_param_power,
                    name=name,
                )
            )
        elif p["type"] == "INT":
            params.append(ParamInt(p["min"], p["max"], name=name))
        elif p["type"] == "STRING":
            params.append(ParamString(p["options"], name=name))
        elif p["type"] == "BOOL":
            params.append(ParamBool(name=name))
        else:
            raise ValueError(f"Unknown parameter type {p['type']}")
    return params


def convert_raw(params, x):
    """Convert a raw vector from [-1, 1] space into named parameters.

    Parameters
    ----------
    params : list[Param]
        The parameter definitions.
    x : array_like
        The raw vector.

    Returns
    -------
    named_params : dict[str, any]
    """
    result = {}
    i = 0
    for p in params:
        if p.size == 1:
            result[p.name] = p.convert_raw(x[i])
            i += 1
        else:
            result[p.name] = p.convert_raw(x[i : i + p.size])
            i += p.size
    return result


def num_params(params):
    """Return the total number of raw dimensions for a list of params."""
    return sum(p.size for p in params)
