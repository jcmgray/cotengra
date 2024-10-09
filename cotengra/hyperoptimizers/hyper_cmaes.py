"""Hyper parameter optimization using cmaes, as implemented by

https://github.com/CyberAgentAILab/cmaes.

"""

import math

from ..utils import get_rng
from .hyper import register_hyper_optlib


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


class HyperCMAESSampler:
    def __init__(
        self,
        space,
        sigma=1.0,
        lr_adapt=True,
        separable=False,
        exponential_param_power=None,
        **kwargs,
    ):
        import numpy as np
        import cmaes

        self.params = []
        for name, p in space.items():
            if p["type"] == "FLOAT":
                self.params.append(ParamFloat(p["min"], p["max"], name=name))
            elif p["type"] == "FLOAT_EXP":
                self.params.append(
                    ParamFloatExp(
                        p["min"],
                        p["max"],
                        power=exponential_param_power,
                        name=name,
                    )
                )
            elif p["type"] == "INT":
                self.params.append(ParamInt(p["min"], p["max"], name=name))
            elif p["type"] == "STRING":
                self.params.append(ParamString(p["options"], name=name))
            elif p["type"] == "BOOL":
                self.params.append(ParamBool(name=name))
            else:
                raise ValueError(f"Unknown parameter type {p['type']}")

        num_params = sum(p.size for p in self.params)

        if separable:
            CMA = cmaes.SepCMA
        else:
            CMA = cmaes.CMA
            kwargs["lr_adapt"] = lr_adapt

        self.opt = CMA(
            mean=np.zeros(num_params),
            sigma=sigma,
            bounds=np.array([(-1.0, 1.0)] * num_params),
            **kwargs,
        )

        self._trial_counter = 0
        self._trial_store = {}
        self._batch = []

    def ask(self):
        # raw vector
        x = self.opt.ask()

        # need to store raw trial to report later
        trial_number = self._trial_counter
        self._trial_store[trial_number] = x
        self._trial_counter += 1

        params = {}
        i = 0
        for p in self.params:
            if p.size == 1:
                params[p.name] = p.convert_raw(x[i])
                i += 1
            else:
                params[p.name] = p.convert_raw(x[i : i + p.size])
                i += p.size

        return trial_number, params

    def tell(self, trial_number, value):
        # retrieve raw vector
        x = self._trial_store.pop(trial_number)
        # wait until batch has reached population size
        self._batch.append((x, value))
        if len(self._batch) == self.opt.population_size:
            self.opt.tell(self._batch)
            self._batch.clear()


def cmaes_init_optimizers(
    self,
    methods,
    space,
    sigma=1.0,
    lr_adapt=True,
    method_exploration=1.0,
    method_temperature=1.0,
    exponential_param_power=None,
    **cmaes_opts,
):
    self._method_chooser = LCBOptimizer(
        options=methods,
        exploration=method_exploration,
        temperature=method_temperature,
    )
    self._optimizers = {
        method: HyperCMAESSampler(
            space[method],
            sigma=sigma,
            lr_adapt=lr_adapt,
            exponential_param_power=exponential_param_power,
            **cmaes_opts,
        )
        for method in methods
    }


def cmaes_get_setting(self):
    method = self._method_chooser.ask()
    params_token, params = self._optimizers[method].ask()

    return {
        "method": method,
        "params_token": params_token,
        "params": params,
    }


def cmaes_report_result(self, settings, trial, score):
    self._method_chooser.tell(settings["method"], score)
    self._optimizers[settings["method"]].tell(settings["params_token"], score)


register_hyper_optlib(
    "cmaes",
    cmaes_init_optimizers,
    cmaes_get_setting,
    cmaes_report_result,
)
