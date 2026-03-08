"""Hyper parameter optimization using cmaes, as implemented by

https://github.com/CyberAgentAILab/cmaes.

"""

from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib


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
        import cmaes
        import numpy as np

        self.params = build_params(
            space, exponential_param_power=exponential_param_power
        )
        ndim = num_params(self.params)

        if separable:
            CMA = cmaes.SepCMA
        else:
            CMA = cmaes.CMA
            kwargs["lr_adapt"] = lr_adapt

        self.opt = CMA(
            mean=np.zeros(ndim),
            sigma=sigma,
            bounds=np.array([(-1.0, 1.0)] * ndim),
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

        return trial_number, convert_raw(self.params, x)

    def tell(self, trial_number, value):
        # retrieve raw vector
        x = self._trial_store.pop(trial_number)
        # wait until batch has reached population size
        self._batch.append((x, value))
        if len(self._batch) == self.opt.population_size:
            self.opt.tell(self._batch)
            self._batch.clear()


class CMAESOptLib(HyperOptLib):
    """Hyper-optimization using CMA-ES with per-method optimizers and
    a Lower Confidence Bound method selector.
    """

    def setup(
        self,
        methods,
        space,
        optimizer=None,
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

    def get_setting(self):
        method = self._method_chooser.ask()
        params_token, params = self._optimizers[method].ask()

        return {
            "method": method,
            "params_token": params_token,
            "params": params,
        }

    def report_result(self, setting, trial, score):
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib("cmaes", CMAESOptLib)
