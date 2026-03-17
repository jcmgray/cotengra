"""Hyper optimization using pymoo single-objective algorithms.

This backend currently supports serial optimization only. Pymoo ask/tell
algorithms operate on generations/batches rather than individual trials, so
the integration buffers one full batch at a time and feeds it back when all
batch members have been evaluated.
"""

from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib


def _get_pymoo_algorithm(name):
    if name == "de":
        from pymoo.algorithms.soo.nonconvex.de import DE

        return DE
    if name == "ga":
        from pymoo.algorithms.soo.nonconvex.ga import GA

        return GA
    if name == "pso":
        from pymoo.algorithms.soo.nonconvex.pso import PSO

        return PSO

    if name == "brkga":
        from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

        return BRKGA

    if name == "es":
        from pymoo.algorithms.soo.nonconvex.es import ES

        return ES

    if name == "sres":
        from pymoo.algorithms.soo.nonconvex.sres import SRES

        return SRES

    if name == "isres":
        from pymoo.algorithms.soo.nonconvex.isres import ISRES

        return ISRES

    if name == "nrbo":
        from pymoo.algorithms.soo.nonconvex.nrbo import NRBO

        return NRBO

    raise ValueError(f"Unknown pymoo sampler {name}.")


class HyperPymooSampler:
    """Per-method ask/tell wrapper around a pymoo algorithm."""

    def __init__(
        self,
        space,
        sampler="de",
        sampler_opts=None,
        exponential_param_power=None,
        seed=None,
    ):
        import numpy as np
        from pymoo.core.evaluator import Evaluator
        from pymoo.core.problem import Problem
        from pymoo.core.termination import NoTermination
        from pymoo.problems.static import StaticProblem

        self._np = np
        self._Evaluator = Evaluator
        self._StaticProblem = StaticProblem

        self.params = build_params(
            space, exponential_param_power=exponential_param_power
        )
        self._ndim = num_params(self.params)
        self._problem = Problem(
            n_var=self._ndim,
            n_obj=1,
            n_constr=0,
            xl=np.full(self._ndim, -1.0),
            xu=np.full(self._ndim, 1.0),
        )

        sampler_opts = {} if sampler_opts is None else dict(sampler_opts)
        Algorithm = _get_pymoo_algorithm(sampler)
        self.algorithm = Algorithm(**sampler_opts)
        self.algorithm.setup(
            self._problem,
            termination=NoTermination(),
            seed=seed,
            verbose=False,
        )

        self._trial_counter = 0
        self._active_batch = None

    def ask(self):
        if self._active_batch is None:
            pop = self.algorithm.ask()
            xs = pop.get("X")
            trial_numbers = tuple(
                range(
                    self._trial_counter,
                    self._trial_counter + len(xs),
                )
            )
            self._trial_counter += len(xs)
            settings = tuple(convert_raw(self.params, x.copy()) for x in xs)
            self._active_batch = {
                "pop": pop,
                "trial_numbers": trial_numbers,
                "settings": settings,
                "scores": {},
                "next_index": 0,
            }

        i = self._active_batch["next_index"]
        self._active_batch["next_index"] += 1
        return (
            self._active_batch["trial_numbers"][i],
            self._active_batch["settings"][i],
        )

    def tell(self, trial_number, score):
        batch = self._active_batch
        batch["scores"][trial_number] = score

        if len(batch["scores"]) != len(batch["trial_numbers"]):
            return

        f = self._np.asarray(
            [batch["scores"][t] for t in batch["trial_numbers"]],
            dtype=float,
        ).reshape(-1, 1)
        static = self._StaticProblem(self._problem, F=f)
        self._Evaluator().eval(static, batch["pop"])
        self.algorithm.tell(infills=batch["pop"])
        self._active_batch = None


class PymooOptLib(HyperOptLib):
    """Hyper-optimization using pymoo algorithms with LCB method choice."""

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        sampler="de",
        sampler_opts=None,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        if getattr(optimizer, "_pool", None) is not None:
            raise ValueError(
                "The 'pymoo' optlib currently only supports serial "
                "hyper-optimization (`parallel=False`)."
            )

        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
            seed=seed,
        )
        self._optimizers = {
            method: HyperPymooSampler(
                space[method],
                sampler=sampler,
                sampler_opts=sampler_opts,
                exponential_param_power=exponential_param_power,
                seed=seed,
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
        method = setting["method"]
        self._method_chooser.tell(method, score)
        self._optimizers[method].tell(setting["params_token"], score)


register_hyper_optlib("pymoo", PymooOptLib)
