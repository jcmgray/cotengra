"""Hyper parameter optimization using SMAC3.

https://automl.github.io/SMAC3/latest/
"""

from .hyper import HyperOptLib, register_hyper_optlib


def build_config_space(method, space):
    """Build a SMAC ``ConfigurationSpace`` from a cotengra space dict.

    Parameters
    ----------
    method : str
        The method name (used as a prefix to avoid name collisions when
        multiple methods share the same parameter name).
    space : dict[str, dict]
        The search space for a single method.

    Returns
    -------
    cs : ConfigurationSpace
    """
    from ConfigSpace import (
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    from smac.configspace import ConfigurationSpace

    cs = ConfigurationSpace()
    for name, param in space.items():
        ptype = param["type"]
        if ptype == "FLOAT":
            hp = UniformFloatHyperparameter(
                name, lower=param["min"], upper=param["max"]
            )
        elif ptype == "FLOAT_EXP":
            hp = UniformFloatHyperparameter(
                name, lower=param["min"], upper=param["max"], log=True
            )
        elif ptype == "INT":
            hp = UniformIntegerHyperparameter(
                name, lower=param["min"], upper=param["max"]
            )
        elif ptype == "STRING":
            hp = CategoricalHyperparameter(name, choices=param["options"])
        elif ptype == "BOOL":
            hp = CategoricalHyperparameter(name, choices=[False, True])
        else:
            raise ValueError(f"Unknown parameter type: {ptype!r}")
        cs.add_hyperparameter(hp)
    return cs


def config_to_params(config):
    """Convert a SMAC ``Configuration`` to a plain dict of parameters."""
    return dict(config)


class SMACOptLib(HyperOptLib):
    """Hyper-optimization using SMAC3 with per-method facades and
    a Lower Confidence Bound method selector.
    """

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        facade="BlackBoxFacade",
        n_trials=10000,
        seed=0,
        method_exploration=1.0,
        method_temperature=1.0,
        **facade_opts,
    ):
        from smac import BlackBoxFacade, HyperparameterOptimizationFacade
        from smac.scenario import Scenario

        from ._param_mapping import LCBOptimizer

        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
        )

        if isinstance(facade, str):
            facade_cls = {
                "BlackBoxFacade": BlackBoxFacade,
                "HyperparameterOptimizationFacade": HyperparameterOptimizationFacade,
            }[facade]
        else:
            facade_cls = facade

        self._facades = {}
        self._trial_infos = {}
        for method in methods:
            cs = build_config_space(method, space[method])
            scenario = Scenario(
                cs,
                n_trials=n_trials,
                seed=seed,
                deterministic=True,
            )
            self._facades[method] = facade_cls(
                scenario,
                target_function=lambda cfg, seed: 0.0,
                overwrite=True,
                logging_level=False,
                **facade_opts,
            )

    def get_setting(self):
        method = self._method_chooser.ask()
        smac = self._facades[method]
        info = smac.ask()
        trial_key = (method, id(info))
        self._trial_infos[trial_key] = info
        return {
            "method": method,
            "trial_key": trial_key,
            "params": config_to_params(info.config),
        }

    def report_result(self, setting, trial, score):
        from smac.runhistory import TrialValue

        method = setting["method"]
        trial_key = setting["trial_key"]
        info = self._trial_infos.pop(trial_key)
        self._method_chooser.tell(method, score)
        value = TrialValue(cost=score)
        self._facades[method].tell(info, value)


register_hyper_optlib("smac", SMACOptLib)
