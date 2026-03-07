"""Hyper optimization using scikit-optimize."""

from .hyper import HyperOptLib, register_hyper_optlib


def convert_param_to_skopt(param, name):
    from skopt.space import Categorical, Integer, Real

    if param["type"] == "BOOL":
        return Categorical([False, True], name=name)
    if param["type"] == "INT":
        return Integer(low=param["min"], high=param["max"], name=name)
    if param["type"] == "STRING":
        return Categorical(param["options"], name=name)
    if param["type"] == "FLOAT":
        return Real(low=param["min"], high=param["max"], name=name)
    if param["type"] == "FLOAT_EXP":
        return Real(
            low=param["min"],
            high=param["max"],
            base=10,
            prior="log-uniform",
            name=name,
        )
    else:
        raise ValueError("Didn't understand space {}.".format(param))


def get_methods_space(methods):
    from skopt.space import Categorical

    return [Categorical(methods)]


def convert_to_skopt_space(method, space):
    return [
        convert_param_to_skopt(param, name=name)
        for name, param in space[method].items()
    ]


class SkoptOptLib(HyperOptLib):
    """Hyper-optimization using ``scikit-optimize``."""

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        sampler="et",
        method_sampler="et",
        sampler_opts=None,
        method_sampler_opts=None,
        **kwargs,
    ):
        """Initialize the ``skopt`` optimizer.

        Parameters
        ----------
        methods : list[str]
            The list of contraction methods to optimize over.
        space : dict[str, dict[str, dict]]
            The search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer instance.
        sampler : str, optional
            The regressor to use to optimize each method's search
            space.
        method_sampler : str, optional
            Meta-optimizer to use to select which overall method to
            use.
        sampler_opts : dict, optional
            Options to supply to the per-method optimizer.
        method_sampler_opts : dict, optional
            Options to supply to the method selector.
        """
        from skopt.optimizer import Optimizer

        sampler_opts = {} if sampler_opts is None else dict(sampler_opts)
        method_sampler_opts = (
            {} if method_sampler_opts is None else dict(method_sampler_opts)
        )

        if method_sampler is None:
            method_sampler = sampler

        self._method_chooser = Optimizer(
            get_methods_space(methods),
            base_estimator=method_sampler,
            **method_sampler_opts,
        )

        skopt_spaces = {m: convert_to_skopt_space(m, space) for m in methods}
        self._param_names = {
            m: [p.name for p in skopt_spaces[m]] for m in methods
        }
        self._optimizers = {
            m: Optimizer(
                skopt_spaces[m],
                base_estimator=sampler,
                **sampler_opts,
            )
            for m in methods
        }

    def get_setting(self):
        """Find the next parameters to test."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="skopt")
            warnings.filterwarnings("ignore", module="sklearn")
            method = self._method_chooser.ask()
            params = self._optimizers[method[0]].ask()
            names = self._param_names[method[0]]

        return {
            "method_token": method,
            "method": method[0],
            "params_token": params,
            "params": dict(zip(names, params)),
        }

    def report_result(self, setting, trial, score):
        """Report the result of a trial to the skopt optimizers."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="skopt")
            warnings.filterwarnings("ignore", module="sklearn")

            self._method_chooser.tell(setting["method_token"], score)
            self._optimizers[setting["method"]].tell(
                setting["params_token"], score
            )


register_hyper_optlib("skopt", SkoptOptLib)
