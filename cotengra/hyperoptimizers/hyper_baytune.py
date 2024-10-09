"""Hyper optimization using baytune."""

from .hyper import register_hyper_optlib


BTB_TYPE_TO_HYPERPARAM = {
    "BOOL": "BooleanHyperParam",
    "INT": "IntHyperParam",
    "INT_CAT": "CategoricalHyperParam",
    "STRING": "CategoricalHyperParam",
    "FLOAT": "FloatHyperParam",
    "FLOAT_EXP": "FloatHyperParam",  # no more EXP support in baytune?
}


def convert_param_to_baytune(param):
    """Convert a search subspace to ``baytune`` form."""
    from btb.tuning import hyperparams

    hp = getattr(hyperparams, BTB_TYPE_TO_HYPERPARAM[param["type"]])

    if param["type"] in ("BOOL",):
        return hp()
    elif param["type"] in ("INT_CAT", "STRING"):
        return hp(param["options"])
    else:
        return hp(
            param["min"], param["max"], include_min=True, include_max=True
        )


def baytune_init_optimizers(
    self,
    methods,
    space,
    sampler="GP",
    method_sampler="UCB1",
    sampler_opts=None,
):
    """
    Set-up the baytune optimizer(s).

    Parameters
    ----------
    space : dict[str, dict[str, dict]]
        The search space.
    sampler : str, optional
        Which ``btb`` parameter fitter to use - default ``'GP'`` means gaussian
        process. Other options include ``'Uniform'`` and ``'GPEi'``.
        See https://hdi-project.github.io/BTB/api/btb.tuning.tuners.html.
    method_sampler : str, optional
        Which ``btb`` selector to use - default 'UCB1'.
        See https://hdi-project.github.io/BTB/api/btb.selection.html.
    sampler_opts : dict, optional
        Options to supply to ``btb``.
    """
    import btb
    import btb.selection
    from btb.tuning import Tunable

    sampler_opts = {} if sampler_opts is None else dict(sampler_opts)
    self._selector = getattr(btb.selection, method_sampler)(
        methods, **sampler_opts
    )

    # for compatability
    if "Tuner" not in sampler:
        sampler += "Tuner"

    tuner_fn = getattr(btb.tuning.tuners, sampler)

    self._tuners = {
        method: tuner_fn(
            Tunable(
                {
                    name: convert_param_to_baytune(param)
                    for name, param in space[method].items()
                }
            )
        )
        for method in methods
    }


def baytune_get_setting(self):
    """Get a setting to trial from one of the baytune optimizers."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="sklearn")

        if len(self._methods) == 1:
            (method,) = self._methods
        else:
            possible_methods = {
                m: getattr(self._tuners[m], "scores", ())
                for m in self._methods
            }
            method = self._selector.select(possible_methods)

        params = self._tuners[method].propose()
        return {"method": method, "params": params}


def baytune_report_result(self, setting, trial, score):
    """Report the result of a trial to the baytune optimizers."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="sklearn")
        self._tuners[setting["method"]].record(setting["params"], -score)


register_hyper_optlib(
    "baytune",
    baytune_init_optimizers,
    baytune_get_setting,
    baytune_report_result,
)
