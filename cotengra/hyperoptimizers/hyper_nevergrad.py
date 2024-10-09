"""Hyper optimization using nevergrad."""

from .hyper import register_hyper_optlib


def convert_param_to_nevergrad(param):
    import nevergrad as ng

    if param["type"] == "BOOL":
        return ng.p.Choice([False, True])
    if param["type"] == "INT":
        return ng.p.TransitionChoice(range(param["min"], param["max"] + 1))
    if param["type"] == "STRING":
        return ng.p.Choice(param["options"])
    if param["type"] == "FLOAT":
        return ng.p.Scalar(lower=param["min"], upper=param["max"])
    if param["type"] == "FLOAT_EXP":
        return ng.p.Log(lower=param["min"], upper=param["max"])
    else:
        raise ValueError("Didn't understand space {}.".format(param))


def get_methods_space(methods):
    import nevergrad as ng

    return ng.p.Choice(methods)


def convert_to_nevergrad_space(method, space):
    import nevergrad as ng

    return ng.p.Dict(
        **{o: convert_param_to_nevergrad(p) for o, p in space[method].items()}
    )


def nevergrad_init_optimizers(
    self,
    methods,
    space,
    sampler="NaiveTBPSA",
    method_sampler=None,
    budget="auto",
    num_workers=1,
    method_budget="auto",
    method_num_workers=1,
    sampler_opts=None,
    method_sampler_opts=None,
):
    """Initialize the ``nevergrad`` optimizer.

    Parameters
    ----------
    space : dict[str, dict[str, dict]]
        The search space.
    sampler : str, optional
        The optimizer to use to search each method's search space, see
        https://facebookresearch.github.io/nevergrad/optimizers_ref.html.
    method_sampler, str, optional
        The meta-optimizer to use to select overall methods.
    budget : int, optional
        Supplied to optimizer.
    num_workers : int, optional
        Supplied to optimizer.
    method_budget : int, optional
        Supplied to meta-optimizer.
    method_num_workers : int, optional
        Supplied to meta-optimizer.
    """
    import nevergrad as ng

    sampler_opts = {} if sampler_opts is None else dict(sampler_opts)
    method_sampler_opts = (
        {} if method_sampler_opts is None else dict(method_sampler_opts)
    )

    if method_sampler is None:
        if len(methods) == 1:
            # some samplers don't support '1D' search spaces -- ignore
            method_sampler = "RandomSearch"
        else:
            method_sampler = sampler
    if method_budget == "auto":
        method_budget = self.max_repeats
    if method_num_workers == "auto":
        method_num_workers = self._num_workers

    self._method_chooser = getattr(ng.optimizers, method_sampler)(
        parametrization=get_methods_space(methods),
        budget=method_budget,
        num_workers=method_num_workers,
        **method_sampler_opts,
    )

    if budget == "auto":
        budget = self.max_repeats
    if num_workers == "auto":
        num_workers = self._num_workers

    self._optimizers = {
        m: getattr(ng.optimizers, sampler)(
            parametrization=convert_to_nevergrad_space(m, space),
            budget=budget,
            num_workers=num_workers,
            **sampler_opts,
        )
        for m in methods
    }


def nevergrad_get_setting(self):
    """Get a setting to trial from one of the nevergrad optimizers."""
    method = self._method_chooser.ask()
    params = self._optimizers[method.args[0]].ask()
    return {
        "method_token": method,
        "method": method.args[0],
        "params_token": params,
        "params": params.args[0],
    }


def nevergrad_report_result(self, setting, trial, score):
    """Report the result of a trial to the ``nevergrad`` optimizers."""
    self._method_chooser.tell(setting["method_token"], score)
    self._optimizers[setting["method"]].tell(setting["params_token"], score)


register_hyper_optlib(
    "nevergrad",
    nevergrad_init_optimizers,
    nevergrad_get_setting,
    nevergrad_report_result,
)
