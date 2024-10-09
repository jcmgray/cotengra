"""Hyper optimization using choco."""

from .hyper import register_hyper_optlib


def convert_param_to_choco(param):
    """Convert a single search parameter suitably for ``chocolate``."""
    from math import log10
    import chocolate as choco

    if param["type"] == "BOOL":
        return choco.choice([False, True])
    if param["type"] == "INT":
        return choco.quantized_uniform(
            low=param["min"], high=param["max"] + 1, step=1
        )
    if param["type"] == "STRING":
        return choco.choice(param["options"])
    if param["type"] == "FLOAT":
        return choco.uniform(low=param["min"], high=param["max"])
    if param["type"] == "FLOAT_EXP":
        return choco.log(
            low=log10(param["min"]), high=log10(param["max"]), base=10
        )
    else:
        raise ValueError("Didn't understand space {}.".format(param))


def get_chocolate_space(methods, space):
    """Get the whole search space defined suitably for ``chocolate``."""
    return [
        {
            "method": m,
            **{o: convert_param_to_choco(p) for o, p in space[m].items()},
        }
        for m in methods
    ]


def chocolate_init_optimizers(
    self,
    methods,
    space,
    sampler="CMAES",
    sampler_opts=None,
):
    """Initialize the ``chocolate`` optimizer.

    Parameters
    ----------
    space : dict[str, dict[str, dict]]
        The search space.
    sampler : str, optional
        The optimizer to search the space with, see
        https://chocolate.readthedocs.io/en/latest/tutorials/algo.html.
    """
    import chocolate as choco

    algo = getattr(choco, sampler)
    conn = choco.DataFrameConnection()
    choco_space = get_chocolate_space(methods, space)
    sampler_opts = {} if sampler_opts is None else dict(sampler_opts)

    self._sampler = choco.ThompsonSampling(
        algo, conn, choco_space, **sampler_opts
    )


def chocolate_get_setting(self):
    """Find the next parameters to test."""
    token, params = self._sampler.next()
    method = params.pop("method")
    return {
        "token": token,
        "method": method,
        "params": params,
    }


def chocolate_report_result(self, setting, trial, score):
    """Report the result of a trial to the ``chocolate`` optimizer."""
    self._sampler.update(setting["token"], score)


register_hyper_optlib(
    "chocolate",
    chocolate_init_optimizers,
    chocolate_get_setting,
    chocolate_report_result,
)
