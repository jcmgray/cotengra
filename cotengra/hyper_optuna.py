
import warnings

from .hyper import register_hyper_optlib


def make_getter(name, param):
    if param['type'] == 'BOOL':
        return lambda trial: trial.suggest_categorical(
            name, [False, True])
    if param['type'] == 'INT':
        return lambda trial: trial.suggest_int(
            name, param['min'], param['max'])
    if param['type'] == 'STRING':
        return lambda trial: trial.suggest_categorical(
            name, param['options'])
    if param['type'] == 'FLOAT':
        return lambda trial: trial.suggest_uniform(
            name, param['min'], param['max'])
    if param['type'] == 'FLOAT_EXP':
        return lambda trial: trial.suggest_loguniform(
            name, param['min'], param['max'])
    raise ValueError("Didn't understand space {}.".format(param))


def make_retriever(methods, space):

    if len(methods) == 1:
        def meth_getter(_):
            return methods[0]
    else:
        def meth_getter(trial):
            return trial.suggest_categorical("method", methods)

    getters = {}
    for meth, meth_space in space.items():
        getters[meth] = {}
        for name, param in meth_space.items():
            getters[meth][name] = make_getter(name, param)

    def retriever(trial):
        meth = meth_getter(trial)
        return {
            'method': meth,
            'params': {
                n: getter(trial) for n, getter in getters[meth].items()
            },
        }

    return retriever


def optuna_init_optimizers(
    self,
    methods,
    space,
    **create_study_opts,
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    self._study = optuna.create_study(**create_study_opts)
    self._retrieve_params = make_retriever(methods, space)


def optuna_get_setting(self):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='.*divide by zero.*'
        )

        otrial = self._study.ask()
        return {
            "trial_number": otrial.number,
            **self._retrieve_params(otrial),
        }


def optuna_report_result(self, settings, trial, score):
    self._study.tell(settings['trial_number'], score)


register_hyper_optlib(
    'optuna',
    optuna_init_optimizers,
    optuna_get_setting,
    optuna_report_result,
)
