"""Hyper optimization using a pure Python differential evolution strategy."""

from ..utils import get_rng
from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib


class HyperDESampler:
    """A lightweight differential evolution optimizer operating in raw
    ``[-1, 1]`` parameter space.

    Each generation maintains a population of candidate vectors. New trial
    vectors are created using ``DE/rand/1/bin`` mutation and binomial
    crossover, then kept only if they improve on their parent.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single contraction method.
    seed : None or int, optional
        Random seed.
    population_size : int or "auto", optional
        The population size. When ``"auto"`` it is chosen based on the mapped
        parameter dimension.
    mutation : float, optional
        The differential weight (F) applied to the difference vector.
    crossover : float, optional
        The crossover probability (CR) for binomial crossover.
    mutation_decay : float, optional
        Multiplicative decay applied to ``mutation`` after each completed
        generation.
    mutation_min : float, optional
        Lower bound for ``mutation``.
    mutation_max : float, optional
        Upper bound for ``mutation``.
    exponential_param_power : float, optional
        Passed through to the shared parameter mapping for ``FLOAT_EXP``
        parameters.
    """

    def __init__(
        self,
        space,
        seed=None,
        population_size="auto",
        mutation=0.8,
        crossover=0.7,
        mutation_decay=1.0,
        mutation_min=0.1,
        mutation_max=1.5,
        exponential_param_power=None,
    ):
        self.rng = get_rng(seed)
        self.params = build_params(
            space, exponential_param_power=exponential_param_power
        )
        self.ndim = num_params(self.params)

        if population_size == "auto":
            population_size = max(8, 5 * self.ndim)
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover
        self.mutation_decay = mutation_decay
        self.mutation_min = mutation_min
        self.mutation_max = mutation_max

        # initialize population uniformly in [-1, 1]
        self._population = [
            tuple(self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim))
            for _ in range(self.population_size)
        ]
        self._scores = [float("inf")] * self.population_size

        self._trial_counter = 0
        self._target_index = 0
        self._generation = None
        self._trial_map = {}

    def _mutate(self, target_idx):
        """Create a trial vector via DE/rand/1/bin."""
        # pick three distinct indices, all different from target
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        r0, r1, r2 = self.rng.sample(indices, 3)

        x_r0 = self._population[r0]
        x_r1 = self._population[r1]
        x_r2 = self._population[r2]

        # mutation: v = x_r0 + F * (x_r1 - x_r2)
        v = []
        for d in range(self.ndim):
            vi = x_r0[d] + self.mutation * (x_r1[d] - x_r2[d])
            v.append(min(max(vi, -1.0), 1.0))

        # binomial crossover
        x_target = self._population[target_idx]
        j_rand = self.rng.randrange(self.ndim)
        trial = []
        for d in range(self.ndim):
            if self.rng.random() < self.crossover or d == j_rand:
                trial.append(v[d])
            else:
                trial.append(x_target[d])

        return tuple(trial)

    def _sample_generation(self):
        """Prepare trial vectors for all population members."""
        self._generation = {
            "trials": [],
            "trial_numbers": [],
            "target_indices": [],
            "scores": {},
            "next_index": 0,
        }
        for i in range(self.population_size):
            self._extend_generation(i)

    def _extend_generation(self, target_idx=None):
        """Append one more trial to the current generation."""
        if target_idx is None:
            # wrap around if we need more trials than population
            target_idx = len(self._generation["trials"]) % self.population_size

        trial_number = self._trial_counter
        self._trial_counter += 1

        trial_vec = self._mutate(target_idx)
        slot = len(self._generation["trials"])

        self._generation["trials"].append(trial_vec)
        self._generation["trial_numbers"].append(trial_number)
        self._generation["target_indices"].append(target_idx)
        self._trial_map[trial_number] = slot

    def ask(self):
        """Return the next candidate from the current generation.

        If all prepared candidates have been issued, grow the generation
        by one more sample.
        """
        if self._generation is None:
            self._sample_generation()

        if self._generation["next_index"] >= len(self._generation["trials"]):
            self._extend_generation()

        i = self._generation["next_index"]
        self._generation["next_index"] += 1

        trial_number = self._generation["trial_numbers"][i]
        x = self._generation["trials"][i]

        return trial_number, convert_raw(self.params, x)

    def tell(self, trial_number, score):
        """Record a completed trial and perform selection if the
        generation is complete.

        For each trial vector, if it scores better than (or equal to) its
        target parent, it replaces the parent in the population.
        """
        slot = self._trial_map.pop(trial_number)
        self._generation["scores"][slot] = score

        if len(self._generation["scores"]) != self._generation["next_index"]:
            return

        # selection: compare each trial against its target
        # (only iterate over issued slots, not all pre-sampled ones)
        for slot_i in range(self._generation["next_index"]):
            target_idx = self._generation["target_indices"][slot_i]
            trial_score = self._generation["scores"][slot_i]
            if trial_score <= self._scores[target_idx]:
                self._population[target_idx] = self._generation["trials"][
                    slot_i
                ]
                self._scores[target_idx] = trial_score

        # decay mutation factor
        self.mutation *= self.mutation_decay
        self.mutation = min(
            max(self.mutation, self.mutation_min), self.mutation_max
        )

        self._generation = None


class DEOptLib(HyperOptLib):
    """Hyper-optimization using differential evolution."""

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        population_size="auto",
        mutation=0.8,
        crossover=0.7,
        mutation_decay=1.0,
        mutation_min=0.1,
        mutation_max=1.5,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        """Initialize DE optimizers for each contraction method.

        Parameters
        ----------
        methods : list[str]
            The contraction methods to optimize over.
        space : dict[str, dict[str, dict]]
            The per-method hyperparameter search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer. Used to size the initial population
            large enough for parallel pre-dispatch.
        population_size : int or "auto", optional
            The population size for each method-specific DE sampler.
        mutation : float, optional
            Differential weight (F).
        crossover : float, optional
            Crossover probability (CR).
        mutation_decay, mutation_min, mutation_max : float, optional
            Parameters controlling mutation scale over generations.
        method_exploration : float, optional
            Exploration strength for the LCB-based method chooser.
        method_temperature : float, optional
            Noise temperature for the LCB-based method chooser.
        exponential_param_power : float, optional
            Passed to the shared parameter mapping for ``FLOAT_EXP``.
        seed : None or int, optional
            Random seed.
        """
        if population_size == "auto":
            max_ndim = max(
                num_params(
                    build_params(
                        space[m],
                        exponential_param_power=exponential_param_power,
                    )
                )
                for m in methods
            )
            population_size = max(
                8,
                max(1, getattr(optimizer, "pre_dispatch", 1)),
                5 * max_ndim,
            )

        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
            seed=seed,
        )
        self._optimizers = {
            method: HyperDESampler(
                space[method],
                seed=seed,
                population_size=population_size,
                mutation=mutation,
                crossover=crossover,
                mutation_decay=mutation_decay,
                mutation_min=mutation_min,
                mutation_max=mutation_max,
                exponential_param_power=exponential_param_power,
            )
            for method in methods
        }

    def get_setting(self):
        """Choose a contraction method, then request its next setting."""
        method = self._method_chooser.ask()
        params_token, params = self._optimizers[method].ask()

        return {
            "method": method,
            "params_token": params_token,
            "params": params,
        }

    def report_result(self, setting, trial, score):
        """Report a completed trial back to the method chooser and DE."""
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib("de", DEOptLib)
register_hyper_optlib("diffev", DEOptLib)
