"""Hyper optimization using parallel evolution with ranked sigma assignment."""

import math

from ..utils import get_rng
from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib


class HyperPESampler:
    """A parallel evolution optimizer operating in raw ``[-1, 1]`` space.

    Multiple workers each maintain their own solution. Perturbation scales
    (sigmas) are distributed across an evenly spaced range and reassigned
    by rank after each generation: the best-scoring worker gets the lowest
    sigma (exploit) and the worst gets the highest (explore).

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single contraction method.
    seed : None or int, optional
        Random seed.
    population_size : int or "auto", optional
        The number of parallel workers. When ``"auto"`` it is chosen
        based on the mapped parameter dimension.
    sigma_min : float, optional
        The smallest perturbation scale (assigned to the best worker).
    sigma_max : float, optional
        The largest perturbation scale (assigned to the worst worker).
    elite_migrate_prob : float, optional
        Probability each generation of copying the best worker's
        solution to the worst worker's slot.
    differential_prob : float, optional
        Per-sample probability of using a differential perturbation
        (``x_best - x_rand``) instead of Gaussian noise.
    patience : int or None, optional
        If a worker has not improved for this many generations,
        re-randomize its solution. ``None`` or ``0`` disables.
    exponential_param_power : float, optional
        Passed through to the shared parameter mapping for ``FLOAT_EXP``
        parameters.
    """

    def __init__(
        self,
        space,
        seed=None,
        population_size=8,
        sigma_min=0.01,
        sigma_max=0.5,
        elite_migrate_prob=0.0,
        differential_prob=0.0,
        patience=None,
        exponential_param_power=None,
    ):
        self.rng = get_rng(seed)
        self.params = build_params(
            space, exponential_param_power=exponential_param_power
        )
        self.ndim = num_params(self.params)

        if population_size == "auto":
            population_size = max(8, 4 * self.ndim)
        self.population_size = population_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.elite_migrate_prob = elite_migrate_prob
        self.differential_prob = differential_prob
        self.patience = patience

        # initialize each worker's solution uniformly in [-1, 1]
        self._solutions = [
            tuple(self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim))
            for _ in range(self.population_size)
        ]
        self._scores = [float("inf")] * self.population_size
        self._stagnation = [0] * self.population_size

        # evenly spaced sigmas, assigned by rank (best -> lowest)
        self._sigmas = self._make_sigmas()

        self._trial_counter = 0
        self._generation = None
        self._trial_map = {}

    def _make_sigmas(self):
        """Create geometrically spaced sigmas from sigma_min to sigma_max."""
        n = self.population_size
        if n == 1:
            return [math.sqrt(self.sigma_min * self.sigma_max)]
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        return [
            math.exp(log_min + i * (log_max - log_min) / (n - 1))
            for i in range(n)
        ]

    def _sample_candidate(self, worker_idx, noise=None):
        """Perturb a worker's current solution with its assigned sigma."""
        sigma = self._sigmas[worker_idx]
        sol = self._solutions[worker_idx]

        if noise is None:
            # possibly use differential perturbation
            if (
                self.differential_prob > 0.0
                and self.population_size >= 3
                and self.rng.random() < self.differential_prob
            ):
                best_idx = min(
                    range(self.population_size),
                    key=lambda i: self._scores[i],
                )
                others = [
                    j
                    for j in range(self.population_size)
                    if j != worker_idx and j != best_idx
                ]
                rand_idx = self.rng.choice(others)
                noise = [
                    self._solutions[best_idx][d] - self._solutions[rand_idx][d]
                    for d in range(self.ndim)
                ]
            else:
                noise = [self.rng.gauss(0.0, 1.0) for _ in range(self.ndim)]

        x = []
        for si, ni in zip(sol, noise):
            xi = si + sigma * ni
            x.append(min(max(xi, -1.0), 1.0))
        return tuple(x)

    def _sample_generation(self):
        """Start a new generation with one trial per worker."""
        self._generation = {
            "worker_indices": [],
            "xs": [],
            "trial_numbers": [],
            "scores": {},
            "next_index": 0,
        }
        for i in range(self.population_size):
            self._extend_generation(i)

    def _extend_generation(self, worker_idx=None, noise=None):
        """Append one more trial to the current generation."""
        if worker_idx is None:
            worker_idx = len(self._generation["xs"]) % self.population_size

        trial_number = self._trial_counter
        self._trial_counter += 1

        slot = len(self._generation["xs"])
        self._generation["worker_indices"].append(worker_idx)
        self._generation["xs"].append(
            self._sample_candidate(worker_idx, noise=noise)
        )
        self._generation["trial_numbers"].append(trial_number)
        self._trial_map[trial_number] = slot

    def ask(self):
        """Return the next candidate from the current generation.

        If all prepared candidates have been issued, grow the generation
        by one more sample.
        """
        if self._generation is None:
            self._sample_generation()

        if self._generation["next_index"] >= len(self._generation["xs"]):
            self._extend_generation()

        i = self._generation["next_index"]
        self._generation["next_index"] += 1

        trial_number = self._generation["trial_numbers"][i]
        x = self._generation["xs"][i]

        return trial_number, convert_raw(self.params, x)

    def tell(self, trial_number, score):
        """Record a completed trial and update workers if the generation
        is complete.

        For each trial, if it scores better than (or equal to) its
        worker's current best, the worker adopts the new solution.
        Then sigmas are reassigned by rank: best worker gets lowest
        sigma, worst gets highest.
        """
        slot = self._trial_map.pop(trial_number)
        self._generation["scores"][slot] = score

        if len(self._generation["scores"]) != self._generation["next_index"]:
            return

        # greedy update: adopt better solutions and track stagnation
        improved = set()
        for slot_i in range(self._generation["next_index"]):
            worker_idx = self._generation["worker_indices"][slot_i]
            trial_score = self._generation["scores"][slot_i]
            if trial_score <= self._scores[worker_idx]:
                self._solutions[worker_idx] = self._generation["xs"][slot_i]
                self._scores[worker_idx] = trial_score
                improved.add(worker_idx)

        for i in range(self.population_size):
            if i in improved:
                self._stagnation[i] = 0
            else:
                self._stagnation[i] += 1

        # stagnation restart
        if self.patience:
            for i in range(self.population_size):
                if self._stagnation[i] >= self.patience:
                    self._solutions[i] = tuple(
                        self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim)
                    )
                    self._scores[i] = float("inf")
                    self._stagnation[i] = 0

        # rank workers by score (best to worst) and reassign sigmas
        ranking = sorted(
            range(self.population_size),
            key=lambda i: self._scores[i],
        )

        # elite migration
        if self.elite_migrate_prob > 0.0:
            if self.rng.random() < self.elite_migrate_prob:
                worst = ranking[-1]
                best = ranking[0]
                self._solutions[worst] = self._solutions[best]

        new_sigmas = [0.0] * self.population_size
        base_sigmas = self._make_sigmas()
        for rank, worker_idx in enumerate(ranking):
            new_sigmas[worker_idx] = base_sigmas[rank]
        self._sigmas = new_sigmas

        self._generation = None


class PEOptLib(HyperOptLib):
    """Hyper-optimization using parallel evolution with ranked sigmas."""

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        population_size="auto",
        sigma_min=0.01,
        sigma_max=0.5,
        elite_migrate_prob=0.5,
        differential_prob=0.5,
        patience=8,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        """Initialize PE optimizers for each contraction method.

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
            The number of parallel workers for each method.
        sigma_min : float, optional
            Smallest perturbation scale (for the best-ranked worker).
        sigma_max : float, optional
            Largest perturbation scale (for the worst-ranked worker).
        elite_migrate_prob : float, optional
            Probability of copying best solution to worst worker.
        differential_prob : float, optional
            Per-sample probability of differential perturbation.
        patience : int or None, optional
            Generations without improvement before restart.
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
                4 * max_ndim,
            )

        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
            seed=seed,
        )
        self._optimizers = {
            method: HyperPESampler(
                space[method],
                seed=seed,
                population_size=population_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                elite_migrate_prob=elite_migrate_prob,
                differential_prob=differential_prob,
                patience=patience,
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
        """Report a completed trial back to the method chooser and PE."""
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib("pe", PEOptLib)
register_hyper_optlib("parallelev", PEOptLib)
