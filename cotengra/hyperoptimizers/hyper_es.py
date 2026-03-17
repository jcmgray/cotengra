"""Hyper optimization using a steady-state diagonal evolutionary strategy."""

import bisect
import math

from ..utils import get_rng
from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib

# E[|N(0,1)|] = sqrt(2/pi), used for CSA path length expectation.
_CHI1 = math.sqrt(2.0 / math.pi)


def _reflect(x):
    """Reflect a scalar into [-1, 1]."""
    # fold back into [-1, 1] by reflecting off boundaries
    while x > 1.0 or x < -1.0:
        if x > 1.0:
            x = 2.0 - x
        if x < -1.0:
            x = -2.0 - x
    return x


class SteadyStateES:
    """A steady-state diagonal evolutionary strategy operating in raw
    ``[-1, 1]`` parameter space.

    Unlike a generational ES, this sampler has no synchronization barrier.
    Every ``ask`` draws from the current distribution immediately, and
    every ``tell`` updates the distribution state from a sliding window
    of recent results. Trials can be in-flight in any number and can
    complete in any order.

    Step sizes are adapted per dimension using a simplified cumulative
    step-size adaptation (CSA) rule, giving the optimizer the character
    of a separable (diagonal) CMA-ES without any matrix operations.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single contraction method.
    seed : None or int, optional
        Random seed.
    population_size : int or "auto", optional
        Size of the sliding archive window. When ``"auto"`` it is
        chosen as ``max(8, 4 * ndim)``.
    sigma : float, optional
        Initial per-dimension step size in raw ``[-1, 1]`` space.
    sigma_min : float, optional
        Lower bound for per-dimension step sizes.
    sigma_max : float, optional
        Upper bound for per-dimension step sizes (also the restart
        value).
    c_sigma : float or "auto", optional
        Learning rate for the evolution path. When ``"auto"`` it is
        set to ``1 / sqrt(ndim)`` (clamped to ``[0.05, 1.0]``).
    d_sigma : float or "auto", optional
        Damping factor for step-size updates. When ``"auto"`` it is
        set to ``1 + sqrt(ndim)``.
    elite_ratio : float, optional
        Fraction of the archive used as elites for weighted
        recombination.
    use_mirror : bool, optional
        If ``True``, every fresh perturbation is paired with its
        antipodal mirror, halving gradient variance.
    inject_best_every : int or "auto", optional
        Inject the best-so-far point every this many asks.  When
        ``"auto"`` it equals ``population_size``.
    restart_patience : int or "auto", optional
        Number of tells without improvement before resetting step
        sizes and re-centering on the best-so-far point. When
        ``"auto"`` it equals ``5 * population_size``.
    explore_prob : float, optional
        Probability of issuing a uniform random point instead of the
        ES-directed point. Maintains global diversity throughout the
        search and helps escape local minima. Random points still feed
        into the archive normally.
    mean_lr : float, optional
        Learning rate for the mean update in ``(0, 1]``. Each tell
        blends the current mean toward the elite weighted mean by this
        factor: ``mean = (1 - mean_lr) * mean + mean_lr * elite_mean``.
        Values below ``1.0`` smooth out noise-driven mean drift. Default
        is ``1.0`` (full update, original behaviour).
    exponential_param_power : float, optional
        Passed through to the shared parameter mapping for
        ``FLOAT_EXP`` parameters.
    """

    def __init__(
        self,
        space,
        seed=None,
        population_size="auto",
        sigma=0.5,
        sigma_min=0.01,
        sigma_max=1.0,
        c_sigma="auto",
        d_sigma="auto",
        elite_ratio=0.5,
        use_mirror=True,
        inject_best_every="auto",
        restart_patience="auto",
        explore_prob=0.05,
        mean_lr=0.5,
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

        self.sigma0 = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if c_sigma == "auto":
            if self.ndim > 0:
                c_sigma = min(1.0, max(0.05, 1.0 / math.sqrt(self.ndim)))
            else:
                c_sigma = 1.0
        self.c_sigma = c_sigma

        if d_sigma == "auto":
            d_sigma = 1.0 + math.sqrt(self.ndim)
        self.d_sigma = d_sigma

        self.elite_ratio = elite_ratio
        self.use_mirror = use_mirror
        self.explore_prob = explore_prob
        self.mean_lr = mean_lr

        if inject_best_every == "auto":
            inject_best_every = population_size
        self.inject_best_every = inject_best_every

        if restart_patience == "auto":
            restart_patience = 2 * population_size
        self.restart_patience = restart_patience

        # precompute log-rank weights for the elite portion of the archive
        mu = max(1, int(round(self.elite_ratio * self.population_size)))
        raw_w = [math.log(mu + 1) - math.log(i + 1) for i in range(mu)]
        w_sum = sum(raw_w)
        self._weights = [w / w_sum for w in raw_w]
        self._mu = mu
        # effective number of elites (for CSA scaling)
        self._mu_eff = 1.0 / sum(w * w for w in self._weights)

        self._init_state()

    def _init_state(self):
        """Initialize or reset the mutable optimizer state."""
        self.mean = [0.0] * self.ndim
        self.sigmas = [self.sigma0] * self.ndim
        self.p_sigma = [0.0] * self.ndim

        # archive: list of (score, x) sorted by score ascending (best
        # first).  we keep at most ``population_size`` entries.
        self._archive = []

        # in-flight trials not yet told
        self._pending = {}

        # global best
        self.best_x = None
        self.best_score = float("inf")

        self._ask_counter = 0
        self._tell_counter = 0
        self._tells_since_improvement = 0
        self._restart_count = 0

        # mirror queue: the eps vector waiting to be issued as -eps
        self._mirror_eps = None

    # ------------------------------------------------------------------ #
    #  ask
    # ------------------------------------------------------------------ #

    def ask(self):
        """Return the next candidate point.

        Draws from the current distribution, optionally using mirrored
        sampling or best-so-far injection.  No synchronization with
        outstanding trials is required.
        """
        trial_number = self._ask_counter
        self._ask_counter += 1

        if self.ndim == 0:
            x = ()
            self._pending[trial_number] = x
            return trial_number, convert_raw(self.params, x)

        # --- global random exploration ---
        if self.explore_prob > 0 and self.rng.random() < self.explore_prob:
            x = tuple(self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim))
            self._pending[trial_number] = x
            return trial_number, convert_raw(self.params, x)

        # --- mirrored partner from previous ask ---
        if self._mirror_eps is not None:
            eps = self._mirror_eps
            self._mirror_eps = None
            x = tuple(
                _reflect(self.mean[d] - eps[d]) for d in range(self.ndim)
            )
            self._pending[trial_number] = x
            return trial_number, convert_raw(self.params, x)

        # --- best-so-far injection ---
        if (
            self.best_x is not None
            and self.inject_best_every > 0
            and trial_number % self.inject_best_every == 0
            and trial_number > 0
        ):
            # inject with a tiny perturbation so it's not an exact
            # duplicate (which would be wasted on a stochastic function)
            x = tuple(
                _reflect(self.rng.gauss(self.best_x[d], 0.05 * self.sigmas[d]))
                for d in range(self.ndim)
            )
            self._pending[trial_number] = x
            return trial_number, convert_raw(self.params, x)

        # --- fresh sample ---
        eps = tuple(
            self.rng.gauss(0.0, self.sigmas[d]) for d in range(self.ndim)
        )
        x = tuple(_reflect(self.mean[d] + eps[d]) for d in range(self.ndim))

        if self.use_mirror:
            self._mirror_eps = eps

        self._pending[trial_number] = x
        return trial_number, convert_raw(self.params, x)

    # ------------------------------------------------------------------ #
    #  tell
    # ------------------------------------------------------------------ #

    def tell(self, trial_number, score):
        """Record a completed trial and update the distribution.

        Updates the sliding archive, recomputes the weighted mean from
        elites, adapts per-dimension step sizes via evolution paths, and
        checks for stalls.
        """
        x = self._pending.pop(trial_number)
        self._tell_counter += 1

        # update global best
        if score < self.best_score:
            self.best_score = score
            self.best_x = x
            self._tells_since_improvement = 0
        else:
            self._tells_since_improvement += 1

        if self.ndim == 0:
            return

        # --- insert into archive (sorted by score, ascending) ---
        pos = bisect.bisect_left(self._archive, score, key=lambda e: e[0])
        self._archive.insert(pos, (score, x))
        if len(self._archive) > self.population_size:
            self._archive.pop()  # drop worst (highest score)

        # need at least mu results before we can update meaningfully
        if len(self._archive) < self._mu:
            return

        # --- weighted recombination from elites ---
        old_mean = self.mean
        elite_mean = [0.0] * self.ndim
        for wi, (_, xi) in zip(self._weights, self._archive):
            for d in range(self.ndim):
                elite_mean[d] += wi * xi[d]
        # blend toward elite mean at rate mean_lr; values < 1.0 smooth
        # out noise-driven drift on highly stochastic objectives
        lr = self.mean_lr
        if lr >= 1.0:
            new_mean = elite_mean
        else:
            new_mean = [
                (1.0 - lr) * old_mean[d] + lr * elite_mean[d]
                for d in range(self.ndim)
            ]
        self.mean = new_mean

        # --- evolution path and sigma adaptation per dimension ---
        c = self.c_sigma
        complement = math.sqrt(c * (2.0 - c) * self._mu_eff)

        for d in range(self.ndim):
            sd = self.sigmas[d]
            if sd < 1e-30:
                sd = self.sigma_min
            delta = (new_mean[d] - old_mean[d]) / sd

            # update conjugate evolution path
            self.p_sigma[d] = (1.0 - c) * self.p_sigma[d] + (
                complement * delta
            )

            # adapt step size: increase if path is longer than expected,
            # decrease if shorter
            self.sigmas[d] = sd * math.exp(
                (c / self.d_sigma) * (abs(self.p_sigma[d]) / _CHI1 - 1.0)
            )

            # clamp
            self.sigmas[d] = min(
                max(self.sigmas[d], self.sigma_min), self.sigma_max
            )

        # --- stall-triggered restart ---
        if self._tells_since_improvement >= self.restart_patience:
            self._restart()

    def _restart(self):
        """Reset step sizes and re-center.

        Alternates between re-centering on the best-so-far point
        (exploitation restart) and a uniformly random point
        (exploration restart) to avoid repeated restarts converging
        to the same local minimum.
        """
        self.sigmas = [self.sigma0] * self.ndim
        self.p_sigma = [0.0] * self.ndim
        if self._restart_count % 2 == 0 and self.best_x is not None:
            # even restarts: exploit best known point
            self.mean = list(self.best_x)
        else:
            # odd restarts: jump to a random point in the search space
            self.mean = [self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim)]
        self._archive.clear()
        self._tells_since_improvement = 0
        self._restart_count += 1


class ESOptLib(HyperOptLib):
    """Hyper-optimization using a steady-state diagonal ES."""

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        population_size="auto",
        sigma=0.5,
        sigma_min=0.01,
        sigma_max=1.0,
        c_sigma="auto",
        d_sigma="auto",
        elite_ratio=0.5,
        use_mirror=True,
        inject_best_every="auto",
        restart_patience="auto",
        explore_prob=0.05,
        mean_lr=1.0,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        """Initialize ES optimizers for each contraction method.

        Parameters
        ----------
        methods : list[str]
            The contraction methods to optimize over.
        space : dict[str, dict[str, dict]]
            The per-method hyperparameter search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer.
        population_size : int or "auto", optional
            Archive window size for each method-specific ES sampler.
        sigma : float, optional
            Initial per-dimension step size.
        sigma_min, sigma_max : float, optional
            Bounds for per-dimension step sizes.
        c_sigma : float or "auto", optional
            Evolution path learning rate.
        d_sigma : float or "auto", optional
            Step-size damping factor.
        elite_ratio : float, optional
            Fraction of the archive used as elites.
        use_mirror : bool, optional
            Enable mirrored sampling.
        inject_best_every : int or "auto", optional
            Best-injection frequency.
        restart_patience : int or "auto", optional
            Tells without improvement before restart.
        explore_prob : float, optional
            Probability of issuing a uniform random point on each ask.
        mean_lr : float, optional
            Learning rate for the mean update in ``(0, 1]``.
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
        self._optimizers = {}
        for i, method in enumerate(methods):
            # diversify seeds across methods
            if seed is not None:
                method_seed = hash((seed, i, method)) % (2**31)
            else:
                method_seed = None
            self._optimizers[method] = SteadyStateES(
                space[method],
                seed=method_seed,
                population_size=population_size,
                sigma=sigma,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                c_sigma=c_sigma,
                d_sigma=d_sigma,
                elite_ratio=elite_ratio,
                use_mirror=use_mirror,
                inject_best_every=inject_best_every,
                restart_patience=restart_patience,
                explore_prob=explore_prob,
                mean_lr=mean_lr,
                exponential_param_power=exponential_param_power,
            )

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
        """Report a completed trial back to the method chooser and ES."""
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib("es", ESOptLib)
