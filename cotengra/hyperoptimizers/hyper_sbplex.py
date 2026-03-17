"""Hyper optimization using the Sbplex method (Rowan, 1990)."""

import warnings

from ..utils import get_rng
from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    generate_lhs_points,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib
from .hyper_neldermead import _clip, _NMCore


class HyperSbplexSampler:
    """Sbplex optimizer in raw ``[-1, 1]`` parameter space.

    This is derived from Subplex (Rowan, 1990) which decomposes the full
    parameter space into low-dimensional subspaces (i.e. subsets of parameters)
    and runs Nelder-Mead (NM) on each in sequence. After all subspaces in a
    *cycle* are optimized, the overall convergence is checked; if the total
    displacement is negligible the search restarts with a jittered center. This
    is generally more robust that vanilla Nelder-Mead especially in higher
    dimensions.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single contraction method.
    seed : None or int, optional
        Random seed.
    adaptive : bool, optional
        Whether to use the adaptive NM coefficients recommended by Gao and
        Han (2010), which scale with problem dimension. If `True` then `alpha`,
        `gamma`, `rho`, and `sigma` are ignored.
    alpha : float, optional
        Reflection coefficient for each sub-NM.
    gamma : float, optional
        Expansion coefficient for each sub-NM.
    rho : float, optional
        Contraction coefficient for each sub-NM.
    sigma : float, optional
        Shrink coefficient for each sub-NM.
    initial_scale : float, optional
        Initial step size in each dimension.
    nsmin : int or None, optional
        Minimum subspace size. Defaults to ``min(2, ndim)``.
    nsmax : int or None, optional
        Maximum subspace size. Defaults to ``min(5, ndim)``.
    psi : float, optional
        Step reduction factor applied when a subspace shows no movement, and
        used to scale the jitter on restart.
    convergence_tol : float, optional
        Convergence threshold for the sub-NM simplex diameter and for the
        overall cycle displacement check.
    filler_scale : float, optional
        Standard deviation of the gaussian noise used for filler points issued
        while the sub-NM is blocked.
    n_initial : int or None, optional
        Number of Latin Hypercube Sampled (LHS) warm-up points to evaluate
        before starting subplex cycling. The best result seeds the starting
        point. Default ``None`` means ``2 * ndim``. Set to ``0`` to disable.
    restart_patience : int or "auto", optional
        Number of completed cycles without a new best score before triggering
        a restart. When ``"auto"`` it is chosen from the number of expected
        subspaces, with a minimum of 3.
    explore_prob : float, optional
        Probability of issuing a uniform random point instead of the
        NM-directed point during normal cycling. Maintains diversity throughout
        the search.
    exponential_param_power : float, optional
        Passed through to the shared parameter mapping.
    """

    def __init__(
        self,
        space,
        seed=None,
        adaptive=True,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
        initial_scale=0.5,
        nsmin=None,
        nsmax=None,
        psi=0.25,
        convergence_tol=0.1,
        filler_scale=0.25,
        n_initial=None,
        restart_patience="auto",
        explore_prob=0.05,
        exponential_param_power=None,
    ):
        self.rng = get_rng(seed)
        self.params = build_params(
            space,
            exponential_param_power=exponential_param_power,
        )
        self.ndim = num_params(self.params)

        self.adaptive = adaptive
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.initial_scale = initial_scale
        self.psi = psi
        self.convergence_tol = convergence_tol
        self.filler_scale = filler_scale
        self.explore_prob = explore_prob

        if nsmin is None:
            nsmin = min(2, self.ndim)
        if nsmax is None:
            nsmax = min(5, self.ndim)
        self.nsmin = nsmin
        self.nsmax = nsmax

        if n_initial is None:
            n_initial = 2 * self.ndim
        self.n_initial = n_initial

        if restart_patience == "auto":
            if self.ndim > 0 and self.nsmax > 0:
                n_subspaces = (self.ndim + self.nsmax - 1) // self.nsmax
            else:
                n_subspaces = 1
            restart_patience = max(3, n_subspaces)
        self.restart_patience = restart_patience

        self._trial_counter = 0
        self._best_x = None
        self._best_score = float("inf")
        # keys: trial_number
        # values: ("nm", core_id, core_token, full_x) or
        #         ("filler", None, None, full_x) or
        #         ("init", None, None, full_x)
        self._trial_map = {}

        # current full point and per-dimension step vector
        self._x = [0.0] * self.ndim
        self._step = [initial_scale] * self.ndim

        self._cycles_since_improvement = 0
        self._restart_count = 0
        self._stagnant_restart_count = 0

        # LHS warm-up phase state
        if self.ndim > 0 and self.n_initial > 0:
            self._init_points = generate_lhs_points(
                self.ndim, self.n_initial, self.rng
            )
            self._init_pending = 0
            self._init_phase = True
        else:
            self._init_points = []
            self._init_pending = 0
            self._init_phase = False

        # subspace cycle state
        self._subspaces = None
        self._sub_idx = 0
        self._sub_dims = None
        self._sub_nm = None
        self._sub_nm_id = None
        self._next_sub_nm_id = 0
        self._x_at_cycle_start = None
        self._best_score_at_cycle_start = float("inf")

    def _partition_dims(self):
        """Partition dimensions into subspaces. Dimensions are sorted by
        ``abs(step[d])`` descending so the most-active dimensions are grouped
        first. Chunks of up to ``nsmax`` dimensions are taken greedily, but
        the last chunk is shortened if it would leave a remainder smaller than
        ``nsmin``.
        """
        order = sorted(
            range(self.ndim),
            key=lambda d: abs(self._step[d]),
            reverse=True,
        )
        subspaces = []
        i = 0
        while i < len(order):
            remaining = len(order) - i
            if remaining <= self.nsmax:
                subspaces.append(order[i:])
                break
            chunk_size = self.nsmax
            leftover = remaining - chunk_size
            # if the leftover would be too small to form a valid subspace,
            # shrink this chunk so the leftover is exactly nsmin
            if 0 < leftover < self.nsmin:
                chunk_size = remaining - self.nsmin
            subspaces.append(order[i : i + chunk_size])
            i += chunk_size
        self._subspaces = subspaces

    def _start_cycle(self):
        """Snapshot the current point and begin a new cycle."""
        self._x_at_cycle_start = list(self._x)
        self._best_score_at_cycle_start = self._best_score
        self._partition_dims()
        self._sub_idx = 0
        self._start_sub_nm()

    def _start_sub_nm(self):
        """Create a ``_NMCore`` for the current subspace."""
        self._sub_dims = self._subspaces[self._sub_idx]
        sub_ndim = len(self._sub_dims)
        center = [self._x[d] for d in self._sub_dims]
        scales = [self._step[d] for d in self._sub_dims]
        self._sub_nm_id = self._next_sub_nm_id
        self._next_sub_nm_id += 1
        self._sub_nm = _NMCore(
            ndim=sub_ndim,
            center=center,
            scales=scales,
            adaptive=self.adaptive,
            alpha=self.alpha,
            gamma=self.gamma,
            rho=self.rho,
            sigma=self.sigma,
            convergence_tol=self.convergence_tol,
        )

    def _embed_sub_vector(self, sub_x):
        """Embed a subspace vector into a full-dimensional point, keeping
        non-subspace dimensions at their current values in ``_x``.
        """
        full_x = list(self._x)
        for i, d in enumerate(self._sub_dims):
            full_x[d] = sub_x[i]
        return full_x

    def _ask_filler(self):
        if self._best_x is not None:
            center = self._best_x
        else:
            center = self._x

        if self.filler_scale == "uniform":
            x = [self.rng.uniform(-1.0, 1.0) for _ in center]
        else:
            x = [_clip(self.rng.gauss(ci, self.filler_scale)) for ci in center]
        trial_number = self._trial_counter
        self._trial_counter += 1
        self._trial_map[trial_number] = ("filler", None, None, x)
        return trial_number, x

    def _rescale_step(self, step, factor, minimum=0.0):
        # shrink or expand the current step by the requested factor
        magnitude = abs(step) * factor

        # enforce a minimum usable step size
        if magnitude < minimum:
            magnitude = minimum

        # if the original step was zero, fall back to a scale based on initialization
        if magnitude == 0.0:
            magnitude = max(minimum, self.initial_scale * factor)

        # preserve the original search direction
        if step < 0.0:
            return -magnitude
        return magnitude

    def _reset_cycle_state(self):
        self._subspaces = None
        self._sub_idx = 0
        self._sub_dims = None
        self._sub_nm = None
        self._sub_nm_id = None
        self._x_at_cycle_start = None
        self._best_score_at_cycle_start = self._best_score

    def _restart(self, mode):
        """Restart the search either locally around the best point or by
        re-expanding globally.

        Parameters
        ----------
        mode : {"local", "global"}
            Restart mode. Local restarts jitter near the best known point while
            preserving step geometry. Global restarts jump to a random point
            and reset step sizes to ``initial_scale``.
        """
        print(
            f"Restarting Sbplex in {mode} mode (restart count={self._restart_count})"
        )

        if mode == "global":
            self._x = [self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim)]
            self._step = [self.initial_scale] * self.ndim
        else:
            # local
            if self._best_x is not None:
                center = self._best_x
            else:
                center = self._x
            self._x = [
                _clip(
                    self.rng.gauss(
                        xi,
                        max(self.psi * abs(si), self.convergence_tol),
                    )
                )
                for xi, si in zip(center, self._step)
            ]
            self._step = [
                self._rescale_step(s, self.psi, self.convergence_tol)
                for s in self._step
            ]

        self._restart_count += 1
        self._stagnant_restart_count += 1
        self._reset_cycle_state()

    def ask(self):
        """Return the next candidate setting. During the LHS warm-up phase,
        pre-generated Latin Hypercube points are issued one at a time. Once all
        warm-up results have been collected, normal subplex cycling begins from
        the best warm-up point. During cycling, with probability
        ``explore_prob`` a uniform random point is returned to maintain
        diversity. If the active sub-NM is blocked waiting for results, a
        filler point is returned instead.
        """
        if self.ndim == 0:
            trial_number = self._trial_counter
            self._trial_counter += 1
            return trial_number, convert_raw(self.params, ())

        # latin hypercube sampling (LHS) warm-up phase
        if self._init_phase:
            if self._init_points:
                x = self._init_points.pop()
                trial_number = self._trial_counter
                self._trial_counter += 1
                self._trial_map[trial_number] = ("init", None, None, x)
                self._init_pending += 1
                return trial_number, convert_raw(self.params, x)

            # all init points issued but some results still pending
            trial_number, x = self._ask_filler()
            return trial_number, convert_raw(self.params, x)

        # random exploration
        current_explore_prob = self.explore_prob + (
            0.05 * self._cycles_since_improvement
        )
        if (
            current_explore_prob > 0
            and self.rng.random() < current_explore_prob
        ):
            x = [self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim)]
            trial_number = self._trial_counter
            self._trial_counter += 1
            self._trial_map[trial_number] = ("filler", None, None, x)
            return trial_number, convert_raw(self.params, x)

        # normal nelder mead subplex cycling, with lazy restarts
        if self._sub_nm is None:
            self._start_cycle()

        result = self._sub_nm.ask()
        if result is not None:
            core_token, sub_x = result
            full_x = self._embed_sub_vector(sub_x)
            trial_number = self._trial_counter
            self._trial_counter += 1
            self._trial_map[trial_number] = (
                "nm",
                self._sub_nm_id,
                core_token,
                full_x,
            )
            return (trial_number, convert_raw(self.params, full_x))

        # sub-NM is blocked, issue filler
        trial_number, x = self._ask_filler()
        return trial_number, convert_raw(self.params, x)

    def tell(self, trial_number, score):
        """Record a completed trial result."""
        if self.ndim == 0:
            if score < self._best_score:
                self._best_score = score
            return

        role, core_id, core_token, full_x = self._trial_map.pop(
            trial_number, ("filler", None, None, None)
        )

        # update global best (includes fillers and init points)
        if score < self._best_score:
            self._best_score = score
            if full_x is not None:
                self._best_x = list(full_x)
            self._cycles_since_improvement = 0
            self._stagnant_restart_count = 0

        # handle LHS warm-up phase completion
        if role == "init":
            self._init_pending -= 1
            if self._init_pending <= 0 and not self._init_points:
                # all warm-up points evaluated — seed _x from best
                if self._best_x is not None:
                    self._x = list(self._best_x)
                self._init_phase = False
            return

        # attempt to inject high-quality filler/exploration results into
        # the active sub-NM simplex so they influence the current cycle
        if (
            role != "nm"
            and full_x is not None
            and self._sub_nm is not None
            and not self._sub_nm.converged
            and self._sub_dims is not None
        ):
            sub_x = [full_x[d] for d in self._sub_dims]
            self._sub_nm.inject_vertex(sub_x, score)

        if role != "nm" or core_token is None:
            return

        # guard: if the sub-NM already converged (from a previous tell) we must
        # not tell it again, as that would skip the _finish_subspace transition
        if (
            self._sub_nm is not None
            and self._sub_nm_id == core_id
            and not self._sub_nm.converged
        ):
            self._sub_nm.tell(core_token, score)
            if self._sub_nm.converged:
                self._finish_subspace()

    def _finish_subspace(self):
        """Extract the sub-NM result, update the full point and step vector,
        then advance to the next subspace or finish the cycle.
        """
        sub_dims = self._sub_dims
        best_sub = self._sub_nm.best_vertex

        if best_sub is not None:
            for i, d in enumerate(sub_dims):
                old = self._x[d]
                self._x[d] = best_sub[i]
                displacement = best_sub[i] - old
                # use the actual displacement as the new step if it is
                # meaningful; otherwise decay the step by psi to avoid stalling
                # on flat dimensions
                if abs(displacement) > self.convergence_tol:
                    self._step[d] = displacement
                else:
                    self._step[d] *= self.psi

        self._sub_idx += 1
        if self._sub_idx < len(self._subspaces):
            self._start_sub_nm()
        else:
            self._finish_cycle()

    def _finish_cycle(self):
        """Check overall convergence across all subspaces. If the total
        displacement over the whole cycle is below ``convergence_tol``, the
        search has stalled and we restart from a jittered copy of the
        best known point with reduced step sizes. Repeated non-improving
        cycles also trigger restarts, escalating to broader exploration.
        """
        max_disp = max(
            abs(self._x[d] - self._x_at_cycle_start[d])
            for d in range(self.ndim)
        )
        improved = self._best_score < self._best_score_at_cycle_start

        if improved:
            self._cycles_since_improvement = 0
        else:
            self._cycles_since_improvement += 1

        should_restart = (
            # simplex has converged
            (max_disp < self.convergence_tol)
            # no score improvement in too many cycles
            or (
                self.restart_patience is not None
                and self._cycles_since_improvement >= self.restart_patience
            )
        )

        if should_restart:
            if self._stagnant_restart_count % 2 == 0:
                restart_mode = "local"
            else:
                restart_mode = "global"
            self._restart(restart_mode)
            return

        self._reset_cycle_state()


class SbplexOptLib(HyperOptLib):
    """Hyper-optimization backend using the 'Sbplex' method adapted from the
    Subplex method (Rowan, 1990).
    """

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        adaptive=True,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
        initial_scale=0.5,
        nsmin=None,
        nsmax=None,
        psi=0.25,
        convergence_tol=0.1,
        filler_scale=0.25,
        n_initial=None,
        restart_patience="auto",
        explore_prob=0.05,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        """Initialize Sbplex optimizers for each method.

        Parameters
        ----------
        methods : list[str]
            The contraction methods to optimize over.
        space : dict[str, dict[str, dict]]
            The per-method hyperparameter search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer.
        adaptive : bool, optional
            Whether to use the adaptive NM coefficients recommended by Gao and
            Han (2010), which scale with problem dimension. If `True` then
            `alpha`, `gamma`, `rho`, and `sigma` are ignored.
        alpha : float, optional
            Reflection coefficient for each sub-NM.
        gamma : float, optional
            Expansion coefficient for each sub-NM.
        rho : float, optional
            Contraction coefficient for each sub-NM.
        sigma : float, optional
            Shrink coefficient for each sub-NM.
        initial_scale : float, optional
            Scale of the initial simplex.
        nsmin : int or None, optional
            Minimum subspace size.
        nsmax : int or None, optional
            Maximum subspace size.
        psi : float, optional
            Step reduction factor.
        convergence_tol : float, optional
            Convergence threshold for sub-NM and cycle check.
        filler_scale : float, optional
            Gaussian noise scale for filler points.
        n_initial : int or None, optional
            Number of LHS warm-up points per method. Default ``None`` means
            ``2 * ndim``. Set to ``0`` to disable.
        restart_patience : int or "auto", optional
            Number of completed non-improving cycles before restarting. When
            ``"auto"`` it is chosen from the expected number of subspaces,
            with a minimum of 3.
        explore_prob : float, optional
            Probability of issuing a uniform random exploration point during
            normal cycling.
        method_exploration : float, optional
            Exploration strength for the LCB method chooser.
        method_temperature : float, optional
            Noise temperature for the LCB method chooser.
        exponential_param_power : float, optional
            Passed to the shared parameter mapping.
        seed : None or int, optional
            Random seed.
        """
        if kwargs:
            warnings.warn(
                f"Sbplex: ignoring unknown keyword arguments: {kwargs}"
            )
        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
            seed=seed,
        )
        self._optimizers = {
            method: HyperSbplexSampler(
                space[method],
                seed=seed,
                adaptive=adaptive,
                alpha=alpha,
                gamma=gamma,
                rho=rho,
                sigma=sigma,
                initial_scale=initial_scale,
                nsmin=nsmin,
                nsmax=nsmax,
                psi=psi,
                convergence_tol=convergence_tol,
                filler_scale=filler_scale,
                n_initial=n_initial,
                restart_patience=restart_patience,
                explore_prob=explore_prob,
                exponential_param_power=exponential_param_power,
            )
            for method in methods
        }

    def get_setting(self):
        """Choose a method, then request its next setting."""
        method = self._method_chooser.ask()
        params_token, params = self._optimizers[method].ask()

        return {
            "method": method,
            "params_token": params_token,
            "params": params,
        }

    def report_result(self, setting, trial, score):
        """Report a completed trial back to the method chooser and the
        per-method Sbplex sampler.
        """
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib("sbplex", SbplexOptLib, defaults={"adaptive": True})
register_hyper_optlib(
    "sbplex-noadapt", SbplexOptLib, defaults={"adaptive": False}
)
