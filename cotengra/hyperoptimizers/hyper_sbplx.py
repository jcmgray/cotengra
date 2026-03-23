"""Hyper optimization using the Sbplx method (Rowan, 1990)."""

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

_SBPLX_OMEGA = 0.1


class HyperSbplxSampler:
    """Sbplx optimizer in raw ``[-1, 1]`` parameter space.

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
    partition : str, optional
        Subspace partitioning strategy. ``"greedy"`` takes equal chunks of
        up to ``nsmax`` dimensions, shrinking only to avoid a remainder
        smaller than ``nsmin``. ``"goodness"`` uses Rowan's heuristic,
        favoring splits where average step magnitude drops sharply.
        Default ``"greedy"``.
    convergence_tol : float, optional
        Relative convergence threshold for the overall cycle check. It is also
        passed as the absolute simplex diameter fallback for the inner
        Nelder-Mead cores.
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
    inject_diameter_fraction : float, optional
        Passed to each sub-``_NMCore`` — controls the maximum allowed
        simplex diameter inflation when injecting an external vertex.
    exponential_param_power : float, optional
        Passed through to the shared parameter mapping.
    """

    def __init__(
        self,
        space,
        seed=None,
        adaptive=False,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
        initial_scale=0.6,
        nsmin=2,
        nsmax=5,
        partition="greedy",
        psi=0.25,
        convergence_tol=0.01,
        filler_scale=0.3,
        n_initial=None,
        restart_patience="auto",
        explore_prob=0.05,
        inject_diameter_fraction=1.5,
        inject_restart_fraction=0.5,
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
        self.inject_diameter_fraction = inject_diameter_fraction
        self.inject_restart_fraction = inject_restart_fraction

        self.nsmin = min(nsmin, self.ndim)
        self.nsmax = min(nsmax, self.ndim)
        self.partition = partition

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
        self._step_at_cycle_start = None
        self._best_score_at_cycle_start = float("inf")

    def _partition_dims(self):
        """Partition dimensions into subspaces.

        Dimensions are first sorted by ``abs(step[d])`` descending. The
        partition strategy is selected by ``self.partition``:

        - ``"greedy"``: equal chunks of up to ``nsmax``, shrinking only
          to avoid a remainder smaller than ``nsmin``.
        - ``"goodness"``: Rowan's heuristic, favoring splits where the
          average step magnitude drops sharply.
        """
        order = sorted(
            range(self.ndim),
            key=lambda d: abs(self._step[d]),
            reverse=True,
        )
        if self.partition == "goodness":
            magnitudes = [abs(self._step[d]) for d in order]
            self._subspaces = self._partition_goodness(order, magnitudes)
        else:
            self._subspaces = self._partition_greedy(order)

    def _partition_greedy(self, order):
        """Equal-chunk partitioning."""
        subspaces = []
        i = 0
        while i < len(order):
            remaining = len(order) - i
            if remaining <= self.nsmax:
                subspaces.append(order[i:])
                break
            chunk_size = self.nsmax
            leftover = remaining - chunk_size
            if 0 < leftover < self.nsmin:
                chunk_size = remaining - self.nsmin
            subspaces.append(order[i : i + chunk_size])
            i += chunk_size
        return subspaces

    def _partition_goodness(self, order, magnitudes):
        """Rowan's goodness heuristic partitioning."""
        subspaces = []
        start = 0
        while start < len(order):
            remaining = len(order) - start
            if remaining <= self.nsmax:
                subspaces.append(order[start:])
                break

            total_remaining = sum(magnitudes[start:])
            prefix_total = 0.0
            best_goodness = float("-inf")
            best_size = self.nsmin
            max_size = min(self.nsmax, remaining)

            for offset in range(max_size):
                prefix_total += magnitudes[start + offset]
                size = offset + 1
                leftover = remaining - size

                if size < self.nsmin:
                    continue
                if leftover and leftover < self.nsmin:
                    continue

                if leftover:
                    suffix_total = total_remaining - prefix_total
                    goodness = prefix_total / size - suffix_total / leftover
                else:
                    goodness = prefix_total / size

                if goodness > best_goodness:
                    best_goodness = goodness
                    best_size = size

            subspaces.append(order[start : start + best_size])
            start += best_size
        return subspaces

    def _clamp_scale_factor(self, factor):
        return min(max(factor, _SBPLX_OMEGA), 1.0 / _SBPLX_OMEGA)

    def _start_cycle(self):
        """Snapshot the current point and begin a new cycle."""
        self._x_at_cycle_start = list(self._x)
        self._step_at_cycle_start = list(self._step)
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
            psi=self.psi,
            inject_diameter_fraction=self.inject_diameter_fraction,
            inject_restart_fraction=self.inject_restart_fraction,
        )

    def _cycle_converged(self):
        if self._x_at_cycle_start is None or self._step is None:
            return False

        for d in range(self.ndim):
            scale = max(abs(self._x[d]), 1.0)
            rel_change = abs(self._x[d] - self._x_at_cycle_start[d]) / scale
            rel_step = abs(self._step[d]) * self.psi / scale
            if max(rel_change, rel_step) > self.convergence_tol:
                return False

        return True

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
            scale = self.filler_scale
            if self._sub_nm is not None and not self._sub_nm.converged:
                step_mag = max((abs(s) for s in self._step), default=0.0)
                scale = max(0.5 * step_mag, self.filler_scale)
            x = [_clip(self.rng.gauss(ci, scale)) for ci in center]
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
        self._step_at_cycle_start = None
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
                self._x[d] = best_sub[i]

        self._sub_idx += 1
        if self._sub_idx < len(self._subspaces):
            self._start_sub_nm()
        else:
            self._finish_cycle()

    def _update_steps_after_cycle(self):
        if self._x_at_cycle_start is None or self._step_at_cycle_start is None:
            return

        deltax = [
            self._x[d] - self._x_at_cycle_start[d] for d in range(self.ndim)
        ]

        if len(self._subspaces) > 1:
            stepnorm = sum(abs(step) for step in self._step_at_cycle_start)
            dxnorm = sum(abs(dx) for dx in deltax)
            if stepnorm > 0.0:
                scale = dxnorm / stepnorm
            else:
                scale = 1.0
            scale = self._clamp_scale_factor(scale)
        else:
            scale = self.psi

        for d in range(self.ndim):
            base_step = self._step_at_cycle_start[d]
            magnitude = abs(base_step) * scale

            if magnitude == 0.0:
                magnitude = self.initial_scale * scale
            if magnitude < self.convergence_tol:
                magnitude = self.convergence_tol

            if deltax[d] > 0.0:
                self._step[d] = magnitude
            elif deltax[d] < 0.0:
                self._step[d] = -magnitude
            elif base_step < 0.0:
                self._step[d] = magnitude
            else:
                self._step[d] = -magnitude

    def _finish_cycle(self):
        """Check overall convergence across all subspaces.

        Following the NLopt Sbplx logic, convergence is based on a relative
        per-dimension test using both the cycle displacement and the current
        step size. Repeated non-improving cycles still trigger restarts to
        preserve the asynchrcoonous wrapper behavior.
        """
        improved = self._best_score < self._best_score_at_cycle_start

        if improved:
            self._cycles_since_improvement = 0
        else:
            self._cycles_since_improvement += 1

        self._update_steps_after_cycle()

        converged = self._cycle_converged()
        patience_exhausted = (
            self.restart_patience is not None
            and self._cycles_since_improvement >= self.restart_patience
        )
        should_restart = converged or patience_exhausted

        if should_restart:
            if self._stagnant_restart_count % 2 == 0:
                restart_mode = "local"
            else:
                restart_mode = "global"
            self._restart(restart_mode)
            return

        self._reset_cycle_state()


class SbplxOptLib(HyperOptLib):
    """Hyper-optimization backend using the 'Sbplx' method adapted from the
    Subplex method (Rowan, 1990).
    """

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        adaptive=False,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
        initial_scale=0.6,
        nsmin=2,
        nsmax=5,
        partition="greedy",
        psi=0.25,
        convergence_tol=0.01,
        filler_scale=0.3,
        n_initial=None,
        restart_patience="auto",
        explore_prob=0.05,
        inject_diameter_fraction=1.5,
        inject_restart_fraction=0.5,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        """Initialize Sbplx optimizers for each method.

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
        partition : str, optional
            Subspace partitioning strategy. ``"greedy"`` or ``"goodness"``.
            Default ``"greedy"``.
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
        inject_diameter_fraction : float, optional
            Passed to each sub-``_NMCore`` — controls the maximum allowed
            simplex diameter inflation when injecting an external vertex.
        inject_restart_fraction : float, optional
            Passed to each sub-``_NMCore`` — controls th XXX
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
                f"Sbplx: ignoring unknown keyword arguments: {kwargs}"
            )
        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
            seed=seed,
        )
        self._optimizers = {
            method: HyperSbplxSampler(
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
                partition=partition,
                psi=psi,
                convergence_tol=convergence_tol,
                filler_scale=filler_scale,
                n_initial=n_initial,
                restart_patience=restart_patience,
                explore_prob=explore_prob,
                inject_diameter_fraction=inject_diameter_fraction,
                inject_restart_fraction=inject_restart_fraction,
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
        per-method Sbplx sampler.
        """
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib("sbplx", SbplxOptLib)
