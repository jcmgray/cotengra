"""Hyper optimization using the Nelder-Mead simplex method."""

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

# state constants
_INIT = "init"
_REFLECT = "reflect"
_EXPAND = "expand"
_CONTRACT = "contract"
_SHRINK = "shrink"


def _clip(x, lo=-1.0, hi=1.0):
    """Clip scalar ``x`` to the interval ``[lo, hi]``."""
    return min(max(x, lo), hi)


def clamp(xs, lo=-1.0, hi=1.0):
    """Clip each element of the list ``xs`` to the interval ``[lo, hi]``."""
    return [_clip(xi, lo, hi) for xi in xs]


class _NMCore:
    """Minimal Nelder-Mead simplex state machine on raw vectors. Manages n+1
    vertices and iteratively improves via reflection, expansion, contraction,
    and shrink. When the simplex diameter drops below ``convergence_tol`` the
    ``converged`` flag is set and no further points are issued.

    Parameters
    ----------
    ndim : int
        Number of raw dimensions.
    center : list[float]
        Center point for the initial simplex.
    scales : list[float]
        Per-dimension scale for axis-aligned perturbations used to construct
        the initial simplex around ``center``.
    adaptive : bool
        Whether to use the adaptive NM coefficients recommended by Gao and Han
        (2010), which scale with problem dimension. If `True` then `alpha`,
        `gamma`, `rho`, and `sigma` are ignored.
    alpha : float
        Reflection coefficient (standard NM default: 1). If ``adaptive`` is
        `True` then this is ignored and set to 1.0.
    gamma : float
        Expansion coefficient (standard NM default: 2). If ``adaptive`` is
        `True` then this is ignored and set to ``1 + 2 / ndim``.
    rho : float
        Contraction coefficient (standard NM default: 0.5). If ``adaptive`` is
        `True` then this is ignored and set to ``0.75 - 1 / (2 * ndim)``.
    sigma : float
        Shrink coefficient (standard NM default: 0.5). If ``adaptive`` is
        `True` then this is ignored and set to ``1 - 1 / ndim``.
    convergence_tol : float
        When the Chebyshev diameter of the simplex falls below this value the
        core signals convergence.
    psi : float, optional
        Relative simplex reduction target. When set, the core also converges
        once the simplex diameter has been reduced below ``psi`` times its
        initialized diameter. This mirrors the internal convergence mode used
        by Sbplex in NLopt.
    inject_diameter_fraction : float
        Maximum allowed simplex diameter inflation when injecting an external
        vertex. The candidate's max distance to any non-worst vertex must be
        at most ``inject_diameter_fraction * current_diameter``. Use ``1.0``
        (default) to prevent any inflation, ``> 1.0`` to allow some growth,
        or ``float('inf')`` to disable the gate entirely.
    inject_restart_fraction : float
        When injecting an external vertex that is far from the current simplex
        (beyond the diameter-based gate above), if its score is better than
        this fraction of the current best score, flag convergence to trigger an
        early restart focused on the better region. Set to ``0.0`` to disable
        this behavior.
    """

    def __init__(
        self,
        ndim,
        center,
        scales,
        adaptive=False,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
        convergence_tol=0.01,
        psi=None,
        inject_diameter_fraction=1.5,
        inject_restart_fraction=0.6,
    ):
        self.ndim = ndim

        if adaptive:
            self.alpha = 1.0
            self.gamma = 1.0 + 2.0 / ndim
            self.rho = 0.75 - 1.0 / (2.0 * ndim)
            self.sigma = 1.0 - 1.0 / ndim
        else:
            self.alpha = alpha
            self.gamma = gamma
            self.rho = rho
            self.sigma = sigma

        self.convergence_tol = convergence_tol
        self.psi = psi
        self.inject_diameter_fraction = inject_diameter_fraction
        self.inject_restart_fraction = inject_restart_fraction

        self._token_counter = 0
        self._tell_count = 0
        self._best_vertex = None
        self._best_score = float("inf")
        self._converged = False
        self._initial_simplex_diameter = None

        self._vertices = []
        self._scores = []
        # pending (token, x, role) tuples waiting to be issued
        self._ask_queue = []
        # token -> (x, role), kept until tell() consumes it
        self._token_map = {}
        # token -> (x, score, role), buffered until the state machine is ready
        self._results = {}

        self._state = _INIT
        self._centroid = None
        self._reflect_x = None
        self._reflect_score = None
        self._contract_inside = False
        self._pending_injection = None

        if self.ndim > 0:
            self._initialize_simplex(center, scales)

    @property
    def converged(self):
        """True when simplex diameter < convergence_tol."""
        return self._converged

    @property
    def best_vertex(self):
        return self._best_vertex

    @property
    def best_score(self):
        return self._best_score

    def _centroid_of(self, vertices):
        n = len(vertices)
        return [sum(v[d] for v in vertices) / n for d in range(self.ndim)]

    def _reflect(self, centroid, worst):
        return clamp(
            [
                centroid[d] + self.alpha * (centroid[d] - worst[d])
                for d in range(self.ndim)
            ]
        )

    def _expand(self, centroid, reflected):
        return clamp(
            [
                centroid[d] + self.gamma * (reflected[d] - centroid[d])
                for d in range(self.ndim)
            ]
        )

    def _contract_outside_pt(self, centroid, reflected):
        return clamp(
            [
                centroid[d] + self.rho * (reflected[d] - centroid[d])
                for d in range(self.ndim)
            ]
        )

    def _contract_inside_pt(self, centroid, worst):
        return clamp(
            [
                centroid[d] + self.rho * (worst[d] - centroid[d])
                for d in range(self.ndim)
            ]
        )

    def _shrink_vertex(self, best, vertex):
        return clamp(
            [
                best[d] + self.sigma * (vertex[d] - best[d])
                for d in range(self.ndim)
            ]
        )

    def _sort_simplex(self):
        paired = sorted(zip(self._scores, self._vertices), key=lambda t: t[0])
        self._scores = [s for s, _ in paired]
        self._vertices = [list(v) for _, v in paired]

    def _simplex_diameter(self):
        """Chebyshev (L-inf) diameter of the current simplex."""
        n = len(self._vertices)
        diam = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = max(
                    abs(self._vertices[i][k] - self._vertices[j][k])
                    for k in range(self.ndim)
                )
                if d > diam:
                    diam = d
        return diam

    def _diameter_converged(self):
        diameter = self._simplex_diameter()

        if diameter < self.convergence_tol:
            # absolute diameter convergence
            return True

        if self.psi is None or self._initial_simplex_diameter is None:
            # relative diameter convergence not enabled or not initialized
            return False

        # relative diameter convergence
        converged = diameter < self.psi * self._initial_simplex_diameter

        return converged

    def _initialize_simplex(self, center, scales):
        self._vertices = []
        self._scores = []
        self._ask_queue = []
        self._token_map = {}
        self._results = {}
        self._state = _INIT

        # n+1 vertices: center plus one axis-aligned perturbation per dimension
        self._enqueue(clamp(list(center)), "init")
        for i in range(self.ndim):
            v = list(center)
            v[i] += scales[i]
            self._enqueue(clamp(v), "init")

    def _enqueue(self, x, role):
        token = self._token_counter
        self._token_counter += 1
        x = list(x)
        self._ask_queue.append((token, x, role))
        self._token_map[token] = (x, role)

    def _try_advance(self):
        if self._state == _INIT:
            self._try_advance_init()
        elif self._state == _REFLECT:
            self._try_advance_reflect()
        elif self._state == _EXPAND:
            self._try_advance_expand()
        elif self._state == _CONTRACT:
            self._try_advance_contract()
        elif self._state == _SHRINK:
            self._try_advance_shrink()

    def _try_advance_init(self):
        # need all n+1 init results before proceeding
        init_results = {
            tn: r for tn, r in self._results.items() if r[2] == "init"
        }
        n_expected = self.ndim + 1
        if len(init_results) < n_expected:
            return

        self._vertices = []
        self._scores = []
        # consume results in token order to preserve the deterministic vertex
        # ordering (center first, then per-dimension perturbations)
        for tn in sorted(init_results):
            x, score, _ = init_results[tn]
            self._vertices.append(list(x))
            self._scores.append(score)
            del self._results[tn]

        self._sort_simplex()
        self._initial_simplex_diameter = max(
            self._simplex_diameter(), self.convergence_tol
        )
        self._begin_reflect()

    def inject_vertex(self, x, score):
        """Defer replacement of the worst vertex with an external point. The
        replacement is applied at the next ``_begin_reflect`` call so that
        in-progress NM operations are not disrupted. Only accepted when the
        simplex is initialized and ``score`` improves on the current worst
        vertex.

        Parameters
        ----------
        x : list[float]
            Raw coordinate vector (same dimensionality as the simplex).
        score : float
            Objective value at ``x``.

        Returns
        -------
        accepted : bool
            Whether the injection was accepted (deferred).
        """
        if self._converged:
            # in finished state
            return False

        if self._state == _INIT:
            # not ready
            return False

        if not self._scores or score >= self._scores[-1]:
            # not better than current worst vertex
            return False

        if self._pending_injection is not None:
            # not better than a previously injected vertex
            if score >= self._pending_injection[1]:
                return False

        # only inject if it won't inflate the simplex beyond the threshold
        threshold = self._simplex_diameter() * self.inject_diameter_fraction
        for v in self._vertices[:-1]:
            d = max(abs(x[k] - v[k]) for k in range(self.ndim))
            if d > threshold:
                # vertex is too far from simplex to accept
                if (
                    self._best_score is not None
                    and score < self.inject_restart_fraction * self._best_score
                    and self._tell_count >= self.ndim + 1
                ):
                    # BUT, early convergence signal: if pointdramatically
                    # better, force convergence to restart from better region
                    self._converged = True
                return False

        self._pending_injection = (list(x), score)
        return True

    def _begin_reflect(self):
        # apply deferred injection if present
        if self._pending_injection is not None:
            inj_x, inj_score = self._pending_injection
            self._pending_injection = None
            self._vertices[-1] = inj_x
            self._scores[-1] = inj_score
            self._sort_simplex()

        if self._diameter_converged():
            # convergence is checked at the start of each NM iteration - if the
            # simplex has collapsed, flag upstream rather than issuing
            self._converged = True
            return

        self._centroid = self._centroid_of(self._vertices[:-1])
        self._reflect_x = self._reflect(self._centroid, self._vertices[-1])
        self._state = _REFLECT
        self._enqueue(self._reflect_x, "reflect")

    def _try_advance_reflect(self):
        reflect_results = {
            tn: r for tn, r in self._results.items() if r[2] == "reflect"
        }
        if not reflect_results:
            return

        tn = next(iter(reflect_results))
        x, score, _ = reflect_results[tn]
        del self._results[tn]
        self._reflect_score = score

        best_score = self._scores[0]
        second_worst_score = self._scores[-2]
        worst_score = self._scores[-1]

        if best_score <= self._reflect_score < second_worst_score:
            # reflection is better than 2nd-worst but not best:
            #     accept and start a new iteration
            self._vertices[-1] = list(self._reflect_x)
            self._scores[-1] = self._reflect_score
            self._sort_simplex()
            self._begin_reflect()
        elif self._reflect_score < best_score:
            # new best: try expanding further
            expand_x = self._expand(self._centroid, self._reflect_x)
            self._state = _EXPAND
            self._enqueue(expand_x, "expand")
        elif self._reflect_score < worst_score:
            # worse than 2nd-worst but better than worst:
            #     try contracting on the reflection side
            contract_x = self._contract_outside_pt(
                self._centroid, self._reflect_x
            )
            self._contract_inside = False
            self._state = _CONTRACT
            self._enqueue(contract_x, "contract")
        else:
            # worse than or equal to worst:
            #     contract toward the worst vertex (inside contraction)
            contract_x = self._contract_inside_pt(
                self._centroid, self._vertices[-1]
            )
            self._contract_inside = True
            self._state = _CONTRACT
            self._enqueue(contract_x, "contract")

    def _try_advance_expand(self):
        expand_results = {
            tn: r for tn, r in self._results.items() if r[2] == "expand"
        }
        if not expand_results:
            return

        tn = next(iter(expand_results))
        x, score, _ = expand_results[tn]
        del self._results[tn]

        if score < self._reflect_score:
            # expansion improved on reflection: accept it
            self._vertices[-1] = list(x)
            self._scores[-1] = score
        else:
            # expansion didn't help: keep the reflection
            self._vertices[-1] = list(self._reflect_x)
            self._scores[-1] = self._reflect_score

        self._sort_simplex()
        self._begin_reflect()

    def _try_advance_contract(self):
        contract_results = {
            tn: r for tn, r in self._results.items() if r[2] == "contract"
        }
        if not contract_results:
            return

        tn = next(iter(contract_results))
        x, score, _ = contract_results[tn]
        del self._results[tn]

        # inside contraction must beat the worst vertex:
        #     outside contraction must beat the reflected point
        if self._contract_inside:
            threshold = self._scores[-1]
        else:
            threshold = self._reflect_score

        if score < threshold:
            self._vertices[-1] = list(x)
            self._scores[-1] = score
            self._sort_simplex()
            self._begin_reflect()
        else:
            # contraction failed:
            #     fall back to shrinking the entire simplex toward best vertex
            self._begin_shrink()

    def _begin_shrink(self):
        self._state = _SHRINK
        best = self._vertices[0]
        for i in range(1, len(self._vertices)):
            new_v = self._shrink_vertex(best, self._vertices[i])
            self._enqueue(new_v, "shrink")

    def _try_advance_shrink(self):
        shrink_results = {
            tn: r for tn, r in self._results.items() if r[2] == "shrink"
        }
        n_expected = len(self._vertices) - 1
        if len(shrink_results) < n_expected:
            return

        new_verts = []
        new_scores = []
        for tn in sorted(shrink_results):
            x, score, _ = shrink_results[tn]
            new_verts.append(list(x))
            new_scores.append(score)
            del self._results[tn]

        self._vertices[1:] = new_verts
        self._scores[1:] = new_scores
        self._sort_simplex()
        self._begin_reflect()

    def ask(self):
        """Return ``(token, raw_x)`` or ``None`` if blocked."""
        if self._converged:
            return None

        self._try_advance()

        if self._ask_queue:
            token, x, _ = self._ask_queue.pop(0)
            return token, x

        return None

    def tell(self, token, score):
        """Report the objective value for a previously issued
        token. Unknown tokens (e.g. from a replaced core) are
        silently ignored so that late-arriving results from a
        previous core instance do not cause errors.
        """
        self._tell_count += 1
        x, role = self._token_map.pop(token, (None, None))

        if x is not None and score < self._best_score:
            self._best_score = score
            self._best_vertex = list(x)

        # unknown token - nothing to feed the state machine
        if role is None:
            return

        self._results[token] = (list(x), score, role)
        self._try_advance()


class HyperNelderMeadSampler:
    """Nelder-Mead simplex optimizer in raw ``[-1, 1]`` space.

    Wraps a ``_NMCore`` state machine and adds parameter mapping, random filler
    point generation while the core is blocked, and automatic restarts when the
    simplex converges.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for a single contraction method.
    seed : None or int or random.Random, optional
        Random seed.
    adaptive : bool, optional
        Whether to use the adaptive NM coefficients recommended by Gao and Han
        (2010), which scale with problem dimension. If `True` then `alpha`,
        `gamma`, `rho`, and `sigma` are ignored.
    alpha : float, optional
        Reflection coefficient.
    gamma : float, optional
        Expansion coefficient.
    rho : float, optional
        Contraction coefficient.
    sigma : float, optional
        Shrink coefficient.
    initial_scale : float, optional
        Scale of the initial simplex around the origin.
    restart_tol : float, optional
        When the simplex diameter falls below this, restart.
    restart_scale : float, optional
        Scale of the restarted simplex around the best point.
    filler_scale : float, optional
        Standard deviation of the gaussian noise used for filler points issued
        while the core Nelder-Mead routine is blocked.
    n_initial : int or None, optional
        Number of Latin Hypercube Sampled (LHS) warm-up points to evaluate
        before starting the simplex. The best result seeds the initial simplex
        center. Default ``None`` means ``2 * ndim``. Set to ``0`` to disable.
    explore_prob : float, optional
        Probability of issuing a uniform random point instead of the
        NM-directed point during normal operation. Maintains diversity
        throughout the search.
    inject_diameter_fraction : float, optional
        Passed to ``_NMCore`` — controls the maximum allowed simplex
        diameter inflation when injecting an external vertex.
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
        restart_tol=0.01,
        restart_scale=0.5,
        filler_scale=0.3,
        n_initial=None,
        explore_prob=0.05,
        inject_diameter_fraction=1.5,
        inject_restart_fraction=0.6,
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
        self.restart_tol = restart_tol
        self.restart_scale = restart_scale
        self.filler_scale = filler_scale
        self.explore_prob = explore_prob
        self.inject_diameter_fraction = inject_diameter_fraction
        self.inject_restart_fraction = inject_restart_fraction

        if n_initial is None:
            n_initial = 2 * self.ndim
        self.n_initial = n_initial

        self._trial_counter = 0
        self._restart_count = 0
        self._best_x = None
        self._best_score = float("inf")
        # trial_number -> ("nm", core_token, raw_x) or
        #                 ("filler", None, raw_x) or
        #                 ("init", None, raw_x)
        self._trial_map = {}

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

        if self.ndim > 0 and not self._init_phase:
            center = [0.0] * self.ndim
            scales = [initial_scale] * self.ndim
            self._core = self._make_core(center, scales)
        else:
            self._core = None

    def _make_core(self, center, scales):
        return _NMCore(
            ndim=self.ndim,
            center=center,
            scales=scales,
            adaptive=self.adaptive,
            alpha=self.alpha,
            gamma=self.gamma,
            rho=self.rho,
            sigma=self.sigma,
            convergence_tol=self.restart_tol,
            inject_diameter_fraction=self.inject_diameter_fraction,
            inject_restart_fraction=self.inject_restart_fraction,
        )

    def _ask_filler(self):
        if self._core is not None and self._core.best_vertex is not None:
            center = self._core.best_vertex
        elif self._best_x is not None:
            center = self._best_x
        else:
            center = [0.0] * self.ndim

        if self.filler_scale == "uniform":
            # global sampling
            x = [self.rng.uniform(-1.0, 1.0) for _ in center]
        else:
            # locally sample around best point
            scale = self.filler_scale
            if self._core is not None and not self._core.converged:
                # if the core is active, scale with the current simplex
                # diameter so that fillers are more likely to be accepted
                diameter = self._core._simplex_diameter()
                scale = max(diameter, self.filler_scale)

            x = [_clip(self.rng.gauss(ci, scale)) for ci in center]

        trial_number = self._trial_counter
        self._trial_counter += 1
        self._trial_map[trial_number] = ("filler", None, x)
        return trial_number, x

    def ask(self):
        """Return the next candidate setting. During the LHS warm-up phase,
        pre-generated Latin Hypercube points are issued one at a time. Once all
        warm-up results have been collected, the NM simplex is initialized
        centered on the best warm-up point. During normal operation, with
        probability ``explore_prob`` a uniform random point is returned to
        maintain diversity. If the NM state machine is blocked waiting for
        results, a filler point is returned instead.
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
                self._trial_map[trial_number] = ("init", None, x)
                self._init_pending += 1
                return trial_number, convert_raw(self.params, x)

            # all init points issued but some results still pending
            trial_number, x = self._ask_filler()
            return trial_number, convert_raw(self.params, x)

        # random exploration
        if self.explore_prob > 0 and self.rng.random() < self.explore_prob:
            x = [self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim)]
            trial_number = self._trial_counter
            self._trial_counter += 1
            self._trial_map[trial_number] = ("filler", None, x)
            return trial_number, convert_raw(self.params, x)

        # normal nelder mead simplex operation
        result = self._core.ask()
        if result is not None:
            core_token, x = result
            trial_number = self._trial_counter
            self._trial_counter += 1
            self._trial_map[trial_number] = ("nm", core_token, x)
            return trial_number, convert_raw(self.params, x)

        # state machine is blocked, issue filler
        trial_number, x = self._ask_filler()
        return trial_number, convert_raw(self.params, x)

    def tell(self, trial_number, score):
        """Record a completed trial result."""
        if self.ndim == 0:
            if score < self._best_score:
                self._best_score = score
            return

        role, core_token, x = self._trial_map.pop(
            trial_number, ("filler", None, None)
        )

        # update global best (includes fillers and init points:
        #     a lucky random point is still a valid best-known solution)
        if score < self._best_score:
            self._best_score = score
            if x is not None:
                self._best_x = list(x)

        # handle LHS warm-up phase completion
        if role == "init":
            self._init_pending -= 1
            if self._init_pending <= 0 and not self._init_points:
                # all warm-up points evaluated — create the NM core
                # centered on the best warm-up result
                center = (
                    list(self._best_x)
                    if self._best_x is not None
                    else [0.0] * self.ndim
                )
                scales = [self.initial_scale] * self.ndim
                self._core = self._make_core(center, scales)
                self._init_phase = False
            return

        # attempt to inject high-quality filler/exploration results into
        # the active NM simplex so they influence the current run
        if (
            role != "nm"
            and x is not None
            and self._core is not None
            and not self._core.converged
        ):
            self._core.inject_vertex(list(x), score)

        if role == "nm" and core_token is not None:
            self._core.tell(core_token, score)

        # when the core converges, replace it with a fresh simplex centered on
        # the best-known point. Any in-flight trials from the old core will be
        # silently ignored by the new core's tell().
        if self._core is not None and self._core.converged:
            self._restart_count += 1
            if self._restart_count % 2 == 1:
                # odd restarts: local refine around best known point
                center = (
                    self._best_x
                    if self._best_x is not None
                    else [0.0] * self.ndim
                )
            else:
                # even restarts: global random point
                center = [
                    self.rng.uniform(-1.0, 1.0) for _ in range(self.ndim)
                ]
            scales = [self.restart_scale] * self.ndim
            self._core = self._make_core(center, scales)


class NelderMeadOptLib(HyperOptLib):
    """Hyper-optimization backend using the Nelder-Mead simplex method."""

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
        restart_tol=0.01,
        restart_scale=0.5,
        filler_scale=0.3,
        n_initial=None,
        explore_prob=0.05,
        inject_diameter_fraction=1.5,
        inject_restart_fraction=0.6,
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        seed=None,
        **kwargs,
    ):
        """Initialize Nelder-Mead optimizers for each method.

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
            Reflection coefficient.
        gamma : float, optional
            Expansion coefficient.
        rho : float, optional
            Contraction coefficient.
        sigma : float, optional
            Shrink coefficient.
        initial_scale : float, optional
            Scale of the initial simplex.
        restart_tol : float, optional
            Simplex diameter threshold for restart.
        restart_scale : float, optional
            Scale of the restarted simplex.
        filler_scale : float, optional
            Gaussian noise scale for filler points.
        n_initial : int or None, optional
            Number of LHS warm-up points per method. Default ``None``
            means ``2 * ndim``. Set to ``0`` to disable.
        explore_prob : float, optional
            Probability of issuing a uniform random exploration point
            during normal operation.
        inject_diameter_fraction : float, optional
            Passed to ``_NMCore`` — controls the maximum allowed simplex
            diameter inflation when injecting an external vertex.
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
                f"NelderMeadOptLib.setup: ignoring unknown "
                f"keyword arguments: {sorted(kwargs)}"
            )
        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
            seed=seed,
        )
        self._optimizers = {
            method: HyperNelderMeadSampler(
                space[method],
                seed=seed,
                adaptive=adaptive,
                alpha=alpha,
                gamma=gamma,
                rho=rho,
                sigma=sigma,
                initial_scale=initial_scale,
                restart_tol=restart_tol,
                restart_scale=restart_scale,
                filler_scale=filler_scale,
                n_initial=n_initial,
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
        """Report a completed trial back to the method chooser and the method
        specific Nelder-Mead sampler.
        """
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )


register_hyper_optlib(
    "neldermead", NelderMeadOptLib, defaults={"adaptive": False}
)
register_hyper_optlib(
    "neldermead-adapt", NelderMeadOptLib, defaults={"adaptive": True}
)
