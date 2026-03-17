"""Hyper optimization using scipy gradient-free optimizers.

Supported methods: ``differential_evolution``, ``dual_annealing``,
``direct``, ``shgo``. Since these optimizers use a callback-style
objective, a background thread is used to invert the control flow into
an ask/tell interface. Multiple workers are maintained per method to
support parallel pre-dispatch (ask-ask-...-tell-tell-...).
"""

import queue
import threading

from ._param_mapping import (
    LCBOptimizer,
    build_params,
    convert_raw,
    num_params,
)
from .hyper import HyperOptLib, register_hyper_optlib

_OPTIMIZER_NAMES = {
    "differential_evolution",
    "dual_annealing",
    "direct",
    "shgo",
}


class _StopOptimization(Exception):
    """Raised inside the objective to abort the scipy optimizer."""


class ScipyAskTell:
    """Ask/tell wrapper around a scipy global optimizer.

    The optimizer runs in a background thread. Each time it needs an objective
    evaluation it posts the candidate vector to ``_ask_q`` and blocks on
    ``_tell_q``. The caller drives progress by alternating
    ``ask()`` / ``tell()`` calls from the main thread.

    Parameters
    ----------
    method : str
        One of the supported scipy optimizer names.
    bounds : list[tuple[float, float]]
        Bounds for every raw dimension.
    kwargs
        Forwarded to the underlying scipy optimizer.
    """

    def __init__(self, method, bounds, **kwargs):
        self.method = method
        self.bounds = bounds
        self.kwargs = kwargs

        self._ask_q = queue.Queue()
        self._tell_q = queue.Queue()
        self._thread = None
        self._stop = threading.Event()
        self.done = False

    # ---- internal --------------------------------------------------------

    def _get_optimizer_fn(self):
        from scipy import optimize

        return getattr(optimize, self.method)

    def _objective(self, x):
        if self._stop.is_set():
            raise _StopOptimization
        self._ask_q.put(x)
        val = self._tell_q.get()
        if self._stop.is_set():
            raise _StopOptimization
        return float(val)

    def _run(self):
        try:
            fn = self._get_optimizer_fn()
            fn(self._objective, self.bounds, **self.kwargs)
        except _StopOptimization:
            pass
        except Exception:
            pass
        finally:
            self.done = True
            # unblock any pending ask()
            self._ask_q.put(None)

    # ---- public ----------------------------------------------------------

    def start(self):
        """Launch the optimizer in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ask(self):
        """Block until the optimizer requests an evaluation.

        Returns
        -------
        x : ndarray or None
            The candidate vector, or ``None`` if the optimizer finished.
        """
        x = self._ask_q.get()
        if self.done and x is None:
            return None
        return x

    def tell(self, score):
        """Provide the objective value back to the optimizer."""
        self._tell_q.put(score)

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop.set()
        # unblock if the thread is waiting in _objective for a tell
        try:
            self._tell_q.put_nowait(float("inf"))
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class HyperScipySampler:
    """Per-method optimizer that wraps a pool of ``ScipyAskTell``
    workers with the ``Param``-based space mapping.

    Each ``ScipyAskTell`` worker is strictly serial
    (ask-tell-ask-tell), so to support parallel pre-dispatch
    (multiple ``ask`` calls before any ``tell``), we maintain a pool
    of workers. Each ``ask`` grabs a candidate from the next idle
    worker; each ``tell`` feeds the score back to the specific worker
    that produced that candidate.

    Parameters
    ----------
    space : dict[str, dict]
        The search space for one method.
    method : str
        Which scipy optimizer to use.
    n_workers : int, optional
        Number of concurrent ``ScipyAskTell`` threads to run.
    exponential_param_power : float, optional
        Power for ``ParamFloatExp``.
    kwargs
        Extra keyword arguments forwarded to the scipy optimizer.
    """

    def __init__(
        self,
        space,
        method="differential_evolution",
        n_workers=1,
        exponential_param_power=None,
        **kwargs,
    ):
        self.params = build_params(
            space, exponential_param_power=exponential_param_power
        )
        self._ndim = num_params(self.params)
        self._method = method
        self._scipy_opts = kwargs
        self._n_workers = n_workers

        # trial bookkeeping
        self._trial_counter = 0
        self._trial_to_worker = {}

        # worker pool: round-robin index for asks
        self._workers = []
        self._worker_idx = 0
        for _ in range(self._n_workers):
            self._workers.append(self._make_worker())

    def _make_worker(self):
        """Create and start a fresh ``ScipyAskTell`` instance."""
        w = ScipyAskTell(
            method=self._method,
            bounds=[(-1.0, 1.0)] * self._ndim,
            **self._scipy_opts,
        )
        w.start()
        return w

    def _next_worker(self):
        """Pick the next worker in round-robin order, restarting any
        that have finished.
        """
        for _ in range(self._n_workers):
            w = self._workers[self._worker_idx]
            self._worker_idx = (self._worker_idx + 1) % self._n_workers
            if w.done:
                # optimizer converged / exhausted – replace it
                w.stop()
                w = self._make_worker()
                # the index was already advanced, so store at previous
                prev = (self._worker_idx - 1) % self._n_workers
                self._workers[prev] = w
            return w

        # should never get here
        return self._workers[0]

    def ask(self):
        """Return ``(trial_number, params_dict)``.

        Picks the next idle worker (round-robin), blocking until it
        has a candidate ready. If a worker has converged, it is
        automatically restarted.
        """
        w = self._next_worker()
        x = w.ask()

        if x is None:
            # worker finished between _next_worker check and ask
            prev = (self._worker_idx - 1) % self._n_workers
            w = self._make_worker()
            self._workers[prev] = w
            x = w.ask()

        trial_number = self._trial_counter
        self._trial_to_worker[trial_number] = w
        self._trial_counter += 1

        return trial_number, convert_raw(self.params, x)

    def tell(self, trial_number, score):
        """Report a score back to the specific worker that produced
        this trial.
        """
        w = self._trial_to_worker.pop(trial_number, None)
        if w is not None and not w.done:
            w.tell(score)

    def stop(self):
        """Stop all background threads."""
        for w in self._workers:
            w.stop()


class ScipyOptLib(HyperOptLib):
    """Hyper-optimization using scipy gradient-free optimizers with
    an LCB method selector.
    """

    def setup(
        self,
        methods,
        space,
        optimizer=None,
        method="differential_evolution",
        method_exploration=1.0,
        method_temperature=1.0,
        exponential_param_power=None,
        **scipy_opts,
    ):
        """Initialize per-method scipy optimizers.

        Parameters
        ----------
        methods : list[str]
            The contraction methods to optimize over.
        space : dict[str, dict[str, dict]]
            The search space.
        optimizer : HyperOptimizer, optional
            The parent optimizer instance.
        method : str, optional
            Which scipy global optimizer to use. One of
            ``'differential_evolution'``, ``'dual_annealing'``,
            ``'direct'``, ``'shgo'``.
        method_exploration : float, optional
            Exploration parameter for the LCB method selector.
        method_temperature : float, optional
            Temperature parameter for the LCB method selector.
        exponential_param_power : float, optional
            Power for ``ParamFloatExp``.
        scipy_opts
            Extra keyword arguments forwarded to the scipy optimizer.
        """
        if method not in _OPTIMIZER_NAMES:
            raise ValueError(
                f"method must be one of {sorted(_OPTIMIZER_NAMES)}, "
                f"got {method!r}"
            )

        n_workers = getattr(optimizer, "_num_workers", 1)
        # need at least as many workers per method as there are
        # parallel pre-dispatch slots, so that asks never block on a
        # worker waiting for its tell
        pre_dispatch = getattr(optimizer, "pre_dispatch", 1)
        n_workers = max(n_workers, pre_dispatch)

        self._method_chooser = LCBOptimizer(
            options=methods,
            exploration=method_exploration,
            temperature=method_temperature,
        )
        self._optimizers = {
            m: HyperScipySampler(
                space[m],
                method=method,
                n_workers=n_workers,
                exponential_param_power=exponential_param_power,
                **scipy_opts,
            )
            for m in methods
        }

    def get_setting(self):
        method = self._method_chooser.ask()
        trial_number, params = self._optimizers[method].ask()

        return {
            "method": method,
            "params_token": trial_number,
            "params": params,
        }

    def report_result(self, setting, trial, score):
        self._method_chooser.tell(setting["method"], score)
        self._optimizers[setting["method"]].tell(
            setting["params_token"], score
        )

    def cleanup(self):
        """Stop all background optimizer threads."""
        for sampler in self._optimizers.values():
            sampler.stop()


register_hyper_optlib("scipy", ScipyOptLib)
