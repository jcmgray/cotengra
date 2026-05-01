"""Tests for parallel pool management behavior."""

import concurrent.futures
import multiprocessing
import os

import pytest

import cotengra.parallel as par

FRESH_START_METHODS = tuple(
    method
    for method in multiprocessing.get_all_start_methods()
    if method != "fork"
)


def _reset_parallel_state():
    """Reset module-level parallel state for test isolation."""
    par.ProcessPoolHandler.shutdown()
    par.ThreadPoolHandler.shutdown()
    par._AUTO_BACKEND = None
    par._AUTO_BACKEND_PID = None
    par._IS_WORKER = False


def _check_worker_flag():
    """Helper that checks _IS_WORKER inside a worker process."""
    return par._IS_WORKER


def _worker_auto_returns_none():
    """Helper that checks nested parallel='auto' in a worker."""
    return par.parse_parallel_arg("auto") is None


def _subprocess_auto_returns_none(q):
    """Helper that runs in a child process."""
    q.put(par.parse_parallel_arg("auto") is None)


def test_auto_creates_pool():
    """parallel='auto' should create a pool by default."""
    _reset_parallel_state()
    try:
        pool = par.parse_parallel_arg("auto")
        assert pool is not None
    finally:
        _reset_parallel_state()


def test_default_backend_preference():
    """Default backend should be loky (if available) else concurrent.futures."""
    if par.have_loky or par.have_joblib:
        assert par._DEFAULT_BACKEND == "loky"
    else:
        assert par._DEFAULT_BACKEND == "concurrent.futures"


def test_explicit_process_backend_reuses_auto():
    """Explicit process backends should become the remembered auto backend."""
    _reset_parallel_state()
    try:
        pool1 = par.parse_parallel_arg("concurrent.futures")
        pool2 = par.parse_parallel_arg("auto")
        assert par._AUTO_BACKEND == "concurrent.futures"
        assert par._AUTO_BACKEND_PID == os.getpid()
        assert pool1 is pool2
    finally:
        _reset_parallel_state()


def test_explicit_loky_reuses_auto():
    """Explicit loky should become the remembered auto backend."""
    if not (par.have_loky or par.have_joblib):
        pytest.skip("loky not available")

    _reset_parallel_state()
    try:
        par.parse_parallel_arg("loky")
        pool = par.parse_parallel_arg("auto")
        assert par._AUTO_BACKEND == "loky"
        assert par._AUTO_BACKEND_PID == os.getpid()
        assert par._infer_backend(pool) == "loky"
    finally:
        _reset_parallel_state()


def test_threads_remain_explicit_only_for_auto():
    """Explicit threads should not become the remembered auto backend."""
    _reset_parallel_state()
    try:
        thread_pool = par.parse_parallel_arg("threads")
        auto_pool = par.parse_parallel_arg("auto")
        assert thread_pool is not None
        assert par._AUTO_BACKEND == par._DEFAULT_BACKEND
        assert par._AUTO_BACKEND != "threads"
        assert auto_pool is not None
    finally:
        _reset_parallel_state()


def test_threads_do_not_clobber_remembered_auto_backend():
    """Explicit threads should not change any remembered process backend."""
    _reset_parallel_state()
    try:
        process_pool = par.parse_parallel_arg("concurrent.futures")
        par.parse_parallel_arg("threads")
        auto_pool = par.parse_parallel_arg("auto")
        assert par._AUTO_BACKEND == "concurrent.futures"
        assert auto_pool is process_pool
    finally:
        _reset_parallel_state()


def test_worker_flag_prevents_auto_pool():
    """parallel='auto' returns None when _IS_WORKER is set."""
    _reset_parallel_state()
    try:
        par._IS_WORKER = True
        pool = par.parse_parallel_arg("auto")
        assert pool is None
    finally:
        _reset_parallel_state()


def test_submit_sets_worker_flag_for_process_pools():
    """Process-backed submit() calls should mark workers as workers."""
    _reset_parallel_state()
    try:
        pool = par.parse_parallel_arg("concurrent.futures")
        future = par.submit(pool, _check_worker_flag)
        assert future.result(timeout=10) is True
    finally:
        _reset_parallel_state()


def test_submit_does_not_mark_thread_workers():
    """Thread-backed submit() calls should not poison top-level auto state."""
    _reset_parallel_state()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = par.submit(pool, _check_worker_flag)
            assert future.result(timeout=10) is False

        assert par._IS_WORKER is False
        assert par.parse_parallel_arg("auto") is not None
    finally:
        _reset_parallel_state()


@pytest.mark.parametrize("start_method", FRESH_START_METHODS)
def test_spawn_like_workers_disable_auto(start_method):
    """Fresh worker processes should disable nested parallel='auto'."""
    _reset_parallel_state()
    try:
        ctx = multiprocessing.get_context(start_method)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=ctx
        ) as pool:
            future = par.submit(pool, _worker_auto_returns_none)
            assert future.result(timeout=20) is True
    finally:
        _reset_parallel_state()


def test_subprocess_no_auto_pool_fork():
    """parallel='auto' returns None in a forked subprocess (PID guard)."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork not available")

    ctx = multiprocessing.get_context("fork")

    _reset_parallel_state()
    try:
        par.parse_parallel_arg(True)

        q = ctx.Queue()
        p = ctx.Process(target=_subprocess_auto_returns_none, args=(q,))
        p.start()
        result = q.get(timeout=10)
        p.join()
        assert result is True
    finally:
        _reset_parallel_state()


def test_random_greedy_parallel_process_backend(monkeypatch):
    """RandomGreedyOptimizer should use the shared process submit path."""
    import cotengra as ctg

    monkeypatch.setenv("COTENGRA_NUM_WORKERS", "1")
    par.choose_default_num_workers.cache_clear()

    _reset_parallel_state()
    try:
        c = ctg.utils.lattice_equation([2, 2], d_min=2, d_max=2, seed=1)
        opt = ctg.RandomGreedyOptimizer(
            max_repeats=2,
            temperature=0.1,
            seed=1,
            accel=False,
            parallel="concurrent.futures",
        )
        assert opt(c.inputs, c.output, c.size_dict)
    finally:
        _reset_parallel_state()
        par.choose_default_num_workers.cache_clear()


def test_pool_persists_across_calls():
    """A created pool should be reused, not recreated each call."""
    _reset_parallel_state()
    try:
        pool1 = par.parse_parallel_arg("auto")
        pool2 = par.parse_parallel_arg("auto")
        assert pool1 is pool2
    finally:
        _reset_parallel_state()


def test_explicit_parallel_true_sets_pid():
    """parallel=True should record the PID for subprocess detection."""
    _reset_parallel_state()
    try:
        par.parse_parallel_arg(True)
        assert par._AUTO_BACKEND_PID == os.getpid()
        assert par._AUTO_BACKEND is not None
    finally:
        _reset_parallel_state()


def test_pid_mismatch_returns_none():
    """Simulated PID mismatch (as after fork) should return None."""
    _reset_parallel_state()
    try:
        par._AUTO_BACKEND = par._DEFAULT_BACKEND
        par._AUTO_BACKEND_PID = -1  # impossible PID
        pool = par.parse_parallel_arg("auto")
        assert pool is None
    finally:
        _reset_parallel_state()
