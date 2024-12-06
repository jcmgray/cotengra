"""Interface for parallelism."""

import atexit
import collections
import functools
import importlib
import inspect
import numbers
import operator
import warnings

_AUTO_BACKEND = None

# check for loky, joblib (vendors loky), then default to concurrent.futures
have_loky = importlib.util.find_spec("loky") is not None
have_joblib = importlib.util.find_spec("joblib") is not None
if have_loky or have_joblib:
    _DEFAULT_BACKEND = "loky"
else:
    _DEFAULT_BACKEND = "concurrent.futures"


@functools.lru_cache(None)
def choose_default_num_workers():
    import os

    if "COTENGRA_NUM_WORKERS" in os.environ:
        return int(os.environ["COTENGRA_NUM_WORKERS"])

    if "OMP_NUM_THREADS" in os.environ:
        return int(os.environ["OMP_NUM_THREADS"])

    return os.cpu_count()


def get_pool(n_workers=None, maybe_create=False, backend=None):
    """Get a parallel pool."""
    if backend is None:
        backend = _DEFAULT_BACKEND

    if backend == "dask":
        return _get_pool_dask(n_workers=n_workers, maybe_create=maybe_create)

    if backend == "ray":
        return _get_pool_ray(n_workers=n_workers, maybe_create=maybe_create)

    # above backends are distributed, don't specify n_workers
    if n_workers is None:
        n_workers = choose_default_num_workers()

    if backend == "loky":
        get_reusable_executor = get_loky_get_reusable_executor()
        return get_reusable_executor(max_workers=n_workers)

    if backend == "concurrent.futures":
        return _get_process_pool_cf(n_workers=n_workers)

    if backend == "threads":
        return _get_thread_pool_cf(n_workers=n_workers)


@functools.lru_cache(None)
def _infer_backed_cached(pool_class):
    if pool_class.__name__ == "RayExecutor":
        return "ray"

    path = pool_class.__module__.split(".")

    if path[0] == "concurrent":
        return "concurrent.futures"

    if path[0] == "joblib":
        return "loky"

    if path[0] == "distributed":
        return "dask"

    return path[0]


def _infer_backend(pool):
    """Return the backend type of ``pool`` - cached for speed."""
    return _infer_backed_cached(pool.__class__)


def get_n_workers(pool=None):
    """Extract how many workers our pool has (mostly for working out how many
    tasks to pre-dispatch).
    """
    if pool is None:
        pool = get_pool()

    try:
        return pool._max_workers
    except AttributeError:
        pass

    backend = _infer_backend(pool)

    if backend == "dask":
        return len(pool.scheduler_info()["workers"])

    if backend == "ray":
        while True:
            try:
                return int(get_ray().available_resources()["CPU"])
            except KeyError:
                import time

                time.sleep(1e-3)

    if backend == "mpi4py":
        from mpi4py import MPI

        return MPI.COMM_WORLD.size

    raise ValueError(f"Can't find number of workers in pool {pool}.")


def parse_parallel_arg(parallel):
    """ """
    global _AUTO_BACKEND

    if parallel == "auto":
        return get_pool(maybe_create=False, backend=_AUTO_BACKEND)

    if parallel is False:
        return None

    if parallel is True:
        if _AUTO_BACKEND is None:
            _AUTO_BACKEND = _DEFAULT_BACKEND
        parallel = _AUTO_BACKEND

    if isinstance(parallel, numbers.Integral):
        _AUTO_BACKEND = _DEFAULT_BACKEND
        return get_pool(
            n_workers=parallel, maybe_create=True, backend=_DEFAULT_BACKEND
        )

    if parallel == "loky":
        return get_pool(maybe_create=True, backend="loky")

    if parallel == "concurrent.futures":
        return get_pool(maybe_create=True, backend="concurrent.futures")

    if parallel == "threads":
        return get_pool(maybe_create=True, backend="threads")

    if parallel == "dask":
        _AUTO_BACKEND = "dask"
        return get_pool(maybe_create=True, backend="dask")

    if parallel == "ray":
        _AUTO_BACKEND = "ray"
        return get_pool(maybe_create=True, backend="ray")

    return parallel


def set_parallel_backend(backend):
    """Create a parallel pool of type ``backend`` which registers it as the
    default for ``'auto'`` parallel.
    """
    return parse_parallel_arg(backend)


def maybe_leave_pool(pool):
    """Logic required for nested parallelism in dask.distributed."""
    if _infer_backend(pool) == "dask":
        return _maybe_leave_pool_dask()


def maybe_rejoin_pool(is_worker, pool):
    """Logic required for nested parallelism in dask.distributed."""
    if is_worker and _infer_backend(pool) == "dask":
        _rejoin_pool_dask()


def submit(pool, fn, *args, **kwargs):
    """Interface for submitting ``fn(*args, **kwargs)`` to ``pool``."""
    if _infer_backend(pool) == "dask":
        kwargs.setdefault("pure", False)
    return pool.submit(fn, *args, **kwargs)


def scatter(pool, data):
    """Interface for maybe turning ``data`` into a remote object or reference."""
    if _infer_backend(pool) in ("dask", "ray"):
        return pool.scatter(data)
    return data


def can_scatter(pool):
    """Whether ``pool`` can make objects remote."""
    return _infer_backend(pool) in ("dask", "ray")


def should_nest(pool):
    """Given argument ``pool`` should we try nested parallelism."""
    if pool is None:
        return False
    backend = _infer_backend(pool)
    if backend in ("ray", "dask"):
        return backend
    return False


# ---------------------------------- loky ----------------------------------- #


@functools.lru_cache(1)
def get_loky_get_reusable_executor():
    try:
        from loky import get_reusable_executor
    except ImportError:
        from joblib.externals.loky import get_reusable_executor
    return get_reusable_executor


# --------------------------- concurrent.futures ---------------------------- #


class CachedProcessPoolExecutor:
    def __init__(self):
        self._pool = None
        self._n_workers = -1
        atexit.register(self.shutdown)

    def __call__(self, n_workers=None):
        if n_workers != self._n_workers:
            from concurrent.futures import ProcessPoolExecutor

            self.shutdown()
            self._pool = ProcessPoolExecutor(n_workers)
            self._n_workers = n_workers
        return self._pool

    def is_initialized(self):
        return self._pool is not None

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __del__(self):
        self.shutdown()


ProcessPoolHandler = CachedProcessPoolExecutor()


def _get_process_pool_cf(n_workers=None):
    return ProcessPoolHandler(n_workers)


class CachedThreadPoolExecutor:
    def __init__(self):
        self._pool = None
        self._n_workers = -1
        atexit.register(self.shutdown)

    def __call__(self, n_workers=None):
        if n_workers != self._n_workers:
            from concurrent.futures import ThreadPoolExecutor

            self.shutdown()
            self._pool = ThreadPoolExecutor(n_workers)
            self._n_workers = n_workers
        return self._pool

    def is_initialized(self):
        return self._pool is not None

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __del__(self):
        self.shutdown()


ThreadPoolHandler = CachedThreadPoolExecutor()


def _get_thread_pool_cf(n_workers=None):
    return ThreadPoolHandler(n_workers)


# ---------------------------------- DASK ----------------------------------- #


def _get_pool_dask(n_workers=None, maybe_create=False):
    """Maybe get an existing or create a new dask.distrbuted client.

    Parameters
    ----------
    n_workers : None or int, optional
        The number of workers to request if creating a new client.
    maybe_create : bool, optional
        Whether to create an new local cluster and client if no existing client
        is found.

    Returns
    -------
    None or dask.distributed.Client
    """
    try:
        from dask.distributed import get_client
    except ImportError:
        if not maybe_create:
            return None
        else:
            raise

    try:
        client = get_client()
    except ValueError:
        if not maybe_create:
            return None

        import shutil
        import tempfile

        from dask.distributed import Client, LocalCluster

        local_directory = tempfile.mkdtemp()
        lc = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            local_directory=local_directory,
            memory_limit=0,
        )
        client = Client(lc)

        warnings.warn(
            "Parallel specified but no existing global dask client found... "
            "created one (with {} workers).".format(get_n_workers(client))
        )

        @atexit.register
        def delete_local_dask_directory():
            shutil.rmtree(local_directory, ignore_errors=True)

    if n_workers is not None:
        current_n_workers = get_n_workers(client)
        if n_workers != current_n_workers:
            warnings.warn(
                "Found existing client (with {} workers which) doesn't match "
                "the requested {}... using it instead.".format(
                    current_n_workers, n_workers
                )
            )

    return client


def _maybe_leave_pool_dask():
    try:
        from dask.distributed import secede

        secede()  # for nested parallelism
        is_dask_worker = True
    except (ImportError, ValueError):
        is_dask_worker = False
    return is_dask_worker


def _rejoin_pool_dask():
    from dask.distributed import rejoin

    rejoin()


# ----------------------------------- RAY ----------------------------------- #


@functools.lru_cache(None)
def get_ray():
    """ """
    import ray

    return ray


class RayFuture:
    """Basic ``concurrent.futures`` like future wrapping a ray ``ObjectRef``."""

    __slots__ = ("_obj", "_cancelled")

    def __init__(self, obj):
        self._obj = obj
        self._cancelled = False

    def result(self, timeout=None):
        return get_ray().get(self._obj, timeout=timeout)

    def done(self):
        return self._cancelled or bool(
            get_ray().wait([self._obj], timeout=0)[0]
        )

    def cancel(self):
        get_ray().cancel(self._obj)
        self._cancelled = True


def _unpack_futures_tuple(x):
    return tuple(map(_unpack_futures, x))


def _unpack_futures_list(x):
    return list(map(_unpack_futures, x))


def _unpack_futures_dict(x):
    return {k: _unpack_futures(v) for k, v in x.items()}


def _unpack_futures_identity(x):
    return x


_unpack_dispatch = collections.defaultdict(
    lambda: _unpack_futures_identity,
    {
        RayFuture: operator.attrgetter("_obj"),
        tuple: _unpack_futures_tuple,
        list: _unpack_futures_list,
        dict: _unpack_futures_dict,
    },
)


def _unpack_futures(x):
    """Allows passing futures by reference - takes e.g. args and kwargs and
    replaces all ``RayFuture`` objects with their underyling ``ObjectRef``
    within all nested tuples, lists and dicts.

    [Subclassing ``ObjectRef`` might avoid needing this.]
    """
    return _unpack_dispatch[x.__class__](x)


@functools.lru_cache(2**14)
def get_remote_fn(fn, **remote_opts):
    """Cached retrieval of remote function."""
    ray = get_ray()
    if remote_opts:
        return ray.remote(**remote_opts)(fn)
    return ray.remote(fn)


@functools.lru_cache(2**14)
def get_fn_as_remote_object(fn):
    ray = get_ray()
    return ray.put(fn)


@functools.lru_cache(None)
def get_deploy(**remote_opts):
    """Alternative for 'non-function' callables - e.g. partial
    functions - pass the callable object too.
    """
    ray = get_ray()

    def deploy(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    if remote_opts:
        return ray.remote(**remote_opts)(deploy)
    return ray.remote(deploy)


class RayExecutor:
    """Basic ``concurrent.futures`` like interface using ``ray``."""

    def __init__(self, *args, default_remote_opts=None, **kwargs):
        ray = get_ray()
        if not ray.is_initialized():
            ray.init(*args, **kwargs)

        self.default_remote_opts = (
            {} if default_remote_opts is None else dict(default_remote_opts)
        )

    def _maybe_inject_remote_opts(self, remote_opts=None):
        """Return the default remote options, possibly overriding some with
        those supplied by a ``submit call``.
        """
        ropts = self.default_remote_opts
        if remote_opts is not None:
            ropts = {**ropts, **remote_opts}
        return ropts

    def submit(self, fn, *args, pure=False, remote_opts=None, **kwargs):
        """Remotely run ``fn(*args, **kwargs)``, returning a ``RayFuture``."""
        # want to pass futures by reference
        args = _unpack_futures_tuple(args)
        kwargs = _unpack_futures_dict(kwargs)

        ropts = self._maybe_inject_remote_opts(remote_opts)

        # this is the same test ray uses to accept functions
        if inspect.isfunction(fn):
            # can use the faster cached remote function
            obj = get_remote_fn(fn, **ropts).remote(*args, **kwargs)
        else:
            fn_obj = get_fn_as_remote_object(fn)
            obj = get_deploy(**ropts).remote(fn_obj, *args, **kwargs)

        return RayFuture(obj)

    def map(self, func, *iterables, remote_opts=None):
        """Remote map ``func`` over arguments ``iterables``."""
        ropts = self._maybe_inject_remote_opts(remote_opts)
        remote_fn = get_remote_fn(func, **ropts)
        objs = tuple(map(remote_fn.remote, *iterables))
        ray = get_ray()
        return map(ray.get, objs)

    def scatter(self, data):
        """Push ``data`` into the distributed store, returning an ``ObjectRef``
        that can be supplied to ``submit`` calls for example.
        """
        ray = get_ray()
        return ray.put(data)

    def shutdown(self):
        """Shutdown the parent ray cluster, this ``RayExecutor`` instance
        itself does not need any cleanup.
        """
        get_ray().shutdown()


_RAY_EXECUTOR = None


def _get_pool_ray(n_workers=None, maybe_create=False):
    """Maybe get an existing or create a new RayExecutor, thus initializing,
    ray.

    Parameters
    ----------
    n_workers : None or int, optional
        The number of workers to request if creating a new client.
    maybe_create : bool, optional
        Whether to create initialize ray and return a RayExecutor if not
        initialized already.

    Returns
    -------
    None or RayExecutor
    """
    try:
        import ray
    except ImportError:
        if not maybe_create:
            return None
        else:
            raise

    global _RAY_EXECUTOR

    if (_RAY_EXECUTOR is None) or (not ray.is_initialized()):
        if not maybe_create:
            return None
        _RAY_EXECUTOR = RayExecutor(num_cpus=n_workers)

    if n_workers is not None:
        current_n_workers = get_n_workers(_RAY_EXECUTOR)
        if n_workers != current_n_workers:
            warnings.warn(
                "Found initialized ray (with {} workers which) doesn't match "
                "the requested {}... sticking with old number.".format(
                    current_n_workers, n_workers
                )
            )

    return _RAY_EXECUTOR
