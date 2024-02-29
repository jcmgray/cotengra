"""`opt_einsum` interface."""

try:
    from opt_einsum.paths import PathOptimizer, get_path_fn, register_path_fn

    opt_einsum_installed = True

except ImportError:

    class PathOptimizer:
        pass

    def get_path_fn(*_, **__):
        raise NotImplementedError("opt_einsum not installed")

    def register_path_fn(*_, **__):
        import warnings

        warnings.warn("opt_einsum not installed")

    opt_einsum_installed = False

__all__ = (
    "get_path_fn",
    "opt_einsum_installed",
    "PathOptimizer",
    "register_path_fn",
)
