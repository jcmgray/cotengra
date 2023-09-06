"""`opt_einsum` interface.
"""

try:
    from opt_einsum.paths import PathOptimizer, get_path_fn, register_path_fn
except ImportError:
    PathOptimizer = object

    def get_path_fn(*_, **__):
        raise NotImplementedError("opt_einsum not installed")

    def register_path_fn(*_, **__):
        import warnings

        warnings.warn("opt_einsum not installed")


DEFAULT_COMBO_FACTOR = 64

__all__ = (
    "PathOptimizer",
    "get_path_fn",
    "register_path_fn",
)
