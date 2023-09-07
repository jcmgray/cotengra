"""High-level interface functions to cotengra.
"""
import functools
import itertools

import autoray as ar

from .core import ContractionTree
from .utils import eq_to_inputs_output
from .oe import (
    get_path_fn,
    opt_einsum_installed,
    register_path_fn,
)

_PRESETS = {}


def register_preset(preset, optimizer, register_opt_einsum="auto"):
    """Register a preset optimizer.
    """
    _PRESETS[preset] = optimizer

    if register_opt_einsum == "auto":
        register_opt_einsum = opt_einsum_installed

    if register_opt_einsum:
        try:
            register_path_fn(preset, optimizer)
        except KeyError:
            pass



@functools.lru_cache(None)
def preset_to_optimizer(preset):
    try:
        return _PRESETS[preset]
    except KeyError:

        if not opt_einsum_installed:
            raise KeyError(
                f"preset {preset!r} not found and can't "
                "check opt_einsum presets as its not installed."
            )

        # allow opt_einsum presets to be explicitly specified
        preset = preset.replace("opt_einsum:", "")
        return get_path_fn(preset)


_find_path_handlers = {}


def find_path_explicit(inputs, output, size_dict, optimize):
    return optimize


def find_path_optimizer(inputs, output, size_dict, optimize, **kwargs):
    return optimize(inputs, output, size_dict, **kwargs)


def find_path_preset(inputs, output, size_dict, optimize, **kwargs):
    optimize = preset_to_optimizer(optimize)
    return find_path(inputs, output, size_dict, optimize, **kwargs)


def find_path_tree(inputs, output, size_dict, optimize, **kwargs):
    return optimize.get_path()


def find_path(inputs, output, size_dict, optimize="auto", **kwargs):
    """Directly find a contraction path for a given set of inputs and output.

    Parameters
    ----------
    inputs : sequence[sequence[str]]
        The inputs terms.
    output : sequence[str]
        The output term.
    size_dict : dict[str, int]
        The size of each index.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

        * A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
        * A ``PathOptimizer`` instance.
        * An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
        * An explicit ``ContractionTree`` instance.

    Returns
    -------
    path : tuple[tuple[int]]
        The contraction path.
    """
    cls = optimize.__class__
    try:
        fn = _find_path_handlers[cls]
    except KeyError:
        if isinstance(optimize, str):
            fn = _find_path_handlers[cls] = find_path_preset
        elif isinstance(optimize, ContractionTree):
            fn = _find_path_handlers[cls] = find_path_tree
        elif isinstance(optimize, (tuple, list)):
            fn = _find_path_handlers[cls] = find_path_explicit
        else:
            fn = _find_path_handlers[cls] = find_path_optimizer

    return fn(inputs, output, size_dict, optimize, **kwargs)


def find_tree_explicit(inputs, output, size_dict, optimize):
    return ContractionTree.from_path(inputs, output, size_dict, path=optimize)


def find_tree_optimizer_search(inputs, output, size_dict, optimize, **kwargs):
    return optimize.search(inputs, output, size_dict, **kwargs)


def find_tree_optimizer_basic(inputs, output, size_dict, optimize, **kwargs):
    path = optimize(inputs, output, size_dict, **kwargs)
    return ContractionTree.from_path(inputs, output, size_dict, path=path)


def find_tree_preset(inputs, output, size_dict, optimize, **kwargs):
    optimize = preset_to_optimizer(optimize)
    return find_tree(inputs, output, size_dict, optimize, **kwargs)


def find_tree_tree(inputs, output, size_dict, optimize, **kwargs):
    # already a tree
    return optimize


_find_tree_handlers = {}


def find_tree(inputs, output, size_dict, optimize="auto", **kwargs):
    """Find a contraction tree for the specific contraction, with fast dispatch
    of ``optimize``, which can be a preset, path, tree, cotengra optimizer or
    opt_einsum optimizer.

    Parameters
    ----------
    inputs : sequence[sequence[str]]
        The inputs terms.
    output : sequence[str]
        The output term.
    size_dict : dict[str, int]
        The size of each index.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

        * A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
        * A ``PathOptimizer`` instance.
        * An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
        * An explicit ``ContractionTree`` instance.

    Returns
    -------
    tree : ContractionTree
    """
    cls = optimize.__class__
    try:
        fn = _find_tree_handlers[cls]
    except KeyError:
        if isinstance(optimize, str):
            fn = _find_tree_handlers[cls] = find_tree_preset
        elif isinstance(optimize, ContractionTree):
            fn = _find_tree_handlers[cls] = find_tree_tree
        elif isinstance(optimize, (tuple, list)):
            fn = _find_tree_handlers[cls] = find_tree_explicit
        elif hasattr(optimize, "search"):
            fn = _find_tree_handlers[cls] = find_tree_optimizer_search
        else:
            fn = _find_tree_handlers[cls] = find_tree_optimizer_basic

    return fn(inputs, output, size_dict, optimize, **kwargs)


class Variadic:
    """Wrapper to make non-variadic (i.e. with signature ``f(arrays)``)
    function variadic (i.e. with signature ``f(*arrays)``).
    """

    __slots__ = ("fn", "kwargs")

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def __call__(self, *arrays, **kwargs):
        return self.fn(arrays, **self.kwargs, **kwargs)


class Via:
    """Wrapper that applies one function to the input arrays and another to
    the output array. For example, moving the tensors from CPU to GPU and back.
    """

    __slots__ = ("fn", "convert_in", "convert_out")

    def __init__(self, fn, convert_in, convert_out):
        self.fn = fn
        self.convert_in = convert_in
        self.convert_out = convert_out

    def __call__(self, *arrays, **kwargs):
        arrays = map(self.convert_in, arrays)
        out = self.fn(*arrays, **kwargs)
        return self.convert_out(out)


def contract_expression(
    eq,
    *shapes,
    optimize="auto",
    constants=None,
    implementation=None,
    prefer_einsum=False,
    autojit=False,
    via=None,
    sort_contraction_indices=False,
):
    """Get an callable 'expression' that will contract tensors with shapes
    ``shapes`` according to equation ``eq``. The ``optimize`` kwarg can be a
    path, optimizer or also a contraction tree. In the latter case sliced
    indices for example will be used if present. The same is true if
    ``optimize`` is an optimizer that can directly produce ``ContractionTree``
    instances (i.e. has a ``.search()`` method).

    Parameters
    ----------
    eq : str
        The equation to use for contraction, for example ``'ab,bc->ac'``.
    shapes : tuple of int
        The shapes of the tensors to contract.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. If a ``HyperOptimizer`` or
        ``ContractionTree`` instance is passed then te expression will make use
        of any sliced indices.
    constants : sequence of int, optional
        The indices of tensors to treat as constant, the final expression will
        take the remaining non-constant tensors as inputs.
    autojit : bool, optional
        Whether to use ``autoray.autojit`` to compile the expression.
    """
    if constants is not None:
        fn = _contract_expression_with_constants(
            eq,
            *shapes,
            optimize=optimize,
            constants=constants,
            autojit=autojit,
            sort_contraction_indices=sort_contraction_indices,
        )
        if via is not None:
            fn = Via(fn, *via)

        return fn
    else:
        constants = ()

    # construct the individual terms and find explicit output
    inputs, output = eq_to_inputs_output(eq)

    if len(inputs) == 1:
        (term,) = inputs
        if term == output:
            # no-op contraction

            def fn(*arrays, backend=None):
                if backend is None:
                    return arrays[0]
                return ar.do("array", arrays[0], like=backend)

        elif len(term) == len(output):
            # transpose contraction
            perm = tuple(map(inputs[0].find, output))

            def fn(*arrays, backend=None):
                return ar.do("transpose", arrays[0], perm, like=backend)

        else:
            # involves traces / reductions

            def fn(*arrays, backend=None):
                return ar.do("einsum", eq, arrays[0], like=backend)

    else:
        size_dict = {
            ix: int(d)
            for ix, d in zip(
                itertools.chain.from_iterable(inputs),
                itertools.chain.from_iterable(
                    s.shape if i in constants else s
                    for i, s in enumerate(shapes)
                ),
            )
        }

        nterms = len(inputs)
        if nterms <= 2:
            # nothing to optimize
            optimize = (tuple(range(nterms)),)

        tree = find_tree(inputs, output, size_dict, optimize)

        if sort_contraction_indices:
            tree.sort_contraction_indices()

        if not tree.sliced_inds:
            fn = tree.get_contractor(
                autojit=autojit,
                prefer_einsum=prefer_einsum,
                implementation=implementation,
            )
        else:
            # can't extract pure sliced contraction function yet...
            fn = Variadic(
                tree.contract,
                autojit=autojit,
                prefer_einsum=prefer_einsum,
                implementation=implementation,
            )

    if via is not None:
        fn = Via(fn, *via)

    return fn


def _contract_expression_with_constants(
    eq,
    *shapes,
    optimize="auto",
    constants=None,
    implementation=None,
    autojit=False,
    prefer_einsum=False,
    sort_contraction_indices=False,
):
    import autoray as ar

    constants = set(constants)

    variables = []
    variables_with_constants = []
    shapes_only = []
    for i, s in enumerate(shapes):
        if i in constants:
            variables_with_constants.append(s)
            shapes_only.append(ar.shape(s))
        else:
            # want to generate function as if it were written with autoray
            v = ar.lazy.Variable(s, backend="autoray.numpy")
            variables.append(v)
            variables_with_constants.append(v)
            shapes_only.append(s)

    # get the full expression, without constants
    full_expr = contract_expression(
        eq,
        *shapes_only,
        optimize=optimize,
        constants=None,
        implementation=implementation,
        # wait to jit until after constants are folded
        autojit=False,
        prefer_einsum=prefer_einsum,
        sort_contraction_indices=sort_contraction_indices,
    )

    # trace through, and then get function with constants folded
    lz_output = full_expr(*variables_with_constants)
    fn = lz_output.get_function(variables, fold_constants=True)

    # now we can jit
    if autojit:
        from autoray import autojit as _autojit

        fn = _autojit(fn)

    return fn
