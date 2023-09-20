"""High-level interface functions to cotengra.
"""
import functools

import autoray as ar

from .core import ContractionTree
from .utils import (
    canonicalize_inputs,
    eq_to_inputs_output,
    find_output_from_inputs,
    inputs_output_to_eq,
    shapes_inputs_to_size_dict,
)
from .oe import (
    get_path_fn,
    opt_einsum_installed,
    register_path_fn,
)

_PRESETS = {}


def register_preset(preset, optimizer, register_opt_einsum="auto"):
    """Register a preset optimizer."""
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


def find_path_explicit_path(inputs, output, size_dict, optimize):
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
    inputs : Sequence[Sequence[str]]
        The inputs terms.
    output : Sequence[str]
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
            fn = _find_path_handlers[cls] = find_path_explicit_path
        else:
            fn = _find_path_handlers[cls] = find_path_optimizer

    return fn(inputs, output, size_dict, optimize, **kwargs)


def _find_tree_explicit(inputs, output, size_dict, optimize):
    return ContractionTree.from_path(inputs, output, size_dict, path=optimize)


def _find_tree_optimizer_search(inputs, output, size_dict, optimize, **kwargs):
    return optimize.search(inputs, output, size_dict, **kwargs)


def _find_tree_optimizer_basic(inputs, output, size_dict, optimize, **kwargs):
    path = optimize(inputs, output, size_dict, **kwargs)
    return ContractionTree.from_path(inputs, output, size_dict, path=path)


def _find_tree_preset(inputs, output, size_dict, optimize, **kwargs):
    optimize = preset_to_optimizer(optimize)
    return find_tree(inputs, output, size_dict, optimize, **kwargs)


def _find_tree_tree(inputs, output, size_dict, optimize, **kwargs):
    # already a tree
    return optimize


_find_tree_handlers = {}


def find_tree(inputs, output, size_dict, optimize="auto", **kwargs):
    """Find a contraction tree for the specific contraction, with fast dispatch
    of ``optimize``, which can be a preset, path, tree, cotengra optimizer or
    opt_einsum optimizer.

    Parameters
    ----------
    inputs : Sequence[Sequence[str]]
        The inputs terms.
    output : Sequence[str]
        The output term.
    size_dict : dict[str, int]
        The size of each index.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

    Returns
    -------
    tree : ContractionTree
    """
    cls = optimize.__class__
    try:
        fn = _find_tree_handlers[cls]
    except KeyError:
        if isinstance(optimize, str):
            fn = _find_tree_handlers[cls] = _find_tree_preset
        elif isinstance(optimize, ContractionTree):
            fn = _find_tree_handlers[cls] = _find_tree_tree
        elif isinstance(optimize, (tuple, list)):
            fn = _find_tree_handlers[cls] = _find_tree_explicit
        elif hasattr(optimize, "search"):
            fn = _find_tree_handlers[cls] = _find_tree_optimizer_search
        else:
            fn = _find_tree_handlers[cls] = _find_tree_optimizer_basic

    return fn(inputs, output, size_dict, optimize, **kwargs)


def array_contract_tree(
    inputs,
    output=None,
    size_dict=None,
    shapes=None,
    optimize="auto",
    canonicalize=True,
    sort_contraction_indices=False,
):
    """Get the `ContractionTree` for the tensor contraction specified by
    ``inputs``, ``output`` and ``size_dict``, with optimization strategy given
    by ``optimize``. The tree can be used to inspect and also perform the
    contraction.

    Parameters
    ----------
    inputs : Sequence[Sequence[Hashable]]
        The inputs terms.
    output : Sequence[Hashable], optional
        The output term.
    size_dict : dict[Hashable, int], optional
        The size of each index, if given, ``shapes`` is ignored.
    shapes : Sequence[tuple[int]], optional
        The shape of each input array. Only needed if ``canonicalize=True``
        and ``size_dict`` is not provided.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

    canonicalize : bool, optional
        If ``True``, canonicalize the inputs and output so that the indices
        are relabelled ``'a', 'b', 'c', ...``, etc. in the order they appear.
    sort_contraction_indices : bool, optional
        If ``True``, call ``tree.sort_contraction_indices()``.

    Returns
    -------
    ContractionTree

    See Also
    --------
    array_contract, array_contract_expression, einsum_tree
    """
    if canonicalize:
        inputs, output, size_dict = canonicalize_inputs(
            inputs,
            output,
            shapes=shapes,
            size_dict=size_dict,
        )
    elif output is None:
        # didn't canonicalize or specify output
        output = find_output_from_inputs(inputs)

    if size_dict is None:
        if shapes is None:
            raise ValueError("Either `size_dict` or `shapes` must be given.")
        else:
            # didn't canonicalize and only shapes given
            size_dict = shapes_inputs_to_size_dict(shapes, inputs)

    nterms = len(inputs)

    if nterms == 1:
        # there is no path
        optimize = ()

    elif nterms <= 2:
        # only a single possible path
        optimize = ((0, 1),)

    tree = find_tree(inputs, output, size_dict, optimize)

    if sort_contraction_indices:
        tree.sort_contraction_indices()

    return tree


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


class WithBackend:
    """Wrapper to make any autoray written function take a ``backend`` kwarg,
    by simply using `autoray.backend_like`.
    """

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, backend=None, **kwargs):
        if backend is None:
            return self.fn(*args, **kwargs)
        with ar.backend_like(backend):
            return self.fn(*args, **kwargs)


def _array_contract_expression_with_constants(
    inputs,
    output,
    size_dict,
    constants,
    optimize="auto",
    implementation=None,
    prefer_einsum=False,
    autojit=False,
    via=None,
    sort_contraction_indices=False,
):
    # make a lazy variable for each non-constant input
    lazy_variables = []
    # and another list of all inputs, including constant arrays
    lazy_variables_and_constants = []

    for i in range(len(inputs)):
        if i in constants:
            constant = constants[i]
            lazy_variables_and_constants.append(constant)
        else:
            term = tuple(inputs[i])
            shape = tuple(size_dict[ix] for ix in term)
            # want to generate function as if it were written with autoray
            v = ar.lazy.Variable(shape, backend="autoray.numpy")
            lazy_variables.append(v)
            lazy_variables_and_constants.append(v)

    # get the full expression, without constants
    full_expr = array_contract_expression(
        inputs,
        output,
        size_dict,
        optimize=optimize,
        constants=None,
        implementation=implementation,
        # wait to jit until after constants are folded
        autojit=False,
        prefer_einsum=prefer_einsum,
        sort_contraction_indices=sort_contraction_indices,
    )

    # trace through, and then get function with constants folded
    lz_output = full_expr(*lazy_variables_and_constants)
    fn = lz_output.get_function(lazy_variables, fold_constants=True)

    # now we can jit
    if autojit:
        from autoray import autojit as _autojit

        fn = _autojit(fn)
    else:
        # allow for backend kwarg (which will set what autoray.numpy uses)
        fn = WithBackend(fn)

    if via is not None:
        fn = Via(fn, *via)

    return fn


def array_contract_expression(
    inputs,
    output=None,
    size_dict=None,
    shapes=None,
    optimize="auto",
    constants=None,
    implementation=None,
    prefer_einsum=False,
    autojit=False,
    via=None,
    sort_contraction_indices=False,
):
    """Get an callable 'expression' that will contract tensors with indices and
    shapes described by ``inputs`` and ``size_dict`` to ``output``. The
    ``optimize`` kwarg can be a path, optimizer or also a contraction tree. In
    the latter case sliced indices for example will be used if present. The
    same is true if ``optimize`` is an optimizer that can directly produce
    ``ContractionTree`` instances (i.e. has a ``.search()`` method).

    Parameters
    ----------
    inputs : Sequence[Sequence[Hashable]]
        The inputs terms.
    output : Sequence[Hashable]
        The output term.
    size_dict : dict[Hashable, int]
        The size of each index.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

        If the optimizer provides sliced indices they will be used.
    constants : dict[int, array_like], optional
        A mapping of constant input positions to constant arrays. If given, the
        final expression will take only the remaining non-constant tensors as
        inputs. Note this is a different format to the ``constants`` kwarg of
        :func:`einsum_expression` since it also provides the constant arrays.
    implementation : str or tuple[callable, callable], optional
        What library to use to actually perform the contractions. Options
        are:

        - None: let cotengra choose.
        - "autoray": dispatch with autoray, using the ``tensordot`` and
            ``einsum`` implementation of the backend.
        - "cotengra": use the ``tensordot`` and ``einsum`` implementation
            of cotengra, which is based on batch matrix multiplication. This
            is faster for some backends like numpy, and also enables
            libraries which don't yet provide ``tensordot`` and ``einsum`` to
            be used.
        - "cuquantum": use the cuquantum library to perform the whole
            contraction (not just individual contractions).
        - tuple[callable, callable]: manually supply the ``tensordot`` and
            ``einsum`` implementations to use.

    autojit : bool, optional
        If ``True``, use :func:`autoray.autojit` to compile the contraction
        function.
    via : tuple[callable, callable], optional
        If given, the first function will be applied to the input arrays and
        the second to the output array. For example, moving the tensors from
        CPU to GPU and back.
    sort_contraction_indices : bool, optional
        If ``True``, call ``tree.sort_contraction_indices()`` before
        constructing the contraction function.

    Returns
    -------
    expr : callable
        A callable, signature ``expr(*arrays)`` that will contract ``arrays``
        with shapes ``shapes``.

    See Also
    --------
    einsum_expression, array_contract, array_contract_tree
    """
    if output is None:
        # didn't canonicalize or specify output
        output = find_output_from_inputs(inputs)

    if size_dict is None:
        if shapes is None:
            raise ValueError("Either `size_dict` or `shapes` must be given.")
        else:
            # didn't canonicalize and only shapes given
            size_dict = shapes_inputs_to_size_dict(shapes, inputs)

    if constants is not None:
        # handle constants specially with autoray
        return _array_contract_expression_with_constants(
            inputs,
            output,
            size_dict,
            constants,
            optimize=optimize,
            implementation=implementation,
            prefer_einsum=prefer_einsum,
            autojit=autojit,
            sort_contraction_indices=sort_contraction_indices,
        )

    if len(inputs) == 1:
        # no need to construct a tree
        term = tuple(inputs[0])
        output = tuple(output)

        if term == output:
            # 'contraction' is a no-op

            def fn(*arrays, backend=None):
                if backend is None:
                    return arrays[0]
                return ar.do("array", arrays[0], like=backend)

        elif len(term) == len(output):
            # 'contraction' is just a transposition
            perm = tuple(map(term.index, output))

            def fn(*arrays, backend=None):
                return ar.do("transpose", arrays[0], perm, like=backend)

        else:
            # contraction involves traces / reductions
            eq = inputs_output_to_eq(inputs, output)

            def fn(*arrays, backend=None):
                return ar.do("einsum", eq, arrays[0], like=backend)

    else:
        # get the contraction tree
        tree = array_contract_tree(
            inputs,
            output,
            size_dict,
            optimize=optimize,
            sort_contraction_indices=sort_contraction_indices,
        )

        if not tree.sliced_inds:
            # can extract pure sliced contraction function, forget tree
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


_CONTRACT_EXPR_CACHE = {}


def array_contract(
    arrays,
    inputs,
    output=None,
    optimize="auto",
    cache_expression=True,
    backend=None,
    **kwargs,
):
    """Perform the tensor contraction specified by ``inputs``, ``output`` and
    ``size_dict``, using strategy given by ``optimize``. By default the path
    finding and expression building is cached, so that if the a matching
    contraction is performed multiple times the overhead is negated.

    Parameters
    ----------
    arrays : Sequence[array_like]
        The arrays to contract.
    inputs : Sequence[Sequence[Hashable]]
        The inputs terms.
    output : Sequence[Hashable]
        The output term.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

        If the optimizer provides sliced indices they will be used.
    cache_expression : bool, optional
        If ``True``, cache the expression used to contract the arrays. This
        negates the overhead of pathfinding and building the expression when
        a contraction is performed multiple times.
    backend : str, optional
        If given, the explicit backend to use for the contraction, by default
        the backend is dispatched automatically.
    kwargs
        Passed to :func:`~cotengra.interface.array_contract_expression`.

    Returns
    -------
    array_like

    See Also
    --------
    array_contract_expression, array_contract_tree, einsum
    """
    # canonicalize
    inputs, output, size_dict = canonicalize_inputs(
        inputs=inputs,
        output=output,
        shapes=tuple(map(ar.shape, arrays)),
    )

    if cache_expression and isinstance(optimize, str):
        key = hash((inputs, output, tuple(size_dict.items()), optimize))
        try:
            try:
                expr = _CONTRACT_EXPR_CACHE[key]
            except KeyError:
                # missing from cache
                expr = _CONTRACT_EXPR_CACHE[key] = array_contract_expression(
                    inputs, output, size_dict, optimize=optimize, **kwargs
                )
        except TypeError:
            # unhashable kwargs
            import warnings

            warnings.warn(
                "Contraction cache disabled as one of the "
                f"arguments is not hashable: {kwargs}."
            )

            expr = array_contract_expression(
                inputs, output, size_dict, optimize=optimize, **kwargs
            )
    else:
        expr = array_contract_expression(
            inputs, output, size_dict, optimize=optimize, **kwargs
        )

    return expr(*arrays, backend=backend)


def einsum_tree(
    eq,
    *shapes,
    optimize="auto",
    canonicalize=False,
    sort_contraction_indices=False,
):
    """Get the `ContractionTree` for the einsum equation ``eq`` and
    optimization strategy ``optimize``. The tree can be used to inspect and
    also perform the contraction.

    Parameters
    ----------
    eq : str
        The equation to use for contraction, for example ``'ab,bc->ac'``.
    shapes : Sequence[tuple[int]]
        The shape of each input array.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

    canonicalize : bool, optional
        If ``True``, canonicalize the inputs and output so that the indices
        are relabelled ``'a', 'b', 'c', ...``, etc. in the order they appear.
    sort_contraction_indices : bool, optional
        If ``True``, call ``tree.sort_contraction_indices()``.

    Returns
    -------
    ContractionTree

    See Also
    --------
    einsum, einsum_expression, array_contract_tree
    """
    # construct the individual terms and find explicit output
    inputs, output = eq_to_inputs_output(eq)

    if canonicalize:
        inputs, output, size_dict = canonicalize_inputs(
            inputs, output, shapes=shapes
        )
    else:
        size_dict = shapes_inputs_to_size_dict(shapes, inputs)

    return array_contract_tree(
        inputs,
        output,
        size_dict,
        optimize=optimize,
        sort_contraction_indices=sort_contraction_indices,
    )


def einsum_expression(
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
        The output will be automatically computed if not supplied, but Ellipses
        (`'...'`) are not supported.
    shapes : Sequence[tuple[int]]
        The shapes of the tensors to contract, or the constant tensor itself
        if marked as constant in ``constants``.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

        If the optimizer provides sliced indices they will be used.
    constants : Sequence of int, optional
        The indices of tensors to treat as constant, the final expression will
        take the remaining non-constant tensors as inputs. Note this is a
        different format to the ``constants`` kwarg of
        :func:`array_contract_expression` since the actual constant arrays are
        inserted into ``shapes``.
    implementation : str or tuple[callable, callable], optional
        What library to use to actually perform the contractions. Options
        are:

        - None: let cotengra choose.
        - "autoray": dispatch with autoray, using the ``tensordot`` and
            ``einsum`` implementation of the backend.
        - "cotengra": use the ``tensordot`` and ``einsum`` implementation
            of cotengra, which is based on batch matrix multiplication. This
            is faster for some backends like numpy, and also enables
            libraries which don't yet provide ``tensordot`` and ``einsum`` to
            be used.
        - "cuquantum": use the cuquantum library to perform the whole
            contraction (not just individual contractions).
        - tuple[callable, callable]: manually supply the ``tensordot`` and
            ``einsum`` implementations to use.

    autojit : bool, optional
        If ``True``, use :func:`autoray.autojit` to compile the contraction
        function.
    via : tuple[callable, callable], optional
        If given, the first function will be applied to the input arrays and
        the second to the output array. For example, moving the tensors from
        CPU to GPU and back.
    sort_contraction_indices : bool, optional
        If ``True``, call ``tree.sort_contraction_indices()`` before
        constructing the contraction function.

    Returns
    -------
    expr : callable
        A callable, signature ``expr(*arrays)`` that will contract ``arrays``
        with shapes matching ``shapes``.

    See Also
    --------
    einsum, einsum_tree, array_contract_expression
    """
    # construct the individual terms and find explicit output
    inputs, output = eq_to_inputs_output(eq)

    if constants is not None:
        # shapes includes the constant arrays
        size_dict = {}
        parsed_constants = {}
        for i, s in enumerate(shapes):
            if i in constants:
                # extract the array and turn into a shape
                parsed_constants[i] = s
                s = ar.shape(s)
            size_dict.update(zip(inputs[i], s))
    else:
        parsed_constants = None
        size_dict = shapes_inputs_to_size_dict(shapes, inputs)

    return array_contract_expression(
        inputs,
        output,
        size_dict,
        optimize=optimize,
        constants=parsed_constants,
        implementation=implementation,
        prefer_einsum=prefer_einsum,
        autojit=autojit,
        via=via,
        sort_contraction_indices=sort_contraction_indices,
    )


@functools.lru_cache(None)
def _cached_eq_to_inputs_output(eq):
    return eq_to_inputs_output(eq)


def einsum(
    eq,
    *arrays,
    optimize="auto",
    cache_expression=True,
    backend=None,
    **kwargs,
):
    """Perform an einsum contraction, using `cotengra`, using strategy given by
    ``optimize``. By default the path finding and expression building is
    cached, so that if a matching contraction is performed multiple times the
    overhead is negated.

    Parameters
    ----------
    eq : str
        The equation to use for contraction, for example ``'ab,bc->ac'``.
    arrays : Sequence[array_like]
        The arrays to contract.
    optimize : str, path_like, PathOptimizer, or ContractionTree
        The optimization strategy to use. This can be:

            - A string preset, e.g. ``'auto'``, ``'greedy'``, ``'optimal'``.
            - A ``PathOptimizer`` instance.
            - An explicit path, e.g. ``[(0, 1), (2, 3), ...]``.
            - An explicit ``ContractionTree`` instance.

        If the optimizer provides sliced indices they will be used.
    cache_expression : bool, optional
        If ``True``, cache the expression used to contract the arrays. This
        negates the overhead of pathfinding and building the expression when
        a contraction is performed multiple times.
    backend : str, optional
        If given, the explicit backend to use for the contraction, by default
        the backend is dispatched automatically.
    kwargs
        Passed to :func:`~cotengra.interface.array_contract_expression`.

    Returns
    -------
    array_like

    See Also
    --------
    einsum_expression, einsum_tree, array_contract
    """
    inputs, output = _cached_eq_to_inputs_output(eq)
    return array_contract(
        arrays,
        inputs,
        output,
        optimize=optimize,
        cache_expression=cache_expression,
        backend=backend,
        **kwargs,
    )
