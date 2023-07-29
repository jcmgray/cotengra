import itertools

import autoray as ar

from .oe import get_path_fn, find_output_str
from .core import ContractionTree


class Variadic:

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def __call__(self, *arrays, **kwargs):
        return self.fn(arrays, **self.kwargs, **kwargs)


def contract_expression(
    eq,
    *shapes,
    optimize='auto',
    constants=None,
    implementation=None,
    autojit=False,
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
        return _contract_expression_with_constants(
            eq,
            *shapes,
            optimize=optimize,
            constants=constants,
            autojit=autojit,
            sort_contraction_indices=sort_contraction_indices,
        )
    else:
        constants = ()

    # construct the individual terms and find explicit output
    try:
        lhs, output = eq.split('->')
    except ValueError:
        lhs = eq
        output = find_output_str(lhs)

    inputs = lhs.split(',')

    if len(inputs) == 1:
        term, = inputs
        if term == output:
            # no-op contraction

            def fn(*arrays, backend=None):
                if backend is None:
                    return arrays[0]
                return ar.do('array', arrays[0], like=backend)

        elif len(term) == len(output):
            # transpose contraction
            perm = tuple(map(inputs[0].find, output))

            def fn(*arrays, backend=None):
                return ar.do('transpose', arrays[0], perm, like=backend)

        else:
            # involves traces / reductions

            def fn(*arrays, backend=None):
                return ar.do('einsum', eq, arrays[0], like=backend)

    else:
        size_dict = {
            ix: int(d) for ix, d in zip(
                itertools.chain.from_iterable(inputs),
                itertools.chain.from_iterable(
                    s.shape if i in constants else s
                    for i, s in enumerate(shapes)
                )
            )
        }

        nterms = len(inputs)
        if nterms <= 2:
            # nothing to optimize in this case
            path = (tuple(range(nterms)),)
            tree = None
        elif isinstance(optimize, (tuple, list)):
            # an explicit path
            path = optimize
            tree = None
        elif hasattr(optimize, "search"):
            # an optimizer that can directly produce trees
            path = None
            tree = optimize.search(inputs, output, size_dict)
        elif isinstance(optimize, ContractionTree):
            # an explicit tree
            path = None
            tree = optimize
        else:
            # opt_einsum style - get the actual path generating function
            if isinstance(optimize, str):
                optimize = get_path_fn(optimize)
            path = optimize(inputs, output, size_dict)
            tree = None

        if tree is None:
            tree = ContractionTree.from_path(
                inputs, output, size_dict, path=path
            )

        if sort_contraction_indices:
            tree.sort_contraction_indices()

        if not tree.sliced_inds:
            fn = tree.get_contractor(
                autojit=autojit,
                implementation=implementation,
            )
        else:
            # can't extract pure sliced contraction function yet...
            fn = Variadic(
                tree.contract,
                autojit=autojit,
                implementation=implementation,
            )

    return fn


def _contract_expression_with_constants(
    eq,
    *shapes,
    optimize='auto',
    constants=None,
    implementation=None,
    autojit=False,
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
            # want to generate function as if it were writtien with autoray
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

