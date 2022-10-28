import functools
import itertools

import opt_einsum as oe
import autoray as ar

from .core import ContractionTree


def _variadic(*arrays, fn, backend=None):
    return fn(arrays, backend=backend)


def _variadic_consts(*arrays, constants, fn, backend=None):
    n = len(arrays) + len(constants)
    arrays = iter(arrays)

    inputs = tuple(
        constants[i] if i in constants else next(arrays)
        for i in range(n)
    )

    return fn(inputs, backend=backend)


def contract_expression(
    eq,
    *shapes,
    optimize='auto',
    constants=None,
    autojit=False,
    sort_contraction_indices=False,
    trace_out_tree=False,
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
    if constants is None:
        constants = {}
    else:
        constants = {i: shapes[i] for i in constants}

    # construct the internal opt_einsum data
    lhs, output = eq.split('->')
    inputs = lhs.split(',')

    if len(inputs) == 1:
        term, = inputs
        if term == output:

            def fn(arrays, backend=None):
                return ar.do('array', arrays[0], like=backend)

        elif len(term) == len(output):
            perm = tuple(map(output.find, inputs[0]))

            def fn(arrays, backend=None):
                return ar.do('transpose', arrays[0], perm, like=backend)

        else:

            def fn(arrays, backend=None):
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
                optimize = oe.paths.get_path_fn(optimize)
            path = optimize(inputs, output, size_dict)
            tree = None

        if tree is None:
            tree = ContractionTree.from_path(
                inputs, output, size_dict, path=path)

        if sort_contraction_indices:
            tree.sort_contraction_indices()

        if trace_out_tree:
            # extract 'pure autoray' function, so we don't need to keep tree
            variables = [
                ar.lazy.Variable(
                    shape=s.shape if i in constants else s,
                    backend='autoray.lazy'
                ) for i, s in enumerate(shapes)
            ]
            x = tree.contract(variables, backend='autoray.lazy')
            fn = x.get_function(variables)
        else:
            fn = tree.contract

    if constants:
        fn = functools.partial(_variadic_consts, constants=constants, fn=fn)
    else:
        fn = functools.partial(_variadic, fn=fn)

    if autojit:
        fn = autojit(fn)

    return fn
