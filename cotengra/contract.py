"""Functionality relating to actually contracting."""

import functools
import itertools
import operator
import contextlib

from autoray import do, shape, infer_backend_multi, get_lib_fn

from .utils import node_from_single


DEFAULT_IMPLEMENTATION = "auto"


def set_default_implementation(impl):
    global DEFAULT_IMPLEMENTATION
    DEFAULT_IMPLEMENTATION = impl


def get_default_implementation():
    return DEFAULT_IMPLEMENTATION


@contextlib.contextmanager
def default_implementation(impl):
    """Context manager for temporarily setting the default implementation."""
    global DEFAULT_IMPLEMENTATION
    old_impl = DEFAULT_IMPLEMENTATION
    DEFAULT_IMPLEMENTATION = impl
    try:
        yield
    finally:
        DEFAULT_IMPLEMENTATION = old_impl


@functools.lru_cache(2**12)
def _sanitize_equation(eq):
    """Get the input and output indices of an equation, computing the output
    implicitly as the sorted sequence of every index that appears exactly once
    if it is not  provided.
    """
    # remove spaces
    eq = eq.replace(" ", "")

    if "..." in eq:
        raise NotImplementedError("Ellipsis not supported.")

    if "->" not in eq:
        lhs = eq
        tmp_subscripts = lhs.replace(",", "")
        out = "".join(
            # sorted sequence of indices
            s
            for s in sorted(set(tmp_subscripts))
            # that appear exactly once
            if tmp_subscripts.count(s) == 1
        )
    else:
        lhs, out = eq.split("->")
    return lhs, out


@functools.lru_cache(2**12)
def _parse_einsum_single(eq, shape):
    """Cached parsing of a single term einsum equation into the necessary
    sequence of arguments for axes diagonals, sums, and transposes.
    """
    lhs, out = _sanitize_equation(eq)

    # parse each index
    need_to_diag = []
    need_to_sum = []
    seen = set()
    for ix in lhs:
        if ix in need_to_diag:
            continue
        if ix in seen:
            need_to_diag.append(ix)
            continue
        seen.add(ix)
        if ix not in out:
            need_to_sum.append(ix)

    # first handle diagonal reductions
    if need_to_diag:
        diag_sels = []
        sizes = dict(zip(lhs, shape))
        while need_to_diag:
            ixd = need_to_diag.pop()
            dinds = tuple(range(sizes[ixd]))

            # construct advanced indexing object
            selector = tuple(dinds if ix == ixd else slice(None) for ix in lhs)
            diag_sels.append(selector)

            # after taking the diagonal what are new indices?
            ixd_contig = ixd * lhs.count(ixd)
            if ixd_contig in lhs:
                # contig axes, new axis is at same position
                lhs = lhs.replace(ixd_contig, ixd)
            else:
                # non-contig, new axis is at beginning
                lhs = ixd + lhs.replace(ixd, "")
    else:
        diag_sels = None

    # then sum reductions
    if need_to_sum:
        sum_axes = tuple(map(lhs.index, need_to_sum))
        for ix in need_to_sum:
            lhs = lhs.replace(ix, "")
    else:
        sum_axes = None

    # then transposition
    if lhs == out:
        perm = None
    else:
        perm = tuple(lhs.index(ix) for ix in out)

    return diag_sels, sum_axes, perm


def _parse_eq_to_pure_multiplication(a_term, shape_a, b_term, shape_b, out):
    """If there are no contracted indices, then we can directly transpose and
    insert singleton dimensions into ``a`` and ``b`` such that (broadcast)
    elementwise multiplication performs the einsum.

    No need to cache this as it is within the cached
    ``_parse_eq_to_batch_matmul``.

    """
    desired_a = ""
    desired_b = ""
    new_shape_a = []
    new_shape_b = []
    for ix in out:
        if ix in a_term:
            desired_a += ix
            new_shape_a.append(shape_a[a_term.index(ix)])
        else:
            new_shape_a.append(1)
        if ix in b_term:
            desired_b += ix
            new_shape_b.append(shape_b[b_term.index(ix)])
        else:
            new_shape_b.append(1)

    if desired_a != a_term:
        eq_a = f"{a_term}->{desired_a}"
    else:
        eq_a = None
    if desired_b != b_term:
        eq_b = f"{b_term}->{desired_b}"
    else:
        eq_b = None

    return (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        None,  # new_shape_ab, not needed since not fusing
        None,  # perm_ab, not needed as we transpose a and b first
        True,  # pure_multiplication=True
    )


@functools.lru_cache(2**12)
def _parse_eq_to_batch_matmul(eq, shape_a, shape_b):
    """Cached parsing of a two term einsum equation into the necessary
    sequence of arguments for contracttion via batched matrix multiplication.
    The steps we need to specify are:

        1. Remove repeated and trivial indices from the left and right terms,
           and transpose them, done as a single einsum.
        2. Fuse the remaining indices so we have two 3D tensors.
        3. Perform the batched matrix multiplication.
        4. Unfuse the output to get the desired final index order.

    """
    lhs, out = eq.split("->")
    a_term, b_term = lhs.split(",")

    if len(a_term) != len(shape_a):
        raise ValueError(f"Term '{a_term}' does not match shape {shape_a}.")
    if len(b_term) != len(shape_b):
        raise ValueError(f"Term '{b_term}' does not match shape {shape_b}.")

    bat_inds = []  # appears on A, B, O
    con_inds = []  # appears on A, B, .
    a_keep = []  # appears on A, ., O
    b_keep = []  # appears on ., B, O
    sizes = {}
    singletons = set()

    # parse left term
    seen = set()
    for ix, d in zip(a_term, shape_a):
        if d == 1:
            # everything (including broadcasting) works nicely if simply ignore
            # such dimensions, but we do need to track if they appear in output
            # and thus should be reintroduced later
            singletons.add(ix)
            continue

        # set or check size
        if sizes.setdefault(ix, d) != d:
            raise ValueError(
                f"Index {ix} has mismatched sizes {sizes[ix]} and {d}."
            )

        if ix in seen:
            continue
        seen.add(ix)

        if ix in b_term:
            if ix in out:
                bat_inds.append(ix)
            else:
                con_inds.append(ix)
        elif ix in out:
            a_keep.append(ix)

    # parse right term
    seen.clear()
    for ix, d in zip(b_term, shape_b):
        if d == 1:
            singletons.add(ix)
            continue
        # broadcast indices don't appear as singletons in output
        singletons.discard(ix)

        # set or check size
        if sizes.setdefault(ix, d) != d:
            raise ValueError(
                f"Index {ix} has mismatched sizes {sizes[ix]} and {d}."
            )

        if ix in seen:
            continue
        seen.add(ix)

        if ix not in a_term:
            if ix in out:
                b_keep.append(ix)

    if not con_inds:
        # contraction is pure multiplication, prepare inputs differently
        return _parse_eq_to_pure_multiplication(
            a_term, shape_a, b_term, shape_b, out
        )

    # only need the size one indices that appear in the output
    singletons = [ix for ix in out if ix in singletons]

    # take diagonal, remove any trivial axes and transpose left
    desired_a = "".join((*bat_inds, *a_keep, *con_inds))
    if a_term != desired_a:
        if set(a_term) == set(desired_a):
            # only need to transpose, don't invoke einsum
            eq_a = tuple(a_term.index(ix) for ix in desired_a)
        else:
            eq_a = f"{a_term}->{desired_a}"
    else:
        eq_a = None

    # take diagonal, remove any trivial axes and transpose right
    desired_b = "".join((*bat_inds, *con_inds, *b_keep))
    if b_term != desired_b:
        if set(b_term) == set(desired_b):
            # only need to transpose, don't invoke einsum
            eq_b = tuple(b_term.index(ix) for ix in desired_b)
        else:
            eq_b = f"{b_term}->{desired_b}"
    else:
        eq_b = None

    # then we want to reshape
    if bat_inds:
        lgroups = (bat_inds, a_keep, con_inds)
        rgroups = (bat_inds, con_inds, b_keep)
        ogroups = (bat_inds, a_keep, b_keep)
    else:
        # avoid size 1 batch dimension if no batch indices
        lgroups = (a_keep, con_inds)
        rgroups = (con_inds, b_keep)
        ogroups = (a_keep, b_keep)

    if any(len(group) != 1 for group in lgroups):
        # need to fuse 'kept' and contracted indices
        # (though could allow batch indices to be broadcast)
        new_shape_a = tuple(
            functools.reduce(operator.mul, (sizes[ix] for ix in ix_group), 1)
            for ix_group in lgroups
        )
    else:
        new_shape_a = None

    if any(len(group) != 1 for group in rgroups):
        # need to fuse 'kept' and contracted indices
        # (though could allow batch indices to be broadcast)
        new_shape_b = tuple(
            functools.reduce(operator.mul, (sizes[ix] for ix in ix_group), 1)
            for ix_group in rgroups
        )
    else:
        new_shape_b = None

    if any(len(group) != 1 for group in ogroups) or singletons:
        new_shape_ab = (1,) * len(singletons) + tuple(
            sizes[ix] for ix_group in ogroups for ix in ix_group
        )
    else:
        new_shape_ab = None

    # then we want to permute the matmul produced output:
    out_produced = "".join((*singletons, *bat_inds, *a_keep, *b_keep))
    perm_ab = tuple(out_produced.index(ix) for ix in out)
    if perm_ab == tuple(range(len(perm_ab))):
        perm_ab = None

    return (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        False,  # pure_multiplication=False
    )


def _einsum_single(eq, x, backend=None):
    """Einsum on a single tensor, via three steps: diagonal selection
    (via advanced indexing), axes summations, transposition. The logic for each
    is cached based on the equation and array shape, and each step is only
    performed if necessary.
    """
    try:
        return do("einsum", eq, x, like=backend)
    except ImportError:
        pass

    diag_sels, sum_axes, perm = _parse_einsum_single(eq, shape(x))

    if diag_sels is not None:
        # diagonal reduction via advanced indexing
        # e.g ababbac->abc
        for selector in diag_sels:
            x = x[selector]

    if sum_axes is not None:
        # trivial removal of axes via summation
        # e.g. abc->c
        x = do("sum", x, sum_axes, like=backend)

    if perm is not None:
        # transpose to desired output
        # e.g. abc->cba
        x = do("transpose", x, perm, like=backend)

    return x


def _do_contraction_via_bmm(
    a,
    b,
    eq_a,
    eq_b,
    new_shape_a,
    new_shape_b,
    new_shape_ab,
    perm_ab,
    pure_multiplication,
    backend,
):
    # prepare left
    if eq_a is not None:
        if isinstance(eq_a, tuple):
            # only transpose
            a = do("transpose", a, eq_a, like=backend)
        else:
            # diagonals, sums, and tranpose
            a = _einsum_single(eq_a, a)
    if new_shape_a is not None:
        a = do("reshape", a, new_shape_a, like=backend)

    # prepare right
    if eq_b is not None:
        if isinstance(eq_b, tuple):
            # only transpose
            b = do("transpose", b, eq_b, like=backend)
        else:
            # diagonals, sums, and tranpose
            b = _einsum_single(eq_b, b)
    if new_shape_b is not None:
        b = do("reshape", b, new_shape_b, like=backend)

    if pure_multiplication:
        # no contracted indices
        return do("multiply", a, b)

    # do the contraction!
    ab = do("matmul", a, b, like=backend)

    # prepare the output
    if new_shape_ab is not None:
        ab = do("reshape", ab, new_shape_ab, like=backend)
    if perm_ab is not None:
        ab = do("transpose", ab, perm_ab, like=backend)

    return ab


def einsum(eq, a, b=None, *, backend=None):
    """Perform arbitrary single and pairwise einsums using only `matmul`,
    `transpose`, `reshape` and `sum`.  The logic for each is cached based on
    the equation and array shape, and each step is only performed if necessary.

    Parameters
    ----------
    eq : str
        The einsum equation.
    a : array_like
        The first array to contract.
    b : array_like, optional
        The second array to contract.
    backend : str, optional
        The backend to use for array operations. If ``None``, dispatch
        automatically based on ``a`` and ``b``.

    Returns
    -------
    array_like
    """
    if b is None:
        return _einsum_single(eq, a, backend=backend)

    (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
    ) = _parse_eq_to_batch_matmul(eq, shape(a), shape(b))

    return _do_contraction_via_bmm(
        a,
        b,
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
        backend,
    )


def gen_nice_inds():
    """Generate the indices from [a-z, A-Z, reasonable unicode...]."""
    for i in range(26):
        yield chr(ord("a") + i)
    for i in range(26):
        yield chr(ord("A") + i)
    for i in itertools.count(192):
        yield chr(i)


@functools.lru_cache(2**12)
def _parse_tensordot_axes_to_matmul(axes, shape_a, shape_b):
    """Parse a tensordot specification into the necessary sequence of arguments
    for contracttion via matrix multiplication. This just converts ``axes``
    into an ``einsum`` eq string then calls ``_parse_eq_to_batch_matmul``.
    """
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)

    if isinstance(axes, int):
        axes_a = tuple(range(ndim_a - axes, ndim_a))
        axes_b = tuple(range(axes))
    else:
        axes_a, axes_b = axes

    num_con = len(axes_a)
    if num_con != len(axes_b):
        raise ValueError(
            f"Axes should have the same length, got {axes_a} and {axes_b}."
        )

    possible_inds = gen_nice_inds()
    inds_a = [next(possible_inds) for _ in range(ndim_a)]
    inds_b = []
    inds_out = inds_a.copy()

    for axb in range(ndim_b):
        if axb not in axes_b:
            # right uncontracted index
            ind = next(possible_inds)
            inds_out.append(ind)
        else:
            # contracted index
            axa = axes_a[axes_b.index(axb)]
            # check that the shapes match
            if shape_a[axa] != shape_b[axb]:
                raise ValueError(
                    f"Dimension mismatch between axes {axa} of {shape_a} and "
                    f"{axb} of {shape_b}: {shape_a[axa]} != {shape_b[axb]}."
                )
            ind = inds_a[axa]
            inds_out.remove(ind)
        inds_b.append(ind)

    eq = f"{''.join(inds_a)},{''.join(inds_b)}->{''.join(inds_out)}"

    return _parse_eq_to_batch_matmul(eq, shape_a, shape_b)


def tensordot(a, b, axes=2, *, backend=None):
    """Perform a tensordot using only `matmul`, `transpose`, `reshape`. The
    logic for each is cached based on the equation and array shape, and each
    step is only performed if necessary.

    Parameters
    ----------
    a, b : array_like
        The arrays to contract.
    axes : int or tuple of (sequence[int], sequence[int])
        The number of axes to contract, or the axes to contract. If an int,
        the last ``axes`` axes of ``a`` and the first ``axes`` axes of ``b``
        are contracted. If a tuple, the axes to contract for ``a`` and ``b``
        respectively.
    backend : str or None, optional
        The backend to use for array operations. If ``None``, dispatch
        automatically based on ``a`` and ``b``.

    Returns
    -------
    array_like
    """
    try:
        # ensure hashable
        axes = tuple(map(int, axes[0])), tuple(map(int, axes[1]))
    except IndexError:
        axes = int(axes)

    (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
    ) = _parse_tensordot_axes_to_matmul(axes, shape(a), shape(b))

    return _do_contraction_via_bmm(
        a,
        b,
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
        backend,
    )


def extract_contractions(
    tree,
    order=None,
    prefer_einsum=False,
):
    """Extract just the information needed to perform the contraction.

    Parameters
    ----------
    order : str or callable, optional
        Supplied to :meth:`ContractionTree.traverse`.
    prefer_einsum : bool, optional
        Prefer to use ``einsum`` for pairwise contractions, even if
        ``tensordot`` can perform the contraction.

    Returns
    -------
    contractions : tuple
        A tuple of tuples, each containing the information needed to
        perform a pairwise contraction. Each tuple contains:

            - ``p``: the parent node,
            - ``l``: the left child node,
            - ``r``: the right child node,
            - ``tdot``: whether to use ``tensordot`` or ``einsum``,
            - ``arg``: the argument to pass to ``tensordot`` or ``einsum``
                i.e. ``axes`` or ``eq``,
            - ``perm``: the permutation required after the contraction, if
                any (only applies to tensordot).

        If both ``l`` and ``r`` are ``None``, the the operation is a single
        term simplification performed with ``einsum``.
    """
    contractions = []

    # pairwise contractions
    contractions.extend(
        (p, l, r, False, tree.get_einsum_eq(p), None)
        if (prefer_einsum or not tree.get_can_dot(p))
        else (
            p,
            l,
            r,
            True,
            tree.get_tensordot_axes(p),
            tree.get_tensordot_perm(p),
        )
        for p, l, r in tree.traverse(order=order)
    )

    if tree.preprocessing:
        # inplace single term simplifications
        # n.b. these are populated lazily when the other information is
        # computed above, so we do it after
        pre_contractions = (
            (node_from_single(i), None, None, False, eq, None)
            for i, eq in tree.preprocessing.items()
        )
        return (*pre_contractions, *contractions)

    return tuple(contractions)


class Contractor:
    """Default cotengra network contractor.

    Parameters
    ----------
    contractions : tuple[tuple]
        The sequence of contractions to perform. Each contraction should be a
        tuple containing:

            - ``p``: the parent node,
            - ``l``: the left child node,
            - ``r``: the right child node,
            - ``tdot``: whether to use ``tensordot`` or ``einsum``,
            - ``arg``: the argument to pass to ``tensordot`` or ``einsum``
                i.e. ``axes`` or ``eq``,
            - ``perm``: the permutation required after the contraction, if
                any (only applies to tensordot).

        e.g. built by calling ``extract_contractions(tree)``.

    strip_exponent : bool, optional
        If ``True``, eagerly strip the exponent (in log10) from
        intermediate tensors to control numerical problems from leaving the
        range of the datatype. This method then returns the scaled
        'mantissa' output array and the exponent separately.
    check_zero : bool, optional
        If ``True``, when ``strip_exponent=True``, explicitly check for
        zero-valued intermediates that would otherwise produce ``nan``,
        instead terminating early if encounteredand returning
        ``(0.0, 0.0)``.
    backend : str, optional
        What library to use for ``tensordot``, ``einsum`` and
        ``transpose``, it will be automatically inferred from the input
        arrays if not given.
    progbar : bool, optional
        Whether to show a progress bar.
    """

    __slots__ = (
        "contractions",
        "strip_exponent",
        "check_zero",
        "implementation",
        "backend",
        "progbar",
        "__weakref__",
    )

    def __init__(
        self,
        contractions,
        strip_exponent=False,
        check_zero=False,
        implementation="auto",
        backend=None,
        progbar=False,
    ):
        self.contractions = contractions
        self.strip_exponent = strip_exponent
        self.check_zero = check_zero
        self.implementation = implementation
        self.backend = backend
        self.progbar = progbar

    def __call__(self, *arrays, **kwargs):
        """Contract ``arrays`` using operations listed in ``contractions``.

        Parameters
        ----------
        arrays : sequence of array-like
            The arrays to contract.
        kwargs : dict
            Override the default settings for this contraction only.

        Returns
        -------
        output : array
            The contracted output, it will be scaled if ``strip_exponent==True``.
        exponent : float
            The exponent of the output in base 10, returned only if
            ``strip_exponent==True``.
        """
        backend = kwargs.pop("backend", self.backend)
        progbar = kwargs.pop("progbar", self.progbar)
        check_zero = kwargs.pop("check_zero", self.check_zero)
        strip_exponent = kwargs.pop("strip_exponent", self.strip_exponent)
        implementation = kwargs.pop("implementation", self.implementation)
        if kwargs:
            raise TypeError(f"Unknown keyword arguments: {kwargs}.")

        if backend is None:
            backend = infer_backend_multi(*arrays)

        if implementation == "auto":
            if backend == "numpy":
                # by default only replace numpy's einsum/tensordot
                implementation = "cotengra"
            else:
                implementation = "autoray"

        if implementation == "cotengra":
            _einsum, _tensordot = einsum, tensordot
        elif implementation == "autoray":
            try:
                _einsum = get_lib_fn(backend, "einsum")
            except ImportError:
                # fallback to cotengra (matmul) implementation
                _einsum = einsum

            try:
                _tensordot = get_lib_fn(backend, "tensordot")
            except ImportError:
                # fallback to cotengra (matmul) implementation
                _tensordot = tensordot
        else:
            # manually supplied
            _einsum, _tensordot = implementation

        # temporary storage for intermediates
        N = len(arrays)
        temps = {
            leaf: array
            for leaf, array in zip(map(node_from_single, range(N)), arrays)
        }

        exponent = 0.0 if (strip_exponent is not False) else None

        if progbar:
            import tqdm

            contractions = tqdm.tqdm(self.contractions, total=N - 1)
        else:
            contractions = self.contractions

        for p, l, r, tdot, arg, perm in contractions:
            if (l is None) and (r is None):
                # single term simplification, perform inplace with einsum
                temps[p] = _einsum(arg, temps[p])
                continue

            # get input arrays for this contraction
            l_array = temps.pop(l)
            r_array = temps.pop(r)

            if tdot:
                p_array = _tensordot(l_array, r_array, arg)
                if perm:
                    p_array = do("transpose", p_array, perm, like=backend)
            else:
                p_array = _einsum(arg, l_array, r_array)

            if exponent is not None:
                factor = do(
                    "max", do("abs", p_array, like=backend), like=backend
                )
                if check_zero and float(factor) == 0.0:
                    return 0.0, float("-inf")
                exponent = exponent + do("log10", factor, like=backend)
                p_array = p_array / factor

            # insert the new intermediate array
            temps[p] = p_array

        if exponent is not None:
            return p_array, exponent

        return p_array


class CuQuantumContractor:
    def __init__(
        self,
        tree,
        handle_slicing=False,
        autotune=False,
        **kwargs,
    ):
        if kwargs.pop("strip_exponent", None):
            raise ValueError(
                "strip_exponent=True not supported with cuQuantum"
            )

        if tree.has_preprocessing():
            raise ValueError("Preprocessing not supported with cuQuantum yet.")

        if kwargs.pop("progbar", None):
            import warnings

            warnings.warn("Progress bar not supported with cuQuantum yet.")

        if handle_slicing:
            self.eq = tree.get_eq()
            self.shapes = tree.get_shapes()
        else:
            self.eq = tree.get_eq_sliced()
            self.shapes = tree.get_shapes_sliced()

        if tree.is_complete():
            kwargs.setdefault("optimize", {})
            kwargs["optimize"].setdefault("path", tree.get_path())

            if handle_slicing and tree.sliced_inds:
                kwargs["optimize"].setdefault(
                    "slicing",
                    [(ix, tree.size_dict[ix] - 1) for ix in tree.sliced_inds],
                )

        self.kwargs = kwargs
        self.autotune = 3 if autotune is True else autotune
        self.handle = None
        self.network = None

    def setup(self, *arrays):
        import cuquantum
        if hasattr(cuquantum, 'bindings'):
            # cuquantum-python >= 25.03
            from cuquantum.tensornet import Network
        else:
            # for cuquantum < 25.03
            from cuquantum import Network

        self.network = Network(
            self.eq,
            *arrays,
        )
        self.network.contract_path(**self.kwargs)
        if self.autotune:
            self.network.autotune(iterations=self.autotune)

    def __call__(
        self,
        *arrays,
        check_zero=False,
        backend=None,
        progbar=False,
    ):
        # can't handle these yet
        assert not check_zero
        assert not progbar
        assert backend is None

        if self.network is None:
            self.setup(*arrays)
        else:
            self.network.reset_operands(*arrays)

        return self.network.contract()

    def __del__(self):
        if self.network is not None:
            self.network.free()


def make_contractor(
    tree,
    order=None,
    prefer_einsum=False,
    strip_exponent=False,
    check_zero=False,
    implementation=None,
    autojit=False,
    progbar=False,
):
    """Get a reusable function which performs the contraction corresponding
    to ``tree``. The various options provide defaults that can also be overrode
    when calling the standard contractor.

    Parameters
    ----------
    tree : ContractionTree
        The contraction tree.
    order : str or callable, optional
        Supplied to :meth:`ContractionTree.traverse`, the order in which
        to perform the pairwise contractions given by the tree.
    prefer_einsum : bool, optional
        Prefer to use ``einsum`` for pairwise contractions, even if
        ``tensordot`` can perform the contraction.
    strip_exponent : bool, optional
        If ``True``, the function will strip the exponent from the output
        array and return it separately.
    check_zero : bool, optional
        If ``True``, when ``strip_exponent=True``, explicitly check for
        zero-valued intermediates that would otherwise produce ``nan``,
        instead terminating early if encountered and returning
        ``(0.0, 0.0)``.
    implementation : str or tuple[callable, callable], optional
        What library to use to actually perform the contractions. Options are

        - "auto": let cotengra choose
        - "autoray": dispatch with autoray, using the ``tensordot`` and
          ``einsum`` implementation of the backend
        - "cotengra": use the ``tensordot`` and ``einsum`` implementation of
          cotengra, which is based on batch matrix multiplication. This is
          faster for some backends like numpy, and also enables libraries
          which don't yet provide ``tensordot`` and ``einsum`` to be used.
        - "cuquantum": use the cuquantum library to perform the whole
          contraction (not just individual contractions).
        - tuple[callable, callable]: manually supply the ``tensordot`` and
          ``einsum`` implementations to use.

    autojit : bool, optional
        If ``True``, use :func:`autoray.autojit` to compile the contraction
        function.
    progbar : bool, optional
            Whether to show progress through the contraction by default.

    Returns
    -------
    fn : callable
        The contraction function, with signature ``fn(*arrays)``.
    """
    if implementation is None:
        implementation = get_default_implementation()

    if implementation == "cuquantum":
        fn = CuQuantumContractor(
            tree,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            progbar=progbar,
        )
    else:
        fn = Contractor(
            contractions=extract_contractions(tree, order, prefer_einsum),
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            implementation=implementation,
            progbar=progbar,
        )
        if autojit:
            from autoray import autojit as _autojit

            fn = _autojit(fn)

    return fn
