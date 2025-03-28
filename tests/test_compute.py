import numpy as np
import pytest
from numpy.testing import assert_allclose

import cotengra as ctg

# these are taken from opt_einsum
test_case_eqs = [
    # Test single-term equations
    "->",
    "a->a",
    "ab->ab",
    "ab->ba",
    "abc->bca",
    "abc->b",
    "baa->ba",
    "aba->b",
    # Test scalar-like operations
    "a,->a",
    "ab,->ab",
    ",ab,->ab",
    ",,->",
    # Test hadamard-like products
    "a,ab,abc->abc",
    "a,b,ab->ab",
    # Test index-transformations
    "ea,fb,gc,hd,abcd->efgh",
    "ea,fb,abcd,gc,hd->efgh",
    "abcd,ea,fb,gc,hd->efgh",
    # Test complex contractions
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
    "abhe,hidj,jgba,hiab,gab",
    "bde,cdh,agdb,hica,ibd,hgicd,hiac",
    "chd,bde,agbc,hiad,hgc,hgi,hiad",
    "chd,bde,agbc,hiad,bdi,cgh,agdb",
    "bdhe,acad,hiab,agac,hibd",
    # Test collapse
    "ab,ab,c->",
    "ab,ab,c->c",
    "ab,ab,cd,cd->",
    "ab,ab,cd,cd->ac",
    "ab,ab,cd,cd->cd",
    "ab,ab,cd,cd,ef,ef->",
    # Test outer prodcuts
    "ab,cd,ef->abcdef",
    "ab,cd,ef->acdf",
    "ab,cd,de->abcde",
    "ab,cd,de->be",
    "ab,bcd,cd->abcd",
    "ab,bcd,cd->abd",
    # Random test cases that have previously failed
    "eb,cb,fb->cef",
    "dd,fb,be,cdb->cef",
    "bca,cdb,dbf,afc->",
    "dcc,fce,ea,dbf->ab",
    "fdf,cdd,ccd,afe->ae",
    "abcd,ad",
    "ed,fcd,ff,bcf->be",
    "baa,dcf,af,cde->be",
    "bd,db,eac->ace",
    "fff,fae,bef,def->abd",
    "efc,dbc,acf,fd->abe",
    # Inner products
    "ab,ab",
    "ab,ba",
    "abc,abc",
    "abc,bac",
    "abc,cba",
    # GEMM test cases
    "ab,bc",
    "ab,cb",
    "ba,bc",
    "ba,cb",
    "abcd,cd",
    "abcd,ab",
    "abcd,cdef",
    "abcd,cdef->feba",
    "abcd,efdc",
    # Inner than dot
    "aab,bc->ac",
    "ab,bcc->ac",
    "aab,bcc->ac",
    "baa,bcc->ac",
    "aab,ccb->ac",
    # Randomly built test caes
    "aab,fa,df,ecc->bde",
    "ecb,fef,bad,ed->ac",
    "bcf,bbb,fbf,fc->",
    "bb,ff,be->e",
    "bcb,bb,fc,fff->",
    "fbb,dfd,fc,fc->",
    "afd,ba,cc,dc->bf",
    "adb,bc,fa,cfc->d",
    "bbd,bda,fc,db->acf",
    "dba,ead,cad->bce",
    "aef,fbc,dca->bde",
]


@pytest.mark.parametrize("eq", test_case_eqs)
@pytest.mark.parametrize(
    "dtype", ("float32", "float64", "complex64", "complex128")
)
@pytest.mark.parametrize("strip_exponent", [False, True])
def test_basic_equations(eq, dtype, strip_exponent):
    arrays = ctg.utils.make_arrays_from_eq(eq, dtype=dtype)
    x = np.einsum(eq, *arrays)
    y = ctg.einsum(eq, *arrays, strip_exponent=strip_exponent)
    if strip_exponent:
        y = y[0] * 10 ** y[1]
    rtol = 5e-3 if dtype in ("float32", "complex64") else 5e-6
    assert_allclose(x, y, rtol=rtol)


@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("d_min", [2])
@pytest.mark.parametrize("d_max", [4])
@pytest.mark.parametrize("reg", [2, 3])
@pytest.mark.parametrize("n_out", [0, 1, 2])
@pytest.mark.parametrize("n_hyper_in", [0, 1, 2])
@pytest.mark.parametrize("n_hyper_out", [0, 1, 2])
@pytest.mark.parametrize("seed", [42, 666])
@pytest.mark.parametrize("indices_sort", [None, "root", "flops"])
def test_contraction_tree_rand_equation(
    n,
    reg,
    n_out,
    n_hyper_in,
    n_hyper_out,
    d_min,
    d_max,
    seed,
    indices_sort,
):
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=n,
        reg=reg,
        n_out=n_out,
        n_hyper_in=n_hyper_in,
        n_hyper_out=n_hyper_out,
        d_min=d_min,
        d_max=d_max,
        seed=seed,
    )
    arrays = [np.random.normal(size=s) for s in shapes]
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    x = np.einsum(eq, *arrays, optimize="greedy")

    tree = ctg.einsum_tree(
        eq, *shapes, optimize="greedy", sort_contraction_indices=indices_sort
    )

    # base contract
    y1 = tree.contract(arrays)
    assert_allclose(x, y1)

    # contract after modifying tree
    tree.subtree_reconfigure_()
    y2 = tree.contract(arrays)
    assert_allclose(x, y2)

    size = tree.max_size()
    if size < 600:
        return

    # contract after slicing and modifying
    tree.slice_and_reconfigure_(target_size=size // 6)
    y3 = tree.contract(arrays)
    assert_allclose(x, y3)

    # contract after slicing some output indices
    remaining_out = list(tree.get_legs(tree.root))
    nsout = np.random.randint(low=0, high=len(remaining_out) + 1)
    so_ix = np.random.choice(remaining_out, replace=False, size=nsout)
    for ind in so_ix:
        tree.remove_ind_(ind)

        if indices_sort:
            tree.sort_contraction_indices(indices_sort)

        y4 = tree.contract(arrays)
        assert_allclose(x, y4)


def test_lazy_sliced_output_reduce():
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=10,
        reg=5,
        n_out=3,
        d_max=2,
        seed=666,
    )
    arrays = [np.random.rand(*s) for s in shapes]
    opt = ctg.HyperOptimizer(max_repeats=32, methods=["greedy"])
    tree = opt.search(inputs, output, size_dict)

    # slice both inner and outer indices
    tree.remove_ind_("a")
    tree.remove_ind_("x")
    tree.remove_ind_("b")
    tree.remove_ind_("y")

    # for such a quantity, sum(f(x)), the inner slice sum must be performed 1st
    x = (tree.contract(arrays) ** 2).sum()

    # ... so that the outer sum can be lazily generated correctly
    y = 0.0
    for chunk in tree.gen_output_chunks(arrays):
        y += (chunk**2).sum()

    assert y == pytest.approx(x)


@pytest.mark.parametrize("autojit", [False, True])
@pytest.mark.parametrize(
    "dtype", ["float32", "float64", "complex64", "complex128"]
)
def test_exponent_stripping(autojit, dtype):
    inputs, output, shapes, size_dict = ctg.utils.lattice_equation([8, 8])

    arrays = ctg.utils.make_arrays_from_inputs(
        inputs, size_dict, seed=42, dtype=dtype
    )

    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    ex = ctg.einsum(eq, *arrays)

    tree = ctg.array_contract_tree(inputs, output, size_dict)

    x1 = tree.contract(arrays, autojit=autojit)
    assert x1 == pytest.approx(ex)

    m, p = tree.contract(arrays, autojit=autojit, strip_exponent=True)
    x2 = m * 10**p
    assert x2 == pytest.approx(ex)

    tree.slice_(target_size=64)
    assert tree.nslices >= 4

    x3 = tree.contract(arrays, autojit=autojit)
    assert x3 == pytest.approx(ex)

    m, p = tree.contract(arrays, autojit=autojit, strip_exponent=True)
    x4 = m * 10**p
    assert x4 == pytest.approx(ex)


@pytest.mark.parametrize("autojit", [False, True])
@pytest.mark.parametrize("constants", [None, True])
@pytest.mark.parametrize("optimize_type", ["path", "tree", "optimizer", "str"])
@pytest.mark.parametrize("sort_contraction_indices", [False, True])
def test_einsum_expression(
    autojit,
    constants,
    optimize_type,
    sort_contraction_indices,
):
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    inputs, output, shapes, size_dict = ctg.utils.lattice_equation([4, 8])
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    arrays = [np.random.rand(*s) for s in shapes]
    x0 = oe.contract(eq, *arrays)

    if optimize_type == "str":
        optimize = "greedy"
    elif optimize_type == "path":
        path = oe.contract_path(eq, *arrays, optimize="greedy")[0]
        optimize = path
    elif optimize_type == "tree":
        tree = ctg.ContractionTree(inputs, output, size_dict)
        tree.contract_nodes(tuple(tree.gen_leaves()), optimize="greedy")
        optimize = tree
    elif optimize_type == "optimizer":
        optimize = ctg.HyperOptimizer(max_repeats=32, methods=["greedy"])

    if constants:
        constants = range(0, len(inputs), 2)
        shapes = list(shapes)
        for c in sorted(constants, reverse=True):
            shapes[c] = arrays.pop(c)

    expr = ctg.einsum_expression(
        eq,
        *shapes,
        optimize=optimize,
        constants=constants,
        autojit=autojit,
        sort_contraction_indices=sort_contraction_indices,
    )

    x1 = expr(*arrays)
    assert_allclose(x0, x1)
