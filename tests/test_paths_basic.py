import pytest
import numpy as np
from numpy.testing import assert_allclose
import cotengra as ctg
import cotengra.pathfinders.path_basic as pb


# these are taken from opt_einsum
test_case_eqs = [
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
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_manual_cases(eq, which):
    inputs, output = ctg.utils.eq_to_inputs_output(eq)
    size_dict = ctg.utils.make_rand_size_dict_from_inputs(inputs)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict)
    expected = np.einsum(eq, *arrays, optimize=True)

    path = {
        "greedy": pb.optimize_greedy,
        "optimal": pb.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert_allclose(tree.contract(arrays), expected)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_basic_rand(seed, which):
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=10,
        reg=4,
        n_out=2,
        n_hyper_in=1,
        n_hyper_out=1,
        d_min=2,
        d_max=3,
        seed=seed,
    )
    eq = ctg.utils.inputs_output_to_eq(inputs, output)

    path = {
        "greedy": pb.optimize_greedy,
        "optimal": pb.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)

    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    arrays = [np.random.randn(*s) for s in shapes]
    assert_allclose(
        tree.contract(arrays), np.einsum(eq, *arrays, optimize=True)
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_basic_perverse(seed, which):
    inputs, output, shapes, size_dict = ctg.utils.perverse_equation(
        10, seed=seed
    )
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    print(eq)
    path = {
        "greedy": pb.optimize_greedy,
        "optimal": pb.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    arrays = [np.random.randn(*s) for s in shapes]
    assert_allclose(
        tree.contract(arrays), np.einsum(eq, *arrays, optimize=True)
    )


def test_optimal_lattice_eq():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [4, 5], d_max=3, seed=42
    )

    path = pb.optimize_optimal(inputs, output, size_dict, minimize="flops")
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert tree.contraction_cost() == 3628

    path = pb.optimize_optimal(inputs, output, size_dict, minimize="size")
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert tree.contraction_width() == pytest.approx(6.754887502163468)
