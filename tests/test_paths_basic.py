import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

import cotengra as ctg
import cotengra.pathfinders.path_basic as pb

test_case_eqs = [
    "a->a",
    "ab->ab",
    "ab->a",
    "ab->ba",
    # these are taken from opt_einsum
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
    }[which](inputs, output, size_dict)
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert_allclose(tree.contract(arrays), expected)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_basic_rand(seed, which):
    c = ctg.utils.rand_equation(
        n=10,
        reg=4,
        n_out=2,
        n_hyper_in=1,
        n_hyper_out=1,
        d_min=2,
        d_max=3,
        seed=seed,
    )
    eq = ctg.utils.inputs_output_to_eq(c.inputs, c.output)

    path = {
        "greedy": pb.optimize_greedy,
        "optimal": pb.optimize_optimal,
    }[which](c.inputs, c.output, c.size_dict)

    tree = ctg.ContractionTree.from_path(
        c.inputs, c.output, c.size_dict, path=path
    )
    arrays = [np.random.randn(*s) for s in c.shapes]
    assert_allclose(
        tree.contract(arrays), np.einsum(eq, *arrays, optimize=True)
    )


@pytest.mark.parametrize("seed", range(3))
def test_random_greedy_track_flops(seed):
    c = ctg.utils.lattice_equation(
        [4, 5],
        d_min=2,
        d_max=3,
        seed=seed,
    )
    opt = ctg.RandomGreedyOptimizer(
        max_repeats=2,
        temperature=0.1,
        seed=seed,
        accel=False,
        parallel=False,
    )
    path = opt(c.inputs, c.output, c.size_dict)
    tree = ctg.ContractionTree.from_path(
        c.inputs, c.output, c.size_dict, path=path
    )
    assert tree.contraction_cost(log=10) == pytest.approx(opt.best_flops)
    # check deterministic
    opt2 = ctg.RandomGreedyOptimizer(
        max_repeats=2,
        temperature=0.1,
        seed=seed,
        accel=False,
        parallel=False,
    )
    opt2(c.inputs, c.output, c.size_dict)
    assert opt.best_ssa_path == opt2.best_ssa_path
    assert opt.best_flops == opt2.best_flops


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_basic_perverse(seed, which):
    c = ctg.utils.perverse_equation(10, seed=seed)
    eq = ctg.utils.inputs_output_to_eq(c.inputs, c.output)
    print(eq)
    path = {
        "greedy": pb.optimize_greedy,
        "optimal": pb.optimize_optimal,
    }[which](c.inputs, c.output, c.size_dict)
    tree = ctg.ContractionTree.from_path(
        c.inputs, c.output, c.size_dict, path=path
    )
    arrays = [np.random.randn(*s) for s in c.shapes]
    assert_allclose(
        tree.contract(arrays), np.einsum(eq, *arrays, optimize=True)
    )


def test_optimal_lattice_eq():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [4, 5], d_max=3, seed=42
    )

    path = pb.optimize_optimal(inputs, output, size_dict, minimize="flops")
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert tree.contraction_cost() == 1464

    path = pb.optimize_optimal(inputs, output, size_dict, minimize="size")
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert tree.contraction_width() == pytest.approx(5.584962500721156)


def test_random_optimize():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [4, 5], d_max=3, seed=42
    )
    tree = ctg.array_contract_tree(
        inputs,
        output,
        size_dict,
        optimize="random",
    )
    assert tree.is_complete()


def test_edgesort_optimize():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [4, 5], d_max=3, seed=42
    )
    tree = ctg.array_contract_tree(
        inputs,
        output,
        size_dict,
        optimize="edgesort",
    )
    assert tree.is_complete()


def test_edgesort_optimize_manual_labelled_reverse():
    # array_contract and friends canonicalize the equation letters by
    # appearance by default, check the manual override to obey the supplied
    # ordering works here:
    tree = ctg.array_contract_tree(
        inputs=[(3, 2), (2, 1), (1, 0)],
        size_dict={0: 2, 1: 2, 2: 2, 3: 2},
        optimize="edgesort",
    )
    assert tree.get_path() == ((1, 2), (0, 1))


# ---- size 1 indices ---- #
# these are simply ignored when finding a path (they contribute a factor of 1
# to every cost), then reintroduced when the tree is rebuilt from the original
# inputs - so the path is valid but the search avoids any blowup from them.


def _size_one_roundtrip(inputs, output, size_dict, which):
    """Strip -> path -> rebuild on original inputs -> contract, vs einsum."""
    path = {
        "greedy": pb.optimize_greedy,
        "optimal": pb.optimize_optimal,
    }[which](inputs, output, size_dict)
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    eq = ctg.utils.inputs_output_to_eq(inputs, output)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict, seed=0)
    assert_allclose(
        tree.contract(arrays), np.einsum(eq, *arrays, optimize=True)
    )
    return tree


@pytest.mark.parametrize("eq", test_case_eqs)
@pytest.mark.parametrize("which", ["greedy", "optimal"])
@pytest.mark.parametrize("seed", range(3))
def test_manual_cases_with_size_one(eq, which, seed):
    # reuse every manual eq but allow size 1 dims (d_min=1)
    inputs, output = ctg.utils.eq_to_inputs_output(eq)
    size_dict = ctg.utils.make_rand_size_dict_from_inputs(
        inputs, d_min=1, d_max=3, seed=seed
    )
    _size_one_roundtrip(inputs, output, size_dict, which)


@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_all_size_one(which):
    # closed loop, every bond size 1 -> whole thing is trivial
    inputs = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
    output = ()
    size_dict = dict.fromkeys("abcd", 1)
    _size_one_roundtrip(inputs, output, size_dict, which)


@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_size_one_in_output(which):
    # 'a' is size 1 AND in the output -> exercises the output-loop guard
    inputs = [("a", "b"), ("b", "c")]
    output = ("a", "c")
    size_dict = {"a": 1, "b": 3, "c": 4}
    tree = _size_one_roundtrip(inputs, output, size_dict, which)
    arrays = ctg.utils.make_arrays_from_inputs(inputs, size_dict, seed=0)
    assert tree.contract(arrays).shape == (1, 4)


@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_size_one_bond_is_outer_product(which):
    # 'b' size-1 bond -> the contraction is really an outer product
    inputs = [("a", "b"), ("b", "c")]
    output = ("a", "c")
    size_dict = {"a": 3, "b": 1, "c": 4}
    _size_one_roundtrip(inputs, output, size_dict, which)


@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_size_one_hyperedge(which):
    # 'h' size-1 shared by all n terms: previously a fully-connected blowup
    # for optimal (no max_neighbors guard), now stripped entirely
    n = 8
    letters = [ctg.utils.get_symbol(i) for i in range(n)]
    inputs = [("h", x) for x in letters]
    output = tuple(letters)
    size_dict = {x: 2 for x in letters}
    size_dict["h"] = 1
    _size_one_roundtrip(inputs, output, size_dict, which)


@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_scalar_after_stripping(which):
    # one term collapses to a scalar once its size-1 dims are dropped
    inputs = [("a", "b"), ("p", "q"), ("a", "b")]
    output = ()
    size_dict = {"a": 3, "b": 2, "p": 1, "q": 1}
    _size_one_roundtrip(inputs, output, size_dict, which)


def test_processor_strips_all_size_one():
    inputs = [("a", "b"), ("b", "c")]
    output = ("a",)
    size_dict = dict.fromkeys("abc", 1)
    cp = pb.ContractionProcessor(inputs, output, size_dict)
    # no edges/indices registered at all...
    assert cp.edges == {}
    assert cp.indmap == {}
    assert cp.sizes == []
    # ...but the nodes (and thus path positions) are preserved
    assert len(cp.nodes) == 2
    assert all(legs == () for legs in cp.nodes.values())


def test_processor_keeps_only_large_indices():
    inputs = [("a", "b"), ("b", "c")]
    # 'a' size 1 and in output -> must not KeyError, must not be registered
    output = ("a",)
    size_dict = {"a": 1, "b": 5, "c": 1}
    cp = pb.ContractionProcessor(inputs, output, size_dict)
    assert set(cp.indmap) == {"b"}
    assert cp.sizes == [5]


def test_processor_no_size_one_blowup():
    # the motivating case: dense all-size-1 graph -> empty edge set, so optimal
    # enumerates no contraction candidates and the build completes instantly
    n = 20
    inputs = [[] for _ in range(n)]
    size_dict = {}
    for k, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        ix = ctg.utils.get_symbol(k)
        size_dict[ix] = 1
        inputs[i].append(ix)
        inputs[j].append(ix)
    inputs = [tuple(t) for t in inputs]
    cp = pb.ContractionProcessor(inputs, (), size_dict)
    assert cp.edges == {}
    path = pb.optimize_optimal(inputs, (), size_dict)
    tree = ctg.ContractionTree.from_path(inputs, (), size_dict, path=path)
    assert tree.is_complete()
