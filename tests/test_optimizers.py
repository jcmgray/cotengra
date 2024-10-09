import subprocess
import functools
from collections import defaultdict

import pytest
import numpy as np
import autoray as ar

import cotengra as ctg


try:
    subprocess.run(["quickbb_64"])
    FOUND_QUICKBB = True
except FileNotFoundError:
    FOUND_QUICKBB = False

try:
    subprocess.run(["flow_cutter_pace17", "--help"])
    FOUND_FLOWCUTTER = True
except FileNotFoundError:
    FOUND_FLOWCUTTER = False


def rand_reg_contract(n, deg, seed=None):
    import networkx as nx

    rG = nx.random_regular_graph(deg, n, seed=seed)
    edge2ind = {
        tuple(sorted(e)): ctg.utils.get_symbol(i)
        for i, e in enumerate(rG.edges)
    }

    inputs = [
        {edge2ind[tuple(sorted(e))] for e in rG.edges(nd)} for nd in rG.nodes
    ]
    output = {}

    eq = ",".join(["".join(i) for i in inputs]) + "->{}".format(
        "".join(output)
    )
    shapes = [(2,) * deg] * n
    views = list(map(ar.lazy.Variable, shapes))

    return eq, shapes, views, inputs, output


@pytest.fixture
def contraction_20_5():
    eq, shapes, views, _, _ = rand_reg_contract(20, 5, seed=42)
    size_dict = defaultdict(lambda: np.random.randint(2, 9))
    return eq, shapes, size_dict, views


@pytest.mark.parametrize(
    ("opt", "requires"),
    [
        (functools.partial(ctg.UniformOptimizer, methods="greedy"), ""),
        (
            functools.partial(
                ctg.UniformOptimizer, methods="greedy-compressed"
            ),
            "",
        ),
        (functools.partial(ctg.UniformOptimizer, methods="greedy-span"), ""),
        (functools.partial(ctg.UniformOptimizer, methods="labels"), ""),
        (
            functools.partial(ctg.UniformOptimizer, methods="kahypar"),
            "kahypar",
        ),
        (
            functools.partial(ctg.UniformOptimizer, methods="kahypar-agglom"),
            "kahypar",
        ),
        (
            functools.partial(ctg.UniformOptimizer, methods="betweenness"),
            "igraph",
        ),
        (
            functools.partial(ctg.UniformOptimizer, methods="labelprop"),
            "igraph",
        ),
        (
            functools.partial(ctg.UniformOptimizer, methods="spinglass"),
            "igraph",
        ),
        (
            functools.partial(ctg.UniformOptimizer, methods="walktrap"),
            "igraph",
        ),
    ],
)
def test_basic(contraction_20_5, opt, requires):
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    if requires:
        pytest.importorskip(requires)

    eq, _, _, arrays = contraction_20_5
    optimizer = opt(max_repeats=16)

    _, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
    assert path_info.speedup > 1


@pytest.mark.parametrize(
    ("optlib", "requires"),
    [
        ("baytune", "btb"),
        ("chocolate", "chocolate"),
        ("nevergrad", "nevergrad"),
        ("skopt", "skopt"),
        ("cmaes", "cmaes"),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_hyper(contraction_20_5, optlib, requires, parallel):
    pytest.importorskip("kahypar")
    pytest.importorskip(requires)
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    eq, _, _, arrays = contraction_20_5
    optimizer = ctg.HyperOptimizer(
        max_repeats=32,
        parallel=parallel,
        optlib=optlib,
    )
    _, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
    assert path_info.speedup > 1
    assert {x[0] for x in optimizer.get_trials()} == {"greedy", "kahypar"}
    optimizer.print_trials()


@pytest.mark.parametrize(
    "optimize",
    [
        pytest.param(
            ctg.QuickBBOptimizer,
            marks=pytest.mark.skipif(
                not FOUND_QUICKBB, reason="quickbb_64 not found on path"
            ),
        ),
        pytest.param(
            ctg.FlowCutterOptimizer,
            marks=pytest.mark.skipif(
                not FOUND_FLOWCUTTER,
                reason="flow_cutter_pace17 not found on path",
            ),
        ),
    ],
)
def test_binaries(contraction_20_5, optimize):
    pytest.importorskip("opt_einsum")

    import opt_einsum as oe

    eq, _, _, arrays = contraction_20_5
    optimizer = optimize(max_time=1)
    _, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
    assert path_info.speedup > 1


@pytest.mark.parametrize("parallel", [False, True])
def test_hyper_slicer(parallel):
    if parallel:
        pytest.importorskip("distributed")

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        30, reg=5, seed=42, d_max=3
    )

    optimizer = ctg.HyperOptimizer(
        max_repeats=16,
        parallel=parallel,
        optlib="random",
        slicing_opts={"target_slices": 1000},
        progbar=True,
    )
    tree = ctg.array_contract_tree(
        inputs, output, size_dict, optimize=optimizer
    )
    assert tree.multiplicity >= 1000
    assert optimizer.best["flops"] > optimizer.best["original_flops"]


@pytest.mark.parametrize("parallel", [False, True])
def test_hyper_reconf(parallel):
    if parallel:
        pytest.importorskip("distributed")

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        30, reg=5, seed=42, d_max=3
    )

    optimizer = ctg.HyperOptimizer(
        max_repeats=16,
        parallel=parallel,
        optlib="random",
        reconf_opts={"subtree_size": 6},
        progbar=True,
    )
    ctg.array_contract_tree(inputs, output, size_dict, optimize=optimizer)
    assert optimizer.best["flops"] < optimizer.best["original_flops"]


@pytest.mark.parametrize("parallel", [False, True])
def test_hyper_slicer_reconf(parallel):
    if parallel:
        pytest.importorskip("distributed")

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        30, reg=5, seed=42, d_max=3
    )

    optimizer = ctg.HyperOptimizer(
        max_repeats=16,
        parallel=parallel,
        optlib="random",
        slicing_reconf_opts={
            "target_size": 2**19,
            "reconf_opts": {
                "subtree_size": 6,
            },
        },
        progbar=True,
    )
    tree = ctg.array_contract_tree(
        inputs, output, size_dict, optimize=optimizer
    )
    assert tree.max_size() <= 2**19


@pytest.mark.parametrize("parallel_backend", ("dask", "ray"))
def test_insane_nested(parallel_backend):
    if parallel_backend == "dask":
        pytest.importorskip("distributed")
    else:
        pytest.importorskip(parallel_backend)

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        30, reg=5, seed=42, d_max=3
    )

    optimizer = ctg.HyperOptimizer(
        max_repeats=16,
        parallel=parallel_backend,
        optlib="random",
        progbar=True,
        slicing_reconf_opts={
            "target_size": 2**20,
            "forested": True,
            "max_repeats": 4,
            "num_trees": 2,
            "reconf_opts": {
                "forested": True,
                "num_trees": 2,
                "subtree_size": 6,
            },
        },
    )
    tree = ctg.array_contract_tree(
        inputs, output, size_dict, optimize=optimizer
    )
    assert tree.max_size() <= 2**20


def test_plotting():
    pytest.importorskip("matplotlib")

    import matplotlib

    matplotlib.use("Template")

    inputs, output, _, size_dict = ctg.utils.rand_equation(
        30, reg=5, seed=42, d_max=3
    )
    hg = ctg.HyperGraph(inputs, output, size_dict)
    hg.plot()
    opt = ctg.HyperOptimizer(max_repeats=16)
    opt.search(inputs, output, size_dict)
    opt.plot_trials()
    opt.plot_scatter()


def test_auto_optimizers_threadsafe():
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(2)

    contractions = [ctg.utils.tree_equation(100, seed=i) for i in range(10)]
    fs = [
        pool.submit(
            ctg.array_contract_tree,
            inputs,
            output,
            size_dict,
            optimize="auto-hq",
        )
        for inputs, output, _, size_dict in contractions
    ]
    [f.result() for f in fs]


def test_reusable_optimizes_overwrite_improved():
    contractions = []

    for seed in range(10):
        con = ctg.utils.rand_equation(20, 3, 2, 2, 2, seed=seed)
        contractions.append(con)

    opt = ctg.ReusableHyperOptimizer(
        methods="greedy",
        overwrite="improved",
        max_repeats=2,
    )

    scores_a = [
        opt.search(inputs, output, size_dict).get_score()
        for inputs, output, _, size_dict in contractions
    ]
    # second time the optimize runs for every contraction, but if an old tree
    # is better than the new one, it will be used instead
    scores_b = [
        opt.search(inputs, output, size_dict).get_score()
        for inputs, output, _, size_dict in contractions
    ]

    for s1, s2 in zip(scores_b, scores_a):
        assert s1 <= s2
