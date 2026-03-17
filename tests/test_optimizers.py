import subprocess
from collections import defaultdict

import autoray as ar
import numpy as np
import pytest

import cotengra as ctg
from cotengra.hyperoptimizers.hyper_sbplex import HyperSbplexSampler

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


methods_requires = [
    ("greedy", ""),
    ("greedy-compressed", ""),
    ("greedy-span", ""),
    ("labels", ""),
    ("kahypar", "kahypar"),
    ("kahypar-agglom", "kahypar"),
    ("betweenness", "igraph"),
    ("labelprop", "igraph"),
    ("spinglass", "igraph"),
    ("walktrap", "igraph"),
]

single_term_cases = [
    ([("a",)], ("a",), {"a": 2}),
    ([("a",)], (), {"a": 2}),
    ([("a", "b")], ("a",), {"a": 2, "b": 3}),
    ([("a", "b")], ("a", "b"), {"a": 2, "b": 3}),
    ([("a", "b")], ("b", "a"), {"a": 2, "b": 3}),
]

SBPLEX_TEST_SPACE = {
    "x": {"type": "FLOAT", "min": -1.0, "max": 1.0},
}


@pytest.mark.parametrize(("method", "requires"), methods_requires)
def test_basic(contraction_20_5, method, requires):
    if requires:
        pytest.importorskip(requires)

    opt = ctg.UniformOptimizer(
        methods=method, max_repeats=16, on_trial_error="raise"
    )
    eq, _, _, arrays = contraction_20_5
    shapes = [a.shape for a in arrays]

    tree = ctg.einsum_tree(eq, *shapes, optimize=opt)
    assert tree.speedup() > 1


@pytest.mark.parametrize(
    ("inputs", "output", "size_dict"),
    single_term_cases,
)
@pytest.mark.parametrize(("method", "requires"), methods_requires)
def test_single_term_uniform(inputs, output, size_dict, method, requires):
    if requires:
        pytest.importorskip(requires)

    opt = ctg.UniformOptimizer(
        methods=method,
        max_repeats=16,
        on_trial_error="raise",
        parallel=False,
    )
    tree = ctg.array_contract_tree(inputs, output, size_dict, optimize=opt)
    tree.describe()


@pytest.mark.parametrize(
    ("inputs", "output", "size_dict"),
    single_term_cases,
)
@pytest.mark.parametrize(("method", "requires"), methods_requires)
def test_single_term_direct(inputs, output, size_dict, method, requires):
    if requires:
        pytest.importorskip(requires)

    opt = ctg.UniformOptimizer(
        methods=method,
        max_repeats=16,
        on_trial_error="raise",
        parallel=False,
    )
    tree = opt.search(inputs, output, size_dict)
    tree.describe()


@pytest.mark.parametrize(
    ("optlib", "requires"),
    [
        ("nevergrad", "nevergrad"),
        ("skopt", "skopt"),
        ("cmaes", "cmaes"),
        ("optuna", "optuna"),
        ("es", ""),
        ("neldermead", ""),
        ("sbplex", ""),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_hyper(contraction_20_5, optlib, requires, parallel):
    pytest.importorskip("kahypar")
    if requires:
        pytest.importorskip(requires)

    eq, _, _, arrays = contraction_20_5
    shapes = [a.shape for a in arrays]
    optimizer = ctg.HyperOptimizer(
        max_repeats=32,
        parallel=parallel,
        optlib=optlib,
    )
    tree = ctg.einsum_tree(eq, *shapes, optimize=optimizer)
    assert tree.speedup() > 1
    assert {x[0] for x in optimizer.get_trials()} == {"greedy", "kahypar"}
    optimizer.print_trials()


def test_hyper_sbplex_restart_patience_triggers_local_restart():
    sampler = HyperSbplexSampler(
        SBPLEX_TEST_SPACE,
        seed=1,
        n_initial=0,
        restart_patience=2,
        explore_prob=0.0,
        convergence_tol=1e-3,
    )
    sampler._best_x = [0.25]
    sampler._best_score = 1.0
    sampler._step = [0.4]

    sampler._x = [0.1]
    sampler._x_at_cycle_start = [0.0]
    sampler._best_score_at_cycle_start = sampler._best_score
    sampler._finish_cycle()

    assert sampler._restart_count == 0
    assert sampler._cycles_since_improvement == 1

    sampler._x = [0.2]
    sampler._x_at_cycle_start = [0.0]
    sampler._best_score_at_cycle_start = sampler._best_score
    sampler._finish_cycle()

    assert sampler._restart_count == 1
    assert sampler._stagnant_restart_count == 1
    assert 0.0 < abs(sampler._step[0]) < 0.4


def test_hyper_sbplex_repeated_restarts_escalate_to_global_restart():
    sampler = HyperSbplexSampler(
        SBPLEX_TEST_SPACE,
        seed=2,
        n_initial=0,
        restart_patience=1,
        explore_prob=0.0,
        convergence_tol=1e-3,
        initial_scale=0.5,
    )
    sampler._best_x = [0.0]
    sampler._best_score = 1.0
    sampler._step = [0.4]

    sampler._x = [0.1]
    sampler._x_at_cycle_start = [0.0]
    sampler._best_score_at_cycle_start = sampler._best_score
    sampler._finish_cycle()

    assert sampler._restart_count == 1
    assert abs(sampler._step[0]) < 0.4

    sampler._x_at_cycle_start = [sampler._x[0] - 0.1]
    sampler._best_score_at_cycle_start = sampler._best_score
    sampler._finish_cycle()

    assert sampler._restart_count == 2
    assert sampler._stagnant_restart_count == 2
    assert sampler._step == [sampler.initial_scale]


def test_hyper_sbplex_improvement_resets_restart_counters():
    sampler = HyperSbplexSampler(
        SBPLEX_TEST_SPACE,
        seed=3,
        n_initial=0,
        restart_patience=2,
        explore_prob=0.0,
    )
    sampler._best_score = 10.0
    sampler._cycles_since_improvement = 4
    sampler._stagnant_restart_count = 3
    sampler._trial_map[0] = ("filler", None, None, [0.5])

    sampler.tell(0, 1.0)

    assert sampler._best_score == 1.0
    assert sampler._best_x == [0.5]
    assert sampler._cycles_since_improvement == 0
    assert sampler._stagnant_restart_count == 0


def test_hyper_sbplex_stale_nm_results_ignored_after_restart():
    sampler = HyperSbplexSampler(
        SBPLEX_TEST_SPACE,
        seed=4,
        n_initial=0,
        restart_patience=1,
        explore_prob=0.0,
    )

    stale_trial, _ = sampler.ask()
    sampler._restart("global")
    current_trial, _ = sampler.ask()

    assert stale_trial != current_trial
    assert sampler._sub_nm_id is not None
    assert 0 in sampler._sub_nm._token_map

    sampler.tell(stale_trial, 123.0)

    assert 0 in sampler._sub_nm._token_map


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
    eq, _, _, arrays = contraction_20_5
    shapes = [a.shape for a in arrays]
    optimizer = optimize(max_time=1)
    tree = ctg.einsum_tree(eq, *shapes, optimize=optimizer)
    assert tree.speedup() > 1


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


@pytest.mark.localonly
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
        max_repeats=4,
        parallel=parallel_backend,
        optlib="random",
        progbar=True,
        slicing_reconf_opts={
            "target_size": 2**20,
            "forested": True,
            "max_repeats": 2,
            "num_trees": 2,
            "reconf_opts": {
                "forested": True,
                "num_trees": 2,
                "subtree_size": 4,
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
