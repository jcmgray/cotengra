import subprocess
from collections import defaultdict

import autoray as ar
import numpy as np
import pytest

import cotengra as ctg
from cotengra.hyperoptimizers.hyper import _OPTLIB_DEFAULTS
from cotengra.hyperoptimizers.hyper_neldermead import _NMCore
from cotengra.hyperoptimizers.hyper_sbplx import HyperSbplxSampler

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

SBPLX_TEST_SPACE = {
    "x": {"type": "FLOAT", "min": -1.0, "max": 1.0},
}

SBPLX_TEST_SPACE_6D = {
    name: {"type": "FLOAT", "min": -1.0, "max": 1.0} for name in "abcdef"
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
        ("sses", ""),
        ("neldermead", ""),
        ("sbplx", ""),
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


def test_hyper_sbplx_restart_patience_triggers_local_restart():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE,
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


def test_hyper_sbplx_partition_uses_goodness_heuristic():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE_6D,
        seed=1,
        n_initial=0,
        nsmin=2,
        nsmax=5,
        partition="goodness",
        explore_prob=0.0,
    )
    sampler._step = [5.0, 4.0, 3.0, 1.0, 1.0, 1.0]

    sampler._partition_dims()

    assert sampler._subspaces == [[0, 1], [2, 3, 4, 5]]


def test_hyper_sbplx_partition_greedy_equal_chunks():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE_6D,
        seed=1,
        n_initial=0,
        nsmin=2,
        nsmax=3,
        partition="greedy",
        explore_prob=0.0,
    )
    sampler._step = [5.0, 4.0, 3.0, 1.0, 1.0, 1.0]

    sampler._partition_dims()

    # greedy: two equal chunks of 3 (sorted by magnitude)
    assert len(sampler._subspaces) == 2
    assert len(sampler._subspaces[0]) == 3
    assert len(sampler._subspaces[1]) == 3


def test_hyper_sbplx_cycle_step_scaling_clamped_by_omega():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE_6D,
        seed=1,
        n_initial=0,
        explore_prob=0.0,
        convergence_tol=1e-3,
    )
    sampler._subspaces = [[0, 1, 2], [3, 4, 5]]
    sampler._x_at_cycle_start = [0.0] * 6
    sampler._step_at_cycle_start = [1.0] * 6
    sampler._x = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    sampler._update_steps_after_cycle()

    assert sampler._step[0] == 10.0
    assert sampler._step[1:] == [-10.0] * 5


def test_hyper_sbplx_cycle_convergence_is_relative_to_scale():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE,
        seed=1,
        n_initial=0,
        explore_prob=0.0,
        convergence_tol=1e-4,
    )
    sampler._x_at_cycle_start = [1000.0]
    sampler._x = [1000.05]
    sampler._step = [0.1]

    assert sampler._cycle_converged()


def test_hyper_sbplx_repeated_restarts_escalate_to_global_restart():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE,
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


def test_hyper_sbplx_improvement_resets_restart_counters():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE,
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


def test_hyper_sbplx_stale_nm_results_ignored_after_restart():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE,
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


def test_nmcore_inject_vertex_diameter_gate_accepts_nearby():
    core = _NMCore(
        ndim=2,
        center=[0.0, 0.0],
        scales=[0.5, 0.5],
        convergence_tol=1e-6,
        inject_diameter_fraction=1.0,
    )
    # manually set up a sorted simplex
    core._vertices = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
    core._scores = [1.0, 2.0, 3.0]
    core._state = "reflect"

    # point close to the simplex — should be accepted
    accepted = core.inject_vertex([0.05, 0.05], 2.5)
    assert accepted
    assert core._pending_injection is not None


def test_nmcore_inject_vertex_diameter_gate_rejects_far():
    core = _NMCore(
        ndim=2,
        center=[0.0, 0.0],
        scales=[0.5, 0.5],
        convergence_tol=1e-6,
        inject_diameter_fraction=1.0,
    )
    core._vertices = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
    core._scores = [1.0, 2.0, 3.0]
    core._state = "reflect"

    # point far from the simplex — should be rejected
    accepted = core.inject_vertex([0.9, 0.9], 2.5)
    assert not accepted
    assert core._pending_injection is None


def test_nmcore_inject_vertex_early_convergence_signal():
    core = _NMCore(
        ndim=2,
        center=[0.0, 0.0],
        scales=[0.5, 0.5],
        convergence_tol=1e-6,
        inject_diameter_fraction=1.0,
    )
    core._vertices = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
    core._scores = [1.0, 2.0, 3.0]
    core._best_score = 1.0
    core._state = "reflect"
    core._tell_count = 10

    # point far away but dramatically better (< 0.5 * best_score)
    accepted = core.inject_vertex([0.9, 0.9], 0.3)
    assert not accepted
    assert core._converged


def test_nmcore_inject_vertex_inf_diameter_fraction():
    core = _NMCore(
        ndim=2,
        center=[0.0, 0.0],
        scales=[0.5, 0.5],
        convergence_tol=1e-6,
        inject_diameter_fraction=float("inf"),
    )
    core._vertices = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
    core._scores = [1.0, 2.0, 3.0]
    core._state = "reflect"

    # even a far point should be accepted with inf fraction
    accepted = core.inject_vertex([0.9, 0.9], 2.5)
    assert accepted


def test_nm_sampler_adaptive_filler_scale():
    from cotengra.hyperoptimizers.hyper_neldermead import (
        HyperNelderMeadSampler,
    )

    sampler = HyperNelderMeadSampler(
        SBPLX_TEST_SPACE,
        seed=42,
        n_initial=0,
        filler_scale=0.01,
        explore_prob=0.0,
    )
    assert sampler._core is not None

    # manually populate the core so it has a real diameter
    sampler._core._vertices = [[0.0], [0.5]]
    sampler._core._scores = [1.0, 2.0]
    sampler._core._state = "reflect"
    sampler._core._initial_simplex_diameter = 0.5

    diameter = sampler._core._simplex_diameter()
    assert diameter > 0.01  # bigger than filler_scale floor

    # ask a filler — effective scale should be max(0.5*0.5, 0.01) = 0.25
    trial_number, params = sampler._ask_filler()
    assert trial_number >= 0

    # filler should center on core's best vertex, not global best
    sampler._core._best_vertex = [0.8]
    sampler._best_x = [-0.9]
    rng = sampler.rng
    samples = [sampler._ask_filler()[1][0] for _ in range(200)]
    mean_sample = sum(samples) / len(samples)
    # centered on 0.8, not -0.9 — mean should be close to 0.8
    assert abs(mean_sample - 0.8) < 0.15


def test_sbplx_sampler_adaptive_filler_scale():
    sampler = HyperSbplxSampler(
        SBPLX_TEST_SPACE,
        seed=42,
        n_initial=0,
        filler_scale=0.01,
        explore_prob=0.0,
    )
    # set up cycling state with a step vector
    sampler._step = [0.4]
    sampler._sub_nm = _NMCore(
        ndim=1,
        center=[0.0],
        scales=[0.4],
        convergence_tol=1e-4,
    )

    # ask a filler — effective scale should be max(0.5*0.4, 0.01) = 0.2
    trial_number, x = sampler._ask_filler()
    assert trial_number >= 0


def test_nmcore_psi_convergence_uses_relative_diameter():
    core = _NMCore(
        ndim=1,
        center=[0.0],
        scales=[1.0],
        convergence_tol=1e-6,
        psi=0.5,
    )
    core._vertices = [[0.0], [0.4]]
    core._scores = [0.0, 1.0]
    core._initial_simplex_diameter = 1.0

    core._begin_reflect()

    assert core.converged


def test_nm_sampler_exits_init_phase_with_inf_scores():
    from cotengra.hyperoptimizers.hyper_neldermead import (
        HyperNelderMeadSampler,
    )

    sampler = HyperNelderMeadSampler(
        SBPLX_TEST_SPACE,
        seed=42,
        n_initial=3,
        explore_prob=0.0,
    )

    assert sampler._init_phase

    # simulate all init trials scoring inf (e.g. BadTrial / timeout)
    tokens = []
    for _ in range(3):
        token, _ = sampler.ask()
        tokens.append(token)

    for token in tokens:
        sampler.tell(token, float("inf"))

    # init phase must have ended despite all-inf scores
    assert not sampler._init_phase


def test_cmaes_report_result_handles_inf():
    cmaes = pytest.importorskip("cmaes")  # noqa: F841
    from cotengra.hyperoptimizers.hyper_cmaes import CMAESOptLib

    space = {"greedy": SBPLX_TEST_SPACE}
    optlib = CMAESOptLib()
    optlib.setup(methods=["greedy"], space=space)

    # ask enough trials to fill one population, report all as inf
    pop_size = optlib._optimizers["greedy"].opt.population_size
    settings = [optlib.get_setting() for _ in range(pop_size)]
    for s in settings:
        optlib.report_result(s, {}, float("inf"))

    # should still be able to ask for more trials afterwards
    s = optlib.get_setting()
    assert s["method"] == "greedy"


def test_optuna_report_result_handles_inf():
    pytest.importorskip("optuna")
    from cotengra.hyperoptimizers.hyper_optuna import OptunaOptLib

    space = {"greedy": SBPLX_TEST_SPACE}
    optlib = OptunaOptLib()
    optlib.setup(methods=["greedy"], space=space)

    # ask a trial and report inf
    s = optlib.get_setting()
    optlib.report_result(s, {}, float("inf"))

    # ask another trial and report a normal score
    s2 = optlib.get_setting()
    optlib.report_result(s2, {}, 1.0)

    # should still be functional
    s3 = optlib.get_setting()
    assert s3["method"] == "greedy"


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
