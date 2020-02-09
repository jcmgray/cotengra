import subprocess
from collections import defaultdict

import pytest
import opt_einsum as oe
from opt_einsum.contract import Shaped
import numpy as np

import cotengra as ctg


try:
    subprocess.run(['quickbb_64'])
    FOUND_QUICKBB = True
except FileNotFoundError:
    FOUND_QUICKBB = False

try:
    subprocess.run(['flow_cutter_pace17', '--help'])
    FOUND_FLOWCUTTER = True
except FileNotFoundError:
    FOUND_FLOWCUTTER = False


def rand_reg_contract(n, deg, seed=None):
    import networkx as nx

    rG = nx.random_regular_graph(deg, n, seed=seed)
    edge2ind = {tuple(sorted(e)): oe.get_symbol(i)
                for i, e in enumerate(rG.edges)}

    inputs = [
        {edge2ind[tuple(sorted(e))]
         for e in rG.edges(nd)}
        for nd in rG.nodes
    ]
    output = {}

    eq = (",".join(["".join(i) for i in inputs]) +
          "->{}".format("".join(output)))
    shapes = [(2,) * deg] * n
    views = list(map(Shaped, shapes))

    return eq, shapes, views, inputs, output


@pytest.fixture
def contraction_20_5():
    eq, shapes, views, _, _ = rand_reg_contract(20, 5, seed=42)
    size_dict = defaultdict(lambda: np.random.randint(2, 9))
    return eq, shapes, size_dict, views


@pytest.mark.parametrize(('opt', 'requires'), [
    (ctg.UniformGreedy, ''),
    (ctg.UniformBetweenness, 'igraph'),
    (ctg.UniformSpinglass, 'igraph'),
    (ctg.UniformKaHyPar, 'kahypar'),
])
def test_basic(contraction_20_5, opt, requires):

    if requires:
        pytest.importorskip(requires)

    eq, _, _, arrays = contraction_20_5
    optimizer = opt(max_repeats=16)

    _, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
    assert path_info.speedup > 1


def test_hyper(contraction_20_5):
    pytest.importorskip('btb')
    pytest.importorskip('psutil')
    pytest.importorskip('kahypar')
    eq, _, _, arrays = contraction_20_5
    optimizer = ctg.HyperOptimizer(
        max_repeats=32,
        parallel=False,
    )
    _, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
    assert path_info.speedup > 1
    assert {x[0] for x in optimizer.get_trials()} == {'greedy', 'kahypar'}


@pytest.mark.parametrize("optimize", [
    pytest.param(ctg.QuickBBOptimizer, marks=pytest.mark.skipif(
        not FOUND_QUICKBB, reason='quickbb_64 not found on path')),
    pytest.param(ctg.FlowCutterOptimizer, marks=pytest.mark.skipif(
        not FOUND_FLOWCUTTER, reason='flow_cutter_pace17 not found on path')),
])
def test_binaries(contraction_20_5, optimize):
    eq, _, _, arrays = contraction_20_5
    optimizer = optimize(max_time=1)
    _, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
    assert path_info.speedup > 1
