import pytest
import cotengra as ctg


def test_shortest_distances():
    inputs, output, _, size_dict = ctg.utils.lattice_equation([3, 3])
    hg = ctg.HyperGraph(inputs, output, size_dict)
    ds = hg.all_shortest_distances()
    assert ds[1, 4] == 1
    assert ds[1, 3] == 2
    assert ds[5, 6] == 3
    assert ds[0, 8] == 4


def test_resistance_centrality():
    inputs, output, _, size_dict = ctg.utils.lattice_equation([3, 3])
    hg = ctg.HyperGraph(inputs, output, size_dict)
    cents = hg.resistance_centrality()
    assert cents[0] == pytest.approx(0.0)
    assert 0.0 < cents[3] < 1.0
    assert cents[4] == 1.0


def test_simple_centrality():
    inputs, output, _, size_dict = ctg.utils.lattice_equation([3, 3])
    hg = ctg.HyperGraph(inputs, output, size_dict)
    cents = hg.simple_centrality()
    assert cents[0] == pytest.approx(0.0)
    assert 0.0 < cents[3] < 1.0
    assert cents[4] == 1.0


def test_compute_loops():
    inputs, output, _, size_dict = ctg.utils.lattice_equation([3, 3])
    hg = ctg.HyperGraph(inputs, output, size_dict)
    loops = list(hg.compute_loops())
    assert set(loops) == {
        (0, 1, 3, 4),
        (1, 2, 4, 5),
        (3, 4, 6, 7),
        (4, 5, 7, 8),
    }
    assert len(list(hg.compute_loops(max_loop_length=6))) == 8


def test_plot_nonconsec():
    import matplotlib as mpl

    mpl.use("Template")

    inputs = [
        ("a", "b", "x"),
        ("b", "c", "d"),
        ("c", "e", "y"),
        ("e", "a", "d"),
    ]
    output = ("x", "y")
    size_dict = {"x": 2, "y": 3, "a": 4, "b": 5, "c": 6, "d": 7, "e": 8}

    hg = ctg.HyperGraph(inputs, output, size_dict)
    hg.contract(0, 1)
    hg.plot()
