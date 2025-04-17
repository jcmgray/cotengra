"""Quickbb based pathfinder."""

import re
import time
import signal
import tempfile
import warnings
import subprocess

from ..oe import PathOptimizer
from ..core import ContractionTree
from ..hypergraph import LineGraph
from ..hyperoptimizers.hyper import register_hyper_function


class QuickBBOptimizer(PathOptimizer):
    def __init__(self, max_time=10, executable="quickbb_64", seed=None):
        self.max_time = max_time
        self.executable = executable

    def run_quickbb(self, fname, outfile, statfile, max_time=None):
        if max_time is None:
            max_time = self.max_time

        args = [
            self.executable,
            "--min-fill-ordering",
            "--time",
            str(max_time),
            "--outfile",
            outfile,
            "--statfile",
            statfile,
            "--cnffile",
            fname,
        ]

        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        t0 = time.time()
        while process.poll() is None:
            time.sleep(0.1)
            if time.time() > t0 + max_time + 0.1:
                process.send_signal(signal.SIGTERM)
                break

        self.out, self.err = (x.decode("utf-8") for x in process.communicate())

        tw_search = re.search(r"Treewidth= (\d+)", self.out)
        if tw_search is None:
            self.treewidth = "n/a"
        else:
            self.treewidth = int(tw_search.group(1))

        # get the last line that started with a digit (i.e. the LG edge order)
        self.qbb_path = next(
            i for i in reversed(self.out.split("\n")) if re.match(r"^\d", i)
        )
        self.ordering = self.qbb_path.strip().split(" ")

        # # DEBUGGING: use consequences code to go via tree-decomposition
        # td_str = generate_td(self.out, fname)
        # td = td_str_to_tree_decomposition(td_str)
        # eo = td_to_eo(td)
        # self.ordering = eo.ordering
        # edges = list(self.LG.nodes)
        # self.edge_path = [edges[e - 1] for e in eo.ordering]
        # # # explicit check
        # # print(eo.ordering ==
        # #       list(map(int, self.qbb_path.strip().split(' '))))

        self.edge_path = [self.lg.nodes[int(i) - 1] for i in self.ordering]
        return self.edge_path

    def build_tree(self, inputs, output, size_dict):
        self.lg = LineGraph(inputs, output)

        with tempfile.NamedTemporaryFile(
            suffix=".cnf"
        ) as file, tempfile.NamedTemporaryFile(
            suffix=".out"
        ) as sfile, tempfile.NamedTemporaryFile(suffix=".out") as ofile:
            self.lg.to_cnf_file(file.name)

            max_time = self.max_time
            while True:
                try:
                    self.run_quickbb(
                        file.name, ofile.name, sfile.name, max_time=max_time
                    )
                    break
                except StopIteration:
                    max_time *= 1.5
                    warnings.warn(
                        "QuickBB produced no input, automatically "
                        "repeating with max_time 1.5x increased to "
                        f" {max_time}."
                    )

            with open(sfile.name, "r") as f:
                self.statfile = f.read()
            with open(ofile.name, "r") as f:
                self.outfile = f.read()

        self.tree = ContractionTree.from_path(
            inputs, output, size_dict, edge_path=self.edge_path
        )

        return self.tree

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return self.build_tree(inputs, output, size_dict).get_path()


def optimize_quickbb(
    inputs, output, size_dict, memory_limit=None, max_time=60, seed=None
):
    opt = QuickBBOptimizer(max_time=max_time, seed=seed)
    return opt(inputs, output, size_dict)


def trial_quickbb(inputs, output, size_dict, max_time=10, seed=None):
    opt = QuickBBOptimizer(max_time=max_time, seed=seed)
    return opt.build_tree(inputs, output, size_dict)


register_hyper_function(
    name="quickbb",
    ssa_func=trial_quickbb,
    space={"max_time": {"type": "FLOAT_EXP", "min": 2.0, "max": 60.0}},
)
