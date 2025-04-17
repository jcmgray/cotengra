"""Flowcutter based pathfinder."""

import re
import time
import signal
import warnings
import tempfile
import subprocess

from ..oe import PathOptimizer
from .treedecomp import td_str_to_tree_decomposition, td_to_eo
from ..core import ContractionTree
from ..hypergraph import LineGraph
from ..hyperoptimizers.hyper import register_hyper_function
from ..utils import get_rng


class FlowCutterOptimizer(PathOptimizer):
    def __init__(
        self, max_time=10, seed=None, executable="flow_cutter_pace17"
    ):
        self.max_time = max_time
        self.rng = get_rng(seed)
        self.executable = executable

    def run_flowcutter(self, file, max_time=None):
        if max_time is None:
            max_time = self.max_time
        seed = self.rng.randint(0, 2**32 - 1)
        args = [self.executable, "-s", str(seed)]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=file)

        t0 = time.time()
        while process.poll() is None:
            time.sleep(0.1)
            if time.time() > t0 + max_time:
                process.send_signal(signal.SIGTERM)
                break

        self.out = process.stdout.read().decode("utf-8")

        # self reported treewidth
        self.treewidth = int(re.findall(r"s td (\d+) (\d+)", self.out)[-1][1])

    def compute_edge_path(self, lg):
        td = td_str_to_tree_decomposition(self.out)
        eo = td_to_eo(td)
        self.edge_path = [lg.nodes[i - 1] for i in eo.ordering]

    def build_tree(self, inputs, output, size_dict, memory_limit=None):
        self.lg = LineGraph(inputs, output)

        with tempfile.NamedTemporaryFile(suffix=".gr") as file:
            self.lg.to_gr_file(file.name)

            max_time = self.max_time
            while True:
                try:
                    self.run_flowcutter(file, max_time=max_time)
                    break
                except IndexError:
                    max_time *= 1.5
                    warnings.warn(
                        "FlowCutter produced no input, automatically"
                        " repeating with max_time 1.5x increased to "
                        f" {max_time}."
                    )

        self.compute_edge_path(self.lg)
        self.tree = ContractionTree.from_path(
            inputs, output, size_dict, edge_path=self.edge_path
        )

        return self.tree

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return self.build_tree(inputs, output, size_dict).get_path()


def optimize_flowcutter(
    inputs, output, size_dict, memory_limit=None, max_time=10, seed=None
):
    opt = FlowCutterOptimizer(max_time=max_time, seed=seed)
    return opt(inputs, output, size_dict)


def trial_flowcutter(inputs, output, size_dict, max_time=10, seed=None):
    opt = FlowCutterOptimizer(max_time=max_time, seed=seed)
    return opt.build_tree(inputs, output, size_dict)


register_hyper_function(
    name="flowcutter",
    ssa_func=trial_flowcutter,
    space={"max_time": {"type": "FLOAT_EXP", "min": 2.0, "max": 60.0}},
)
