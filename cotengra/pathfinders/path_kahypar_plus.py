#
# Based on https://github.com/ti2-group/hybrid_contraction_tree_optimizer/
#

import random
from dataclasses import dataclass, field
from typing import (
    Hashable,
    Optional,
    Union,
    List,
    Tuple,
    Dict,
)

from cotengra.pathfinders.path_basic import (
    OptimalOptimizer,
    RandomGreedyOptimizer,
)
from .path_kahypar import kahypar_subgraph_find_membership
from ..core import ContractionTree
from ..hyperoptimizers.hyper import register_hyper_function


Inputs = List[List[Hashable]]
Output = List[Hashable]
SizeDict = Dict[Hashable, int]
Path = List[Tuple[int, ...]]


@dataclass
class BasicInputNode:
    indices: List[Hashable]


@dataclass
class OriginalInputNode(BasicInputNode):
    id: int

    def get_id(self):
        return self.id


@dataclass
class SubNetworkInputNode(BasicInputNode):
    sub_network: "TensorNetwork"

    def get_id(self):
        return self.sub_network._ssa_id

    def __repr__(self) -> str:
        return f"Sub network Input({self.sub_network.output_indices},"


InputNode = Union[OriginalInputNode, SubNetworkInputNode]
InputNodes = List["InputNode"]


def greedy_optimizer(tn: "TensorNetwork") -> Tuple[Path, float]:
    inputs = [input.indices for input in tn.inputs]
    output = tn.output_indices
    size_dict = tn.size_dict

    if len(inputs) <= 12:
        optimal_opt = OptimalOptimizer()
        return optimal_opt.ssa_path(inputs, output, size_dict)

    greedy_opt = RandomGreedyOptimizer(max_repeats=512)
    return greedy_opt.ssa_path(inputs, output, size_dict)


@dataclass
class TensorNetwork:
    name: str
    parent_name: str
    inputs: InputNodes
    size_dict: SizeDict
    output_indices: Output
    _ssa_id: Optional[int] = field(default=None, init=False)


def get_sub_networks(
    tensor_network: TensorNetwork,
    imbalance: float,
    weight_nodes: str = "const",
):
    input_nodes = tensor_network.inputs
    output = tensor_network.output_indices
    num_input_nodes = len(input_nodes)
    assert (
        num_input_nodes > 2
    ), f"You need to pass at least two input nodes, {input_nodes}"

    inputs = [input.indices for input in input_nodes]

    if len(output) > 0:
        inputs.append(output)

    block_ids = kahypar_subgraph_find_membership(
        inputs,
        set(),
        tensor_network.size_dict,
        weight_nodes=weight_nodes,
        weight_edges="log",
        fix_output_nodes=False,
        parts=2,
        imbalance=imbalance,
        compress=0,
        mode="recursive",
        objective="cut",
        quiet=True,
    )

    ## Noramlize block ids

    # Check if all input nodes were assigned to the same block
    input_block_ids = block_ids[:num_input_nodes]
    min_block_id = min(input_block_ids)
    max_block_id = max(input_block_ids)
    if min_block_id == max_block_id:
        # If there is only one block just distribute them with modulo
        block_ids = [i % 2 for i in range(num_input_nodes + 1)]
        input_block_ids = block_ids[:num_input_nodes]
    else:
        if min_block_id != 0 or max_block_id != 1:
            block_ids = [0 if id == min_block_id else 1 for id in block_ids]

    assert (
        len(set(input_block_ids)) == 2
    ), f"There should be two blocks, {input_block_ids}, {min_block_id}, {max_block_id}"

    # Group inputs by block id
    block_inputs: list[InputNodes] = [[], []]
    for block_id, input_node in zip(block_ids, input_nodes):
        block_inputs[block_id].append(input_node)

    block_indices = [
        frozenset(
            set.union(*[set(input_node.indices) for input_node in block])
        )
        for block in block_inputs
    ]

    cut_indices = block_indices[0].intersection(block_indices[1])

    if len(output) > 0:
        parent_block_id = block_ids[-1]
    else:
        parent_block_id = random.choice([0, 1])

    child_block_id = 1 - parent_block_id

    parent_sub_network = TensorNetwork(
        f"{tensor_network.name}.{parent_block_id}",
        tensor_network.name,
        block_inputs[parent_block_id],
        tensor_network.size_dict,
        output,
    )

    child_sub_network = TensorNetwork(
        f"{tensor_network.name}.{child_block_id}",
        parent_sub_network.name,
        block_inputs[child_block_id],
        tensor_network.size_dict,
        cut_indices,
    )

    sub_network_node = SubNetworkInputNode(
        child_sub_network.output_indices,
        child_sub_network,
    )
    parent_sub_network.inputs.append(sub_network_node)

    return parent_sub_network, child_sub_network


def extend_path(tn: TensorNetwork, sub_path: Path, last_id, path: Path):
    n = len(tn.inputs)
    for pair in sub_path:
        new_pair = []
        for element in pair:
            if element < n:
                new_pair.append(int(tn.inputs[element].get_id()))
            else:
                new_pair.append(last_id - n + element + 1)
        path.append(tuple(new_pair))

    return last_id + len(sub_path)


def hybrid_hypercut_greedy(
    inputs: Inputs,
    output: Output,
    size_dict: SizeDict,
    imbalance,
    weight_nodes="const",
    cutoff=15,
):
    # Noramlize parameters
    inputs = [list(input) for input in inputs]
    output = list(output)

    input_nodes: InputNodes = [
        OriginalInputNode(input, id) for id, input in enumerate(inputs)
    ]

    tensor_network = TensorNetwork("tn", None, input_nodes, size_dict, output)

    stack = [tensor_network]
    path = []
    last_id = len(inputs) - 1
    network_by_name = {tensor_network.name: tensor_network}
    cost = 0
    while stack:
        tn = stack.pop()
        if len(tn.inputs) <= cutoff:
            sub_path, sub_cost = greedy_optimizer(tn)
            cost += sub_cost
            last_id = extend_path(tn, sub_path, last_id, path)
            tn._ssa_id = last_id
            while tn.parent_name and len(tn.parent_name) < len(tn.name):
                network_by_name[tn.parent_name]._ssa_id = last_id
                tn = network_by_name[tn.parent_name]
            continue
        parent_sub_network, child_sub_network = get_sub_networks(
            tn,
            imbalance=imbalance,
            weight_nodes=weight_nodes,
        )
        stack.append(parent_sub_network)
        network_by_name[parent_sub_network.name] = parent_sub_network
        stack.append(child_sub_network)
        network_by_name[child_sub_network.name] = child_sub_network

    # print(f"{cost:.6e}", math.log10(cost / 2), imbalance, cutoff, weight_nodes)
    return ContractionTree.from_path(inputs, output, size_dict, ssa_path=path)


hyper_space = {
    "imbalance": {"type": "FLOAT", "min": 0.001, "max": 0.2},
    "weight_nodes": {"type": "STRING", "options": ["log"]},
    "cutoff": {"type": "INT", "min": 60, "max": 100},
}
register_hyper_function("kahypar+", hybrid_hypercut_greedy, hyper_space)
