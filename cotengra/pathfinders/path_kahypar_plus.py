#
# Based on https://github.com/ti2-group/hybrid_contraction_tree_optimizer/
#

import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Hashable,
    Optional,
    Union,
    List,
    Tuple,
    Dict,
    Set,
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
GreedyOptimizer = Callable[["TensorNetwork"], Path]


@dataclass
class BasicInputNode:
    indices: List[Hashable]


@dataclass
class OriginalInputNode(BasicInputNode):
    id: int

    def get_id(self):
        return str(self.id)

    def __repr__(self) -> str:
        return f"Original Input({self.indices})"


@dataclass
class SubNetworkInputNode(BasicInputNode):
    sub_network: "TensorNetwork"

    def get_id(self):
        return f"sn-{self.sub_network.name}"

    def __repr__(self) -> str:
        return f"Sub network Input({self.sub_network.output_indices},"


InputNode = Union[OriginalInputNode, SubNetworkInputNode]
InputNodes = List["InputNode"]


@dataclass
class IntermediateContractNode:
    all_indices: Set[Hashable]  # Union all indices of children
    scale: int
    indices: Output
    children: List["ContractTreeNode"]
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_id(self):
        return self.uuid

    def __repr__(self) -> str:
        return (
            f"IntermediateContractNode({self.uuid}), "
            + f"children: {[child.get_id() for child in self.children]}"
        )


ContractTreeNode = Union[
    OriginalInputNode, SubNetworkInputNode, IntermediateContractNode
]

ContractTree = List[ContractTreeNode]


def safe_log2(x):
    if x < 1:
        return 0
    return math.log2(x)


def get_contract_tree_and_cost_from_path(
    tn: "TensorNetwork", ssa_path
) -> Tuple[ContractTree, int]:
    contract_tree: ContractTree = []
    histogram = defaultdict(lambda: 0)
    for input in tn.inputs:
        contract_tree.append(input)
        for edge in input.indices:
            histogram[edge] += 1

    for index in tn.output_indices:
        histogram[index] += 1

    # If there is only one input thats the whole tree
    if len(contract_tree) == 1:
        return contract_tree, 1
    total_cost = 0

    for pair in ssa_path:
        if len(pair) == 1:
            left_node: ContractTreeNode = contract_tree[pair[0]]
            all_indices = set(left_node.indices)
            cost = 1
            remove = set()
            for index in left_node.indices:
                cost = cost * tn.size_dict[index]
                if histogram[index] == 0:
                    remove.add(index)
            total_cost += cost
            intermediate = all_indices - remove
            for index in intermediate:
                histogram[index] += 1
            contract_tree.append(
                IntermediateContractNode(
                    all_indices,
                    int(safe_log2(cost)),
                    list(intermediate),
                    [contract_tree[pair[0]]],
                )
            )
        if len(pair) == 2:
            left_node: ContractTreeNode = contract_tree[pair[0]]
            right_node: ContractTreeNode = contract_tree[pair[1]]
            all_indices = set(left_node.indices).union(right_node.indices)
            cost = 1
            remove = set()
            for index in all_indices:
                cost = cost * tn.size_dict[index]

            for index in left_node.indices:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            for index in right_node.indices:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            total_cost += cost
            intermediate = all_indices - remove

            for index in intermediate:
                histogram[index] += 1

            contract_tree.append(
                IntermediateContractNode(
                    all_indices,
                    int(safe_log2(cost)),
                    list(intermediate),
                    [contract_tree[pair[0]], contract_tree[pair[1]]],
                )
            )
    total_cost = 2 * total_cost

    return contract_tree, total_cost


def greedy_optimizer(tn: "TensorNetwork") -> Path:
    inputs = [input.indices for input in tn.inputs]
    output = tn.output_indices
    size_dict = tn.size_dict

    if len(inputs) <= 15:
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
    _contract_tree: ContractTree = None
    _cost: Optional[int] = None
    _ssa_id: Optional[int] = None

    def get_contract_tree(self):
        if self._contract_tree is None:
            self.find_path()
        return self._contract_tree

    def find_path(self):
        """
        Finds the path for the tensor network.
        """
        path = greedy_optimizer(self)
        contract_tree, cost = get_contract_tree_and_cost_from_path(self, path)
        self._contract_tree = contract_tree
        self._cost = cost


def get_sub_networks(
    tensor_network: TensorNetwork,
    imbalance: float,
    weight_nodes: str = "const",
):
    input_nodes = tensor_network.inputs
    output = tensor_network.output_indices
    size_dict = tensor_network.size_dict
    tensor_network_name = tensor_network.name
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
        size_dict,
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

    cut_indices = set()
    cut_indices = cut_indices.union(
        block_indices[0].intersection(block_indices[1])
    )

    cut_indices = frozenset(cut_indices.union(output))
    output_indices: list[Output] = [
        list(cut_indices.intersection(block)) for block in block_indices
    ]

    if len(output) > 0:
        output_block_id = block_ids[-1]
    else:
        output_block_id = random.choice([0, 1])

    sub_networks = [
        TensorNetwork(
            f"{tensor_network_name}.{key}",
            tensor_network_name,
            block_inputs[key],
            block_indices[key],
            size_dict,
            output_indices[key],
        )
        for key, _ in enumerate(block_inputs)
    ]

    parent_sub_network = sub_networks.pop(output_block_id)
    child_sub_network = sub_networks.pop()
    child_sub_network.parent_name = parent_sub_network.name
    sub_network_node = SubNetworkInputNode(
        child_sub_network.output_indices,
        child_sub_network,
    )
    parent_sub_network.inputs.append(sub_network_node)
    parent_sub_network.output_indices = output

    return parent_sub_network, child_sub_network


def contract_tree_to_path(root: ContractTreeNode, last_id, path: Path):
    if isinstance(root, IntermediateContractNode):
        sub_ids = []
        for child in root.children:
            sub_root = contract_tree_to_path(child, last_id, path)
            last_id = max(last_id, sub_root)
            sub_ids.append(sub_root)
        path.append(tuple(sub_ids))
        new_root_id = max(last_id, max(sub_ids)) + 1
        return new_root_id

    if isinstance(root, OriginalInputNode):
        return int(root.get_id())
    if isinstance(root, SubNetworkInputNode):
        return root.sub_network._ssa_id


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
        # print(f"Popped {tn.name}")
        if len(tn.inputs) <= cutoff:
            tree = tn.get_contract_tree()
            cost += tn._cost
            sub_id = contract_tree_to_path(tree[-1], last_id, path)
            last_id = max(last_id, sub_id)
            tn._ssa_id = sub_id
            while tn.parent_name and len(tn.parent_name) < len(tn.name):
                network_by_name[tn.parent_name]._ssa_id = sub_id
                tn = network_by_name[tn.parent_name]
            continue
        parent_sub_network, child_sub_network = get_sub_networks(
            tn,
            imbalance=imbalance,
            weight_nodes=weight_nodes,
        )
        stack.append(parent_sub_network)
        network_by_name[parent_sub_network.name] = parent_sub_network
        # print(f"Pushed parent {parent_sub_network.name}")
        stack.append(child_sub_network)
        network_by_name[child_sub_network.name] = child_sub_network
        # print(f"Pushed child {child_sub_network.name}")

    # contract_tree = tensor_network.build_tree(imbalance, cutoff, weight_nodes)
    # path = []
    # contract_tree_to_path(contract_tree[-1], len(inputs) - 1, path)
    path = [tuple([int(i) for i in pair]) for pair in path]
    # print(f"{cost:.6e}", math.log10(cost / 2), imbalance, cutoff, weight_nodes)
    return ContractionTree.from_path(inputs, output, size_dict, ssa_path=path)


hyper_space = {
    "imbalance": {"type": "FLOAT", "min": 0.001, "max": 0.1},
    "weight_nodes": {"type": "STRING", "options": ["log"]},
    "cutoff": {"type": "INT", "min": 50, "max": 100},
}
register_hyper_function("kahypar+", hybrid_hypercut_greedy, hyper_space)
