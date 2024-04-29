import copy
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import total_ordering
from queue import PriorityQueue
from typing import (
    Callable,
    Hashable,
    Optional,
    Sequence,
    Union,
    List,
    Tuple,
    Dict,
    Set,
    FrozenSet,
)

from cotengra.pathfinders.path_basic import (
    OptimalOptimizer,
    RandomGreedyOptimizer,
)
from .path_kahypar import kahypar_subgraph_find_membership
from ..core import ContractionTree
from ..hyperoptimizers.hyper import register_hyper_function


Inputs = List[List[Hashable]]
Shape = Tuple[int, ...]
Shapes = List[Shape]
Output = List[Hashable]
SizeDict = Dict[Hashable, int]
Path = List[Tuple[int, ...]]
GreedyOptimizer = Callable[["TensorNetwork"], Path]


@dataclass
class BasicInputNode:
    indices: List[Hashable]
    shape: Shape


@dataclass
class OriginalInputNode(BasicInputNode):
    id: int

    def get_id(self):
        return str(self.id)

    def __repr__(self) -> str:
        return f"Original Input({self.indices}, {self.shape})"


@dataclass
class SubNetworkInputNode(BasicInputNode):
    sub_network: "SubTensorNetwork"

    def get_id(self):
        return f"sn-{self.sub_network.name}"

    def __repr__(self) -> str:
        return f"Sub network Input({self.sub_network.output_indices}, {self.sub_network.get_output_shape()})"


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
        return f"IntermediateContractNode({self.uuid}), children: {[child.get_id() for child in self.children]}"


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
    for input in tn.get_all_input_nodes():
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
    inputs = [input.indices for input in tn.get_all_input_nodes()]
    output = tn.output_indices
    size_dict = tn.size_dict

    if len(inputs) <= 15:
        optimal_opt = OptimalOptimizer()
        return optimal_opt.ssa_path(inputs, output, size_dict)

    greedy_opt = RandomGreedyOptimizer(max_repeats=512)
    return greedy_opt.ssa_path(inputs, output, size_dict)


@dataclass
class SubTensorNetwork:
    name: str
    key: int
    parent_name: str
    inputs: InputNodes
    indices: FrozenSet[Hashable]
    size_dict: SizeDict
    cut_indices: FrozenSet[Hashable]
    output_indices: Output

    def get_all_input_nodes(self) -> InputNodes:
        return self.inputs

    def get_all_networks(self):
        return [self]

    def get_output_shape(self):
        return tuple([self.size_dict[index] for index in self.output_indices])

    def find_path(self):
        """
        Finds the path for the sub-tensor network.

        Returns:
            SubTensorNetworkWithContractTree: The sub-tensor network with the computed contract tree and its cost.
        """
        path = greedy_optimizer(self)

        contract_tree, cost = get_contract_tree_and_cost_from_path(self, path)

        tn_with_tree = SubTensorNetworkWithContractTree(
            name=self.name,
            key=self.key,
            parent_name=self.parent_name,
            inputs=self.inputs,
            indices=self.indices,
            size_dict=self.size_dict,
            cut_indices=self.cut_indices,
            output_indices=self.output_indices,
            cost=cost,
            contract_tree=contract_tree,
        )
        return tn_with_tree


@total_ordering
@dataclass
class SubTensorNetworkWithContractTree(SubTensorNetwork):
    cost: int
    contract_tree: ContractTree

    def get_total_cost(self):
        return self.cost

    def get_contract_tree(self):
        return self.contract_tree

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return self.cost == other.cost

    def __lt__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        # Yes this might seem weird, but we want the one with the highest cost to be the first in the priority queue
        return self.cost > other.cost

    def find_path(self):
        """
        Since the path is already computed, this function just returns the current object.

        Returns:
            The current object.
        """
        return self


@dataclass
class SuperTensorNetwork(SubTensorNetwork):
    parent_name: str
    sub_networks: Sequence[SubTensorNetwork]

    def get_all_input_nodes(self) -> InputNodes:
        sub_input_nodes = [
            SubNetworkInputNode(
                sub_network.output_indices,
                sub_network.get_output_shape(),
                sub_network,
            )
            for sub_network in self.sub_networks
        ]

        return sub_input_nodes + self.inputs

    def get_all_networks(self):
        return [self] + [sub_network for sub_network in self.sub_networks]

    def find_path(self):
        sub_networks_with_path: List[SubTensorNetworkWithContractTree] = []
        for sub_network in self.sub_networks:
            sub_networks_with_path.append(sub_network.find_path())

        self.sub_networks = sub_networks_with_path

        path = greedy_optimizer(self)
        contract_tree, cost = get_contract_tree_and_cost_from_path(self, path)

        tn_with_tree = SuperTensorNetworkWithTree(
            self.name,
            self.key,
            self.parent_name,
            self.inputs,
            self.indices,
            self.size_dict,
            self.cut_indices,
            self.output_indices,
            cost,
            contract_tree,
            sub_networks_with_path,
        )

        return tn_with_tree


@dataclass
class SuperTensorNetworkWithTree(
    SuperTensorNetwork,
    SubTensorNetworkWithContractTree,
):
    sub_networks: List[SubTensorNetworkWithContractTree]

    # def find_path(self, greedy_optimizer: GreedyOptimizer):
    #     """
    #     Since the path is already computed, this function just returns the current object.

    #     Returns:
    #         The current object.
    #     """
    #     return self

    def get_total_cost(self):
        return (
            sum([sub_network.cost for sub_network in self.sub_networks])
            + self.cost
        )

    def get_parent_tree(self):
        super_tree = self.get_contract_tree()

        parent_tree = []
        sub_tree_root: Dict[str, ContractTreeNode] = {}
        for node in super_tree:
            if (
                isinstance(node, SubNetworkInputNode)
                and node.sub_network.parent_name == self.name
            ):
                assert isinstance(
                    node.sub_network, SubTensorNetworkWithContractTree
                ), "The subnetworks should have a contract tree, when calling get_parent_tree"
                sub_tree = None
                for sn in self.sub_networks:
                    if sn.name == node.sub_network.name:
                        sub_tree = sn.contract_tree
                assert (
                    sub_tree is not None
                ), f"Sub tree {node.sub_network.name} not found in {self.name}"
                for sub_node in sub_tree:
                    parent_tree.append(sub_node)

                sub_tree_root[node.sub_network.name] = sub_tree[-1]
            elif isinstance(node, IntermediateContractNode):
                for key, child in enumerate(node.children):
                    if (
                        isinstance(child, SubNetworkInputNode)
                        and child.sub_network.parent_name == self.name
                    ):
                        node.children[key] = sub_tree_root[
                            child.sub_network.name
                        ]

                parent_tree.append(node)
            else:
                parent_tree.append(node)

        return parent_tree

    def update_tree(self, name, tree: ContractTree):
        if name == self.name:
            self.contract_tree = tree
        else:
            for sub_network in self.sub_networks:
                if sub_network.name == name:
                    sub_network.contract_tree = tree
                    return
            raise Exception(f"name {name} not found in {self.name}")

    def find_path(self):
        return self


TensorNetwork = Union[SubTensorNetwork, SuperTensorNetwork]
TensorNetworkWithTree = Union[
    SubTensorNetworkWithContractTree, SuperTensorNetworkWithTree
]


def get_sub_networks(
    tensor_network_name: str,
    input_nodes,
    output,
    size_dict: SizeDict,
    imbalance: float,
    weight_nodes: str = "const",
):
    num_input_nodes = len(input_nodes)
    assert (
        num_input_nodes > 2
    ), f"Not enough input nodes to split, pass at least two input nodes, {input_nodes}"

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
        cut_indices.intersection(block) for block in block_indices
    ]

    output_block_id = 0
    if len(output) > 0:
        output_block_id = block_ids[-1]
    else:
        output_block_id = random.choice([0, 1])

    sub_networks = [
        SubTensorNetwork(
            f"{tensor_network_name}.{key}",
            key,
            None,
            block_inputs[key],
            block_indices[key],
            size_dict,
            cut_indices,
            output_indices[key],
        )
        for key, _ in enumerate(block_inputs)
    ]

    super_sub_network = sub_networks.pop(output_block_id)
    super_network = build_super_network(
        super_sub_network,
        tensor_network_name,
        cut_indices,
        output,
        sub_networks,
    )

    return super_network.find_path()


def build_super_network(
    super_network: TensorNetwork,
    parent_name: str,
    cut_indices: FrozenSet[Hashable],
    output: Output,
    sub_networks: List[SubTensorNetwork],
):
    # Set parent for sub_networks
    for sub_network in sub_networks:
        sub_network.parent_name = super_network.name

    super_network = SuperTensorNetwork(
        name=super_network.name,
        key=super_network.key,
        parent_name=parent_name,
        inputs=super_network.inputs,
        indices=super_network.indices,
        size_dict=super_network.size_dict,
        cut_indices=cut_indices,
        output_indices=output,
        sub_networks=sub_networks,
    )
    return super_network


def get_remapped_id(id, input_remap: Optional[Dict[str, int]]):
    return input_remap[id] if input_remap != None and id in input_remap else id


def contract_tree_to_path(
    tree: ContractTree, remap: Optional[Dict[str, int]] = None
):
    root = tree[-1]

    if isinstance(root, BasicInputNode):
        assert (
            len(tree) == 1
        ), "Tree should only contain one node, if root is basic input"
        return [(int(get_remapped_id(root.get_id(), remap)),)]

    path = []

    counter = (len(tree) + 1) // 2
    uuid_to_ssa_id = {}
    for node in tree:
        if isinstance(node, IntermediateContractNode):
            uuid_to_ssa_id[node.get_id()] = counter
            counter += 1
            pair = []
            if isinstance(node.children[0], BasicInputNode):
                pair.append(get_remapped_id(node.children[0].get_id(), remap))
            else:
                pair.append(uuid_to_ssa_id[node.children[0].get_id()])

            if len(node.children) > 1:
                if isinstance(node.children[1], BasicInputNode):
                    pair.append(
                        get_remapped_id(node.children[1].get_id(), remap)
                    )
                else:
                    pair.append(uuid_to_ssa_id[node.children[1].get_id()])
            path.append(tuple(pair))
    return path


def hybrid_hypercut_greedy(
    inputs: Inputs,
    output: Output,
    size_dict: SizeDict,
    imbalance,
    weight_nodes="const",
    cutoff=15,
):
    # Problem setup, transform arguments to tanser network with path
    shapes = [tuple([size_dict[i] for i in input]) for input in inputs]
    inputs = [list(input) for input in inputs]
    output = list(output)
    indices = frozenset.union(*[frozenset(input) for input in inputs])

    input_nodes: InputNodes = [
        OriginalInputNode(in_sh[0], in_sh[1], id)
        for id, in_sh in enumerate(zip(inputs, shapes))
    ]

    tensor_network_without_path = SubTensorNetwork(
        "tn", 0, None, input_nodes, indices, size_dict, frozenset(), output
    )

    tensor_network = tensor_network_without_path.find_path()

    root_name = tensor_network.name

    # Initialize queues
    to_partition: PriorityQueue[TensorNetwork] = PriorityQueue()
    to_partition.put(tensor_network)

    # Initialize dictoionary for finalized partitions
    partitioned: Dict[str, TensorNetworkWithTree] = {}

    while not to_partition.empty():
        next_network = to_partition.get()
        if len(next_network.inputs) <= cutoff:
            with_path = next_network.find_path()  #
            partitioned[next_network.name] = with_path
            continue

        super_network = get_sub_networks(
            next_network.name,
            next_network.get_all_input_nodes(),
            next_network.output_indices,
            next_network.size_dict,
            imbalance=imbalance,
            weight_nodes=weight_nodes,
        )

        partitioned[super_network.parent_name] = copy.deepcopy(super_network)

        to_partition.put(super_network)

        for sub_network in super_network.sub_networks:
            to_partition.put(sub_network)

    def merge(network_name) -> ContractTree:
        partitioned_network = partitioned[network_name]
        # Check if we reached a leave
        if partitioned_network.name == network_name or not (
            isinstance(partitioned_network, SuperTensorNetworkWithTree)
        ):
            with_tree = partitioned_network.find_path()
            return with_tree.contract_tree
        for sub_network in partitioned_network.get_all_networks():
            merged_sub_tree = merge(sub_network.name)
            partitioned_network.update_tree(sub_network.name, merged_sub_tree)

        parent_tree = partitioned_network.get_parent_tree()

        return parent_tree

    parent_tree = merge(root_name)
    path = contract_tree_to_path(parent_tree)
    path = [tuple([int(i) for i in pair]) for pair in path]
    return ContractionTree.from_path(inputs, output, size_dict, ssa_path=path)


hyper_space = {
    "imbalance": {"type": "FLOAT", "min": 0.01, "max": 0.2},
    "weight_nodes": {"type": "STRING", "options": ["const", "log"]},
    "cutoff": {"type": "INT", "min": 40, "max": 200},
}
register_hyper_function("random-kahypar+", hybrid_hypercut_greedy, hyper_space)
