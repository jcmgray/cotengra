"""Compressed contraction tree search using monte carlo tree search."""

import math

from ..pathfinders.path_basic import ssa_to_linear
from ..core import ContractionTreeCompressed, get_hypergraph
from ..utils import GumbelBatchedGenerator


class Node:
    __slots__ = (
        "hg",
        "n",
        "graph_key",
        "nid_path",
        "size",
        "local_score",
        "forward_score",
        "mean",
        "count",
        "leaf_score",
    )

    def __init__(
        self,
        hg,
        nid_path,
        size,
        local_score,
        forward_score,
    ):
        self.hg = hg
        self.n = hg.get_num_nodes()
        self.graph_key = hash(frozenset(hg.nodes))
        self.nid_path = nid_path
        self.size = size
        self.local_score = local_score
        self.forward_score = forward_score

        self.count = 0
        self.mean = float("inf")
        # self.mean = 0.0
        self.leaf_score = None

    def update(self, x):
        """Report the score ``x``, presumably from a child node, updating this
        nodes score.
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        # self.mean = min(self.mean, x)

        phi = math.log2(self.mean)
        phi -= (phi / self.count) ** 0.5
        self.leaf_score = phi

    def __hash__(self):
        return hash((self.graph_key, self.size))

    def __lt__(self, other):
        return self.leaf_score < other.leaf_score

    def __repr__(self):
        return (
            f"<Node(|V|={self.n}, "
            f"fscore={math.log2(self.forward_score)}, "
            f"lscore={self.leaf_score}, "
            f"count={self.count}, "
            f"id={id(self)})>"
        )


class MCTS:
    def __init__(
        self,
        chi,
        T=0.1,
        prune=True,
        optimize=None,
        optimize_factory=False,
        seed=None,
    ):
        self.chi = chi
        self.T = T
        self.prune = prune
        self.optimize = optimize
        self.optimize_factory = optimize_factory
        self.best_score = float("inf")
        self.best_nid_path = None
        self.children = {}
        self.parents = {}
        self.seen = {}
        self.to_delete = set()
        self.leaves = None
        self.root = None
        self.N = None
        self.gmblgen = GumbelBatchedGenerator(seed)

    def __repr__(self):
        return (
            "<MCTS("
            f"bs: {self.best_score}, "
            f"nT: {len(self.parents)}, "
            f"lq: {len(self.leaves)}"
            ")>"
        )

    def setup(self, inputs, output, size_dict):
        """ """
        if self.leaves is None:
            H = get_hypergraph(
                {1 << i: term for i, term in enumerate(inputs)},
                output,
                size_dict,
                accel=False,
            )
            self.N = H.get_num_nodes()
            size0 = sum(map(H.node_size, H.nodes))

            root = Node(
                hg=H,
                nid_path=(),
                size=size0,
                local_score=0,
                forward_score=size0,
            )

            self.root = root
            self.leaves = set()
            self.check_node(root)

    def get_ssa_path(self):
        """Convert unique node identifiers to ssa."""
        ssa_path = []
        ssa = self.N
        ssa_lookup = {1 << i: i for i in range(ssa)}
        for i, j in self.best_nid_path:
            ij = i | j
            ssa_lookup[ij] = ssa
            ssa += 1
            ssa_path.append((ssa_lookup[i], ssa_lookup[j]))
        return ssa_path

    def check_node(self, node):
        """ """
        if node in self.children:
            return

        hg = node.hg
        cnodes = self.children[node] = set()

        # for all possible next contractions
        for i, j in hg.edges.values():
            hg_next = hg.copy()

            # compress then contract
            hg_next.compress(
                self.chi, hg_next.get_node(i) + hg_next.get_node(j)
            )
            ij = hg_next.contract(i, j, node=i | j)

            # measure change in total memory
            dsize = hg_next.neighborhood_size([ij]) - hg.neighborhood_size(
                [i, j]
            )

            # score is peak total size encountered
            new_size = node.size + dsize
            new_score = max(node.forward_score, new_size)

            if self.prune and (new_score >= self.best_score):
                # all subsequent paths will be worse - skip
                continue

            new_node = Node(
                hg=hg_next,
                nid_path=node.nid_path + ((i, j),),
                size=new_size,
                local_score=dsize,
                forward_score=new_score,
            )

            graph_key = new_node.graph_key
            if self.prune and (
                new_score >= self.seen.get(graph_key, float("inf"))
            ):
                # we've reached this graph before with better score
                continue
            self.seen[graph_key] = min(
                new_score, self.seen.get(graph_key, float("inf"))
            )

            # add to tree
            cnodes.add(new_node)
            self.parents[new_node] = node

        # hypergraph only needed to generate children
        # node.hg = None

    def delete_node(self, node):
        """ """
        if node is self.root:
            raise KeyError("Cannot delete root node")

        dnodes = []

        # get all children
        queue = [node]
        while queue:
            cnode = queue.pop()
            queue.extend(self.children.get(cnode, ()))
            dnodes.append(cnode)

        while node is not self.root:
            # get childless parents
            pnode = self.parents[node]
            siblings = self.children[pnode]
            siblings.remove(node)

            # NB: could also contract single children?
            if siblings:
                break

            dnodes.append(pnode)
            node = pnode

        # wipe nodes
        for dnode in dnodes:
            self.children.pop(dnode, None)
            self.parents.pop(dnode, None)
            self.seen.pop(dnode.graph_key, None)
            self.leaves.discard(dnode)

        return pnode

    def backprop(self, node):
        final_score = node.forward_score
        if final_score < self.best_score:
            self.best_score = final_score
            self.best_nid_path = node.nid_path
        self.pbar.update()
        self.pbar.set_description(
            f"lq:{len(self.leaves) if self.leaves is not None else None} "
            f"best:{math.log2(self.best_score):.2f} "
            f"tree:{len(self.parents)} ",
            refresh=False,
        )
        # back track upwards updating backward_score
        while node is not self.root:
            pnode = self.parents[node]
            pnode.update(final_score)
            node = pnode

    def simulate_node(self, node):
        """ """
        # greedily descend to bottom based on heuristic local_score
        while True:
            self.check_node(node)
            if node.n == 1:
                # we've reached a fully contracted graph
                break
            if self.prune and self.is_deadend(node):
                node = self.delete_node(node)
                continue
            node = min(
                self.children[node],
                key=lambda node: node.local_score - self.T * self.gmblgen(),
            )
        self.backprop(node)

    def simulate_optimized(self, node):
        H = node.hg

        if self.optimize_factory:
            optimize = self.optimize()
        else:
            optimize = self.optimize

        path = optimize(
            inputs=tuple(H.nodes.values()),
            output=H.output,
            size_dict=H.size_dict,
        )
        nids = list(H.nodes.keys())
        for i, j in path:
            self.check_node(node)
            ni, nj = map(nids.pop, sorted((i, j), reverse=True))
            nids.append(ni | nj)
            try:
                node = next(
                    iter(
                        (
                            x
                            for x in self.children[node]
                            if set(x.nid_path[-1]) == {ni, nj}
                        )
                    )
                )
            except StopIteration:
                # next path node already pruned
                self.simulate_node(node)
                return

        self.backprop(node)

    def is_deadend(self, node):
        """ """
        return (
            not bool(self.children[node])
            or node.forward_score >= self.best_score
            or node.forward_score > self.seen.get(node.graph_key, float("inf"))
        )

    def descend(self):
        """ """
        if self.optimize is not None:
            simulate = self.simulate_optimized
        else:
            simulate = self.simulate_node

        node = self.root
        while node in self.leaves:
            node = min(self.children[node])
            if self.prune and self.is_deadend(node):
                node = self.delete_node(node)
                continue
        for cnode in tuple(self.children[node]):
            if cnode in self.children[node]:
                simulate(cnode)
        self.leaves.add(node)
        for node in tuple(self.children):
            try:
                if self.is_deadend(node):
                    self.delete_node(node)
            except KeyError:
                pass

    @property
    def ssa_path(self):
        """ """
        return self.get_ssa_path()

    @property
    def path(self):
        """ """
        return ssa_to_linear(self.ssa_path)

    def run(self, inputs, output, size_dict):
        """ """
        import tqdm

        self.pbar = tqdm.tqdm()
        self.setup(inputs, output, size_dict)

        try:
            while self.children:
                self.descend()
        except KeyboardInterrupt:
            pass
        finally:
            self.pbar.close()

    def search(self, inputs, output, size_dict):
        """ """
        self.run(inputs, output, size_dict)
        return ContractionTreeCompressed.from_path(
            inputs,
            output,
            size_dict,
            ssa_path=self.ssa_path,
        )

    def __call__(self, inputs, output, size_dict):
        """ """
        self.run(inputs, output, size_dict)
        return self.path
