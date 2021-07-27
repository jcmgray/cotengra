import random
import math
import heapq

from cotengra.core import get_hypergraph


class Node:

    __slots__ = (
        'hg',
        'ssa_path',
        'size',
        'local_score',
        'forward_score',
        'backward_score',
        'n',
        '_key',
    )

    def __init__(
        self,
        hg,
        ssa_path,
        size,
        local_score,
        forward_score,
    ):
        self.hg = hg
        self.ssa_path = ssa_path
        self.size = size
        self.local_score = local_score
        self.forward_score = forward_score
        self.n = 1

        if hg.get_num_nodes() > 1:
            self.backward_score = float('inf')
        else:
            self.backward_score = forward_score

        self._key = None

    @property
    def key(self):
        k = self._key
        if k is None:
            k = self._key = hash(frozenset(self.hg.nodes))
        return k

    def update(self, other):
        self.backward_score = min(self.backward_score, other.backward_score)
        self.n += 1

    @property
    def score(self):
        ls = math.log2(self.backward_score)
        ls -= (ls / self.n)**0.5
        return (ls, self.hg.get_num_nodes())

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return (f"<Node(|V|={self.hg.get_num_nodes()}, "
                f"fscore={self.forward_score}, "
                f"bscore={self.backward_score}, "
                f"id={id(self)})>")


def gumbel():
    return -math.log(-math.log(random.random()))


class MCTS:

    __slots__ = (
        'chi',
        'T',
        'best_score',
        'best_ssa_path',
        'children',
        'parents',
        'seen',
        'hits',
        'leaves',
        'root',
        'N',
        'pbar',
    )

    def __init__(self, chi, T=0.1):
        self.chi = chi
        self.T = T
        self.best_score = float('inf')
        self.best_ssa_path = None
        self.children = {}
        self.parents = {}
        self.seen = {}
        self.hits = 0
        self.leaves = None
        self.root = None
        self.N = None

    def setup(self, inputs, output, size_dict):
        """
        """
        if self.leaves is None:
            H = get_hypergraph(
                {1 << i: term
                 for i, term in enumerate(inputs)},
                output,
                size_dict,
                accel=False,
            )
            self.N = H.get_num_nodes()
            size0 = sum(map(H.node_size, H.nodes))

            root = Node(
                hg=H,
                ssa_path=(),
                size=size0,
                local_score=0,
                forward_score=size0,
            )

            self.add_node(root)
            self.seen[root.key] = size0
            self.root = root
            self.leaves = [root]
            self.simulate_node(root)

    def get_ssa_path(self):
        """Convert unique node identifiers to ssa.
        """
        ssa_path = []
        ssa = self.N
        ssa_lookup = {1 << i: i for i in range(ssa)}
        for i, j in self.best_ssa_path:
            ij = i | j
            ssa_lookup[ij] = ssa
            ssa += 1
            ssa_path.append((ssa_lookup[i], ssa_lookup[j]))
        return ssa_path

    def add_node(self, node):
        """
        """
        if node in self.children:

            # if (
            #     (node.forward_score >= self.best_score) or
            #     (node.forward_score) > self.seen[node.key]
            # ):
            #     self.kill(node)

            return

        hg = node.hg
        cnodes = set()

        for e, (i, j) in hg.edges.items():
            hg_next = hg.copy()

            hg_next.compress(self.chi,
                             hg_next.get_node(i) + hg_next.get_node(j))
            ij = hg_next.contract(i, j, node=i | j)

            dsize = hg_next.neighborhood_size([ij]) - hg.neighborhood_size(
                [i, j])
            new_size = node.size + dsize
            new_score = max(node.forward_score, new_size)
            if new_score >= self.best_score:
                # all subsequent paths with be worse - terminate
                continue

            new_ssa_path = node.ssa_path + ((i, j), )
            new_node = Node(
                hg=hg_next,
                ssa_path=new_ssa_path,
                size=new_size,
                local_score=dsize,
                forward_score=new_score,
            )

            key = new_node.key
            if new_score >= self.seen.get(key, float('inf')):
                self.hits += 1
                continue

            self.seen[key] = new_score
            cnodes.add(new_node)
            self.parents[new_node] = node

        self.children[node] = cnodes
        # if not cnodes:
        #     self.kill(node)

    def kill(self, node):
        """
        """
        self.children[node].clear()
        try:
            pnode = self.parents.pop(node)
            self.children[pnode].discard(node)
            self.maybe_prune(pnode)
        except KeyError:
            pass

    def maybe_prune(self, node):
        """
        """
        if not self.children[node]:
            self.kill(node)

    def simulate_node(self, node):
        """
        """
        # make sure we know node
        self.add_node(node)

        # greedily descend to bottom based on heuristic local_score
        while node.hg.get_num_nodes() > 1:

            # if all children are bad, kill this node
            cnodes = self.children[node]
            if not cnodes:
                # self.kill(node)
                return

            node = min(
                cnodes,
                key=lambda node: node.local_score - self.T * gumbel()
            )

            # make sure we know node
            self.add_node(node)

        # we've reached a fully contracted graph
        self.pbar.update()
        if node.backward_score < self.best_score:
            self.best_score = node.backward_score
            self.best_ssa_path = node.ssa_path
            self.pbar.set_description(
                f"lq:{len(self.leaves) if self.leaves is not None else None} "
                f"best:{math.log2(self.best_score):.2f} "
                f"hits:{self.hits} ")

        # back track upwards updating backward_score
        while True:
            try:
                pnode = self.parents[node]
                pnode.update(node)
                node = pnode
            except KeyError:
                break

    def explore(self):
        """
        """
        while self.leaves:
            # get the most favourable leaf node
            heapq.heapify(self.leaves)
            leaf = heapq.heappop(self.leaves)

            # simulate all its children
            for node in tuple(self.children[leaf]):
                self.simulate_node(node)
                if self.children[node]:
                    heapq.heappush(self.leaves, node)

            # if all children were bad then remove
            # self.maybe_prune(leaf)

    def __call__(self, inputs, output, size_dict):
        """
        """
        import tqdm
        self.pbar = tqdm.tqdm()
        self.setup(inputs, output, size_dict)

        try:
            self.explore()
        except KeyboardInterrupt:
            pass
        finally:
            self.pbar.close()

        return self.get_ssa_path()
