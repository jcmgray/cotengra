"""Simple hypergraph (and also linegraph) representations for simulating
contractions.
"""

import math
import itertools
import collections

from .plot import (
    plot_hypergraph,
)

from .utils import (
    compute_size_by_dict,
    prod,
    unique,
)


try:
    from cotengra.cotengra import HyperGraph as HyperGraphRust
except ImportError:
    HyperGraphRust = None


class HyperGraph:
    """Simple hypergraph builder and writer.

    Parameters
    ----------
    inputs : sequence of list[str] or dict[int, list[str]]
        The nodes. If given as a dict, the keys will be taken as the node
        enumeration rather than ``range(len(inputs))``.
    output : str, optional
        Output indices.
    size_dict : dict[str, int], optional
        Size of each index.

    Attributes
    ----------
    nodes : dict[int, list[str]]
        Mapping of node to the list of edges incident to it.
    edges : dict[str, list[int]]
        Mapping of hyper edges to list of nodes it is incident to.
    num_nodes : int
        The number of nodes.
    num_edges : int
        The number of hyper-edges.
    """

    __slots__ = (
        "inputs",
        "output",
        "size_dict",
        "nodes",
        "edges",
        "node_counter",
    )

    def __init__(self, inputs, output=None, size_dict=None):
        self.inputs = inputs
        self.output = [] if output is None else list(output)
        self.size_dict = {} if size_dict is None else dict(size_dict)

        if isinstance(inputs, dict):
            self.nodes = {int(k): tuple(v) for k, v in inputs.items()}
        else:
            self.nodes = dict(enumerate(map(tuple, inputs)))

        self.edges = {}
        for i, term in self.nodes.items():
            for e in term:
                self.edges[e] = (*self.edges.setdefault(e, ()), i)

        self.node_counter = self.num_nodes - 1

    def copy(self):
        """Copy this ``HyperGraph``."""
        new = object.__new__(self.__class__)
        new.inputs = self.inputs
        new.output = self.output
        new.size_dict = self.size_dict.copy()
        new.nodes = self.nodes.copy()
        new.edges = self.edges.copy()
        new.node_counter = self.node_counter
        return new

    @classmethod
    def from_edges(cls, edges, output=(), size_dict=()):
        self = cls.__new__(cls)
        self.edges = {}
        for e, e_nodes in edges.items():
            self.edges[e] = tuple(e_nodes)
        self.output = output
        self.size_dict = dict(size_dict)

        self.nodes = {}
        for e, e_nodes in self.edges.items():
            for i in e_nodes:
                self.nodes[i] = (*self.nodes.setdefault(i, ()), e)

        self.node_counter = self.num_nodes - 1

        return self

    def get_num_nodes(self):
        return len(self.nodes)

    @property
    def num_nodes(self):
        return len(self.nodes)

    def get_num_edges(self):
        return len(self.edges)

    @property
    def num_edges(self):
        return len(self.edges)

    def __len__(self):
        return self.num_nodes

    def edges_size(self, es):
        """Get the combined, i.e. product, size of all edges in ``es``."""
        return prod(map(self.size_dict.__getitem__, es))

    def bond_size(self, i, j):
        """Get the combined, i.e. product, size of edges shared by nodes ``i``
        and ``j``.
        """
        return self.edges_size(set(self.nodes[i]).intersection(self.nodes[j]))

    def node_size(self, i):
        """Get the size of the term represented by node ``i``."""
        return self.edges_size(self.nodes[i])

    def neighborhood_size(self, nodes):
        """Get the size of nodes in the immediate neighborhood of ``nodes``."""
        neighborhood = {
            nn
            for n in nodes
            for e in self.get_node(n)
            for nn in self.get_edge(e)
        }
        return sum(map(self.node_size, neighborhood))

    def contract_pair_cost(self, i, j):
        """Get the cost of contracting nodes ``i`` and ``j`` - the product of
        the dimensions of the indices involved.
        """
        return self.edges_size(set(self.get_node(i) + self.get_node(j)))

    def neighborhood_compress_cost(self, chi, nodes):
        region_edges = {e for n in nodes for e in self.get_node(n)}

        # group edges that are incident to the same set of nodes
        incidences = {}
        for e in region_edges:
            if e not in self.output:
                e_nodes = frozenset(self.get_edge(e))
                incidences.setdefault(e_nodes, []).append(e)

        # ignore intra-region bonds (assuming we are about to contract these)
        incidences.pop(frozenset(nodes), None)

        # compute the cost dominated by QR reductions onto bond
        C = 0
        for e_nodes, edges in incidences.items():
            da = self.edges_size(edges)

            if da > chi:
                # large multibond shared by e_nodes -> should compress
                for node in e_nodes:
                    # get outer edges and size
                    outer_edges = [
                        e for e in self.get_node(node) if e not in edges
                    ]
                    db = self.edges_size(outer_edges)

                    # estimate QR cost
                    da, db = sorted((da, db))
                    C += da**2 * db

        if C < 0:
            raise ValueError("Negative cost!?", C)

        return C

    def total_node_size(self):
        """Get the total size of all nodes."""
        return sum(map(self.node_size, self.nodes))

    def output_nodes(self):
        """Get the nodes with output indices."""
        return unique(i for e in self.output for i in self.edges[e])

    def neighbors(self, i):
        """Get the neighbors of node ``i``."""
        return unique(
            j for e in self.nodes[i] for j in self.edges[e] if (j != i)
        )

    def neighbor_edges(self, i):
        """Get the edges incident to all neighbors of node ``i``, (including
        its own edges).
        """
        return unique(
            itertools.chain.from_iterable(
                map(self.get_node, self.neighbors(i))
            )
        )

    def has_node(self, i):
        """Does this hypergraph have node ``i``?"""
        return i in self.nodes

    def get_node(self, i):
        """Get the edges node ``i`` is incident to."""
        return self.nodes[i]

    def get_edge(self, e):
        """Get the nodes edge ``e`` is incident to."""
        return self.edges[e]

    def has_edge(self, e):
        """Does this hypergraph have edge ``e``?"""
        return e in self.edges

    def next_node(self):
        """Get the next available node identifier."""
        # always increment to try and generate unique ids
        self.node_counter += 1
        # ... but also check node is valid
        while self.node_counter in self.nodes:
            self.node_counter += 1
        return self.node_counter

    def add_node(self, inds, node=None):
        """Add a node with ``inds``, and optional identifier ``node``. The
        identifier will be generated if not given and returned.
        """
        if node is None:
            node = self.next_node()
        inds = tuple(inds)
        self.nodes[node] = inds
        for e in inds:
            try:
                self.edges[e] += (node,)
            except KeyError:
                # if we just contracted a node with output index, can be empty
                self.edges[e] = (node,)
        return node

    def remove_node(self, i):
        """Remove node ``i`` from this hypergraph."""
        inds = self.nodes.pop(i)
        for e in inds:
            e_nodes = self.edges[e] = tuple(j for j in self.edges[e] if j != i)
            if not e_nodes:
                del self.edges[e]
        return inds

    def remove_edge(self, e):
        """Remove edge ``e`` from this hypergraph."""
        for i in self.edges[e]:
            self.nodes[i] = tuple(d for d in self.nodes[i] if d != e)
        del self.edges[e]

    def contract(self, i, j, node=None):
        """Combine node ``i`` and node ``j``."""
        inds_i = self.remove_node(i)
        inds_j = self.remove_node(j)
        inds_ij = unique(
            ind
            for ind in inds_i + inds_j
            # index will only still be here if its not only on i and j
            if (ind in self.edges) or (ind in self.output)
        )
        return self.add_node(inds_ij, node=node)

    def compress(self, chi, edges=None):
        """'Compress' multiedges, combining their size up to a maximum of
        ``chi``.
        """
        if edges is None:
            edges = self.edges

        # find edges which are incident to the same set of nodes
        incidences = collections.defaultdict(list)
        for e in unique(edges):
            if e not in self.output:
                nodes = frozenset(self.edges[e])
                incidences[nodes].append(e)

        for es in incidences.values():
            if len(es) > 1:
                # combine edges into first, capping size at `chi`
                new_size = self.edges_size(es)
                e_keep, *es_del = es
                for e in es_del:
                    self.remove_edge(e)
                self.size_dict[e_keep] = min(new_size, chi)

    def compute_contracted_inds(self, nodes):
        """Generate the output indices if one were to contract ``nodes``."""
        snodes = set(nodes)
        return unique(
            e
            for i in nodes
            for e in self.get_node(i)
            # keep index if it appears on any other nodes or in output
            if set(self.edges[e]) - snodes or e in self.output
        )

    def candidate_contraction_size(self, i, j, chi=None):
        """Get the size of the node created if ``i`` and ``j`` were contracted,
        optionally including the effect of first compressing bonds to size
        ``chi``.
        """
        # figure out the indices of the contracted nodes
        new_es = tuple(self.compute_contracted_inds((i, j)))

        if chi is None:
            return self.edges_size(new_es)

        incidences = collections.defaultdict(list)
        for e in new_es:
            # compressable indices -> those which will not be incident to the
            # exact same set of nodes
            contracted_neighbs = frozenset(
                i if k == j else k for k in self.edges[e]
            )
            incidences[contracted_neighbs].append(e)

        # each group of compressed inds maxes out at size `chi`
        return prod(
            min(chi, self.edges_size(es)) for es in incidences.values()
        )

    def all_shortest_distances(
        self,
        nodes=None,
    ):
        if nodes is None:
            nodes = set(self.nodes)
        elif not isinstance(nodes, set):
            nodes = set(nodes)

        # build lazily
        neighbors_map = {}

        n = len(nodes)
        ncomb = n * (n - 1) // 2
        distances = {}

        visitors = {node: {node} for node in nodes}
        for d in range(1, self.num_nodes):
            any_change = False
            previous_visitors = {k: v.copy() for k, v in visitors.items()}
            for i, ivis in previous_visitors.items():
                try:
                    ineighbs = neighbors_map[i]
                except KeyError:
                    ineighbs = neighbors_map[i] = tuple(self.neighbors(i))

                for j in ineighbs:
                    try:
                        visitors[j] |= ivis
                    except KeyError:
                        visitors[j] = ivis.copy()
                        # won't get caught in the later any_change check
                        any_change = True

            for i in nodes:
                for j in visitors[i] - previous_visitors[i]:
                    if (i < j) and (j in nodes):
                        distances[i, j] = d
                    any_change = True

            if not any_change:
                # also ened to check non target nodes
                any_change |= any(
                    ivis != visitors[i]
                    for i, ivis in previous_visitors.items()
                )

            if (len(distances) == ncomb) or (not any_change):
                break

        return distances

    def all_shortest_distances_condensed(
        self,
        nodes=None,
    ):
        if nodes is None:
            nodes = tuple(self.nodes)
        distances = self.all_shortest_distances(nodes=nodes)

        default_distance = 10 * self.num_nodes

        condensed = []
        for i, ni in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                key = (ni, nj) if ni < nj else (nj, ni)
                condensed.append(distances.get(key, default_distance))

        return condensed

    def simple_distance(self, region, p=2):
        """Compute a simple distance metric from nodes in ``region`` to all
        others. Unlike graph distance, relative connectedness is taken into
        account.
        """
        region = set(region)
        distances = {i: 0 for i in region}
        queue = list(region)
        surface = collections.defaultdict(lambda: 0)

        for d in itertools.count(1):
            surface.clear()

            while queue:
                i = queue.pop()
                for j in self.neighbors(i):
                    if j not in region:
                        surface[j] += 1

            for j, c in surface.items():
                region.add(j)
                queue.append(j)
                distances[j] = d + (1 / c) ** p

            if not queue:
                break

        return dict_affine_renorm(distances)

    def simple_closeness(self, p=0.75, mu=0.5):
        """Compute a rough hypergraph 'closeness'.

        Parameters
        ----------
        p : float, optional
            Once any node has had ``H.num_nodes**p`` visitors terminate. Set
            greater than 1.0 for no limit (slower).
        mu : float, optional
            Let the visitor score decay with this power. The higher this is,
            the more local connectivity is favored.

        Returns
        -------
        scores : dict[int, float]
            The simple hypergraph closenesses - higher being more central.
        """
        sz_stop = self.num_nodes**p
        should_stop = False

        # which nodes have reached which other nodes (bitmap set)
        visitors = {i: 1 << i for i in self.nodes}

        # store the number of unique visitors - the change is this each step
        #    is the number of new shortest paths of length ``d``
        num_visitors = {i: 1 for i in self.nodes}

        # the total weighted score - combining num visitors and their distance
        scores = {i: 0.0 for i in self.nodes}

        # pre-cache the lists of neighbors
        neighbors = {i: list(self.neighbors(i)) for i in self.nodes}

        # at each iteration expand all nodes visitors to their neighbors
        for d in self.nodes:
            # do a parallel update
            previous_visitors = visitors.copy()

            for i in self.nodes:
                for j in neighbors[i]:
                    visitors[i] |= previous_visitors[j]

                # visitors are worth less the further they've come from
                new_nv = popcount(visitors[i])
                scores[i] += (new_nv - num_visitors[i]) / (d + 1) ** mu
                num_visitors[i] = new_nv

                # once any node has reached a certain number of visitors stop
                should_stop |= new_nv >= sz_stop

            if should_stop:
                break

        # finally rescale the values between 0.0 and 1.0
        return dict_affine_renorm(scores)

    def simple_centrality(self, r=None, smoothness=2, **closeness_opts):
        """A simple algorithm for large hypergraph centrality. First we find
        a rough closeness centrality, then relax / smooth this by nodes
        iteratively radiating their centrality to their neighbors.

        Parameters
        ----------
        r : None or int, optional
            Number of iterations. Defaults to
            ``max(10, int(self.num_nodes**0.5))``.
        smoothness : float, optional
            The smoothness. In conjunction with a high value of ``r`` this will
            create a smooth gradient from one of the hypergraph to the other.
        closeness_opts
            Supplied to ``HyperGraph.simple_closeness`` as the starting point.

        Returns
        -------
        dict[int, float]
        """
        # take a rough closeness as the starting point
        c = self.simple_closeness(**closeness_opts)

        # pre-cache the lists of neighbors
        neighbors = {i: list(self.neighbors(i)) for i in self.nodes}

        if r is None:
            # take the propagation time as sqrt hypergraph size
            r = max(10, int(self.num_nodes**0.5))

        for _ in range(r):
            # do a parallel update
            previous_c = c.copy()

            # spread the centrality of each node into its neighbors
            for i in self.nodes:
                ci = previous_c[i]
                for j in neighbors[i]:
                    c[j] += smoothness * ci / r

            # then rescale all the values between 0.0 and 1.0
            c = dict_affine_renorm(c)

        return c

    def compute_loops(self, start=None, max_loop_length=None):
        """Generate all loops up to a certain length in this hypergraph.

        Parameters
        ----------
        start : sequence of int, optional
            Only generate loops including these nodes, defaults to all.
        max_loop_length : None or int, optional
            The maximum loop length to search for. If ``None``, then this is
            set automatically by the length of the first loop found.

        Yields
        ------
        loop : tuple[int]
            A set of nodes that form a loop.
        """
        if start is None:
            start = self.nodes

        # start paths beginning at every node
        queue = [(i,) for i in start]
        # cache neighbors for speed
        neighbors = {}

        seen = set()
        while queue:
            # consider all the ways to extend each path
            path = queue.pop(0)

            jf = path[-1]
            try:
                j_neighbs = neighbors[jf]
            except KeyError:
                j_neighbs = neighbors[jf] = tuple(self.neighbors(jf))

            for j in j_neighbs:
                i0 = path[0]
                # check for valid loop ...
                if (
                    # is not trivial
                    (len(path) > 2)
                    and
                    # ends where it starts
                    (j == i0)
                    and
                    # and is not just a cyclic permutation of existing loop
                    (frozenset(path) not in seen)
                ):
                    yield tuple(sorted(path))
                    seen.add(frozenset(path))
                    if max_loop_length is None:
                        # automatically set the max loop length
                        max_loop_length = len(path) + 1

                # path hits itself too early
                elif j in path:
                    continue

                # keep extending path, but only if
                elif (
                    # we haven't found any loops yet
                    (max_loop_length is None)
                    or
                    # or this loops is short
                    (len(path) < max_loop_length)
                ):
                    queue.append(path + (j,))

    def get_laplacian(self):
        """Get the graph Laplacian."""
        import numpy as np

        lp = np.zeros((self.num_nodes, self.num_nodes))

        for i, term in self.nodes.items():
            lp[i, i] = len(term)

        for i, j in self.edges.values():
            lp[i, j] = lp[j, i] = -1

        return lp

    def get_resistance_distances(self):
        """Get the resistance distance between all nodes of the raw graph."""
        import numpy as np

        lp = self.get_laplacian()
        lp += 1 / self.num_nodes
        lp = np.linalg.inv(lp)
        d = np.array(np.diag(lp))  # needs to be copy
        lp *= -2
        lp += d.reshape(1, -1)
        lp += d.reshape(-1, 1)

        return lp

    def resistance_centrality(self, rescale=True):
        """Compute the centrality in terms of the total resistance distance
        to all other nodes.
        """
        rd = self.get_resistance_distances()
        cents = dict(enumerate(-rd.sum(axis=1)))
        if rescale:
            cents = dict_affine_renorm(cents)
        return cents

    def to_networkx(H, as_tree_leaves=False):
        """Convert to a networkx Graph, with hyperedges represented as nodes.

        Parameters
        ----------
        as_tree_leaves : bool, optional
            If true, then the nodes are converted to 'tree leaf' form, i.e.
            map node ``i`` to ``frozenset([i])``, to match the nodes in a
            ``ContractionTree``.
        """
        import networkx as nx

        # any_hyper is just a custom attribute
        G = nx.Graph(any_hyper=False)
        for ix, nodes in H.edges.items():
            if as_tree_leaves:
                nodes = [frozenset([node]) for node in nodes]

            output = ix in H.output

            if len(nodes) == 2 and (not output):
                # regular edge
                if not G.has_edge(*nodes):
                    G.add_edge(*nodes, ind=ix, hyperedge=False, output=False)
                else:
                    multi = G.edges[nodes].setdefault("multi", {})
                    multi.setdefault("inds", []).append(ix)
            else:
                # hyperedge
                G.graph["any_hyper"] = True
                output = ix in H.output

                if output:
                    hyperedge = len(nodes) != 1
                else:
                    hyperedge = True

                G.add_node(ix, ind=ix, hyperedge=hyperedge, output=output)
                for nd in nodes:
                    G.add_edge(
                        ix, nd, ind=ix, hyperedge=hyperedge, output=output
                    )

                if hyperedge and output:
                    # attach extra dummy output node to hyperedge center
                    G.add_node(
                        f"__output__{ix}",
                        ind=ix,
                        hyperedge=False,
                        output=output,
                    )
                    G.add_edge(
                        ix,
                        f"__output__{ix}",
                        ind=ix,
                        hyperedge=hyperedge,
                        output=output,
                    )

        for nd in G.nodes:
            G.nodes[nd].setdefault("hyperedge", False)

        return G

    def compute_weights(
        self,
        weight_edges="const",
        weight_nodes="const",
    ):
        winfo = {}

        winfo["node_weights"] = tuple(
            calc_node_weight(term, self.size_dict, weight_nodes)
            for term in self.nodes.values()
        )

        winfo["edge_list"] = tuple(self.edges)
        winfo["edge_weight_map"] = {
            e: calc_edge_weight(e, self.size_dict, weight_edges)
            for e in winfo["edge_list"]
        }
        winfo["edge_weights"] = tuple(
            winfo["edge_weight_map"][e] for e in winfo["edge_list"]
        )

        winfo["has_edge_weights"] = weight_edges in ("log", "linear")
        winfo["has_node_weights"] = weight_nodes in ("log", "linear")
        winfo["fmt"] = {
            (False, False): "",
            (False, True): "10",
            (True, False): "1",
            (True, True): "11",
        }[winfo["has_edge_weights"], winfo["has_node_weights"]]

        return winfo

    plot = plot_hypergraph

    def __repr__(self):
        return f"<HyperGraph(|V|={self.num_nodes}, |E|={self.num_edges})>"


def get_hypergraph(inputs, output=None, size_dict=None, accel=False):
    """Single entry-point for creating a, possibly accelerated, HyperGraph."""
    if accel == "auto":
        accel = HyperGraphRust is not None

    if accel:
        if not isinstance(inputs, dict):
            inputs = {i: list(term) for i, term in enumerate(inputs)}
        if not isinstance(output, list):
            output = [] if output is None else list(output)
        if not isinstance(size_dict, dict):
            size_dict = {} if size_dict is None else dict(size_dict)
        return HyperGraphRust(inputs, output, size_dict)

    return HyperGraph(inputs, output, size_dict)


def calc_edge_weight(ix, size_dict, scale="log"):
    if scale in ("const", None, False):
        return 1

    w = size_dict[ix]

    if scale == "linear":
        w = 1000 * w
    elif scale == "log":
        w = int(1000 * math.log2(w)) + 1
    elif scale == "exp":
        w = 2**w

    return int(w)


def calc_edge_weight_float(ix, size_dict, scale="log"):
    if scale in ("const", None, False):
        return 1.0

    w = size_dict[ix]

    if scale == "linear":
        w = float(w)
    elif scale == "log":
        w = math.log2(w)
    elif scale == "exp":
        w = 2**w

    return w


def calc_node_weight(term, size_dict, scale="linear"):
    if scale in ("const", None, False):
        return 1

    w = compute_size_by_dict(term, size_dict)

    # scale up by a thousand so we can add small integer jitter
    if scale == "linear":
        w = 1000 * w
    elif scale == "log":
        w = 1000 * math.log2(w)
    elif scale == "exp":
        w = 2**w

    return int(w)


def calc_node_weight_float(term, size_dict, scale="linear"):
    if scale in ("const", None, False):
        return 1.0

    w = compute_size_by_dict(term, size_dict)

    # scale up by a thousand so we can add small integer jitter
    if scale == "linear":
        w
    elif scale == "log":
        w = math.log2(w)
    elif scale == "exp":
        w = 2**w

    return w


class LineGraph:
    """Very simple line-graph builder and file writer."""

    def __init__(self, inputs, output):
        self.inputs = inputs
        self.nodes = tuple(set.union(*inputs))
        self.nodemap = {ix: i for i, ix in enumerate(self.nodes)}

        # num nodes in dual = num edges in real graph
        self.number_of_nodes = len(self.nodemap)

        self.edges = []
        for term in inputs:
            for ix1, ix2 in itertools.combinations(term, 2):
                self.edges.append((self.nodemap[ix1], self.nodemap[ix2]))
        for ix1, ix2 in itertools.combinations(output, 2):
            self.edges.append((self.nodemap[ix1], self.nodemap[ix2]))
        self.number_of_edges = len(self.edges)

    def to_gr_str(self):
        ls = [f"p tw {self.number_of_nodes} {self.number_of_edges}"]
        for i, j in self.edges:
            ls.append(f"{i + 1} {j + 1}")
        return "\n".join(ls)

    def to_gr_file(self, fname):
        contents = self.to_gr_str()
        with open(fname, "w") as f:
            f.write(contents)

    def to_cnf_str(self):
        ls = [f"p cnf {self.number_of_nodes} {self.number_of_edges}"]
        for i, j in self.edges:
            ls.append(f"{i + 1} {j + 1} 0")
        return "\n".join(ls)

    def to_cnf_file(self, fname):
        contents = self.to_cnf_str()
        with open(fname, "w") as f:
            f.write(contents)


# best: use built in
if hasattr(int, "bit_count"):

    def popcount(x):
        return x.bit_count()

else:
    # second best, gmpy2 is installed
    try:
        from gmpy2 import popcount

    except ImportError:
        # finally, use string method

        def popcount(x):
            return bin(x).count("1")


def dict_affine_renorm(d):
    dmax = max(d.values())
    dmin = min(d.values())
    if dmax == dmin:
        dmin = 0
        if dmax == 0.0:
            dmax = 1.0
    return {k: (v - dmin) / (dmax - dmin) for k, v in d.items()}
