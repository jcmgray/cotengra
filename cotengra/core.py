import math
import random
import operator
import itertools
import functools
import collections

from opt_einsum.helpers import compute_size_by_dict, flop_count
from opt_einsum.paths import get_path_fn


def calc_edge_weight(ix, size_dict, scale='log'):

    if scale in ('const', None, False):
        return 1

    w = size_dict[ix]

    if scale == 'linear':
        w = 1000 * w
    elif scale == 'log':
        w = int(1000 * math.log2(w)) + 1
    elif scale == 'exp':
        w = 2**w

    return int(w)


def calc_edge_weight_float(ix, size_dict, scale='log'):

    if scale in ('const', None, False):
        return 1.0

    w = size_dict[ix]

    if scale == 'linear':
        w
    elif scale == 'log':
        w = math.log2(w)
    elif scale == 'exp':
        w = 2**w

    return w


def calc_node_weight(term, size_dict, scale='linear'):
    if scale in ('const', None, False):
        return 1

    w = compute_size_by_dict(term, size_dict)

    # scale up by a thousand so we can add small integer jitter
    if scale == 'linear':
        w = 1000 * w
    elif scale == 'log':
        w = 1000 * math.log2(w)
    elif scale == 'exp':
        w = 2**w

    return int(w)


def calc_node_weight_float(term, size_dict, scale='linear'):
    if scale in ('const', None, False):
        return 1.0

    w = compute_size_by_dict(term, size_dict)

    # scale up by a thousand so we can add small integer jitter
    if scale == 'linear':
        w
    elif scale == 'log':
        w = math.log2(w)
    elif scale == 'exp':
        w = 2**w

    return w


def jitter(x, strength):
    return x * (1 + strength * random.expovariate(1.0))


def jitter_dict(d, strength):
    return {k: jitter(v, strength) for k, v in d.items()}


def separate(xs, blocks):
    """Partition ``xs`` into ``n`` different list based on the corresponding
    labels in ``blocks``.
    """
    sorter = collections.defaultdict(list)
    for x, b in zip(xs, blocks):
        sorter[b].append(x)
    x_b = list(sorter.items())
    x_b.sort()
    return [x[1] for x in x_b]


def is_valid_node(node):
    """Check ``node`` is of type frozenset[frozenset[str]].
    """
    try:
        if not isinstance(node, frozenset):
            return False
        term = next(iter(node))
        if not isinstance(term, frozenset):
            return False
        index = next(iter(term))
        if not isinstance(index, str):
            return False
        return True
    except TypeError:
        return False


def symm_diff(xs):
    return functools.reduce(operator.xor, xs)


class ContractionTree:
    """Binary tree representing a tensor contraction order. Currently useful
    for:

        1. Building paths with hybrid methods - e.g. community + greedy/dp.
        2. Keeping dynamic track of path cost for early pruning.
        3. Checking equivalency of two paths.
        4. Making sure contraction of tree is depth first

    Might in fugure be useful for:

        5. Grafting the best bits of many trees into each other.
        6. Reordering paths to e.g. put as many constant contractions first.
        7. Plotting and exploring what contraction paths are actually doing.

    Parameters
    ----------
    inputs : sequence of str
        The list of input tensor's indices.
    output : str
        The output indices.
    size_dict : dict[str, int]
        The size of each index.
    track_childless : bool, optional
        Whether to dynamically keep track of which nodes are childless. Useful
        if you are 'divisively' building the tree.
    track_size : bool, optional
        Whether to dynamically keep track of the largest tensor so far. If
        ``False`` You can still compute this once the tree is complete.
    track_flops : bool, optional
        Whether to dynamically keep track of the total number of flops. If
        ``False`` You can still compute this once the tree is complete.

    Properties
    ----------
    info : dict[frozenset[frozenset[str]], dict]
        Information about the tree nodes. The key is the set of inputs (itself
        each a set of indices) the node contains. Or in other words, the
        subgraph of the node. The value is a dictionary to cache information
        about effective 'leg' indices, size, flops of formation etc.
    children : dict[frozenset[frozenset[str]],
                    tuple[frozenset[frozenset[str]]]
        Mapping of each node to two children.
    """

    def __init__(self, inputs, output, size_dict,
                 track_childless=False, track_size=False, track_flops=False):

        self.inputs = tuple(map(frozenset, inputs))
        self.N = len(self.inputs)
        self.size_dict = size_dict

        # mapping of parents to children - the core binary tree object
        self.children = {}

        # information about all the nodes
        self.info = {}

        # ... which we can fill in already for final / top node i.e.
        # the collection of all nodes
        self.root = frozenset(self.inputs)
        self.output = frozenset(output)
        self.add_node(self.root)
        self.info[self.root]['legs'] = self.output
        self.info[self.root]['size'] = compute_size_by_dict(self.output,
                                                            size_dict)

        # whether to keep track of dangling nodes/subgraphs
        self.track_childless = track_childless
        if self.track_childless:
            # the set of dangling nodes
            self.childless = {self.root}

        # running largest_intermediate and total flops
        self._track_flops = track_flops
        if track_flops:
            self._flops = 0
        self._track_size = track_size
        if track_size:
            self._size = 0

    @classmethod
    def from_path(cls, inputs, output, size_dict, *,
                  path=None, ssa_path=None, **kwargs):
        """
        """
        if int(path is None) + int(ssa_path is None) != 1:
            raise ValueError("Exactly one of ``path`` or ``ssa_path`` must be "
                             "supplied.")

        if ssa_path is not None:
            path = ssa_path

        contree = cls(inputs, output, size_dict, **kwargs)
        terms = [frozenset([frozenset(i)]) for i in inputs]

        for p in path:
            if ssa_path is not None:
                merge = [terms[i] for i in p]
            else:
                merge = [terms.pop(i) for i in sorted(p, reverse=True)]
            terms.append(contree.contract(merge))

        return contree

    @classmethod
    def from_pathinfo(cls, path_info, **kwargs):
        """
        """
        return cls.from_path(inputs=path_info.input_subscripts.split(','),
                             output=path_info.output_subscript,
                             size_dict=path_info.size_dict,
                             path=path_info.path, **kwargs)

    @classmethod
    def from_eq(cls, eq, size_dict, **kwargs):
        """
        """
        lhs, output = eq.split('->')
        inputs = lhs.split(',')
        return cls(inputs, output, size_dict, **kwargs)

    @classmethod
    def from_edge_path(cls, edge_path, inputs, output, size_dict, **kwargs):
        """Create a binary contraction tree from a edge elimination ordering.
        """
        contree = cls(inputs, output, size_dict, **kwargs)
        terms = [frozenset([frozenset(i)]) for i in inputs]

        for e in edge_path:

            # filter out the subgraph induced by edge `e` (generally a pair)
            new_terms, merge = [], []
            for term in terms:
                (merge if e in frozenset.union(*term)
                 else new_terms).append(term)

            # contract the subgraph
            if merge:
                terms = new_terms + [contree.contract(merge)]

        # make sure we are generating a full contraction tree
        nt = len(terms)
        if nt > 1:
            # this seems to happen when the initial contraction contains a
            # scalar? Or disconnected subgraphs?
            import warnings
            warnings.warn(
                f"Ended up with {nt} terms - contracting all remaining.")
            contree.contract(terms)

        return contree

    def add_node(self, node, check=False):
        if check:
            if len(self.info) > 2 * self.N - 1:
                raise ValueError("There are too many children already.")
            if len(self.children) > self.N - 1:
                raise ValueError("There are too many branches already.")
            if not is_valid_node(node):
                raise ValueError("{} is not a valid node - should be "
                                 "frozenset[frozenset[str]].".format(node))

        if node not in self.info:
            self.info[node] = {}

    def get_keep(self, node):
        """Get a set of at least the indices that should be explicitly kept if
        they appear on ``node`` (or below).
        """
        try:
            keep = self.info[node]['keep']
        except KeyError:
            keep = frozenset.union(self.output, *(self.root - node))
            self.info[node]['keep'] = keep
        return keep

    def get_legs(self, node, sub_legs=None):
        """Get the effective 'outer' indices for the collection of tensors
        in ``node``.
        """
        try:
            legs = self.info[node]['legs']
        except KeyError:
            if len(node) == 1:
                legs = next(iter(node))
            else:
                # legs = symm_diff(self.get_legs(x)
                #                  for x in self.children[node])
                try:
                    involved = self.get_involved(node, sub_legs=sub_legs)
                except KeyError:
                    involved = frozenset.union(*node)
                keep = self.get_keep(node)
                legs = involved & keep
            self.info[node]['legs'] = legs
        return legs

    def get_involved(self, node, sub_legs=None):
        """Get all the indices involved in the formation of subgraph ``node``.
        """
        try:
            involved = self.info[node]['involved']
        except KeyError:
            if len(node) == 1:
                involved = frozenset()
            else:
                if sub_legs is None:
                    sub_legs = map(self.get_legs, self.children[node])
                involved = frozenset.union(*sub_legs)
            self.info[node]['involved'] = involved
        return involved

    def get_removed(self, node):
        """Get the indices that will be removed by the creation of ``node``.
        """
        try:
            removed = self.info[node]['removed']
        except KeyError:
            removed = self.get_involved(node) - self.get_legs(node)
            self.info[node]['removed'] = removed
        return removed

    def get_size(self, node):
        """Get the tensor size of ``node``.
        """
        try:
            size = self.info[node]['size']
        except KeyError:
            size = compute_size_by_dict(self.get_legs(node), self.size_dict)
            self.info[node]['size'] = size
        return size

    def get_flops(self, node):
        """Get the FLOPs for the pairwise contraction that will create
        ``node``.
        """
        try:
            flops = self.info[node]['flops']
        except KeyError:
            if len(node) == 1:
                flops = 0
            else:
                involved = self.get_involved(node)
                removed = self.get_removed(node)
                flops = flop_count(involved, removed, 2, self.size_dict)
            self.info[node]['flops'] = flops
        return flops

    def total_flops(self):
        """Sum the flops contribution from every node in the tree.
        """
        if self._track_flops:
            return self._flops

        flops = 0
        queue = [self.root]
        while queue:
            node = queue.pop()
            flops += self.get_flops(node)
            if len(node) > 1:
                queue.extend(self.children[node])

        return flops

    def max_size(self):
        """Compute the maximum tensor size of the path.
        """
        if self._track_size:
            return self._size

        size = 0
        queue = [self.root]
        while queue:
            node = queue.pop()
            size = max(size, self.get_size(node))
            if len(node) > 1:
                queue.extend(self.children[node])

        return size

    def contract_pair(self, x, y, check=False):
        """
        """
        parent = x | y

        # make sure info entries exist for all (default dict)
        for node in (x, y, parent):
            self.add_node(node, check=check)

        # enforce left ordering of 'heaviest' subtrees
        nx, ny = len(x), len(y)
        # deterministcally break ties
        hx, hy = hash(x), hash(y)

        if (nx, hx) > (ny, hy):
            lr = (x, y)
        else:
            lr = (y, x)
        self.children[parent] = lr

        if self.track_childless:
            self.childless.discard(parent)
            if x not in self.children and nx > 1:
                self.childless.add(x)
            if y not in self.children and ny > 1:
                self.childless.add(y)

        if self._track_size:
            self._size = max(self._size, self.get_size(parent))
        if self._track_flops:
            self._flops += self.get_flops(parent)

        return parent

    def contract(self, nodes, optimize='auto-hq', check=False):
        """
        """
        if len(nodes) == 1:
            return next(iter(nodes))

        if len(nodes) == 2:
            return self.contract_pair(*nodes, check=check)

        # create the bottom and top nodes
        grandparent = frozenset.union(*nodes)
        self.add_node(grandparent, check=check)
        for node in nodes:
            self.add_node(node, check=check)

        # if more than two nodes need to find the path to fill in between
        #         \
        #         GN             <- 'grandparent'
        #        /  \
        #      ?????????
        #    ?????????????       <- to be filled with 'temp nodes'
        #   /  \    /   / \
        #  N0  N1  N2  N3  N4    <- ``nodes``, or, subgraphs
        #  /    \  /   /    \
        path_inputs = [set(self.get_legs(x)) for x in nodes]
        path_output = set(self.get_legs(grandparent))
        path_fn = get_path_fn(optimize)
        path = path_fn(path_inputs, path_output, self.size_dict)

        # now we have path create the nodes in between
        temp_nodes = list(nodes)
        for p in path:
            to_contract = [
                frozenset(temp_nodes.pop(i)) for i in sorted(p, reverse=True)
            ]
            temp_nodes.append(
                self.contract(to_contract, optimize=optimize, check=check)
            )

        parent, = temp_nodes

        if check:
            # final remaining temp input should be the 'grandparent'
            assert parent == grandparent

        return parent

    def is_complete(self):
        """Check every node has two children, unless it is a leaf.
        """
        too_many_nodes = len(self.info) > 2 * self.N - 1
        too_many_branches = len(self.children) > self.N - 1

        if too_many_nodes or too_many_branches:
            raise ValueError("Contraction tree seems to be over complete!")

        queue = [self.root]
        while queue:
            x = queue.pop()
            if len(x) == 1:
                continue
            try:
                queue.extend(self.children[x])
            except KeyError:
                return False

        return True

    def traverse(self):
        """Generate, in order, all the node merges in this tree. Non-recursive!

        Returns
        -------
        generator[tuple[frozenset[frozenset[str]]]]
            The bottom up ordered sequence of tree merges, each a
            tuple of ``(parent, left_child, right_child)``.
        """
        queue = [self.root]
        ready = {frozenset([term]) for term in self.inputs}

        while queue:
            node = queue[-1]
            l, r = self.children[node]

            if (l in ready) and (r in ready):
                ready.add(queue.pop())
                yield node, l, r
                continue

            if r not in ready:
                queue.append(r)
            if l not in ready:
                queue.append(l)

    def flat_tree(self):
        """Create a nested tuple representation of the contraction tree like::

            ((0, (1, 2)), ((3, 4), ((5, (6, 7)), (8, 9))))

        Such that the contraction will progress like::

            ((0, (1, 2)), ((3, 4), ((5, (6, 7)), (8, 9))))
            ((0, 12), (34, ((5, 67), 89)))
            (012, (34, (567, 89)))
            (012, (34, 56789))
            (012, 3456789)
            0123456789

        Where each integer represents a leaf (i.e. frozenset[str]).
        """
        tups = {frozenset([x]): i for i, x in enumerate(self.inputs)}

        for parent, l, r in self.traverse():
            tups[parent] = tups[l], tups[r]

        return tups[self.root]

    def get_leaves_ordered(self):
        """Return the list of leaves as ordered by the contraction tree.

        Returns
        -------
        tuple[frozenset[str]]
        """
        if not self.is_complete():
            raise ValueError("Can't order the leaves until tree is complete.")

        return tuple(
            nd for nd in itertools.chain.from_iterable(self.traverse())
            if len(nd) == 1
        )

    def path(self):
        """Generate a standard path from the contraction tree.
        """
        path = []
        terms = [frozenset([k]) for k in self.inputs]

        for parent, l, r in self.traverse():
            i, j = sorted((terms.index(l), terms.index(r)))
            terms.pop(j)
            terms.pop(i)
            path.append((i, j))
            terms.append(parent)

        return tuple(path)

    def ssa_path(self):
        """Generate a ssa path from the contraction tree.
        """
        ssa_path = []
        pos = {frozenset([x]): i for i, x in enumerate(self.inputs)}

        for parent, l, r in self.traverse():
            i, j = sorted((pos[l], pos[r]))
            ssa_path.append((i, j))
            pos[parent] = len(ssa_path) + self.N - 1

        return tuple(ssa_path)

    def __repr__(self):
        s = "<ContractionTree(N={}, branches={}, complete={})>"
        return s.format(self.N, len(self.children), self.is_complete())


class PartitionTreeBuilder:
    """Function wrapper that takes a function that partitions graphs and
    uses it to build a contraction tree. ``partition_fn`` should have
    signature:

        def partition_fn(inputs, output, size_dict,
                         weight_nodes, weight_edges, **kwargs):
            ...
            return membership

    Where ``weight_nodes`` and ``weight_edges`` decsribe how to weight the
    nodes and edges of the graph respectively and ``membership`` should be a
    list of integers of length ``len(inputs)`` labelling which partitition
    each input node should be put it.
    """

    def __init__(self, partition_fn):
        self.partition_fn = partition_fn

    def __call__(self, inputs, output, size_dict, random_strength=0.01,
                 weight_nodes='constant', weight_edges='log',
                 cutoff=10, parts=2, parts_decay=0.5,
                 sub_optimize='auto', super_optimize='auto-hq',
                 check=False, **kwargs):

        tree = ContractionTree(inputs, output, size_dict,
                               track_flops=True,
                               track_childless=True)
        rand_size_dict = jitter_dict(size_dict, random_strength)

        while tree.childless:
            subgraph = tuple(next(iter(tree.childless)))
            subsize = len(subgraph)

            # skip straight to better method
            if subsize <= cutoff:
                tree.contract([frozenset([x]) for x in subgraph],
                              optimize=sub_optimize, check=check)
                continue

            # let the target number of communities depend on subgraph size
            sub_parts = max(int((subsize / tree.N)**parts_decay * parts), 2)

            # partition! get community membership list e.g.
            # [0, 0, 1, 0, 1, 0, 0, 2, 2, ...]
            membership = self.partition_fn(
                subgraph, output, rand_size_dict,
                weight_nodes=weight_nodes, weight_edges=weight_edges,
                parts=sub_parts, **kwargs,
            )

            # divide subgraph up e.g. if we enumerate the subgraph index sets
            # (0, 1, 2, 3, 4, 5, 6, 7, 8, ...) ->
            # ({0, 1, 3, 5, 6}, {2, 4}, {7, 8})
            new_subgs = tuple(map(frozenset, separate(subgraph, membership)))

            if len(new_subgs) == 1:
                # no communities found - contract all remaining
                tree.contract([frozenset([x]) for x in subgraph],
                              optimize=sub_optimize, check=check)
                continue

            tree.contract(new_subgs, optimize=super_optimize, check=check)

        if check:
            assert tree.is_complete()

        return tree

    def trial_fn(self, inputs, output, size_dict, **kwargs):
        tree = self(inputs, output, size_dict, **kwargs)
        return {'tree': tree, 'ssa_path': tree.ssa_path(),
                'flops': tree.total_flops(), 'size': tree.max_size()}


class LineGraph:
    """Very simple line-graph builder and file writer.
    """

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
        with open(fname, 'w') as f:
            f.write(contents)

    def to_cnf_str(self):
        ls = [f"p cnf {self.number_of_nodes} {self.number_of_edges}"]
        for i, j in self.edges:
            ls.append(f"{i + 1} {j + 1} 0")
        return "\n".join(ls)

    def to_cnf_file(self, fname):
        contents = self.to_cnf_str()
        with open(fname, 'w') as f:
            f.write(contents)


class HyperGraph:
    """Simple hypergraph builder and writer. For compatibility with KaHyPar
    vertices are enumerated from 1.
    """

    def __init__(self, inputs, output, size_dict,
                 weight_edges='log', weight_nodes='constant',
                 fuse_output_inds=True):

        self.node_weights = []
        self.edgemap = collections.defaultdict(list)

        if output and fuse_output_inds:
            nodes = itertools.chain(inputs, [output])
        else:
            nodes = inputs

        for i, term in enumerate(nodes):
            for ix in term:
                self.edgemap[ix].append(i)
            self.node_weights.append(
                calc_node_weight(term, size_dict, weight_nodes))

        self.edge_list = tuple(self.edgemap)
        self.edge_weight_map = {
            e: calc_edge_weight(e, size_dict, weight_edges)
            for e in self.edge_list}
        self.edge_weights = [self.edge_weight_map[e] for e in self.edge_list]

        self.num_vertices = len(self.node_weights)
        self.num_edges = len(self.edge_list)

        self.has_edge_weights = weight_edges in ('log', 'linear')
        self.has_node_weights = weight_nodes in ('log', 'linear')
        self.fmt = {
            (False, False): "",
            (False, True): "10",
            (True, False): "1",
            (True, True): "11",
        }[self.has_edge_weights, self.has_node_weights]

    def to_hmetis_str(self):
        lns = [f"{self.num_edges} {self.num_vertices} {self.fmt}"]

        for e in self.edge_list:
            ln = " ".join(str(v + 1) for v in self.edgemap[e])
            if self.has_edge_weights:
                ln = f"{self.edge_weight_map[e]} {ln}"
            lns.append(ln)

        if self.has_node_weights:
            for v in self.node_weights:
                lns.append(str(v))

        return "\n".join(lns)

    def to_hmetis_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.to_hmetis_str())

    def to_sparse(self):
        hyperedge_indices = []
        hyperedges = []
        for e in self.edge_list:
            hyperedge_indices.append(len(hyperedges))
            hyperedges.extend(self.edgemap[e])
        hyperedge_indices.append(len(hyperedges))
        return hyperedge_indices, hyperedges

    def __repr__(self):
        return f"<HyperGraph(|V|={self.num_vertices}, |E|={self.num_edges})>"
