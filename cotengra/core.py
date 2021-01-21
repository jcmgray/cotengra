import math
import random
import warnings
import itertools
import functools
import collections

try:
    from cytoolz import groupby, interleave
except ImportError:
    from toolz import groupby, interleave

from opt_einsum.helpers import compute_size_by_dict, flop_count
from opt_einsum.paths import get_path_fn, DynamicProgramming

from .utils import MaxCounter, oset
from .parallel import (
    parse_parallel_arg,
)
from .plot import (
    plot_tree_ring,
    plot_tree_tent,
    plot_tree_span,
    plot_contractions,
    plot_contractions_alt,
    plot_hypergraph,
)


def is_valid_node(node):
    """Check ``node`` is of type frozenset[int].
    """
    try:
        if not isinstance(node, frozenset):
            return False
        el = next(iter(node))
        if not isinstance(el, int):
            return False
        return True
    except TypeError:
        return False


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
    track_flops : bool, optional
        Whether to dynamically keep track of the total number of flops. If
        ``False`` You can still compute this once the tree is complete.
    track_size : bool, optional
        Whether to dynamically keep track of the largest tensor so far. If
        ``False`` You can still compute this once the tree is complete.

    Properties
    ----------
    info : dict[frozenset[int], dict]
        Information about the tree nodes. The key is the set of inputs (a
        frozenset of inputs indices) the node contains. Or in other words, the
        subgraph of the node. The value is a dictionary to cache information
        about effective 'leg' indices, size, flops of formation etc.
    children : dict[frozenset[int], tuple[frozenset[int]]
        Mapping of each node to two children.
    """

    def __init__(
        self, inputs, output, size_dict,
        track_childless=False,
        track_flops=False,
        track_write=False,
        track_size=False,
    ):

        self.inputs = tuple(map(oset, inputs))
        self.output = oset(output)
        self.size_dict = size_dict
        self.N = len(self.inputs)

        # mapping of parents to children - the core binary tree object
        self.children = {}

        # information about all the nodes
        self.info = {}

        # ... which we can fill in already for final / top node i.e.
        # the collection of all nodes
        self.root = frozenset(range(self.N))
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

        self._track_write = track_write
        if track_write:
            self._write = 0

        self._track_size = track_size
        if track_size:
            self._sizes = MaxCounter()

        # container for caching subtree reconfiguration condidates
        self.already_optimized = dict()

        self.multiplicity = 1
        self.sliced_inds = ()

    def set_state_from(self, other):
        """Set the internal state of this tree to that of ``other``.
        """
        # immutable properties
        for attr in ('inputs', 'N', 'root', 'output',
                     'multiplicity', 'sliced_inds'):
            setattr(self, attr, getattr(other, attr))

        # mutable properties
        for attr in ('size_dict', 'children'):
            setattr(self, attr, getattr(other, attr).copy())

        # dicts of mutable
        for attr in ('info', 'already_optimized'):
            setattr(self, attr,
                    {k: v.copy() for k, v in getattr(other, attr).items()})

        self.track_childless = other.track_childless
        if other.track_childless:
            self.childless = other.childless.copy()

        self._track_flops = other._track_flops
        if other._track_flops:
            self._flops = other._flops

        self._track_write = other._track_write
        if other._track_write:
            self._write = other._write

        self._track_size = other._track_size
        if other._track_size:
            self._sizes = other._sizes.copy()

    def copy(self):
        """Create a copy of this ``ContractionTree``.
        """
        tree = object.__new__(ContractionTree)
        tree.set_state_from(self)
        return tree

    def node_to_terms(self, node):
        """Turn a node -- a frozen set of ints -- into the corresponding terms
        -- a sequence of sets of str corresponding to input indices.
        """
        return map(self.inputs.__getitem__, node)

    def gen_leaves(self):
        """Generate the nodes representing leaves of the contraction tree, i.e.
        of size 1 each corresponding to a single input tensor.
        """
        for i in range(self.N):
            yield frozenset((i,))

    @classmethod
    def from_path(cls, inputs, output, size_dict, *,
                  path=None, ssa_path=None, check=False, **kwargs):
        """Create a (completed) ``ContractionTree`` from the usual inputs plus
        a standard contraction path or 'ssa_path' - you need to supply one.
        """
        if int(path is None) + int(ssa_path is None) != 1:
            raise ValueError("Exactly one of ``path`` or ``ssa_path`` must be "
                             "supplied.")

        if ssa_path is not None:
            path = ssa_path

        tree = cls(inputs, output, size_dict, **kwargs)
        terms = list(tree.gen_leaves())

        for p in path:
            if ssa_path is not None:
                merge = [terms[i] for i in p]
            else:
                merge = [terms.pop(i) for i in sorted(p, reverse=True)]
            terms.append(tree.contract(merge, check=check))

        return tree

    @classmethod
    def from_info(cls, info, **kwargs):
        """Create a ``ContractionTree`` from an ``opt_einsum.PathInfo`` object.
        """
        return cls.from_path(inputs=info.input_subscripts.split(','),
                             output=info.output_subscript,
                             size_dict=info.size_dict,
                             path=info.path, **kwargs)

    @classmethod
    def from_eq(cls, eq, size_dict, **kwargs):
        """Create a empty ``ContractionTree`` directly from an equation and set
        of shapes.
        """
        lhs, output = eq.split('->')
        inputs = lhs.split(',')
        return cls(inputs, output, size_dict, **kwargs)

    @classmethod
    def from_edge_path(cls, edge_path, inputs, output, size_dict,
                       check=False, **kwargs):
        """Create a ``ContractionTree`` from an edge elimination ordering.
        """
        tree = cls(inputs, output, size_dict, **kwargs)
        nodes = list(tree.gen_leaves())

        for e in edge_path:

            # filter out the subgraph induced by edge `e` (generally a pair)
            new_terms, merge = [], []
            for node in nodes:
                term = oset.union(*tree.node_to_terms(node))
                if e in term:
                    merge.append(node)
                else:
                    new_terms.append(node)

            # contract the subgraph
            if merge:
                nodes = new_terms + [tree.contract(merge, check=check)]

        # make sure we are generating a full contraction tree
        nt = len(nodes)
        if nt > 1:
            # this seems to happen when the initial contraction contains a
            # scalar? Or disconnected subgraphs?
            warnings.warn(
                f"Ended up with {nt} nodes - contracting all remaining.")
            tree.contract(nodes, check=check)

        return tree

    def add_node(self, node, check=False):
        if check:
            if len(self.info) > 2 * self.N - 1:
                raise ValueError("There are too many children already.")
            if len(self.children) > self.N - 1:
                raise ValueError("There are too many branches already.")
            if not is_valid_node(node):
                raise ValueError("{} is not a valid node - should be "
                                 "frozenset[int].".format(node))

        self.info.setdefault(node, dict())

    def remove_node(self, node):
        """Remove ``node`` from this tree and update the flops and maximum size
        if tracking them respectively. Inplace operation.
        """
        if self._track_flops:
            self._flops -= self.get_flops(node)

        if self._track_write:
            self._write -= self.get_size(node)

        if self._track_size:
            self._sizes.discard(self.get_size(node))

        del self.info[node]
        del self.children[node]

    def get_keep(self, node):
        """Get a set of at least the indices that should be explicitly kept if
        they appear on ``node`` (or below).
        """
        try:
            keep = self.info[node]['keep']
        except KeyError:
            nodes_above = self.root - node
            terms_above = self.node_to_terms(nodes_above)
            keep = oset.union(self.output, *terms_above)
            self.info[node]['keep'] = keep
        return keep

    def get_legs(self, node):
        """Get the effective 'outer' indices for the collection of tensors
        in ``node``.
        """
        try:
            legs = self.info[node]['legs']
        except KeyError:
            if len(node) == 1:
                legs = self.inputs[next(iter(node))]
            else:
                try:
                    involved = self.get_involved(node)
                except KeyError:
                    involved = oset.union(*self.node_to_terms(node))
                keep = self.get_keep(node)
                legs = involved & keep
            self.info[node]['legs'] = legs
        return legs

    def get_involved(self, node):
        """Get all the indices involved in the formation of subgraph ``node``.
        """
        try:
            involved = self.info[node]['involved']
        except KeyError:
            if len(node) == 1:
                involved = oset()
            else:
                sub_legs = map(self.get_legs, self.children[node])
                involved = oset.union(*sub_legs)
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

    def get_centrality(self, node):
        try:
            return self.info[node]['centrality']
        except KeyError:
            self.compute_centralities()
            return self.info[node]['centrality']

    def total_flops(self):
        """Sum the flops contribution from every node in the tree.
        """
        if self._track_flops:
            return self.multiplicity * self._flops

        self._flops = 0
        for node, _, _ in self.traverse():
            self._flops += self.get_flops(node)

        self._track_flops = True
        return self.multiplicity * self._flops

    def total_write(self):
        """Sum the total amount of memory that will be created and operated on.
        """
        if self._track_write:
            return self.multiplicity * self._write

        self._write = 0
        for node, _, _ in self.traverse():
            self._write += self.get_size(node)

        self._track_write = True
        return self.multiplicity * self._write

    def max_size(self):
        """The size of the largest intermediate tensor.
        """
        if self._track_size:
            return self._sizes.max()

        self._sizes = MaxCounter()
        for node, _, _ in self.traverse():
            self._sizes.add(self.get_size(node))

        self._track_size = True
        return self._sizes.max()

    def peak_size(self, order=None):
        """Get the peak concurrent size of tensors needed - this depends on the
        traversal order, i.e. the exact contraction path, not just the
        contraction tree.
        """
        tot_size = sum(self.get_size(node) for node in self.gen_leaves())
        peak = tot_size
        for p, l, r in self.traverse(order=order):
            tot_size -= self.get_size(l)
            tot_size -= self.get_size(r)
            tot_size += self.get_size(p)
            peak = max(peak, tot_size)
        return peak

    def arithmetic_intensity(self):
        """The ratio of total flops to total write - the higher the better for
        extracting good computational performance.
        """
        return self.total_flops() / self.total_write()

    def remove_ind(self, ind, inplace=False):
        tree = self if inplace else self.copy()

        tree.total_flops()
        tree.total_write()
        tree.max_size()

        d = tree.size_dict[ind]
        s_ind = oset([ind])

        for node, node_info in tree.info.items():

            # if ind doesn't feature in this node (contraction) nothing to do
            involved = tree.get_involved(node)

            # inputs can have leg indices that are not involved so
            legs = tree.get_legs(node)

            if (ind not in involved) and (ind not in legs):
                continue

            # else update all the relevant information about this node
            node_info['involved'] = involved - s_ind
            removed = tree.get_removed(node)

            if ind in legs:
                node_info['legs'] = legs - s_ind

                old_size = tree.get_size(node)
                tree._sizes.discard(old_size)
                new_size = old_size // d
                tree._sizes.add(new_size)
                node_info['size'] = new_size
                tree._write += (-old_size + new_size)

                # modifying keep not stricly necessarily as its only called as
                #     ``legs = keep & involved`` ?
                keep = tree.get_keep(node)
                node_info['keep'] = keep - s_ind
            else:
                node_info['removed'] = removed - s_ind

            old_flops = tree.get_flops(node)
            new_flops = old_flops // d
            if len(removed) == 1:
                # if ind was the last contracted index then have outer product
                new_flops //= 2
            node_info['flops'] = new_flops
            tree._flops += (-old_flops + new_flops)

        def term_without(t):
            if ind in t:
                return t - s_ind
            return t

        tree.output = term_without(tree.output)
        tree.inputs = tuple(map(term_without, tree.inputs))

        tree.already_optimized.clear()

        tree.multiplicity = tree.multiplicity * d
        tree.sliced_inds = tree.sliced_inds + (ind,)

        return tree

    remove_ind_ = functools.partialmethod(remove_ind, inplace=True)

    def contract_pair(self, x, y, check=False):
        """Contract node ``x`` with node ``y`` in the tree to create a new
        parent node.
        """
        parent = x | y

        # make sure info entries exist for all (default dict)
        for node in (x, y, parent):
            self.add_node(node, check=check)

        # enforce left ordering of 'heaviest' subtrees
        nx, ny = len(x), len(y)
        # deterministically break ties
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

        if self._track_flops:
            self._flops += self.get_flops(parent)
        if self._track_write:
            self._write += self.get_size(parent)
        if self._track_size:
            self._sizes.add(self.get_size(parent))

        return parent

    def contract(self, nodes, optimize='auto-hq', check=False):
        """Contract an arbitrary number of ``nodes`` in the tree to build up a
        subtree. The root of this subtree (a new intermediate) is returned.
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
        path_inputs = [oset(self.get_legs(x)) for x in nodes]
        path_output = oset(self.get_legs(grandparent))

        if isinstance(optimize, str):
            path_fn = get_path_fn(optimize)
        else:
            path_fn = optimize

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

    def _traverse_ordered(self, order):
        """Traverse the tree in the order that minimizes ``order(node)``, but
        still contrained to produce children before parents.
        """
        from bisect import bisect

        seen = set()
        queue = [self.root]
        scores = [order(self.root)]

        while len(seen) != len(self.children):
            i = 0
            while i < len(queue):
                node = queue[i]
                if node not in seen:
                    for child in self.children[node]:
                        if len(child) > 1:
                            # insert child into queue by score + before parent
                            score = order(child)
                            ci = bisect(scores[:i], score)
                            scores.insert(ci, score)
                            queue.insert(ci, child)
                            # parent moves extra place to right
                            i += 1
                    seen.add(node)
                i += 1

        for node in queue:
            yield (node, *self.children[node])

    def traverse(self, order=None):
        """Generate, in order, all the node merges in this tree. Non-recursive!
        This ensures children are always visited before their parent.

        Parameters
        ----------
        order : None or callable, optional
            How to order the contractions within the tree. If a callable is
            given (which should take a node as its argument), try to contract
            nodes that minimize this function first.

        Returns
        -------
        generator[tuple[frozenset[frozenset[str]]]]
            The bottom up ordered sequence of tree merges, each a
            tuple of ``(parent, left_child, right_child)``.

        See Also
        --------
        descend
        """
        if order is not None:
            yield from self._traverse_ordered(order=order)
            return

        ready = set(self.gen_leaves())
        queue = [self.root]

        while queue:
            node = queue[-1]
            l, r = self.children[node]

            # both node's children are ready -> we can yield this contraction
            if (l in ready) and (r in ready):
                ready.add(queue.pop())
                yield node, l, r
                continue

            if r not in ready:
                queue.append(r)
            if l not in ready:
                queue.append(l)

    def descend(self, mode='dfs'):
        """Generate, from root to leaves, all the node merges in this tree.
        Non-recursive! This ensures parents are visited before their children.

        Parameters
        ----------
        mode : {'dfs', bfs}, optional
            How expand from a parent.

        Returns
        -------
        generator[tuple[frozenset[frozenset[str]]]]
            The top down ordered sequence of tree merges, each a
            tuple of ``(parent, left_child, right_child)``.

        See Also
        --------
        traverse
        """
        queue = [self.root]
        while queue:
            if mode == 'dfs':
                parent = queue.pop(-1)
            elif mode == 'bfs':
                parent = queue.pop(0)
            l, r = self.children[parent]
            yield parent, l, r
            if len(l) > 1:
                queue.append(l)
            if len(r) > 1:
                queue.append(r)

    def get_subtree(self, node, size, search='bfs'):
        """Get a subtree spanning down from ``node`` which will have ``size``
        leaves (themselves not necessarily leaves of the actual tree).

        Parameters
        ----------
        node : frozenset[int]
            The node of the tree to start with.
        size : int
            How many subtree leaves to aim for.
        search : {'bfs', 'dfs', 'random'}, optional
            How to build the tree:

                - 'bfs': breadth first expansion
                - 'dfs': depth first expansion (largest nodes first)
                - 'random': random expansion

        Returns
        -------
        sub_leaves : tuple[frozenset[int]]
            Nodes which are subtree leaves.
        branches : tuple[frozenset[int]]
            Nodes which are between the subtree leaves and root.
        """
        # nodes which are subtree leaves
        branches = []

        # actual tree leaves - can't expand
        real_leaves = []

        # nodes to expand
        queue = [node]

        while (len(queue) + len(real_leaves) < size) and queue:

            if search == 'bfs':
                p = queue.pop(0)
            elif search == 'dfs':
                p = queue.pop(-1)
            elif search == 'random':
                p = queue.pop(random.randint(0, len(queue) - 1))

            if len(p) == 1:
                real_leaves.append(p)
                continue

            # the left child is always >= in weight that right child
            #     if we append it last then ``.pop(-1)`` above perform the
            #     depth first search sorting by node subgraph size
            l, r = self.children[p]
            queue.append(r)
            queue.append(l)
            branches.append(p)

        # nodes at the bottom of the subtree
        sub_leaves = queue + real_leaves

        return tuple(sub_leaves), tuple(branches)

    def calc_subtree_candidates(self, pwr=2, what='flops'):
        candidates = list(self.children)

        if what == 'size':
            weights = [self.get_size(x) for x in candidates]

        elif what == 'flops':
            weights = [self.get_flops(x) for x in candidates]

        max_weight = max(weights)

        # can be bigger than numpy int/float allows
        weights = [float(w / max_weight)**(1 / pwr) for w in weights]

        # sort by descending score
        candidates, weights = zip(
            *sorted(zip(candidates, weights), key=lambda x: -x[1]))

        return list(candidates), list(weights)

    def subtree_reconfigure(
        self,
        subtree_size=8,
        subtree_search='bfs',
        weight_what='flops',
        weight_pwr=2,
        select='max',
        maxiter=500,
        seed=None,
        minimize='flops',
        inplace=False,
        progbar=False,
    ):
        """Reconfigure subtrees of this tree with locally optimal paths.

        Parameters
        ----------
        subtree_size : int, optional
            The size of subtree to consider. Cost is exponential in this.
        subtree_search : {'bfs', 'dfs', 'random'}, optional
            How to build the subtrees:

                - 'bfs': breadth-first-search creating balanced subtrees
                - 'dfs': depth-first-search creating imbalanced subtrees
                - 'random': random subtree building

        weight_what : {'flops', 'size'}, optional
            When assessing nodes to build and optimize subtrees from whether to
            score them by the (local) contraction cost, or tensor size.
        weight_pwr : int, optional
            When assessing nodes to build and optimize subtrees from, how to
            scale their score into a probability: ``score**(1 / weight_pwr)``.
            The larger this is the more explorative the algorithm is when
            ``select='random'``.
        select : {'max', 'min', 'random'}, optional
            What order to select node subtrees to optimize:

                - 'max': choose the highest score first
                - 'min': choose the lowest score first
                - 'random': choose randomly weighted on score -- see
                  ``weight_pwr``.

        maxiter : int, optional
            How many subtree optimizations to perform, the algorithm can
            terminate before this if all subtrees have been optimized.
        seed : int, optional
            A random seed (seeds python system random module).
        minimize : {'flops', 'size'}, optional
            Whether to minimize with respect to contraction flops or size.
        inplace : bool, optional
            Whether to perform the reconfiguration inplace or not.
        progbar : bool, optional
            Whether to show live progress of the reconfiguration.

        Returns
        -------
        ContractionTree
        """
        tree = self if inplace else self.copy()

        # ensure these have been computed and thus are being tracked
        tree.total_flops()
        tree.max_size()

        optimizer = DynamicProgramming(minimize=minimize)

        # different caches as we might want to reconfigure one before other
        self.already_optimized.setdefault(minimize, set())
        already_optimized = self.already_optimized[minimize]

        if seed is not None:
            random.seed(seed)

        candidates, weights = self.calc_subtree_candidates(
            pwr=weight_pwr, what=weight_what)

        if progbar:
            import tqdm
            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree))

        r = 0
        try:
            while candidates and r < maxiter:
                if select == 'max':
                    i = 0
                elif select == 'min':
                    i = -1
                elif select == 'random':
                    i, = random.choices(
                        range(len(candidates)), weights=weights)

                weights.pop(i)
                sub_root = candidates.pop(i)

                # get a subtree to possibly reconfigure
                sub_leaves, sub_branches = tree.get_subtree(
                    sub_root, size=subtree_size, search=subtree_search)

                sub_leaves = frozenset(sub_leaves)

                # check if its already been optimized
                if sub_leaves in already_optimized:
                    continue

                # else remove the branches, keeping track of current cost
                if minimize == 'flops':
                    current_cost = tree.get_flops(sub_root) // 2
                elif minimize == 'write':
                    current_cost = tree.get_size(sub_root)
                elif minimize == 'size':
                    current_cost = tree.get_size(sub_root)
                elif minimize == 'combo':
                    current_cost = (
                        tree.get_flops(sub_root) // 2 +
                        500 * tree.get_size(sub_root))

                for node in sub_branches:
                    if minimize == 'flops':
                        current_cost += tree.get_flops(node) // 2
                    elif minimize == 'write':
                        current_cost += tree.get_size(node)
                    elif minimize == 'size':
                        current_cost = max(
                            current_cost, tree.get_size(node))
                    elif minimize == 'combo':
                        current_cost += (
                            tree.get_flops(node) // 2 +
                            500 * tree.get_size(node))

                    tree.remove_node(node)

                # make the optimizer more efficient by supplying accurate cap
                optimizer.cost_cap = current_cost

                # and reoptimize the leaves
                tree.contract(sub_leaves, optimize=optimizer)
                already_optimized.add(sub_leaves)

                r += 1

                if progbar:
                    pbar.update()
                    pbar.set_description(_describe_tree(tree))

                # if we have reconfigured simply re-add all candidates
                candidates, weights = tree.calc_subtree_candidates(
                    pwr=weight_pwr, what=weight_what)
        finally:
            if progbar:
                pbar.close()

        return tree

    subtree_reconfigure_ = functools.partialmethod(
        subtree_reconfigure, inplace=True)

    def subtree_reconfigure_forest(
        self,
        num_trees=8,
        num_restarts=10,
        restart_fraction=0.5,
        subtree_maxiter=100,
        subtree_size=10,
        subtree_search=('random', 'bfs'),
        subtree_select=('random',),
        subtree_weight_what=('flops', 'size'),
        subtree_weight_pwr=(2,),
        parallel='auto',
        parallel_maxiter_steps=4,
        minimize='flops',
        progbar=False,
        inplace=False,
    ):
        """'Forested' version of ``subtree_reconfigure`` which is more
        explorative and can be parallelized. It stochastically generates
        a 'forest' reconfigured trees, then only keeps some fraction of these
        to generate the next forest.

        Parameters
        ----------
        num_trees : int, optional
            The number of trees to reconfigure at each stage.
        num_restarts : int, optional
            The number of times to halt, prune and then restart the
            tree reconfigurations.
        restart_fraction : float, optional
            The fraction of trees to keep at each stage and generate the next
            forest from.
        subtree_maxiter : int, optional
            Number of subtree reconfigurations per step.
            ``num_restarts * subtree_maxiter`` is the max number of total
            subtree reconfigurations for the final tree produced.
        subtree_size : int, optional
            The size of subtrees to search for and reconfigure.
        subtree_search : tuple[{'random', 'bfs', 'dfs'}], optional
            Tuple of options for the ``search`` kwarg of
            :meth:`ContractionTree.subtree_reconfigure` to randomly sample.
        subtree_select : tuple[{'random', 'max', 'min'}], optional
            Tuple of options for the ``select`` kwarg of
            :meth:`ContractionTree.subtree_reconfigure` to randomly sample.
        subtree_weight_what : tuple[{'flops', 'size'}], optional
            Tuple of options for the ``weight_what`` kwarg of
            :meth:`ContractionTree.subtree_reconfigure` to randomly sample.
        subtree_weight_pwr : tuple[int], optional
            Tuple of options for the ``weight_pwr`` kwarg of
            :meth:`ContractionTree.subtree_reconfigure` to randomly sample.
        parallel : 'auto', False, True, int, or distributed.Client
            Whether to parallelize the search.
        parallel_maxiter_steps : int, optional
            If parallelizing, how many steps to break each reconfiguration into
            in order to evenly saturate many processes.
        minimize : {'flops', 'size'}, optional
            Whether to minimize the total flops or maximum size of the
            contraction tree.
        progbar : bool, optional
            Whether to show live progress.
        inplace : bool, optional
            Whether to perform the subtree reconfiguration inplace.

        Returns
        -------
        ContractionTree
        """
        tree = self if inplace else self.copy()

        # candidate trees
        num_keep = max(1, int(num_trees * restart_fraction))

        # how to rank the trees
        score = get_score_fn(minimize)

        # set up the initial 'forest' and parallel machinery
        pool = parse_parallel_arg(parallel)
        if pool is not None:
            try:
                from dask.distributed import secede, rejoin
                secede()  # for nested parallelism
                is_dask_worker = True
            except (ImportError, ValueError):
                is_dask_worker = False

            # store the trees as futures for the entire process
            forest = [pool.scatter(tree)]
            maxiter = subtree_maxiter // parallel_maxiter_steps
        else:
            forest = [tree]
            maxiter = subtree_maxiter

        if progbar:
            import tqdm
            pbar = tqdm.tqdm(total=num_restarts)
            pbar.set_description(_describe_tree(tree))

        try:
            for _ in range(num_restarts):

                # on the next round take only the best trees
                forest = itertools.cycle(forest[:num_keep])

                # select some random configurations
                saplings = [{
                    'tree': next(forest),
                    'maxiter': maxiter,
                    'minimize': minimize,
                    'subtree_size': subtree_size,
                    'subtree_search': random.choice(subtree_search),
                    'select': random.choice(subtree_select),
                    'weight_pwr': random.choice(subtree_weight_pwr),
                    'weight_what': random.choice(subtree_weight_what),
                } for _ in range(num_trees)]

                if pool is None:
                    forest = [_reconfigure_tree(**s) for s in saplings]
                    res = [{'tree': t, **_get_tree_info(t)} for t in forest]
                else:
                    # submit in smaller steps to saturate processes
                    for _ in range(parallel_maxiter_steps):
                        for s in saplings:
                            s['tree'] = pool.submit(
                                _reconfigure_tree, pure=False, **s)

                    # compute scores remotely then gather
                    forest_futures = [s['tree'] for s in saplings]
                    res_futures = [pool.submit(_get_tree_info, t, pure=False)
                                   for t in forest_futures]
                    res = [{'tree': tree_future, **res_future.result()}
                           for tree_future, res_future in
                           zip(forest_futures, res_futures)]

                # update the order of the new forest
                res.sort(key=score)
                forest = [r['tree'] for r in res]

                if progbar:
                    pbar.update()
                    if pool is None:
                        d = _describe_tree(forest[0])
                    else:
                        d = pool.submit(_describe_tree, forest[0],
                                        pure=False).result()
                    pbar.set_description(d)

        finally:
            if progbar:
                pbar.close()

        if pool is None:
            tree.set_state_from(forest[0])
        else:
            tree.set_state_from(forest[0].result())

            if is_dask_worker:
                rejoin()

        return tree

    subtree_reconfigure_forest_ = functools.partialmethod(
        subtree_reconfigure_forest, inplace=True)

    def slice(self, max_repeats=16, inplace=False, **slicing_opts):
        """Slice this tree (turn some indices into indices which are explicitly
        summed over rather than being part of contractions).
        """
        from .slicer import SliceFinder

        tree = self if inplace else self.copy()

        sf = SliceFinder(tree, **slicing_opts)
        ix_sl, _ = sf.search(max_repeats)
        for ix in ix_sl:
            tree.remove_ind_(ix)

        return tree

    slice_ = functools.partialmethod(slice, inplace=True)

    def slice_and_reconfigure(
        self,
        target_size,
        step_size=2,
        temperature=0.01,
        minimize='flops',
        max_repeats=16,
        reconf_opts=None,
        progbar=False,
        inplace=False,
    ):
        """Interleave slicing (removing indices into an exterior sum) with
        subtree reconfiguration to minimize the overhead induced by this
        slicing.

        Parameters
        ----------
        target_size : int
            Slice the tree until the maximum intermediate size is this or
            smaller.
        step_size : int, optional
            The minimum size reduction to try and achieve before switching to a
            round of subtree reconfiguration.
        temperature : float, optional
            The temperature to supply to ``SliceFinder`` for searching for
            indices.
        max_repeats : int, optional
            The number of slicing attempts to perform per search.
        progbar : bool, optional
            Whether to show live progress.
        inplace : bool, optional
            Whether to perform the slicing and reconfiguration inplace.
        reconf_opts : None or dict, optional
            Supplied to
            :meth:`ContractionTree.subtree_reconfigure` or
            :meth:`ContractionTree.subtree_reconfigure_forest`, depending on
            `'forested'` key value.
        """
        tree = self if inplace else self.copy()

        reconf_opts = {} if reconf_opts is None else dict(reconf_opts)
        reconf_opts.setdefault('minimize', minimize)
        forested_reconf = reconf_opts.pop('forested', False)

        if progbar:
            import tqdm
            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree))

        try:
            while tree.max_size() > target_size:
                tree.slice_(
                    temperature=temperature,
                    target_slices=step_size,
                    minimize=minimize,
                )
                if forested_reconf:
                    tree.subtree_reconfigure_forest_(**reconf_opts)
                else:
                    tree.subtree_reconfigure_(**reconf_opts)

                if progbar:
                    pbar.update()
                    pbar.set_description(_describe_tree(tree))
        finally:
            if progbar:
                pbar.close()

        return tree

    slice_and_reconfigure_ = functools.partialmethod(
        slice_and_reconfigure, inplace=True)

    def slice_and_reconfigure_forest(
        self,
        target_size,
        step_size=2,
        num_trees=8,
        restart_fraction=0.5,
        temperature=0.02,
        max_repeats=32,
        minimize='flops',
        parallel='auto',
        progbar=False,
        inplace=False,
        reconf_opts=None,
    ):
        """'Forested' version of :meth:`ContractionTree.slice_and_reconfigure`.
        This maintains a 'forest' of trees with different slicing and subtree
        reconfiguration attempts, pruning the worst at each step and generating
        a new forest from the best.

        Parameters
        ----------
        target_size : int
            Slice the tree until the maximum intermediate size is this or
            smaller.
        step_size : int, optional
            The minimum size reduction to try and achieve before switching to a
            round of subtree reconfiguration.
        num_restarts : int, optional
            The number of times to halt, prune and then restart the
            tree reconfigurations.
        restart_fraction : float, optional
            The fraction of trees to keep at each stage and generate the next
            forest from.
        temperature : float, optional
            The temperature at which to randomize the sliced index search.
        max_repeats : int, optional
            The number of slicing attempts to perform per search.
        parallel : 'auto', False, True, int, or distributed.Client
            Whether to parallelize the search.
        progbar : bool, optional
            Whether to show live progress.
        inplace : bool, optional
            Whether to perform the slicing and reconfiguration inplace.
        reconf_opts : None or dict, optional
            Supplied to
            :meth:`ContractionTree.slice_and_reconfigure`.

        Returns
        -------
        ContractionTree
        """
        tree = self if inplace else self.copy()

        # candidate trees
        num_keep = max(1, int(num_trees * restart_fraction))

        # how to rank the trees
        score = get_score_fn(minimize)

        # set up the initial 'forest' and parallel machinery
        pool = parse_parallel_arg(parallel)
        if pool is not None:
            try:
                from dask.distributed import secede, rejoin
                secede()  # for nested parallelism
                is_dask_worker = True
            except (ImportError, ValueError):
                is_dask_worker = False

            # store the trees as futures for the entire process
            forest = [pool.scatter(tree)]
        else:
            forest = [tree]

        if progbar:
            import tqdm
            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree))

        next_size = tree.max_size()

        try:
            while True:
                next_size //= step_size

                # on the next round take only the best trees
                forest = itertools.cycle(forest[:num_keep])

                saplings = [{
                    'tree': next(forest),
                    'target_size': next_size,
                    'step_size': step_size,
                    'temperature': temperature,
                    'max_repeats': max_repeats,
                    'reconf_opts': reconf_opts,
                } for _ in range(num_trees)]

                if pool is None:
                    forest = [
                        _slice_and_reconfigure_tree(**s) for s in saplings
                    ]
                    res = [{'tree': t, **_get_tree_info(t)} for t in forest]

                else:
                    forest_futures = [
                        pool.submit(
                            _slice_and_reconfigure_tree, pure=False, **s)
                        for s in saplings
                    ]

                    # compute scores remotely then gather
                    res_futures = [
                        pool.submit(_get_tree_info, t, pure=False)
                        for t in forest_futures
                    ]
                    res = [
                        {'tree': tree_future, **res_future.result()}
                        for tree_future, res_future in
                        zip(forest_futures, res_futures)
                    ]

                # we want to sort by flops, but also favour sampling as
                # many different sliced index combos as possible
                #    ~ [1, 1, 1, 2, 2, 3] -> [1, 2, 3, 1, 2, 1]
                res.sort(key=score)
                res = list(interleave(groupby(
                    lambda r: r['sliced_inds'], res
                ).values()))

                # update the order of the new forest
                forest = [r['tree'] for r in res]

                if progbar:
                    pbar.update()
                    if pool is None:
                        d = _describe_tree(forest[0])
                    else:
                        d = pool.submit(_describe_tree, forest[0],
                                        pure=False).result()
                    pbar.set_description(d)

                if res[0]['size'] <= target_size:
                    break

        finally:
            if progbar:
                pbar.close()

        if pool is None:
            tree.set_state_from(forest[0])
        else:
            tree.set_state_from(forest[0].result())

            if is_dask_worker:
                rejoin()

        return tree

    slice_and_reconfigure_forest_ = functools.partialmethod(
        slice_and_reconfigure_forest, inplace=True)

    def flat_tree(self, order=None):
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
        tups = {frozenset([i]): i for i in range(self.N)}

        for parent, l, r in self.traverse(order=order):
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

    def path(self, order=None):
        """Generate a standard path from the contraction tree.
        """
        path = []
        terms = list(self.gen_leaves())

        for parent, l, r in self.traverse(order=order):
            i, j = sorted((terms.index(l), terms.index(r)))
            terms.pop(j)
            terms.pop(i)
            path.append((i, j))
            terms.append(parent)

        return tuple(path)

    def ssa_path(self, order=None):
        """Generate a ssa path from the contraction tree.
        """
        ssa_path = []
        pos = {frozenset([i]): i for i in range(self.N)}

        for parent, l, r in self.traverse(order=order):
            i, j = sorted((pos[l], pos[r]))
            ssa_path.append((i, j))
            pos[parent] = len(ssa_path) + self.N - 1

        return tuple(ssa_path)

    def surface_order(self, node):
        return (len(node), self.get_centrality(node))

    def path_surface(self):
        return self.path(order=self.surface_order)

    def ssa_path_surface(self):
        return self.ssa_path(order=self.surface_order)

    def get_spans(self):
        """Get all (which could mean none) potential embeddings of this
        contraction tree into a spanning tree of the original graph.

        Returns
        -------
        tuple[dict[frozenset[int], frozenset[int]]]
        """
        ind_to_term = collections.defaultdict(set)
        for i, term in enumerate(self.inputs):
            for ix in term:
                ind_to_term[ix].add(i)

        def boundary_pairs(node):
            """Get nodes along the boundary of the bipartition represented by
            ``node``.
            """
            pairs = set()
            for ix in self.get_removed(node):
                # for every index across the contraction
                l1, l2 = ind_to_term[ix]

                # can either span from left to right or right to left
                pairs.add((l1, l2))
                pairs.add((l2, l1))

            return pairs

        # first span choice is any nodes across the top level bipart
        candidates = [
            {
                # which intermedate nodes map to which leaf nodes
                'map': {self.root: frozenset([l2])},
                # the leaf nodes in the spanning tree
                'spine': {l1, l2},
            }
            for l1, l2 in boundary_pairs(self.root)
        ]

        for p, l, r in self.descend():
            for child in (r, l):

                # for each current candidate check all the possible extensions
                for _ in range(len(candidates)):
                    cand = candidates.pop(0)

                    # don't need to do anything for
                    if len(child) == 1:
                        candidates.append({
                            'map': {child: child, **cand['map']},
                            'spine': cand['spine'].copy(),
                        })

                    for l1, l2 in boundary_pairs(child):
                        if (l1 in cand['spine']) or (l2 not in cand['spine']):
                            # pair does not merge inwards into spine
                            continue

                        # valid extension of spanning tree
                        candidates.append({
                            'map': {child: frozenset([l2]), **cand['map']},
                            'spine': cand['spine'] | {l1, l2},
                        })

        return tuple(c['map'] for c in candidates)

    def compute_centralities(self, combine='mean'):
        """Compute a centrality for every node in this contraction tree.
        """
        H = self.get_hypergraph()
        Cs = H.simple_centrality()

        for i, leaf in enumerate(self.gen_leaves()):
            self.info[leaf]['centrality'] = Cs[i]

        if combine == 'mean':
            def combine(x, y):
                return (x + y) / 2

        elif combine == 'sum':
            def combine(x, y):
                return (x + y)

        elif combine == 'max':
            def combine(x, y):
                return max(x, y)

        elif combine == 'min':
            def combine(x, y):
                return min(x, y)

        for p, l, r in self.traverse():
            self.info[p]['centrality'] = combine(
                self.info[l]['centrality'],
                self.info[r]['centrality'])

    def get_hypergraph(self):
        if not hasattr(self, '_hypergraph'):
            self._hypergraph = HyperGraph(
                self.inputs, self.output, self.size_dict)
        return self._hypergraph

    plot_ring = plot_tree_ring
    plot_tent = plot_tree_tent
    plot_span = plot_tree_span
    plot_contractions = plot_contractions
    plot_contractions_alt = plot_contractions_alt

    @functools.wraps(plot_hypergraph)
    def plot_hypergraph(self, **kwargs):
        H = self.get_hypergraph()
        H.plot(**kwargs)

    def __repr__(self):
        s = "<ContractionTree(N={}, branches={}, complete={})>"
        return s.format(self.N, len(self.children), self.is_complete())


def _reconfigure_tree(tree, *args, **kwargs):
    return tree.subtree_reconfigure(*args, **kwargs)


def _slice_and_reconfigure_tree(tree, *args, **kwargs):
    return tree.slice_and_reconfigure(*args, **kwargs)


def _get_tree_info(tree):
    return {
        'flops': tree.total_flops(),
        'write': tree.total_flops(),
        'size': tree.max_size(),
        'sliced_inds': frozenset(tree.sliced_inds),
    }


def score_flops(trial):
    return (
        math.log2(trial['flops']) +
        math.log2(trial['write']) / 1000 +
        math.log2(trial['size']) / 1000
    )


def score_write(trial):
    return (
        math.log2(trial['flops']) / 1000 +
        math.log2(trial['write']) +
        math.log2(trial['size']) / 1000
    )


def score_size(trial):
    return (
        math.log2(trial['flops']) / 1000 +
        math.log2(trial['write']) / 1000 +
        math.log2(trial['size'])
    )


def score_combo(trial):
    return (
        math.log2(trial['flops'] + 256 * trial['write']) +
        math.log2(trial['size']) / 1000
    )


def get_score_fn(minimize):
    if minimize == 'flops':
        return score_flops
    if minimize == 'write':
        return score_write
    if minimize == 'size':
        return score_size
    if minimize == 'combo':
        return score_combo


def _score_flops(tree):
    """Score by flops but split ties with size.
    """
    return math.log2(tree.total_flops()) + math.log2(tree.max_size()) / 1000


def _score_size(tree):
    """Score by size but split ties with flops.
    """
    return math.log2(tree.total_flops()) / 1000 + math.log2(tree.max_size())


def _describe_tree(tree):
    return (
        f"log2[SIZE]: {math.log2(tree.max_size()):.2f} "
        f"log10[FLOPs]: {math.log10(tree.total_flops()):.2f}"
    )


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
    list of integers of length ``len(inputs)`` labelling which partition
    each input node should be put it.
    """

    def __init__(self, partition_fn):
        self.partition_fn = partition_fn

    def __call__(self, inputs, output, size_dict, random_strength=0.01,
                 weight_nodes='const', weight_edges='log',
                 cutoff=10, parts=2, parts_decay=0.5,
                 sub_optimize='auto', super_optimize='auto-hq',
                 check=False, **kwargs):

        tree = ContractionTree(inputs, output, size_dict,
                               track_flops=True,
                               track_childless=True)
        rand_size_dict = jitter_dict(size_dict, random_strength)

        dynamic_imbalance = (
            ('imbalance' in kwargs) and ('imbalance_decay' in kwargs)
        )

        if dynamic_imbalance:
            imbalance = kwargs.pop('imbalance')
            imbalance_decay = kwargs.pop('imbalance_decay')

        while tree.childless:
            tree_node = next(iter(tree.childless))
            subgraph = tuple(tree_node)
            subsize = len(subgraph)

            # skip straight to better method
            if subsize <= cutoff:
                tree.contract([frozenset([x]) for x in subgraph],
                              optimize=sub_optimize, check=check)
                continue

            # relative subgraph size
            s = subsize / tree.N

            # let the target number of communities depend on subgraph size
            parts_s = max(int(s**parts_decay * parts), 2)

            # let the imbalance either rise or fall
            if dynamic_imbalance:
                if imbalance_decay >= 0:
                    imbalance_s = s ** imbalance_decay * imbalance
                else:
                    imbalance_s = 1 - s ** -imbalance_decay * (1 - imbalance)
                kwargs['imbalance'] = imbalance_s

            # partition! get community membership list e.g.
            # [0, 0, 1, 0, 1, 0, 0, 2, 2, ...]
            inputs = tuple(tree.node_to_terms(subgraph))
            output = tree.get_legs(tree_node)
            membership = self.partition_fn(
                inputs, output, rand_size_dict,
                weight_nodes=weight_nodes, weight_edges=weight_edges,
                parts=parts_s, **kwargs,
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
        return self(inputs, output, size_dict, **kwargs)


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


class LineGraph:
    """Very simple line-graph builder and file writer.
    """

    def __init__(self, inputs, output):
        self.inputs = inputs
        self.nodes = tuple(oset.union(*inputs))
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


try:
    from gmpy2 import popcount

except ImportError:

    def popcount(x):
        return bin(x).count('1')


def dict_affine_renorm(d):
    dmax = max(d.values())
    dmin = min(d.values())
    if dmax == dmin:
        dmin = 0
    return {k: (v - dmin) / (dmax - dmin) for k, v in d.items()}


class HyperGraph:
    """Simple hypergraph builder and writer.
    """

    def __init__(
        self,
        inputs,
        output=(),
        size_dict=None,
        weight_edges='const',
        weight_nodes='const',
        fuse_output_inds=False,
    ):
        self.inputs = inputs
        self.output = output
        self.size_dict = dict() if size_dict is None else size_dict

        if output and fuse_output_inds:
            self.nodes = (tuple(itertools.chain(inputs, [output])))
        else:
            self.nodes = tuple(inputs)

        self.fuse_output_inds = fuse_output_inds

        self.indmap = collections.defaultdict(list)
        for i, term in enumerate(self.nodes):
            for ix in term:
                self.indmap[ix].append(i)

        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.indmap)

        self.weight_nodes = weight_nodes
        self.weight_edges = weight_edges

        # compute these lazily
        self.node_weights = None
        self.edge_list = None
        self.edge_weight_map = None
        self.edge_weights = None
        self.has_edge_weights = None
        self.has_node_weights = None
        self.fmt = None

    def _compute_weights(self):
        if self.node_weights is not None:
            # only compute once
            return

        self.node_weights = [
            calc_node_weight(term, self.size_dict, self.weight_nodes)
            for term in self.nodes
        ]

        self.edge_list = tuple(self.indmap)
        self.edge_weight_map = {
            e: calc_edge_weight(e, self.size_dict, self.weight_edges)
            for e in self.edge_list}
        self.edge_weights = [self.edge_weight_map[e] for e in self.edge_list]

        self.has_edge_weights = self.weight_edges in ('log', 'linear')
        self.has_node_weights = self.weight_nodes in ('log', 'linear')
        self.fmt = {
            (False, False): "",
            (False, True): "10",
            (True, False): "1",
            (True, True): "11",
        }[self.has_edge_weights, self.has_node_weights]

    def __len__(self):
        return self.num_nodes

    def neighbors(self, i):
        """Get the neighbors of node ``i``.
        """
        return oset(j for ix in self.nodes[i]
                    for j in self.indmap[ix] if j != i)

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
        visitors = {i: 1 << i for i in range(self.num_nodes)}

        # store the number of unique visitors - the change is this each step
        #    is the number of new shortest paths of length ``d``
        num_visitors = {i: 1 for i in range(self.num_nodes)}

        # the total weighted score - combining num visitors and their distance
        scores = {i: 0.0 for i in range(self.num_nodes)}

        # at each iteration expand all nodes visitors to their neighbors
        for d in range(self.num_nodes):

            # do a parallel update
            previous_visitors = visitors.copy()

            for i in range(self.num_nodes):
                for j in self.neighbors(i):
                    visitors[i] |= previous_visitors[j]

                # visitors are worth less the further they've come from
                new_nv = popcount(visitors[i])
                scores[i] += (new_nv - num_visitors[i]) / (d + 1)**mu
                num_visitors[i] = new_nv

                # once any node has reached a certain number of visitors stop
                should_stop |= (new_nv >= sz_stop)

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

        if r is None:
            # take the propagation time as sqrt hypergraph size
            r = max(10, int(self.num_nodes**0.5))

        for _ in range(r):
            # do a parallel update
            previous_c = c.copy()

            # spread the centrality of each node into its neighbors
            for i in range(self.num_nodes):
                ci = previous_c[i]
                for j in self.neighbors(i):
                    c[j] += smoothness * ci / r

            # then rescale all the values between 0.0 and 1.0
            c = dict_affine_renorm(c)

        return c

    def get_laplacian(self):
        """Get the graph Laplacian.
        """
        import numpy as np

        L = np.zeros((self.num_nodes, self.num_nodes))

        for i, term in enumerate(self.nodes):
            L[i, i] = len(term)

        for i, j in self.indmap.values():
            L[i, j] = L[j, i] = -1

        return L

    def get_resistance_distances(self):
        """Get the resistance distance between all nodes of the raw graph.
        """
        import numpy as np

        L = self.get_laplacian()
        L += (1 / self.num_nodes)
        L = np.linalg.inv(L)
        d = np.array(np.diag(L))  # needs to be copy
        L *= -2
        L += d.reshape(1, -1)
        L += d.reshape(-1, 1)

        return L

    def resistance_centrality(self, rescale=True):
        """Compute the centrality in terms of the total resistance distance
        to all other nodes.
        """
        L = self.get_resistance_distances()

        Cs = dict(enumerate(-L.sum(axis=1)))

        if rescale:
            Cs = dict_affine_renorm(Cs)

        return Cs

    def to_networkx(H):
        """Convert to a networkx Graph, with hyperedges represented as nodes.
        """
        import networkx as nx

        G = nx.Graph(any_hyper=False)
        for ix, nodes in H.indmap.items():
            if len(nodes) == 2:
                # regular edge
                G.add_edge(*nodes, ind=ix, hyperedge=False)
            else:
                # hyperedge
                G.graph['any_hyper'] = True
                G.add_node(ix, hyperedge=True)
                for nd in nodes:
                    G.add_edge(ix, nd, ind=ix, hyperedge=True)

        for nd in G.nodes:
            G.nodes[nd].setdefault('hyperedge', False)

        return G

    def to_hmetis_str(self):
        """Note that vertices are enumerated from 1 not 0.
        """
        self._compute_weights()

        lns = [f"{self.num_edges} {self.num_nodes} {self.fmt}"]

        for e in self.edge_list:
            ln = " ".join(str(v + 1) for v in self.indmap[e])
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
        self._compute_weights()

        hyperedge_indices = []
        hyperedges = []
        for e in self.edge_list:
            hyperedge_indices.append(len(hyperedges))
            hyperedges.extend(self.indmap[e])
        hyperedge_indices.append(len(hyperedges))
        return hyperedge_indices, hyperedges

    plot = plot_hypergraph

    def __repr__(self):
        return f"<HyperGraph(|V|={self.num_nodes}, |E|={self.num_edges})>"
