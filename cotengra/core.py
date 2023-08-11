import math
import random
import warnings
import operator
import itertools
import functools
import collections

from .oe import (
    compute_size_by_dict,
    DEFAULT_COMBO_FACTOR,
    DynamicProgramming,
    flop_count,
    get_path_fn,
    linear_to_ssa,
    get_symbol,
)
from autoray import do

from .utils import (
    BitSet,
    deprecated,
    dynary,
    groupby,
    interleave,
    is_valid_node,
    MaxCounter,
    node_from_seq,
    node_from_single,
    node_get_single_el,
    node_supremum,
    oset,
    prod,
    unique,
)
from .parallel import (
    can_scatter,
    maybe_leave_pool,
    maybe_rejoin_pool,
    parse_parallel_arg,
    scatter,
    submit,
)
from .hypergraph import get_hypergraph
from .scoring import get_score_fn, CompressedStatsTracker
from .contract import make_contractor
from .plot import (
    plot_contractions_alt,
    plot_contractions,
    plot_hypergraph,
    plot_tree_ring,
    plot_tree_rubberband,
    plot_tree_span,
    plot_tree_tent,
)


def cached_node_property(name):
    """Decorator for caching information about nodes."""

    def wrapper(meth):
        @functools.wraps(meth)
        def getter(self, node):
            try:
                return self.info[node][name]
            except KeyError:
                self.info[node][name] = value = meth(self, node)
                return value

        return getter

    return wrapper


def union_it(bs):
    """Non-variadic version of various set type unions."""
    b0, *bs = bs
    return b0.union(*bs)


def get_with_default(k, obj, default):
    return obj.get(k, default)


class ContractionTree:
    """Binary tree representing a tensor network contraction.

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

    Attributes
    ----------
    children : dict[node, tuple[node]]
        Mapping of each node to two children.
    info : dict[node, dict]
        Information about the tree nodes. The key is the set of inputs (a
        set of inputs indices) the node contains. Or in other words, the
        subgraph of the node. The value is a dictionary to cache information
        about effective 'leg' indices, size, flops of formation etc.
    """

    def __init__(
        self,
        inputs,
        output,
        size_dict,
        track_childless=False,
        track_flops=False,
        track_write=False,
        track_size=False,
    ):
        self.inputs = inputs
        self.output = output

        if not isinstance(next(iter(size_dict.values()), 1), int):
            # make sure we are working with python integers to avoid overflow
            # comparison errors with inf etc.
            self.size_dict = {k: int(v) for k, v in size_dict.items()}
        else:
            self.size_dict = size_dict

        self.N = len(self.inputs)

        self.bitset_edges = BitSet(size_dict.keys())
        self.inputs_legs = list(map(self.bitset_edges, self.inputs))
        self.output_legs = self.bitset_edges(self.output)

        # mapping of parents to children - the core binary tree object
        self.children = {}

        # information about all the nodes
        self.info = {}

        # ... which we can fill in already for final / top node i.e.
        # the collection of all nodes
        self.root = node_supremum(self.N)
        self.info[self.root] = {
            "legs": self.output_legs,
            "keep": self.output_legs,
            "size": compute_size_by_dict(self.output, size_dict),
        }

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

        # info relating to slicing (base constructor is always unsliced)
        self.multiplicity = 1
        self.sliced_inds = self.sliced_sizes = ()
        self.sliced_inputs = frozenset()

        # cache for compiled contraction cores
        self.contraction_cores = {}

    def set_state_from(self, other):
        """Set the internal state of this tree to that of ``other``."""
        # immutable properties
        for attr in (
            "inputs",
            "output",
            "size_dict",
            "N",
            "root",
            "multiplicity",
            "sliced_inds",
            "sliced_sizes",
            "sliced_inputs",
            "bitset_edges",
        ):
            setattr(self, attr, getattr(other, attr))

        # mutable properties
        for attr in (
            "children",
            "inputs_legs",
            "output_legs",
            "contraction_cores",
        ):
            setattr(self, attr, getattr(other, attr).copy())

        # dicts of mutable
        for attr in ("info", "already_optimized"):
            setattr(
                self,
                attr,
                {k: v.copy() for k, v in getattr(other, attr).items()},
            )

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
        """Create a copy of this ``ContractionTree``."""
        tree = object.__new__(self.__class__)
        tree.set_state_from(self)
        return tree

    @property
    def nslices(self):
        """Simple alias for how many independent contractions this tree
        represents overall.
        """
        return self.multiplicity

    @property
    def nchunks(self):
        """The number of 'chunks' - determined by the number of sliced output
        indices.
        """
        return prod(
            d
            for ix, d in zip(self.sliced_inds, self.sliced_sizes)
            if ix in self.output
        )

    def node_to_terms(self, node):
        """Turn a node -- a frozen set of ints -- into the corresponding terms
        -- a sequence of sets of str corresponding to input indices.
        """
        return map(self.inputs_legs.__getitem__, node)

    def gen_leaves(self):
        """Generate the nodes representing leaves of the contraction tree, i.e.
        of size 1 each corresponding to a single input tensor.
        """
        return map(node_from_single, range(self.N))

    @classmethod
    def from_path(
        cls,
        inputs,
        output,
        size_dict,
        *,
        path=None,
        ssa_path=None,
        check=False,
        **kwargs,
    ):
        """Create a (completed) ``ContractionTree`` from the usual inputs plus
        a standard contraction path or 'ssa_path' - you need to supply one.
        """
        if int(path is None) + int(ssa_path is None) != 1:
            raise ValueError(
                "Exactly one of ``path`` or ``ssa_path`` must be " "supplied."
            )

        if ssa_path is not None:
            path = ssa_path

        tree = cls(inputs, output, size_dict, **kwargs)
        terms = list(tree.gen_leaves())

        for p in path:
            if ssa_path is not None:
                merge = [terms[i] for i in p]
            else:
                merge = [terms.pop(i) for i in sorted(p, reverse=True)]
            terms.append(tree.contract_nodes(merge, check=check))

        return tree

    @classmethod
    def from_info(cls, info, **kwargs):
        """Create a ``ContractionTree`` from an ``opt_einsum.PathInfo`` object.
        """
        return cls.from_path(
            inputs=info.input_subscripts.split(","),
            output=info.output_subscript,
            size_dict=info.size_dict,
            path=info.path,
            **kwargs,
        )

    @classmethod
    def from_eq(cls, eq, size_dict, **kwargs):
        """Create a empty ``ContractionTree`` directly from an equation and set
        of shapes.

        Parameters
        ----------
        eq : str
            The einsum string equation.
        size_dict : dict[str, int]
            The size of each index.
        """
        lhs, output = eq.split("->")
        inputs = lhs.split(",")
        return cls(inputs, output, size_dict, **kwargs)

    def get_eq(self):
        """Get the einsum equation corresponding to this tree. Note that this
        is the total (or original) equation, so includes indices which have
        been sliced.

        Returns
        -------
        eq : str
        """
        return ",".join(self.inputs) + "->" + self.output

    def get_shapes(self):
        """Get the shapes of the input tensors corresponding to this tree.

        Returns
        -------
        shapes : tuple[tuple[int]]
        """
        return tuple(
            tuple(self.size_dict[ix] for ix in term) for term in self.inputs
        )

    def get_eq_sliced(self):
        """Get the einsum equation corresponding to a single sliced of this
        tree, i.e. with sliced indices removed.

        Returns
        -------
        eq : str
        """
        return (
            ",".join(
                (
                    "".join(ix for ix in term if ix not in self.sliced_inds)
                    for term in self.inputs
                )
            )
            + "->"
            + "".join(ix for ix in self.output if ix not in self.sliced_inds)
        )

    def get_shapes_sliced(self):
        """Get the shapes of the input tensors corresponding to a single sliced
        of this tree, i.e. with sliced indices removed.

        Returns
        -------
        shapes : tuple[tuple[int]]
        """
        return tuple(
            tuple(
                self.size_dict[ix] for ix in term if ix not in self.sliced_inds
            )
            for term in self.inputs
        )

    @classmethod
    def from_edge_path(
        cls, edge_path, inputs, output, size_dict, check=False, **kwargs
    ):
        """Create a ``ContractionTree`` from an edge elimination ordering."""
        tree = cls(inputs, output, size_dict, **kwargs)
        nodes = list(tree.gen_leaves())

        for e in edge_path:
            # filter out the subgraph induced by edge `e` (generally a pair)
            new_terms, merge = [], []
            for node in nodes:
                term = union_it(tree.node_to_terms(node))
                if e in term:
                    merge.append(node)
                else:
                    new_terms.append(node)

            # contract the subgraph
            if merge:
                nodes = new_terms + [tree.contract_nodes(merge, check=check)]

        # make sure we are generating a full contraction tree
        nt = len(nodes)
        if nt > 1:
            # this seems to happen when the initial contraction contains a
            # scalar? Or disconnected subgraphs?
            warnings.warn(
                f"Ended up with {nt} nodes - contracting all remaining."
            )
            tree.contract_nodes(nodes, check=check)

        return tree

    def _add_node(self, node, check=False):
        if check:
            if len(self.info) > 2 * self.N - 1:
                raise ValueError("There are too many children already.")
            if len(self.children) > self.N - 1:
                raise ValueError("There are too many branches already.")
            if not is_valid_node(node):
                raise ValueError("{} is not a valid node.".format(node))

        self.info.setdefault(node, dict())

    def _remove_node(self, node):
        """Remove ``node`` from this tree and update the flops and maximum size
        if tracking them respectively. Inplace operation.
        """
        if self._track_flops:
            self._flops -= self.get_flops(node)

        if self._track_write and len(node) > 1:
            # only non-leaf nodes contribute to write
            self._write -= self.get_size(node)

        if self._track_size:
            self._sizes.discard(self.get_size(node))

        del self.info[node]
        del self.children[node]

    @cached_node_property("keep")
    def get_keep(self, node):
        """Get a set of at least the indices that should be explicitly kept if
        they appear on ``node`` (or below).
        """
        nodes_above = self.root.difference(node)
        terms_above = self.node_to_terms(nodes_above)
        return union_it((self.output_legs, *terms_above))

    @cached_node_property("legs")
    def get_legs(self, node):
        """Get the effective 'outer' indices for the collection of tensors
        in ``node``.
        """
        if len(node) == 1:
            return self.inputs_legs[node_get_single_el(node)]
        try:
            involved = self.get_involved(node)
        except KeyError:
            involved = union_it(self.node_to_terms(node))
        keep = self.get_keep(node)
        return involved.intersection(keep)

    @cached_node_property("involved")
    def get_involved(self, node):
        """Get all the indices involved in the formation of subgraph ``node``.
        """
        if len(node) == 1:
            return self.bitset_edges.infimum
        sub_legs = map(self.get_legs, self.children[node])
        return union_it(sub_legs)

    @cached_node_property("removed")
    def get_removed(self, node):
        """Get the indices that will be removed by the creation of ``node``."""
        return self.get_involved(node).difference(self.get_legs(node))

    @cached_node_property("size")
    def get_size(self, node):
        """Get the tensor size of ``node``."""
        return compute_size_by_dict(self.get_legs(node), self.size_dict)

    @cached_node_property("ops")
    def get_ops(self, node):
        """Get the number of scalar ops for the pairwise contraction."""
        if len(node) == 1:
            return 0
        involved = self.get_involved(node)
        return compute_size_by_dict(involved, self.size_dict)

    @cached_node_property("flops")
    def get_flops(self, node):
        """Get the FLOPs for the pairwise contraction that will create
        ``node``.
        """
        if len(node) == 1:
            return 0
        involved = self.get_involved(node)
        removed = self.get_removed(node)
        return flop_count(involved, removed, 2, self.size_dict)

    @cached_node_property("can_dot")
    def get_can_dot(self, node):
        """Get whether this contraction can be performed as a dot product (i.e.
        with ``tensordot``), or else requires ``einsum``, as it has indices
        that don't appear exactly twice in either the inputs or the output.
        """
        l, r = self.children[node]
        sp, sl, sr = map(self.get_legs, (node, l, r))
        return sl.symmetric_difference(sr) == sp

    @cached_node_property("inds")
    def get_inds(self, node):
        """Get the indices of this node - an ordered string version of
        ``get_legs`` that starts with ``tree.inputs`` and maintains the order
        they appear in each contraction 'ABC,abc->ABCabc', to match tensordot.
        """
        # NB: self.inputs and self.output contain the full (unsliced) indices
        #     thus we filter even the input legs and output legs

        if len(node) == 1:
            # leaf indices are fixed
            i = node_get_single_el(node)
            if not self.sliced_inds:
                return "".join(self.inputs[i])
            legs = self.get_legs(node)
            return "".join(filter(legs.__contains__, self.inputs[i]))

        if len(node) == self.N:
            # root (output) indices are fixed
            if not self.sliced_inds:
                return "".join(self.output)
            return "".join(filter(self.output_legs.__contains__, self.output))

        legs = self.get_legs(node)
        l_inds, r_inds = map(self.get_inds, self.children[node])
        # the filter here takes care of contracted indices
        return "".join(
            unique(filter(legs.__contains__, itertools.chain(l_inds, r_inds)))
        )

    @cached_node_property("tensordot_axes")
    def get_tensordot_axes(self, node):
        """Get the ``axes`` arg for a tensordot ocontraction that produces
        ``node``. The pairs are sorted in order of appearance on the left
        input.
        """
        l_inds, r_inds = map(self.get_inds, self.children[node])
        l_axes, r_axes = [], []
        for i, ind in enumerate(l_inds):
            j = r_inds.find(ind)
            if j != -1:
                l_axes.append(i)
                r_axes.append(j)
        return tuple(l_axes), tuple(r_axes)

    @cached_node_property("tensordot_perm")
    def get_tensordot_perm(self, node):
        """Get the permutation required, if any, to bring the tensordot output
        of this nodes contraction into line with ``self.get_inds(node)``.
        """
        l_inds, r_inds = map(self.get_inds, self.children[node])
        # the target output inds
        p_inds = self.get_inds(node)
        # the tensordot output inds
        td_inds = "".join(sorted(p_inds, key=f"{l_inds}{r_inds}".find))
        if td_inds == p_inds:
            return None
        return tuple(map(td_inds.find, p_inds))

    @cached_node_property("einsum_eq")
    def get_einsum_eq(self, node):
        """Get the einsum string describing the contraction that produces
        ``node``, unlike ``get_inds`` the characters are mapped into [a-zA-Z],
        for compatibility with ``numpy.einsum`` for example.
        """
        l, r = self.children[node]
        l_inds, r_inds, p_inds = map(self.get_inds, (l, r, node))
        # we need to map any extended unicode characters into ascii
        char_mapping = {
            ord(ix): get_symbol(i)
            for i, ix in enumerate(unique(itertools.chain(l_inds, r_inds)))
        }
        return f"{l_inds},{r_inds}->{p_inds}".translate(char_mapping)

    def get_centrality(self, node):
        try:
            return self.info[node]["centrality"]
        except KeyError:
            self.compute_centralities()
            return self.info[node]["centrality"]

    def total_flops(self, dtype="float"):
        """Sum the flops contribution from every node in the tree.

        Parameters
        ----------
        dtype : {'float', 'complex', None}, optional
            Scale the answer depending on the assumed data type.
        """
        if self._track_flops:
            real_flops = self.multiplicity * self._flops

        else:
            self._flops = 0
            for node, _, _ in self.traverse():
                self._flops += self.get_flops(node)

            self._track_flops = True
            real_flops = self.multiplicity * self._flops

        if dtype is None:
            return real_flops // 2

        if "float" in dtype:
            return real_flops

        if "complex" in dtype:
            return real_flops * 4

    def total_write(self):
        """Sum the total amount of memory that will be created and operated on.
        """
        if not self._track_write:
            self._write = 0
            for node, _, _ in self.traverse():
                self._write += self.get_size(node)

            self._track_write = True

        return self.multiplicity * self._write

    def total_cost(self, factor=DEFAULT_COMBO_FACTOR, combine=sum):
        t = 0
        for p in self.children:
            f = self.get_flops(p) // 2
            w = self.get_size(p)
            t += combine((f, factor * w))
        return self.multiplicity * t

    def max_size(self):
        """The size of the largest intermediate tensor."""
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
        return self.total_flops(dtype=None) / self.total_write()

    def contraction_cost(self):
        """Get the total number of scalar operations ~ time complexity."""
        return float(self.total_flops(dtype=None))

    def contraction_width(self):
        """Get log2 of the size of the largest tensor."""
        return math.log2(self.max_size())

    def compressed_contract_stats(
        self,
        chi,
        order="surface_order",
        compress_late=False,
    ):
        hg = self.get_hypergraph(accel="auto")

        # conversion between tree nodes <-> hypergraph nodes during contraction
        tree_map = dict(zip(self.gen_leaves(), range(hg.get_num_nodes())))

        tracker = CompressedStatsTracker(hg, chi)

        for p, l, r in self.traverse(order):
            li = tree_map[l]
            ri = tree_map[r]

            tracker.update_pre_step()

            if compress_late:
                tracker.update_pre_compress(hg, li, ri)
                # compress just before we contract tensors
                hg.compress(chi=chi, edges=hg.get_node(li))
                hg.compress(chi=chi, edges=hg.get_node(ri))
                tracker.update_post_compress(hg, li, ri)

            tracker.update_pre_contract(hg, li, ri)
            pi = tree_map[p] = hg.contract(li, ri)
            tracker.update_post_contract(hg, pi)

            if not compress_late:
                # compress as soon as we can after contracting tensors
                tracker.update_pre_compress(hg, pi)
                hg.compress(chi=chi, edges=hg.get_node(pi))
                tracker.update_post_compress(hg, pi)

            tracker.update_post_step()

        return tracker

    def total_flops_compressed(
        self, chi, order="surface_order", compress_late=False, dtype=None,
    ):
        """Estimate the total flops for a compressed contraction of this tree
        with maximum bond size ``chi``. This includes basic estimates of the
        ops to perform contractions, QRs and SVDs.
        """
        if dtype is not None:
            raise ValueError(
                "Can only estimate cost in terms of "
                "number of abstract scalar ops."
            )

        return self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).flops

    def total_write_compressed(
        self, chi, order="surface_order", compress_late=False, accel="auto"
    ):
        """Compute the total size of all intermediate tensors when a
        compressed contraction is performed with maximum bond size ``chi``,
        ordered by ``order``. This is relevant maybe for time complexity and
        e.g. autodiff space complexity (since every intermediate is kept).
        """
        return self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).write

    def total_cost_compressed(
        self,
        chi,
        order="surface_order",
        compress_late=False,
        factor=DEFAULT_COMBO_FACTOR,
    ):
        return self.total_flops_compressed(
            chi=chi, order=order, compress_late=compress_late
        ) + factor * self.total_write_compressed(
            chi=chi, order=order, compress_late=compress_late
        )

    def max_size_compressed(
        self, chi, order="surface_order", compress_late=False
    ):
        """Compute the maximum sized tensor produced when a compressed
        contraction is performed with maximum bond size ``chi``, ordered by
        ``order``. This is close to the ideal space complexity if only
        tensors that are being directly operated on are kept in memory.
        """
        return self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).max_size

    def peak_size_compressed(
        self, chi, order="surface_order", compress_late=False, accel="auto"
    ):
        """Compute the peak size of combined intermediate tensors when a
        compressed contraction is performed with maximum bond size ``chi``,
        ordered by ``order``. This is the practical space complexity if one is
        not swapping intermediates in and out of memory.
        """
        return self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).peak_size

    contraction_cost_compressed = total_cost_compressed

    def contraction_width_compressed(
        self, chi, order="surface_order", compress_late=False
    ):
        """Compute log2 of the maximum sized tensor produced when a compressed
        contraction is performed with maximum bond size ``chi``, ordered by
        ``order``.
        """
        return math.log2(self.max_size_compressed(chi, order, compress_late))

    def contract_nodes_pair(self, x, y, check=False):
        """Contract node ``x`` with node ``y`` in the tree to create a new
        parent node.
        """
        parent = x.union(y)

        # make sure info entries exist for all (default dict)
        for node in (x, y, parent):
            self._add_node(node, check=check)

        # enforce left ordering of 'heaviest' subtrees
        nx, ny = len(x), len(y)
        hx, hy = hash(x), hash(y)

        # deterministically break ties
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

    def contract_nodes(
        self,
        nodes,
        optimize="auto-hq",
        check=False,
        extra_opts=None,
    ):
        """Contract an arbitrary number of ``nodes`` in the tree to build up a
        subtree. The root of this subtree (a new intermediate) is returned.
        """
        if len(nodes) == 1:
            return next(iter(nodes))

        if len(nodes) == 2:
            return self.contract_nodes_pair(*nodes, check=check)

        # create the bottom and top nodes
        grandparent = union_it(nodes)
        self._add_node(grandparent, check=check)
        for node in nodes:
            self._add_node(node, check=check)

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

        if extra_opts is None:
            path = path_fn(path_inputs, path_output, self.size_dict)
        else:
            path = path_fn(
                path_inputs, path_output, self.size_dict, **extra_opts
            )

        # now we have path create the nodes in between
        temp_nodes = list(nodes)
        for p in path:
            to_contract = [temp_nodes.pop(i) for i in sorted(p, reverse=True)]
            temp_nodes.append(
                self.contract_nodes(
                    to_contract, optimize=optimize, check=check
                )
            )

        (parent,) = temp_nodes

        if check:
            # final remaining temp input should be the 'grandparent'
            assert parent == grandparent

        return parent

    def is_complete(self):
        """Check every node has two children, unless it is a leaf."""
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

    def get_default_order(self):
        return "dfs"

    def _traverse_ordered(self, order):
        """Traverse the tree in the order that minimizes ``order(node)``, but
        still contrained to produce children before parents.
        """
        from bisect import bisect

        if order == "surface_order":
            order = self.surface_order

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
        generator[tuple[node]]
            The bottom up ordered sequence of tree merges, each a
            tuple of ``(parent, left_child, right_child)``.

        See Also
        --------
        descend
        """
        if order is None:
            order = self.get_default_order()

        if order != "dfs":
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

    def descend(self, mode="dfs"):
        """Generate, from root to leaves, all the node merges in this tree.
        Non-recursive! This ensures parents are visited before their children.

        Parameters
        ----------
        mode : {'dfs', bfs}, optional
            How expand from a parent.

        Returns
        -------
        generator[tuple[node]
            The top down ordered sequence of tree merges, each a
            tuple of ``(parent, left_child, right_child)``.

        See Also
        --------
        traverse
        """
        queue = [self.root]
        while queue:
            if mode == "dfs":
                parent = queue.pop(-1)
            elif mode == "bfs":
                parent = queue.pop(0)
            l, r = self.children[parent]
            yield parent, l, r
            if len(l) > 1:
                queue.append(l)
            if len(r) > 1:
                queue.append(r)

    def get_subtree(self, node, size, search="bfs"):
        """Get a subtree spanning down from ``node`` which will have ``size``
        leaves (themselves not necessarily leaves of the actual tree).

        Parameters
        ----------
        node : node
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
        sub_leaves : tuple[node]
            Nodes which are subtree leaves.
        branches : tuple[node]
            Nodes which are between the subtree leaves and root.
        """
        # nodes which are subtree leaves
        branches = []

        # actual tree leaves - can't expand
        real_leaves = []

        # nodes to expand
        queue = [node]

        while (len(queue) + len(real_leaves) < size) and queue:
            if search == "bfs":
                p = queue.pop(0)
            elif search == "dfs":
                p = queue.pop(-1)
            elif search == "random":
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

    def remove_ind(self, ind, inplace=False):
        """Remove (i.e. slice) index ``ind`` from this contraction tree,
        taking care to update all relevant information about each node.
        """
        tree = self if inplace else self.copy()

        # make sure all flops and size information has been populated
        tree.total_flops()
        tree.total_write()
        tree.max_size()

        d = tree.size_dict[ind]
        s_ind = self.bitset_edges.frommembers((ind,))

        for node, node_info in tree.info.items():
            # if ind doesn't feature in this node (contraction) nothing to do
            involved = tree.get_involved(node)

            # inputs can have leg indices that are not involved so
            legs = tree.get_legs(node)

            if not ((s_ind & involved) or (s_ind & legs)):
                continue

            # else update all the relevant information about this node
            node_info["involved"] = involved.difference(s_ind)
            removed = tree.get_removed(node)

            # update information regarding node indices sets
            if s_ind & legs:
                # removing indices changes both flops and size of node
                node_info["legs"] = legs.difference(s_ind)

                old_size = tree.get_size(node)
                tree._sizes.discard(old_size)
                new_size = old_size // d
                tree._sizes.add(new_size)
                node_info["size"] = new_size

                if len(node) > 1:
                    # only non-leaf nodes contribute to write
                    tree._write += -old_size + new_size

                # XXX: modifying 'keep' not stricly necessarily as its only
                #     needed for ``legs = keep.intersection(involved)``?
                keep = tree.get_keep(node)
                node_info["keep"] = keep.difference(s_ind)
            else:
                # removing indices only changes flops
                node_info["removed"] = removed.difference(s_ind)

            old_flops = tree.get_flops(node)
            new_flops = old_flops // d
            if len(removed) == 1:
                # if ind was the last contracted index then have outer product
                new_flops //= 2
            node_info["flops"] = new_flops
            tree._flops += -old_flops + new_flops

            if len(node) == 1:
                # its a leaf - corresponding input will be sliced
                i = node_get_single_el(node)
                tree.sliced_inputs = tree.sliced_inputs | frozenset([i])
                tree.inputs_legs[i] = tree.inputs_legs[i] - s_ind
            elif len(node) == tree.N:
                # root node
                tree.output_legs = tree.output_legs - s_ind

            # delete info we can't change
            for k in (
                "inds",
                "einsum_eq",
                "can_dot",
                "tensordot_axes",
                "tensordot_perm",
            ):
                tree.info[node].pop(k, None)

        tree.multiplicity = tree.multiplicity * d

        # update the tuples of sliced indices and sliced sizes, but maintain
        # the order such that output sliced indices always appear first
        tree.sliced_inds, tree.sliced_sizes = zip(
            *sorted(
                zip(
                    itertools.chain(tree.sliced_inds, (ind,)),
                    itertools.chain(tree.sliced_sizes, (tree.size_dict[ind],)),
                ),
                key=lambda x: (x[0] not in tree.output, x),
            )
        )

        tree.already_optimized.clear()
        tree.contraction_cores.clear()

        return tree

    remove_ind_ = functools.partialmethod(remove_ind, inplace=True)

    def calc_subtree_candidates(self, pwr=2, what="flops"):
        candidates = list(self.children)

        if what == "size":
            weights = [self.get_size(x) for x in candidates]

        elif what == "flops":
            weights = [self.get_flops(x) for x in candidates]

        max_weight = max(weights)

        # can be bigger than numpy int/float allows
        weights = [float(w / max_weight) ** (1 / pwr) for w in weights]

        # sort by descending score
        candidates, weights = zip(
            *sorted(zip(candidates, weights), key=lambda x: -x[1])
        )

        return list(candidates), list(weights)

    def subtree_reconfigure(
        self,
        subtree_size=8,
        subtree_search="bfs",
        weight_what="flops",
        weight_pwr=2,
        select="max",
        maxiter=500,
        seed=None,
        minimize="flops",
        optimize=None,
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

        scorer = get_score_fn(minimize)

        if optimize is None:
            opt = DynamicProgramming(
                minimize=scorer.get_dynamic_programming_minimize()
            )
        else:
            opt = optimize

        cost = getattr(scorer, "cost_local_tree_node", lambda _: 1)

        # different caches as we might want to reconfigure one before other
        self.already_optimized.setdefault(minimize, set())
        already_optimized = self.already_optimized[minimize]

        if seed is not None:
            random.seed(seed)

        candidates, weights = self.calc_subtree_candidates(
            pwr=weight_pwr, what=weight_what
        )

        if progbar:
            import tqdm

            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree), refresh=False)

        r = 0
        try:
            while candidates and r < maxiter:
                if select == "max":
                    i = 0
                elif select == "min":
                    i = -1
                elif select == "random":
                    (i,) = random.choices(
                        range(len(candidates)), weights=weights
                    )

                weights.pop(i)
                sub_root = candidates.pop(i)

                # get a subtree to possibly reconfigure
                sub_leaves, sub_branches = tree.get_subtree(
                    sub_root, size=subtree_size, search=subtree_search
                )

                sub_leaves = frozenset(sub_leaves)

                # check if its already been optimized
                if sub_leaves in already_optimized:
                    continue

                # else remove the branches, keeping track of current cost
                current_cost = cost(tree, sub_root)
                for node in sub_branches:
                    if minimize == "size":
                        current_cost = max(current_cost, cost(tree, node))
                    else:
                        current_cost += cost(tree, node)
                    tree._remove_node(node)

                # make the optimizer more efficient by supplying accurate cap
                opt.cost_cap = max(1, current_cost)

                # and reoptimize the leaves
                tree.contract_nodes(sub_leaves, optimize=opt)
                already_optimized.add(sub_leaves)

                r += 1

                if progbar:
                    pbar.update()
                    pbar.set_description(_describe_tree(tree), refresh=False)

                # if we have reconfigured simply re-add all candidates
                candidates, weights = tree.calc_subtree_candidates(
                    pwr=weight_pwr, what=weight_what
                )
        finally:
            if progbar:
                pbar.close()

        # invalidate any compiled contractions
        tree.contraction_cores.clear()

        return tree

    subtree_reconfigure_ = functools.partialmethod(
        subtree_reconfigure, inplace=True
    )

    def subtree_reconfigure_forest(
        self,
        num_trees=8,
        num_restarts=10,
        restart_fraction=0.5,
        subtree_maxiter=100,
        subtree_size=10,
        subtree_search=("random", "bfs"),
        subtree_select=("random",),
        subtree_weight_what=("flops", "size"),
        subtree_weight_pwr=(2,),
        parallel="auto",
        parallel_maxiter_steps=4,
        minimize="flops",
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
        minimize : {'flops', 'size', ..., Objective}, optional
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
        is_scatter_pool = can_scatter(pool)
        if is_scatter_pool:
            is_worker = maybe_leave_pool(pool)
            # store the trees as futures for the entire process
            forest = [scatter(pool, tree)]
            maxiter = subtree_maxiter // parallel_maxiter_steps
        else:
            forest = [tree]
            maxiter = subtree_maxiter

        if progbar:
            import tqdm

            pbar = tqdm.tqdm(total=num_restarts)
            pbar.set_description(_describe_tree(tree), refresh=False)

        try:
            for _ in range(num_restarts):
                # on the next round take only the best trees
                forest = itertools.cycle(forest[:num_keep])

                # select some random configurations
                saplings = [
                    {
                        "tree": next(forest),
                        "maxiter": maxiter,
                        "minimize": minimize,
                        "subtree_size": subtree_size,
                        "subtree_search": random.choice(subtree_search),
                        "select": random.choice(subtree_select),
                        "weight_pwr": random.choice(subtree_weight_pwr),
                        "weight_what": random.choice(subtree_weight_what),
                    }
                    for _ in range(num_trees)
                ]

                if pool is None:
                    forest = [_reconfigure_tree(**s) for s in saplings]
                    res = [{"tree": t, **_get_tree_info(t)} for t in forest]
                elif not is_scatter_pool:
                    forest_futures = [
                        submit(pool, _reconfigure_tree, **s) for s in saplings
                    ]
                    forest = [f.result() for f in forest_futures]
                    res = [{"tree": t, **_get_tree_info(t)} for t in forest]
                else:
                    # submit in smaller steps to saturate processes
                    for _ in range(parallel_maxiter_steps):
                        for s in saplings:
                            s["tree"] = submit(pool, _reconfigure_tree, **s)

                    # compute scores remotely then gather
                    forest_futures = [s["tree"] for s in saplings]
                    res_futures = [
                        submit(pool, _get_tree_info, t) for t in forest_futures
                    ]
                    res = [
                        {"tree": tree_future, **res_future.result()}
                        for tree_future, res_future in zip(
                            forest_futures, res_futures
                        )
                    ]

                # update the order of the new forest
                res.sort(key=score)
                forest = [r["tree"] for r in res]

                if progbar:
                    pbar.update()
                    if pool is None:
                        d = _describe_tree(forest[0])
                    else:
                        d = submit(pool, _describe_tree, forest[0]).result()
                    pbar.set_description(d, refresh=False)

        finally:
            if progbar:
                pbar.close()

        if is_scatter_pool:
            tree.set_state_from(forest[0].result())
            maybe_rejoin_pool(is_worker, pool)
        else:
            tree.set_state_from(forest[0])

        return tree

    subtree_reconfigure_forest_ = functools.partialmethod(
        subtree_reconfigure_forest, inplace=True
    )

    def slice(
        self,
        target_size=None,
        target_overhead=None,
        target_slices=None,
        temperature=0.01,
        minimize="flops",
        allow_outer=True,
        max_repeats=16,
        inplace=False,
    ):
        """Slice this tree (turn some indices into indices which are explicitly
        summed over rather than being part of contractions). The indices are
        stored in ``tree.sliced_inds``, and the contraction width updated to
        take account of the slicing. Calling ``tree.contract(arrays)`` moreover
        which automatically perform the slicing and summation.

        Parameters
        ----------
        target_size : int, optional
            The target number of entries in the largest tensor of the sliced
            contraction. The search algorithm will terminate after this is
            reached.
        target_slices : int, optional
            The target or minimum number of 'slices' to consider - individual
            contractions after slicing indices. The search algorithm will
            terminate after this is breached.
        target_overhead : float, optional
            The target increase in total number of floating point operations.
            For example, a value of ``2.0`` will terminate the search just
            before the cost of computing all the slices individually breaches
            twice that of computing the original contraction all at once.
        temperature : float, optional
            How much to randomize the repeated search.
        minimize : {'flops', 'size', ..., Objective}, optional
            Which metric to score the overhead increase against.
        allow_outer : bool, optional
            Whether to allow slicing of outer indices.
        max_repeats : int, optional
            How many times to repeat the search with a slight randomization.
        inplace : bool, optional
            Whether the remove the indices from this tree inplace or not.

        Returns
        -------
        ContractionTree

        See Also
        --------
        SliceFinder, ContractionTree.slice_and_reconfigure
        """
        from .slicer import SliceFinder

        tree = self if inplace else self.copy()

        sf = SliceFinder(
            tree,
            target_size=target_size,
            target_overhead=target_overhead,
            target_slices=target_slices,
            temperature=temperature,
            minimize=minimize,
            allow_outer=allow_outer,
        )

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
        minimize="flops",
        allow_outer=True,
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
        minimize : {'flops', 'size', ..., Objective}, optional
            The metric to minimize when slicing and reconfiguring subtrees.
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
        minimize = get_score_fn(minimize)
        reconf_opts.setdefault("minimize", minimize)
        forested_reconf = reconf_opts.pop("forested", False)

        if progbar:
            import tqdm

            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree), refresh=False)

        try:
            while tree.max_size() > target_size:
                tree.slice_(
                    temperature=temperature,
                    target_slices=step_size,
                    minimize=minimize,
                    allow_outer=allow_outer,
                    max_repeats=max_repeats,
                )
                if forested_reconf:
                    tree.subtree_reconfigure_forest_(**reconf_opts)
                else:
                    tree.subtree_reconfigure_(**reconf_opts)

                if progbar:
                    pbar.update()
                    pbar.set_description(_describe_tree(tree), refresh=False)
        finally:
            if progbar:
                pbar.close()

        return tree

    slice_and_reconfigure_ = functools.partialmethod(
        slice_and_reconfigure, inplace=True
    )

    def slice_and_reconfigure_forest(
        self,
        target_size,
        step_size=2,
        num_trees=8,
        restart_fraction=0.5,
        temperature=0.02,
        max_repeats=32,
        minimize="flops",
        allow_outer=True,
        parallel="auto",
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
        is_scatter_pool = can_scatter(pool)
        if is_scatter_pool:
            is_worker = maybe_leave_pool(pool)
            # store the trees as futures for the entire process
            forest = [scatter(pool, tree)]
        else:
            forest = [tree]

        if progbar:
            import tqdm

            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree), refresh=False)

        next_size = tree.max_size()

        try:
            while True:
                next_size //= step_size

                # on the next round take only the best trees
                forest = itertools.cycle(forest[:num_keep])

                saplings = [
                    {
                        "tree": next(forest),
                        "target_size": next_size,
                        "step_size": step_size,
                        "temperature": temperature,
                        "max_repeats": max_repeats,
                        "reconf_opts": reconf_opts,
                        "allow_outer": allow_outer,
                    }
                    for _ in range(num_trees)
                ]

                if pool is None:
                    forest = [
                        _slice_and_reconfigure_tree(**s) for s in saplings
                    ]
                    res = [{"tree": t, **_get_tree_info(t)} for t in forest]

                elif not is_scatter_pool:
                    # simple pool with no pass by reference
                    forest_futures = [
                        submit(pool, _slice_and_reconfigure_tree, **s)
                        for s in saplings
                    ]
                    forest = [f.result() for f in forest_futures]
                    res = [{"tree": t, **_get_tree_info(t)} for t in forest]

                else:
                    forest_futures = [
                        submit(pool, _slice_and_reconfigure_tree, **s)
                        for s in saplings
                    ]

                    # compute scores remotely then gather
                    res_futures = [
                        submit(pool, _get_tree_info, t) for t in forest_futures
                    ]
                    res = [
                        {"tree": tree_future, **res_future.result()}
                        for tree_future, res_future in zip(
                            forest_futures, res_futures
                        )
                    ]

                # we want to sort by flops, but also favour sampling as
                # many different sliced index combos as possible
                #    ~ [1, 1, 1, 2, 2, 3] -> [1, 2, 3, 1, 2, 1]
                res.sort(key=score)
                res = list(
                    interleave(
                        groupby(lambda r: r["sliced_inds"], res).values()
                    )
                )

                # update the order of the new forest
                forest = [r["tree"] for r in res]

                if progbar:
                    pbar.update()
                    if pool is None:
                        d = _describe_tree(forest[0])
                    else:
                        d = submit(pool, _describe_tree, forest[0]).result()
                    pbar.set_description(d, refresh=False)

                if res[0]["size"] <= target_size:
                    break

        finally:
            if progbar:
                pbar.close()

        if is_scatter_pool:
            tree.set_state_from(forest[0].result())
            maybe_rejoin_pool(is_worker, pool)
        else:
            tree.set_state_from(forest[0])

        return tree

    slice_and_reconfigure_forest_ = functools.partialmethod(
        slice_and_reconfigure_forest, inplace=True
    )

    def compressed_reconfigure(
        self,
        chi,
        minimize="peak",
        compress_late=True,
        order_only=False,
        max_nodes="auto",
        max_time=None,
        local_score=None,
        exploration_power=0,
        best_score=None,
        progbar=False,
        inplace=False,
    ):
        """Reconfigure this tree according to ``peak_size_compressed``.

        Parameters
        ----------
        chi : int
            The maximum bond dimension to consider.
        order_only : bool, optional
            Whether to only consider the ordering of the current tree
            contractions, or all possible contractions, starting with the
            current.
        max_nodes : int, optional
            Set the maximum number of contraction steps to consider.
        max_time : float, optional
            Set the maximum time to spend on the search.
        local_score : callable, optional
            A function that assigns a score to a potential contraction, with a
            lower score giving more priority to explore that contraction
            earlier. It should have signature::

                local_score(step, new_score, dsize, new_size)

            where ``step`` is the number of steps so far, ``new_score`` is the
            score of the contraction so far, ``dsize`` is the change in memory
            by the current step, and ``new_size`` is the new memory size after
            contraction.
        exploration_power : float, optional
            If not ``0.0``, the inverse power to which the step is raised in
            the default local score function. Higher values favor exploring
            more promising branches early on - at the cost of increased memory.
            Ignored if ``local_score`` is supplied.
        best_score : float, optional
            Manually specify an upper bound for best score found so far.
        progbar : bool, optional
            If ``True``, display a progress bar.
        inplace : bool, optional
            Whether to perform the reconfiguration inplace on this tree.

        Returns
        -------
        ContractionTree
        """
        from .pathfinders.path_compressed import CompressedExhaustive

        minimize = minimize.replace("-compressed", "")

        if max_nodes == "auto":
            if max_time is None:
                max_nodes = max(10_000, self.N**2)
            else:
                max_nodes = float("inf")

        opt = CompressedExhaustive(
            chi=chi,
            minimize=minimize,
            compress_late=compress_late,
            local_score=local_score,
            max_nodes=max_nodes,
            max_time=max_time,
            exploration_power=exploration_power,
            best_score=best_score,
            progbar=progbar,
        )
        opt.setup(self.inputs, self.output, self.size_dict)
        opt.explore_path(self.get_path_surface(), restrict=order_only)

        # rtree = opt.search(self.inputs, self.output, self.size_dict)

        opt.run(self.inputs, self.output, self.size_dict)
        ssa_path = opt.ssa_path
        # ssa_path = opt(self.inputs, self.output, self.size_dict)
        rtree = self.__class__.from_path(
            self.inputs,
            self.output,
            self.size_dict,
            ssa_path=ssa_path,
        )
        if inplace:
            self.set_state_from(rtree)
            rtree = self
        rtree.set_surface_order_from_path(ssa_path)
        return rtree

    compressed_reconfigure_ = functools.partialmethod(
        compressed_reconfigure, inplace=True
    )

    def windowed_reconfigure(
        self,
        minimize,
        order_only=False,
        window_size=20,
        max_iterations=100,
        max_window_tries=1000,
        score_temperature=0.0,
        queue_temperature=1.0,
        scorer=None,
        queue_scorer=None,
        seed=None,
        inplace=False,
        progbar=False,
        **kwargs,
    ):
        from .pathfinders.path_compressed import WindowedOptimizer

        wo = WindowedOptimizer(
            self.inputs,
            self.output,
            self.size_dict,
            minimize=minimize,
            ssa_path=self.get_ssa_path(),
            seed=seed,
        )

        wo.refine(
            window_size=window_size,
            max_iterations=max_iterations,
            order_only=order_only,
            max_window_tries=max_window_tries,
            score_temperature=score_temperature,
            queue_temperature=queue_temperature,
            scorer=scorer,
            queue_scorer=queue_scorer,
            progbar=progbar,
            **kwargs,
        )
        ssa_path = wo.get_ssa_path()

        rtree = self.__class__.from_path(
            self.inputs,
            self.output,
            self.size_dict,
            ssa_path=ssa_path,
        )

        if inplace:
            self.set_state_from(rtree)
            rtree = self
        rtree.set_surface_order_from_path(ssa_path)

        return rtree

    windowed_reconfigure_ = functools.partialmethod(
        windowed_reconfigure, inplace=True
    )

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

        Where each integer represents a leaf (i.e. single element node).
        """
        tups = dict(zip(self.gen_leaves(), range(self.N)))

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
            nd
            for nd in itertools.chain.from_iterable(self.traverse())
            if len(nd) == 1
        )

    def get_path(self, order=None):
        """Generate a standard path from the contraction tree."""
        path = []
        terms = list(self.gen_leaves())

        for parent, l, r in self.traverse(order=order):
            i, j = sorted((terms.index(l), terms.index(r)))
            terms.pop(j)
            terms.pop(i)
            path.append((i, j))
            terms.append(parent)

        return tuple(path)

    path = deprecated(get_path, "path", "get_path")

    def get_numpy_path(self, order=None):
        """Generate a path compatible with the `optimize` kwarg of
        `numpy.einsum`.
        """
        return ["einsum_path", *self.get_path(order=order)]

    def get_ssa_path(self, order=None):
        """Generate a ssa path from the contraction tree."""
        ssa_path = []
        pos = dict(zip(self.gen_leaves(), range(self.N)))

        for parent, l, r in self.traverse(order=order):
            i, j = sorted((pos[l], pos[r]))
            ssa_path.append((i, j))
            pos[parent] = len(ssa_path) + self.N - 1

        return tuple(ssa_path)

    ssa_path = deprecated(get_ssa_path, "ssa_path", "get_ssa_path")

    def surface_order(self, node):
        return (len(node), self.get_centrality(node))

    def set_surface_order_from_path(self, ssa_path):
        o = {}
        nodes = list(self.gen_leaves())
        for j, p in enumerate(ssa_path):
            l, r = (nodes[i] for i in p)
            p = l.union(r)
            nodes.append(p)
            o[p] = j

        self.surface_order = functools.partial(
            get_with_default, obj=o, default=float("inf")
        )

    def get_path_surface(self):
        return self.get_path(order=self.surface_order)

    path_surface = deprecated(
        get_path_surface, "path_surface", "get_path_surface"
    )

    def get_ssa_path_surface(self):
        return self.get_ssa_path(order=self.surface_order)

    ssa_path_surface = deprecated(
        get_ssa_path_surface, "ssa_path_surface", "get_ssa_path_surface"
    )

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
                "map": {self.root: node_from_single(l2)},
                # the leaf nodes in the spanning tree
                "spine": {l1, l2},
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
                        candidates.append(
                            {
                                "map": {child: child, **cand["map"]},
                                "spine": cand["spine"].copy(),
                            }
                        )

                    for l1, l2 in boundary_pairs(child):
                        if (l1 in cand["spine"]) or (l2 not in cand["spine"]):
                            # pair does not merge inwards into spine
                            continue

                        # valid extension of spanning tree
                        candidates.append(
                            {
                                "map": {
                                    child: node_from_single(l2),
                                    **cand["map"],
                                },
                                "spine": cand["spine"] | {l1, l2},
                            }
                        )

        return tuple(c["map"] for c in candidates)

    def compute_centralities(self, combine="mean"):
        """Compute a centrality for every node in this contraction tree."""
        hg = self.get_hypergraph(accel="auto")
        cents = hg.simple_centrality()

        for i, leaf in enumerate(self.gen_leaves()):
            self.info[leaf]["centrality"] = cents[i]

        combine = {
            "mean": lambda x, y: (x + y) / 2,
            "sum": lambda x, y: (x + y),
            "max": max,
            "min": min,
        }.get(combine, combine)

        for p, l, r in self.traverse("dfs"):
            self.info[p]["centrality"] = combine(
                self.info[l]["centrality"], self.info[r]["centrality"]
            )

    def get_hypergraph(self, accel=False):
        """Get a hypergraph representing the uncontracted network (i.e. the
        leaves).
        """
        return get_hypergraph(self.inputs, self.output, self.size_dict, accel)

    def reset_contraction_indices(self):
        """Reset all information regarding the explicit contraction indices
        ordering.
        """
        # delete all derived information
        for node in self.children:
            for k in (
                "inds",
                "einsum_eq",
                "can_dot",
                "tensordot_axes",
                "tensordot_perm",
            ):
                self.info[node].pop(k, None)

        # invalidate any compiled contractions
        self.contraction_cores.clear()

    def sort_contraction_indices(
        self,
        priority="flops",
        make_output_contig=True,
        make_contracted_contig=True,
        reset=True,
    ):
        """Set explicit orders for the contraction indices of this self to
        optimize for one of two things: contiguity in contracted ('k') indices,
        or contiguity of left and right output ('m' and 'n') indices.

        Parameters
        ----------
        priority : {'flops', 'size', 'root', 'leaves'}, optional
            Which order to process the intermediate nodes in. Later nodes
            re-sort previous nodes so are more likely to keep their ordering.
            E.g. for 'flops' the mostly costly contracton will be process last
            and thus will be guaranteed to have its indices exactly sorted.
        make_output_contig : bool, optional
            When processing a pairwise contraction, sort the parent contraction
            indices so that the order of indices is the order they appear
            from left to right in the two child (input) tensors.
        make_contracted_contig : bool, optional
            When processing a pairwise contraction, sort the child (input)
            tensor indices so that all contracted indices appear contiguously.
        reset : bool, optional
            Reset all indices to the default order before sorting.
        """
        if reset:
            self.reset_contraction_indices()

        if priority == "flops":
            nodes = sorted(
                self.children.items(), key=lambda x: self.get_flops(x[0])
            )
        elif priority == "size":
            nodes = sorted(
                self.children.items(), key=lambda x: self.get_size(x[0])
            )
        elif priority == "root":
            nodes = ((p, (l, r)) for p, l, r in self.traverse())
        elif priority == "leaves":
            nodes = ((p, (l, r)) for p, l, r in self.descend())
        else:
            raise ValueError(priority)

        for p, (l, r) in nodes:
            p_inds, l_inds, r_inds = map(self.get_inds, (p, l, r))

            if make_output_contig and len(p) != self.N:
                # sort indices by whether they appear in the left or right
                # whether this happens before or after the sort below depends
                # on the order we are processing the nodes
                # (avoid root as don't want to modify output)

                def psort(ix):
                    # group by whether in left or right input
                    return (r_inds.find(ix), l_inds.find(ix))

                p_inds = "".join(sorted(p_inds, key=psort))
                self.info[p]["inds"] = p_inds

            if make_contracted_contig:
                # sort indices by:
                # 1. if they are going to be contracted
                # 2. what order they appear in the parent indices
                # (but ignore leaf indices)
                if len(l) != 1:

                    def lsort(ix):
                        return (r_inds.find(ix), p_inds.find(ix))

                    l_inds = "".join(sorted(self.get_legs(l), key=lsort))
                    self.info[l]["inds"] = l_inds

                if len(r) != 1:

                    def rsort(ix):
                        return (p_inds.find(ix), l_inds.find(ix))

                    r_inds = "".join(sorted(self.get_legs(r), key=rsort))
                    self.info[r]["inds"] = r_inds

        # invalidate any compiled contractions
        self.contraction_cores.clear()

    def print_contractions(self, sort=None, show_brackets=True):
        """Print each pairwise contraction, with colorized indices (if
        `colorama` is installed), and other information.
        """
        try:
            from colorama import Fore

            RESET = Fore.RESET
            GREY = Fore.WHITE
            PINK = Fore.MAGENTA
            RED = Fore.RED
            BLUE = Fore.BLUE
            GREEN = Fore.GREEN
        except ImportError:
            RESET = GREY = PINK = RED = BLUE = GREEN = ""

        entries = []

        for i, (p, l, r) in enumerate(self.traverse()):
            p_legs, l_legs, r_legs = map(self.get_legs, [p, l, r])
            p_inds, l_inds, r_inds = map(self.get_inds, [p, l, r])

            # print sizes and flops
            p_flops = self.get_flops(p)
            p_sz, l_sz, r_sz = (
                math.log2(self.get_size(node)) for node in [p, l, r]
            )
            # print whether tensordottable
            if self.get_can_dot(p):
                type_msg = "tensordot"
                perm = self.get_tensordot_perm(p)
                if perm is not None:
                    # and whether indices match tensordot
                    type_msg += "+perm"
            else:
                type_msg = "einsum"

            pa = "".join(
                PINK + f"({ix})"
                if ix in l_legs & r_legs
                else GREEN + f"({ix})"
                if ix in r_legs
                else BLUE + ix
                for ix in p_inds
            ).replace(f"){GREEN}(", "")
            la = "".join(
                PINK + f"[{ix}]"
                if ix in p_legs & r_legs
                else RED + f"[{ix}]"
                if ix in r_legs
                else BLUE + ix
                for ix in l_inds
            ).replace(f"]{RED}[", "")
            ra = "".join(
                PINK + f"[{ix}]"
                if ix in p_legs & l_legs
                else RED + f"[{ix}]"
                if ix in l_legs
                else GREEN + ix
                for ix in r_inds
            ).replace(f"]{RED}[", "")

            entries.append(
                (
                    p,
                    f"{GREY}({i}) cost: {RESET}{p_flops:.1e} "
                    f"{GREY}widths: {RESET}{l_sz:.1f},{r_sz:.1f}->{p_sz:.1f} "
                    f"{GREY}type: {RESET}{type_msg}\n"
                    f"{GREY}inputs: {la},{ra}{RESET}->\n"
                    f"{GREY}output: {pa}\n",
                )
            )

        if sort == "flops":
            entries.sort(key=lambda x: self.get_flops(x[0]), reverse=True)
        if sort == "size":
            entries.sort(key=lambda x: self.get_size(x[0]), reverse=True)

        entries.append((None, f"{RESET}"))

        o = "\n".join(entry for _, entry in entries)
        print(o)

    # --------------------- Performing the Contraction ---------------------- #

    def extract_contractions(
        self,
        order=None,
        prefer_einsum=False,
    ):
        """Extract just the information needed to perform the contraction.

        Parameters
        ----------
        order : str or callable, optional
            Supplied to :meth:`ContractionTree.traverse`.
        prefer_einsum : bool, optional
            Prefer to use ``einsum`` for pairwise contractions, even if
            ``tensordot`` can perform the contraction.

        Returns
        -------
        contractions : tuple
            A tuple of tuples, each containing the information needed to
            perform a pairwise contraction. Each tuple contains:

                - ``p``: the parent node,
                - ``l``: the left child node,
                - ``r``: the right child node,
                - ``tdot``: whether to use ``tensordot`` or ``einsum``,
                - ``arg``: the argument to pass to ``tensordot`` or ``einsum``
                  i.e. ``axes`` or ``eq``,
                - ``perm``: the permutation required after the contraction, if
                  any (only applies to tensordot).

        """
        return tuple(
            (p, l, r, False, self.get_einsum_eq(p), None)
            if (prefer_einsum or not self.get_can_dot(p))
            else (
                p,
                l,
                r,
                True,
                self.get_tensordot_axes(p),
                self.get_tensordot_perm(p),
            )
            for p, l, r in self.traverse(order=order)
        )

    def get_contractor(
        self,
        order=None,
        prefer_einsum=False,
        strip_exponent=False,
        implementation=None,
        autojit=False,
    ):
        """Get a reusable function which performs the contraction corresponding
        to this tree, cached.

        Parameters
        ----------
        tree : ContractionTree
            The contraction tree.
        order : str or callable, optional
            Supplied to :meth:`ContractionTree.traverse`, the order in which
            to perform the pairwise contractions given by the tree.
        prefer_einsum : bool, optional
            Prefer to use ``einsum`` for pairwise contractions, even if
            ``tensordot`` can perform the contraction.
        strip_exponent : bool, optional
            If ``True``, the function will strip the exponent from the output
            array and return it separately.
        implementation : str or tuple[callable, callable], optional
            What library to use to actually perform the contractions. Options
            are:

            - "auto": let cotengra choose
            - "autoray": dispatch with autoray, using the ``tensordot`` and
              ``einsum`` implementation of the backend
            - "cotengra": use the ``tensordot`` and ``einsum`` implementation
              of cotengra, which is based on batch matrix multiplication. This
              is faster for some backends like numpy, and also enables
              libraries which don't yet provide ``tensordot`` and ``einsum`` to
              be used.
            - "cuquantum": use the cuquantum library to perform the whole
              contraction (not just individual contractions).
            - tuple[callable, callable]: manually supply the ``tensordot`` and
              ``einsum`` implementations to use.

        autojit : bool, optional
            If ``True``, use :func:`autoray.autojit` to compile the contraction
            function.

        Returns
        -------
        fn : callable
            The contraction function, with signature ``fn(*arrays)``.
        """
        key = (
            autojit,
            order,
            prefer_einsum,
            strip_exponent,
            implementation,
        )
        try:
            fn = self.contraction_cores[key]
        except KeyError:
            fn = self.contraction_cores[key] = make_contractor(
                tree=self,
                order=order,
                prefer_einsum=prefer_einsum,
                strip_exponent=strip_exponent,
                implementation=implementation,
                autojit=autojit,
            )

        return fn

    def contract_core(
        self,
        arrays,
        order=None,
        prefer_einsum=False,
        strip_exponent=False,
        check_zero=False,
        backend=None,
        implementation=None,
        autojit=False,
        progbar=False,
    ):
        """Contract ``arrays`` with this tree. The order of the axes and
        output is assumed to be that of ``tree.inputs`` and ``tree.output``,
        but with sliced indices removed. This functon contracts the core tree
        and thus if indices have been sliced the arrays supplied need to be
        sliced as well.

        Parameters
        ----------
        arrays : sequence of array
            The arrays to contract.
        order : str or callable, optional
            Supplied to :meth:`ContractionTree.traverse`.
        prefer_einsum : bool, optional
            Prefer to use ``einsum`` for pairwise contractions, even if
            ``tensordot`` can perform the contraction.
        backend : str, optional
            What library to use for ``einsum`` and ``transpose``, will be
            automatically inferred from the arrays if not given.
        autojit : bool, optional
            Whether to use ``autoray.autojit`` to jit compile the expression.
        progbar : bool, optional
            Show progress through the contraction.
        """
        fn = self.get_contractor(
            order=order,
            prefer_einsum=prefer_einsum,
            strip_exponent=strip_exponent is not False,
            implementation=implementation,
            autojit=autojit,
        )
        result = fn(
            *arrays,
            check_zero=check_zero,
            backend=backend,
            progbar=progbar,
        )

        # handle exponent outside of potential jit
        if isinstance(strip_exponent, dict):
            result, exponent = result
            if "exponent" not in strip_exponent:
                # set the exponent (e.g. first slice)
                strip_exponent["exponent"] = exponent
            else:
                # match the exponent (e.g. subsequent slices)
                target = strip_exponent["exponent"]
                result = result * 10 ** (exponent - target)

        return result

    def slice_key(self, i):
        """Get the combination of sliced index values for overall slice ``i``.

        Parameters
        ----------
        i : int
            The overall slice index.

        Returns
        -------
        key : dict[str, int]
            The value each sliced index takes for slice ``i``.
        """
        return dict(zip(self.sliced_inds, dynary(i, self.sliced_sizes)))

    def slice_arrays(self, arrays, i):
        """Take ``arrays`` and slice the relevant inputs according to
        ``tree.sliced_inds`` and the dynary representation of ``i``.
        """
        temp_arrays = list(arrays)

        # e.g. {'a': 2, 'd': 7, 'z': 0}
        locations = self.slice_key(i)

        for c in self.sliced_inputs:
            # the indexing object, e.g. [:, :, 7, :, 2, :, :, 0]
            selector = tuple(
                locations.get(ix, slice(None)) for ix in self.inputs[c]
            )
            # re-insert the sliced array
            temp_arrays[c] = temp_arrays[c][selector]

        return temp_arrays

    def contract_slice(self, arrays, i, **kwargs):
        """Get slices ``i`` of ``arrays`` and then contract them."""
        return self.contract_core(self.slice_arrays(arrays, i), **kwargs)

    def gather_slices(self, slices, backend=None, progbar=False):
        """Gather all the output contracted slices into a single full result.
        If none of the sliced indices appear in the output, then this is a
        simple sum - otherwise the slices need to be partially summed and
        partially stacked.
        """
        if progbar:
            import tqdm

            slices = tqdm.tqdm(slices, total=self.multiplicity)

        output_pos = {
            ix: i for i, ix in enumerate(self.output) if ix in self.sliced_inds
        }

        if not output_pos:
            # we can just sum everything
            return functools.reduce(operator.add, slices)

        # first we sum over non-output sliced indices
        chunks = {}
        for i, s in enumerate(slices):
            key_slice = self.slice_key(i)
            key = tuple(key_slice[ix] for ix in output_pos)
            try:
                chunks[key] = chunks[key] + s
            except KeyError:
                chunks[key] = s

        # then we stack these summed chunks over output sliced indices
        def recursively_stack_chunks(loc, rem):
            if not rem:
                return chunks[loc]
            arrays = [
                recursively_stack_chunks(loc + (d,), rem[1:])
                for d in range(self.size_dict[rem[0]])
            ]
            axes = output_pos[rem[0]] - len(loc)
            return do("stack", arrays, axes, like=backend)

        return recursively_stack_chunks((), tuple(output_pos))

    def gen_output_chunks(self, arrays, progbar=False, **contract_opts):
        """Generate each output chunk of the contraction - i.e. take care of
        summing internally sliced indices only first. This assumes that the
        ``sliced_inds`` are sorted by whether they appear in the output or not
        (the default order). Useful for performing some kind of reduction over
        the final tensor object like  ``fn(x).sum()`` without constructing the
        entire thing.
        """
        # consecutive slices of size ``stepsize`` all belong to the same output
        # block because the sliced indices are sorted output first
        stepsize = prod(
            d
            for ix, d in zip(self.sliced_inds, self.sliced_sizes)
            if ix not in self.output
        )

        if progbar:
            import tqdm

            it = tqdm.trange(self.nslices // stepsize)
        else:
            it = range(self.nslices // stepsize)

        for o in it:
            chunk = self.contract_slice(arrays, o * stepsize, **contract_opts)
            for j in range(1, stepsize):
                i = o * stepsize + j
                chunk = chunk + self.contract_slice(arrays, i, **contract_opts)
            yield chunk

    def contract(
        self,
        arrays,
        order=None,
        prefer_einsum=False,
        strip_exponent=False,
        check_zero=False,
        backend=None,
        implementation="auto",
        autojit=False,
        progbar=False,
    ):
        """Contract ``arrays`` with this tree. This function takes *unsliced*
        arrays and handles the slicing, contractions and gathering. The order
        of the axes and output is assumed to match that of ``tree.inputs`` and
        ``tree.output``.

        Parameters
        ----------
        arrays : sequence of array
            The arrays to contract.
        order : str or callable, optional
            Supplied to :meth:`ContractionTree.traverse`.
        prefer_einsum : bool, optional
            Prefer to use ``einsum`` for pairwise contractions, even if
            ``tensordot`` can perform the contraction.
        strip_exponent : bool, optional
            If ``True``, eagerly strip the exponent (in log10) from
            intermediate tensors to control numerical problems from leaving the
            range of the datatype. This method then returns the scaled
            'mantissa' output array and the exponent separately.
        check_zero : bool, optional
            If ``True``, when ``strip_exponent=True``, explicitly check for
            zero-valued intermediates that would otherwise produce ``nan``,
            instead terminating early if encounteredand returning
            ``(0.0, 0.0)``.
        backend : str, optional
            What library to use for ``tensordot``, ``einsum`` and
            ``transpose``, it will be automatically inferred from the input
            arrays if not given.
        autojit : bool, optional
            Whether to use the 'autojit' feature of `autoray` to compile the
            contraction expression.
        progbar : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        output : array
            The contracted output, it will be scaled if
            ``strip_exponent==True``.
        exponent : float
            The exponent of the output in base 10, returned only if
            ``strip_exponent==True``.

        See Also
        --------
        contract_core, contract_slice, slice_arrays, gather_slices
        """
        if isinstance(self.inputs[0], set) or isinstance(self.output, set):
            warnings.warn("The inputs or output of this tree are not ordered.")

        if not self.sliced_inds:
            return self.contract_core(
                arrays,
                order=order,
                prefer_einsum=prefer_einsum,
                strip_exponent=strip_exponent,
                check_zero=check_zero,
                backend=backend,
                implementation=implementation,
                autojit=autojit,
                progbar=progbar,
            )

        if strip_exponent:
            # first slice will set the exponent for others to match
            strip_exponent = {}

        slices = (
            self.contract_slice(
                arrays,
                i,
                order=order,
                prefer_einsum=prefer_einsum,
                strip_exponent=strip_exponent,
                check_zero=check_zero,
                backend=backend,
                implementation=implementation,
                autojit=autojit,
            )
            for i in range(self.multiplicity)
        )

        result = self.gather_slices(slices, backend=backend, progbar=progbar)

        if strip_exponent:
            return result, strip_exponent["exponent"]

        return result

    def contract_mpi(self, arrays, comm=None, root=None, **kwargs):
        """Contract the slices of this tree and sum them in parallel -
        *assuming* we are already running under MPI.

        Parameters
        ----------
        arrays : sequence of array
            The input (unsliced arrays)
        comm : None or mpi4py communicator
            Defaults to ``mpi4py.MPI.COMM_WORLD`` if not given.
        root : None or int, optional
            If ``root=None``, an ``Allreduce`` will be performed such that
            every process has the resulting tensor, else if an integer e.g.
            ``root=0``, the result will be exclusively gathered to that
            process using ``Reduce``, with every other process returning
            ``None``.
        kwargs
            Supplied to :meth:`~cotengra.ContractionTree.contract_slice`.
        """
        if not set(self.sliced_inds).isdisjoint(set(self.output)):
            raise NotImplementedError(
                "Sliced and output indices overlap - currently only a simple "
                "sum of result slices is supported currently."
            )

        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD

        if self.multiplicity < comm.size:
            raise ValueError(
                f"Need to have more slices than MPI processes, but have "
                f"{self.multiplicity} and {comm.size} respectively."
            )

        # round robin compute each slice, eagerly summing
        result_i = None
        for i in range(comm.rank, self.multiplicity, comm.size):
            # note: fortran ordering is needed for the MPI reduce
            x = do("asfortranarray", self.contract_slice(arrays, i, **kwargs))
            if result_i is None:
                result_i = x
            else:
                result_i += x

        if root is None:
            # everyone gets the summed result
            result = do("empty_like", result_i)
            comm.Allreduce(result_i, result)
            return result

        # else we only sum reduce the result to process ``root``
        if comm.rank == root:
            result = do("empty_like", result_i)
        else:
            result = None
        comm.Reduce(result_i, result, root=root)
        return result

    plot_ring = plot_tree_ring
    plot_tent = plot_tree_tent
    plot_span = plot_tree_span
    plot_rubberband = plot_tree_rubberband
    plot_contractions = plot_contractions
    plot_contractions_alt = plot_contractions_alt

    @functools.wraps(plot_hypergraph)
    def plot_hypergraph(self, **kwargs):
        hg = self.get_hypergraph(accel=False)
        hg.plot(**kwargs)

    def __repr__(self):
        s = "<{}(N={}, branches={}, complete={})>"
        return s.format(
            self.__class__.__name__,
            self.N,
            len(self.children),
            self.is_complete(),
        )


def _reconfigure_tree(tree, *args, **kwargs):
    return tree.subtree_reconfigure(*args, **kwargs)


def _slice_and_reconfigure_tree(tree, *args, **kwargs):
    return tree.slice_and_reconfigure(*args, **kwargs)


def _get_tree_info(tree):
    return {
        "flops": tree.total_flops(),
        "write": tree.total_write(),
        "size": tree.max_size(),
        "sliced_inds": frozenset(tree.sliced_inds),
    }


def _describe_tree(tree):
    return (
        f"log2[SIZE]: {math.log2(tree.max_size()):.2f} "
        f"log10[FLOPs]: {math.log10(tree.total_flops()):.2f}"
    )


class ContractionTreeCompressed(ContractionTree):
    """A contraction tree for compressed contractions. Currently the only
    difference is that this defaults to the 'surface' traversal ordering.
    """

    @classmethod
    def from_path(
        cls,
        inputs,
        output,
        size_dict,
        *,
        path=None,
        ssa_path=None,
        check=False,
        **kwargs,
    ):
        """Create a (completed) ``ContractionTreeCompressed`` from the usual
        inputs plus a standard contraction path or 'ssa_path' - you need to
        supply one. This also set the default 'surface' traversal ordering to
        be the initial path.
        """
        if int(path is None) + int(ssa_path is None) != 1:
            raise ValueError(
                "Exactly one of ``path`` or ``ssa_path`` must be " "supplied."
            )

        if path is not None:
            ssa_path = linear_to_ssa(path)

        tree = cls(inputs, output, size_dict, **kwargs)
        terms = list(tree.gen_leaves())

        for p in ssa_path:
            merge = [terms[i] for i in p]
            terms.append(tree.contract_nodes(merge, check=check))

        tree.set_surface_order_from_path(ssa_path)

        return tree

    def get_default_order(self):
        return "surface_order"

    total_flops = ContractionTree.total_flops_compressed
    total_write = ContractionTree.total_write_compressed
    total_cost = ContractionTree.total_cost_compressed
    max_size = ContractionTree.max_size_compressed
    peak_size = ContractionTree.peak_size_compressed
    contraction_cost = ContractionTree.contraction_cost_compressed
    contraction_width = ContractionTree.contraction_width_compressed

    total_flops_exact = ContractionTree.total_flops
    total_write_exact = ContractionTree.total_write
    total_cost_exact = ContractionTree.total_cost
    max_size_exact = ContractionTree.max_size
    peak_size_exact = ContractionTree.peak_size

    def get_contractor(self, *_, **__):
        raise NotImplementedError(
            "`cotengra` doesn't implement compressed contraction itself. "
            "If you want to use compressed contractions, you need to use "
            "`quimb` and the `TensorNetwork.contract_compressed` method, "
            "with e.g. `optimize=tree.get_path()`."
        )


class ContractionTreeMulti(ContractionTree):
    def set_varmults(self, varmults):
        self._varmults = varmults

    def get_varmults(self):
        return self._varmults

    def set_numconfigs(self, numconfigs):
        self._numconfigs = numconfigs

    def get_numconfigs(self):
        return self._numconfigs


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

    def build_divide(
        self,
        inputs,
        output,
        size_dict,
        random_strength=0.01,
        cutoff=10,
        parts=2,
        parts_decay=0.5,
        sub_optimize="auto",
        super_optimize="auto-hq",
        check=False,
        **partition_opts,
    ):
        tree = ContractionTree(inputs, output, size_dict, track_childless=True)
        rand_size_dict = jitter_dict(size_dict, random_strength)

        dynamic_imbalance = ("imbalance" in partition_opts) and (
            "imbalance_decay" in partition_opts
        )
        if dynamic_imbalance:
            imbalance = partition_opts.pop("imbalance")
            imbalance_decay = partition_opts.pop("imbalance_decay")

        dynamic_fix = partition_opts.get("fix_output_nodes", None) == "auto"

        while tree.childless:
            tree_node = next(iter(tree.childless))
            subgraph = tuple(tree_node)
            subsize = len(subgraph)

            # skip straight to better method
            if subsize <= cutoff:
                tree.contract_nodes(
                    [node_from_single(x) for x in subgraph],
                    optimize=sub_optimize,
                    check=check,
                )
                continue

            # relative subgraph size
            s = subsize / tree.N

            # let the target number of communities depend on subgraph size
            parts_s = max(int(s**parts_decay * parts), 2)

            # let the imbalance either rise or fall
            if dynamic_imbalance:
                if imbalance_decay >= 0:
                    imbalance_s = s**imbalance_decay * imbalance
                else:
                    imbalance_s = 1 - s**-imbalance_decay * (1 - imbalance)
                partition_opts["imbalance"] = imbalance_s

            if dynamic_fix:
                # for the top level subtree (s==1.0) we partition the outputs
                # nodes first into their own bi-partition
                parts_s = 2
                partition_opts["fix_output_nodes"] = s == 1.0

            # partition! get community membership list e.g.
            # [0, 0, 1, 0, 1, 0, 0, 2, 2, ...]
            inputs = tuple(map(oset, tree.node_to_terms(subgraph)))
            output = oset(tree.get_legs(tree_node))
            membership = self.partition_fn(
                inputs,
                output,
                rand_size_dict,
                parts=parts_s,
                **partition_opts,
            )

            # divide subgraph up e.g. if we enumerate the subgraph index sets
            # (0, 1, 2, 3, 4, 5, 6, 7, 8, ...) ->
            # ({0, 1, 3, 5, 6}, {2, 4}, {7, 8})
            new_subgs = tuple(
                map(node_from_seq, separate(subgraph, membership))
            )

            if len(new_subgs) == 1:
                # no communities found - contract all remaining
                tree.contract_nodes(
                    tuple(map(node_from_single, subgraph)),
                    optimize=sub_optimize,
                    check=check,
                )
                continue

            # update tree structure with newly contracted subgraphs
            tree.contract_nodes(
                new_subgs, optimize=super_optimize, check=check
            )

        if check:
            assert tree.is_complete()

        return tree

    def build_agglom(
        self,
        inputs,
        output,
        size_dict,
        random_strength=0.01,
        groupsize=4,
        check=False,
        sub_optimize="greedy",
        **partition_opts,
    ):
        tree = ContractionTree(inputs, output, size_dict, track_childless=True)
        rand_size_dict = jitter_dict(size_dict, random_strength)
        leaves = tuple(tree.gen_leaves())
        for node in leaves:
            tree._add_node(node)
        output = oset(tree.output)

        while len(leaves) > groupsize:
            parts = max(2, len(leaves) // groupsize)

            inputs = [oset(tree.get_legs(node)) for node in leaves]
            membership = self.partition_fn(
                inputs,
                output,
                rand_size_dict,
                parts=parts,
                **partition_opts,
            )
            leaves = [
                tree.contract_nodes(group, check=check, optimize=sub_optimize)
                for group in separate(leaves, membership)
            ]

        if len(leaves) > 1:
            tree.contract_nodes(leaves, check=check, optimize=sub_optimize)

        if check:
            assert tree.is_complete()

        return tree

    def trial_fn(self, inputs, output, size_dict, **partition_opts):
        return self.build_divide(inputs, output, size_dict, **partition_opts)

    def trial_fn_agglom(self, inputs, output, size_dict, **partition_opts):
        return self.build_agglom(inputs, output, size_dict, **partition_opts)


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
