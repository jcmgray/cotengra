"""Core contraction tree data structure and methods."""

import collections
import functools
import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Optional

from autoray import do

from .contract import make_contractor
from .hypergraph import get_hypergraph
from .parallel import (
    can_scatter,
    maybe_leave_pool,
    maybe_rejoin_pool,
    parse_parallel_arg,
    scatter,
    submit,
)
from .pathfinders.path_simulated_annealing import (
    parallel_temper_tree,
    simulated_anneal_tree,
)
from .plot import (
    plot_contractions,
    plot_contractions_alt,
    plot_hypergraph,
    plot_tree_circuit,
    plot_tree_flat,
    plot_tree_ring,
    plot_tree_rubberband,
    plot_tree_span,
    plot_tree_tent,
)
from .scoring import (
    DEFAULT_COMBO_FACTOR,
    CompressedStatsTracker,
    get_score_fn,
)
from .utils import (
    MaxCounter,
    compute_size_by_dict,
    deprecated,
    get_rng,
    get_symbol,
    groupby,
    inputs_output_to_eq,
    interleave,
    is_valid_node,
    node_from_seq,
    node_from_single,
    node_get_single_el,
    node_supremum,
    oset,
    prod,
    unique,
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


def legs_union(legs_seq):
    """Combine a sequence of legs into a single set of legs, summing their
    appearances.
    """
    new_legs, *rem_legs = legs_seq
    new_legs = new_legs.copy()
    for legs in rem_legs:
        for ix, ix_count in legs.items():
            new_legs[ix] = new_legs.get(ix, 0) + ix_count
    return new_legs


def legs_without(legs, ind):
    """Discard ``ind`` from legs to create a new set of legs."""
    new_legs = legs.copy()
    new_legs.pop(ind, None)
    return new_legs


def get_with_default(k, obj, default):
    return obj.get(k, default)


@dataclass(order=True, frozen=True)
class SliceInfo:
    inner: bool
    ind: str
    size: int
    project: Optional[int]

    @property
    def sliced_range(self):
        if self.project is None:
            return range(self.size)
        else:
            return [self.project]


def get_slice_strides(sliced_inds):
    """Compute the 'strides' given the (ordered) dictionary of sliced indices."""
    slice_infos = list(sliced_inds.values())
    nsliced = len(slice_infos)
    strides = [1] * nsliced
    # backwards cumulative product
    for i in range(nsliced - 2, -1, -1):
        strides[i] = strides[i + 1] * slice_infos[i + 1].size
    return strides


def add_maybe_exponent_stripped(x, y):
    """Add two arrays, or tuples of (array, exponent) together in a stable
    and branchless way.
    """
    xistup = isinstance(x, tuple)
    yistup = isinstance(y, tuple)
    if not (xistup or yistup):
        # simple sum without exponent
        return x + y

    if xistup:
        xm, xe = x
    else:
        xm = x
        xe = 0.0

    if yistup:
        ym, ye = y
    else:
        ym = y
        ye = 0.0

    # perform branchless for jit etc.
    e = max(xe, ye)
    m = xm * 10 ** (xe - e) + ym * 10 ** (ye - e)

    return (m, e)


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
    track_write : bool, optional
        Whether to dynamically keep track of the total number of elements
        written. If ``False`` You can still compute this once the tree is
        complete.
    track_size : bool, optional
        Whether to dynamically keep track of the largest tensor so far. If
        ``False`` You can still compute this once the tree is complete.
    objective : str or Objective, optional
        An default objective function to use for further optimization and
        scoring, for example reconfiguring or computing the combo cost. If not
        supplied the default is to create a flops objective when needed.

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
        objective=None,
    ):
        self.inputs = inputs
        self.output = output

        if isinstance(self.inputs[0], set) or isinstance(self.output, set):
            warnings.warn(
                "The inputs or output of this tree are not ordered."
                "Costs will be accurate but actually contracting requires "
                "ordered indices corresponding to array axes."
            )

        if not isinstance(next(iter(size_dict.values()), 1), int):
            # make sure we are working with python integers to avoid overflow
            # comparison errors with inf etc.
            self.size_dict = {k: int(v) for k, v in size_dict.items()}
        else:
            self.size_dict = size_dict

        self.N = len(self.inputs)

        # the index representation for each input is an ordered mapping of
        # each index to the number of times it has appeared on children. By
        # also tracking the total number of appearances one can efficiently
        # and locally compute which indices should be kept or contracted
        self.appearances = {}
        for term in self.inputs:
            for ix in term:
                self.appearances[ix] = self.appearances.get(ix, 0) + 1
        # adding output appearances ensures these are never contracted away,
        # N.B. if after this step every appearance count is exactly 2,
        # then there are no 'hyper' indices in the contraction
        for ix in self.output:
            self.appearances[ix] = self.appearances.get(ix, 0) + 1

        # this stores potentialy preprocessing steps that are not part of the
        # main contraction tree, but assumed to have been applied, for example
        # tracing or summing over indices that appear only once
        self.preprocessing = {}

        # mapping of parents to children - the core binary tree object
        self.children = {}

        # information about all the nodes
        self.info = {}

        # add constant nodes: the leaves
        for leaf in self.gen_leaves():
            self._add_node(leaf)
        # and the root or top node
        self.root = node_supremum(self.N)
        self._add_node(self.root)

        # whether to keep track of dangling nodes/subgraphs
        self.track_childless = track_childless
        if self.track_childless:
            # the set of dangling nodes
            self.childless = oset([self.root])

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
        self.sliced_inds = {}
        self.sliced_inputs = frozenset()

        # cache for compiled contraction cores
        self.contraction_cores = {}

        # a default objective function useful for
        # further optimization and scoring
        self._default_objective = objective

    def set_state_from(self, other):
        """Set the internal state of this tree to that of ``other``."""
        # immutable or never mutated properties
        for attr in (
            "appearances",
            "inputs",
            "multiplicity",
            "N",
            "output",
            "root",
            "size_dict",
            "sliced_inputs",
            "_default_objective",
        ):
            setattr(self, attr, getattr(other, attr))

        # mutable properties
        for attr in (
            "children",
            "contraction_cores",
            "sliced_inds",
            "preprocessing",
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

    def set_default_objective(self, objective):
        """Set the objective function for this tree."""
        self._default_objective = get_score_fn(objective)

    def get_default_objective(self):
        """Get the objective function for this tree."""
        if self._default_objective is None:
            self._default_objective = get_score_fn("flops")
        return self._default_objective

    def get_default_combo_factor(self):
        """Get the default combo factor for this tree."""
        objective = self.get_default_objective()
        try:
            return objective.factor
        except AttributeError:
            return DEFAULT_COMBO_FACTOR

    def get_score(self, objective=None):
        """Score this tree using the default objective function."""
        from .scoring import get_score_fn

        if objective is None:
            objective = self.get_default_objective()

        objective = get_score_fn(objective)

        return objective({"tree": self})

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
            si.size for si in self.sliced_inds.values() if not si.inner
        )

    def node_to_terms(self, node):
        """Turn a node -- a frozen set of ints -- into the corresponding terms
        -- a sequence of sets of str corresponding to input indices.
        """
        return (self.get_legs(node_from_single(i)) for i in node)

    def gen_leaves(self):
        """Generate the nodes representing leaves of the contraction tree, i.e.
        of size 1 each corresponding to a single input tensor.
        """
        return map(node_from_single, range(self.N))

    def get_incomplete_nodes(self):
        """Get the set of current nodes that have no children and the set of
        nodes that have no parents. These are the 'childless' and 'parentless'
        nodes respectively, that need to be contracted to complete the tree.
        The parentless nodes are grouped into the childless nodes that contain
        them as subgraphs.

        Returns
        -------
        groups : dict[frozenet[int], list[frozenset[int]]]
            A mapping of childless nodes to the list of parentless nodes are
            beneath them.

        See Also
        --------
        autocomplete
        """
        childless = dict.fromkeys(
            node
            for node in self.info
            # start wth all but leaves
            if len(node) != 1
        )
        parentless = dict.fromkeys(
            node
            for node in self.info
            # start with all but root
            if len(node) != self.N
        )
        for p, (l, r) in self.children.items():
            parentless.pop(l)
            parentless.pop(r)
            childless.pop(p)

        groups = {node: [] for node in childless}
        for node in parentless:
            # get the smallest node that contains this node
            ancestor = min(
                filter(node.issubset, childless),
                key=len,
            )
            groups[ancestor].append(node)

        return groups

    def autocomplete(self, **contract_opts):
        """Contract all remaining node groups (as computed by
        ``tree.get_incomplete_nodes``) in the tree to complete it.

        Parameters
        ----------
        contract_opts
            Options to pass to ``tree.contract_nodes``.

        See Also
        --------
        get_incomplete_nodes, contract_nodes
        """
        groups = self.get_incomplete_nodes()
        for _, parentless_subnodes in groups.items():
            self.contract_nodes(parentless_subnodes, **contract_opts)

    @classmethod
    def from_path(
        cls,
        inputs,
        output,
        size_dict,
        *,
        path=None,
        ssa_path=None,
        edge_path=None,
        optimize="auto-hq",
        autocomplete="auto",
        check=False,
        **kwargs,
    ):
        """Create a (completed) ``ContractionTree`` from the usual inputs plus
        a standard contraction path or 'ssa_path' - you need to supply one.

        Parameters
        ----------
        inputs : Sequence[Sequence[str]]
            The input indices of each tensor, as single unicode characters.
        output : Sequence[str]
            The output indices.
        size_dict : dict[str, int]
            The size of each index.
        path : Sequence[Sequence[int]], optional
            The contraction path, a sequence of pairs of tensor ids to
            contract. The ids are linear indices into the list of temporary
            tensors, which are recycled as each contraction pops a pair and
            appends the result. One of ``path``, ``ssa_path`` or ``edge_path``
            must be supplied.
        ssa_path : Sequence[Sequence[int]], optional
            The contraction path, a sequence of pairs of indices to contract.
            The indices are single use, as if the result of each contraction is
            appended to the end of the list of temporary tensors without
            popping. One of ``path``, ``ssa_path`` or ``edge_path`` must be
            supplied.
        edge_path : Sequence[str], optional
            The contraction path, a sequence of indices to contract in order.
            One of ``path``, ``ssa_path`` or ``edge_path`` must be supplied.
        optimize : str, optional
            If a contraction within the path contains 3 or more tensors, how to
            optimize this subcontraction into a binary tree.
        autocomplete : "auto" or bool, optional
            Whether to automatically complete the path, i.e. contract all
            remaining nodes. If "auto" then a warning is issued if the path is
            not complete.
        check : bool, optional
            Whether to perform some basic checks while creating the contraction
            nodes.

        Returns
        -------
        ContractionTree
        """
        if (path is None) + (ssa_path is None) + (edge_path is None) != 2:
            raise ValueError(
                "Exactly one of ``path`` or ``ssa_path`` must be supplied."
            )

        contract_opts = {"optimize": optimize, "check": check}

        if edge_path is not None:
            from .pathfinders.path_basic import edge_path_to_ssa

            ssa_path = edge_path_to_ssa(edge_path, inputs)

        if ssa_path is not None:
            path = ssa_path

        tree = cls(inputs, output, size_dict, **kwargs)

        if ssa_path is not None:
            # ssa path (single use ids)
            nodes = dict(enumerate(tree.gen_leaves()))
            ssa = len(nodes)
            for p in path:
                merge = [nodes.pop(i) for i in p]
                nodes[ssa] = tree.contract_nodes(merge, **contract_opts)
                ssa += 1
            nodes = nodes.values()
        else:
            # regular path ('recycled' ids)
            nodes = list(tree.gen_leaves())
            for p in path:
                merge = [nodes.pop(i) for i in sorted(p, reverse=True)]
                nodes.append(tree.contract_nodes(merge, **contract_opts))

        if len(nodes) > 1 and autocomplete:
            if autocomplete == "auto":
                # warn that we are completing
                warnings.warn(
                    "Path was not complete - contracting all remaining. "
                    "You can silence this warning with `autocomplete=True`."
                    "Or produce an incomplete tree with `autocomplete=False`."
                )

            tree.contract_nodes(nodes, **contract_opts)

        return tree

    @classmethod
    def from_info(cls, info, **kwargs):
        """Create a ``ContractionTree`` from an ``opt_einsum.PathInfo`` object."""
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
        return inputs_output_to_eq(self.inputs, self.output)

    def get_shapes(self):
        """Get the shapes of the input tensors corresponding to this tree.

        Returns
        -------
        shapes : tuple[tuple[int]]
        """
        return tuple(
            tuple(self.size_dict[ix] for ix in term) for term in self.inputs
        )

    def get_inputs_sliced(self):
        """Get the input indices corresponding to a single slice of this tree,
        i.e. with sliced indices removed.

        Returns
        -------
        inputs : tuple[tuple[str]]
        """
        return tuple(
            tuple(ix for ix in term if ix not in self.sliced_inds)
            for term in self.inputs
        )

    def get_output_sliced(self):
        """Get the output indices corresponding to a single slice of this tree,
        i.e. with sliced indices removed.

        Returns
        -------
        output : tuple[str]
        """
        return tuple(ix for ix in self.output if ix not in self.sliced_inds)

    def get_eq_sliced(self):
        """Get the einsum equation corresponding to a single slice of this
        tree, i.e. with sliced indices removed.

        Returns
        -------
        eq : str
        """
        return inputs_output_to_eq(
            self.get_inputs_sliced(), self.get_output_sliced()
        )

    def get_shapes_sliced(self):
        """Get the shapes of the input tensors corresponding to a single slice
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
        cls,
        edge_path,
        inputs,
        output,
        size_dict,
        optimize="auto-hq",
        autocomplete="auto",
        check=False,
        **kwargs,
    ):
        """Create a ``ContractionTree`` from an edge elimination ordering."""
        warnings.warn(
            "ContractionTree.from_edge_path(edge_path, ...) is deprecated. Use"
            " ContractionTree.from_path(edge_path=edge_path, ...) instead."
        )
        return cls.from_path(
            inputs,
            output,
            size_dict,
            edge_path=edge_path,
            optimize=optimize,
            autocomplete=autocomplete,
            check=check,
            **kwargs,
        )

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
        if tracking them respectively, as well as input pre-processing.
        """
        node_extent = len(node)

        if node_extent == 1:
            # leaf nodes should always exist
            self.info[node].clear()
            # input: remove any associated preprocessing
            self.preprocessing.pop(node_get_single_el(node), None)
        else:
            # only non-leaf nodes contribute to size, flops and write
            if self._track_size:
                self._sizes.discard(self.get_size(node))

            if self._track_flops:
                self._flops -= self.get_flops(node)

            if self._track_write:
                self._write -= self.get_size(node)

            del self.children[node]
            if node_extent == self.N:
                # root node should always exist
                self.info[node].clear()
            else:
                del self.info[node]

    def compute_leaf_legs(self, i):
        """Compute the effective 'outer' indices for the ith input tensor. This
        is not always simply the ith input indices, due to A) potential slicing
        and B) potential preprocessing.
        """
        # indices of input tensor (after slicing which is done immediately)
        if self.sliced_inds:
            term = tuple(
                ix for ix in self.inputs[i] if ix not in self.sliced_inds
            )
        else:
            term = self.inputs[i]

        legs = {}
        for ix in term:
            legs[ix] = legs.get(ix, 0) + 1

        # check for single term simplifications, these are treated as a simple
        # preprocessing step that only is taken into account during actual
        # contraction, and are not represented in the binary tree
        # N.B. need to compute simplifiability *after* slicing
        is_simplifiable = (
            # repeated indices (diag or traces)
            (len(term) != len(legs))
            or
            # reduced indices (are summed immediately)
            any(
                ix_count == self.appearances[ix]
                for ix, ix_count in legs.items()
            )
        )

        if is_simplifiable:
            # compute the simplified legs -> the new effective input legs
            legs = {
                ix: ix_count
                for ix, ix_count in legs.items()
                if ix_count != self.appearances[ix]
            }
            # add a preprocessing step to the list of contractions
            eq = inputs_output_to_eq((term,), legs, canonicalize=True)
            self.preprocessing[i] = eq

        return legs

    def has_preprocessing(self):
        # touch all inputs legs, since preprocessing is lazily computed
        for node in self.gen_leaves():
            self.get_legs(node)
        return bool(self.preprocessing)

    def has_hyper_indices(self):
        """Check if there are any 'hyper' indices in the contraction, i.e.
        indices that don't appear exactly twice, when considering the inputs
        and output.
        """
        return any(ix_count != 2 for ix_count in self.appearances.values())

    @cached_node_property("legs")
    def get_legs(self, node):
        """Get the effective 'outer' indices for the collection of tensors
        in ``node``.
        """
        node_extent = len(node)

        if node_extent == 1:
            # leaf legs are inputs
            return self.compute_leaf_legs(node_get_single_el(node))
        elif node_extent == self.N:
            # root legs are output, after slicing
            # n.b. the index counts are irrelevant for the output
            return {ix: 0 for ix in self.output if ix not in self.sliced_inds}

        try:
            involved = self.get_involved(node)
        except KeyError:
            involved = legs_union(self.node_to_terms(node))

        return {
            ix: ix_count
            for ix, ix_count in involved.items()
            if ix_count < self.appearances[ix]
        }

    @cached_node_property("involved")
    def get_involved(self, node):
        """Get all the indices involved in the formation of subgraph ``node``."""
        if len(node) == 1:
            return {}
        sub_legs = map(self.get_legs, self.children[node])
        return legs_union(sub_legs)

    @cached_node_property("size")
    def get_size(self, node):
        """Get the tensor size of ``node``."""
        return compute_size_by_dict(self.get_legs(node), self.size_dict)

    @cached_node_property("flops")
    def get_flops(self, node):
        """Get the FLOPs for the pairwise contraction that will create
        ``node``.
        """
        if len(node) == 1:
            return 0
        involved = self.get_involved(node)
        return compute_size_by_dict(involved, self.size_dict)

    @cached_node_property("can_dot")
    def get_can_dot(self, node):
        """Get whether this contraction can be performed as a dot product (i.e.
        with ``tensordot``), or else requires ``einsum``, as it has indices
        that don't appear exactly twice in either the inputs or the output.
        """
        l, r = self.children[node]
        sp, sl, sr = map(self.get_legs, (node, l, r))
        return set(sp) == set(sl).symmetric_difference(sr)

    @cached_node_property("inds")
    def get_inds(self, node):
        """Get the indices of this node - an ordered string version of
        ``get_legs`` that starts with ``tree.inputs`` and maintains the order
        they appear in each contraction 'ABC,abc->ABCabc', to match tensordot.
        """
        # NB: self.inputs and self.output contain the full (unsliced) indices
        #     thus we filter even the input legs and output legs

        if len(node) in (1, self.N):
            return "".join(self.get_legs(node))

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

    def total_flops(self, dtype=None, log=None):
        """Sum the flops contribution from every node in the tree.

        Parameters
        ----------
        dtype : {'float', 'complex', None}, optional
            Scale the answer depending on the assumed data type.
        """
        if self._track_flops:
            C = self.multiplicity * self._flops

        else:
            self._flops = 0
            for node, _, _ in self.traverse():
                self._flops += self.get_flops(node)

            self._track_flops = True
            C = self.multiplicity * self._flops

        if dtype is None:
            pass
        elif "float" in dtype:
            C *= 2
        elif "complex" in dtype:
            C *= 4
        else:
            raise ValueError(f"Unknown dtype {dtype}")

        if log is not None:
            C = math.log(C, log)

        return C

    def total_write(self):
        """Sum the total amount of memory that will be created and operated on."""
        if not self._track_write:
            self._write = 0
            for node, _, _ in self.traverse():
                self._write += self.get_size(node)

            self._track_write = True

        return self.multiplicity * self._write

    def combo_cost(self, factor=DEFAULT_COMBO_FACTOR, combine=sum, log=None):
        t = 0
        for p in self.children:
            f = self.get_flops(p)
            w = self.get_size(p)
            t += combine((f, factor * w))

        t *= self.multiplicity

        if log is not None:
            t = math.log(t, log)

        return t

    total_cost = combo_cost

    def max_size(self, log=None):
        """The size of the largest intermediate tensor."""
        if self.N == 1:
            return self.get_size(self.root)

        if not self._track_size:
            self._sizes = MaxCounter()
            for node, _, _ in self.traverse():
                self._sizes.add(self.get_size(node))
            self._track_size = True

        size = self._sizes.max()

        if log is not None:
            size = math.log(size, log)

        return size

    def peak_size(self, order=None, log=None):
        """Get the peak concurrent size of tensors needed - this depends on the
        traversal order, i.e. the exact contraction path, not just the
        contraction tree.
        """
        tot_size = sum(self.get_size(node) for node in self.gen_leaves())
        peak = tot_size
        for p, l, r in self.traverse(order=order):
            tot_size += self.get_size(p)
            # measure peak assuming we need both inputs and output
            peak = max(peak, tot_size)
            tot_size -= self.get_size(l)
            tot_size -= self.get_size(r)

        if log is not None:
            peak = math.log(peak, log)

        return peak

    def contract_stats(self, force=False):
        """Simulteneously compute the total flops, write and size of the
        contraction tree. This is more efficient than calling each of the
        individual methods separately. Once computed, each quantity is then
        automatically tracked.

        Returns
        -------
        stats : dict[str, int]
            The total flops, write and size.
        """
        if force or not (
            self._track_flops and self._track_write and self._track_size
        ):
            self._flops = self._write = 0
            self._sizes = MaxCounter()

            for node, _, _ in self.traverse():
                self._flops += self.get_flops(node)
                node_size = self.get_size(node)
                self._write += node_size
                self._sizes.add(node_size)

            self._track_flops = self._track_write = self._track_size = True

        return {
            "flops": self.multiplicity * self._flops,
            "write": self.multiplicity * self._write,
            "size": self._sizes.max(),
        }

    def arithmetic_intensity(self):
        """The ratio of total flops to total write - the higher the better for
        extracting good computational performance.
        """
        return self.total_flops(dtype=None) / self.total_write()

    def contraction_scaling(self):
        """This is computed simply as the maximum number of indices involved
        in any single contraction, which will match the scaling assuming that
        all dimensions are equal.
        """
        return max(len(self.get_involved(node)) for node in self.info)

    def contraction_cost(self, log=None):
        """Get the total number of scalar operations ~ time complexity."""
        return self.total_flops(dtype=None, log=log)

    def contraction_width(self, log=2):
        """Get log2 of the size of the largest tensor."""
        return self.max_size(log=log)

    def compressed_contract_stats(
        self,
        chi=None,
        order="surface_order",
        compress_late=None,
    ):
        if chi is None:
            chi = self.get_default_chi()

        if compress_late is None:
            compress_late = self.get_default_compress_late()

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
        self,
        chi=None,
        order="surface_order",
        compress_late=None,
        dtype=None,
        log=None,
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

        F = self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).flops

        if log is not None:
            F = math.log(F, log)

        return F

    contraction_cost_compressed = total_flops_compressed

    def total_write_compressed(
        self,
        chi=None,
        order="surface_order",
        compress_late=None,
        accel="auto",
        log=None,
    ):
        """Compute the total size of all intermediate tensors when a
        compressed contraction is performed with maximum bond size ``chi``,
        ordered by ``order``. This is relevant maybe for time complexity and
        e.g. autodiff space complexity (since every intermediate is kept).
        """
        W = self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).write

        if log is not None:
            W = math.log(W, log)

        return W

    def combo_cost_compressed(
        self,
        chi=None,
        order="surface_order",
        compress_late=None,
        factor=None,
        log=None,
    ):
        if factor is None:
            factor = self.get_default_combo_factor()

        C = self.total_flops_compressed(
            chi=chi, order=order, compress_late=compress_late
        ) + factor * self.total_write_compressed(
            chi=chi, order=order, compress_late=compress_late
        )

        if log is not None:
            C = math.log(C, log)

        return C

    total_cost_compressed = combo_cost_compressed

    def max_size_compressed(
        self, chi=None, order="surface_order", compress_late=None, log=None
    ):
        """Compute the maximum sized tensor produced when a compressed
        contraction is performed with maximum bond size ``chi``, ordered by
        ``order``. This is close to the ideal space complexity if only
        tensors that are being directly operated on are kept in memory.
        """
        S = self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).max_size

        if log is not None:
            S = math.log(S, log)

        return S

    def peak_size_compressed(
        self,
        chi=None,
        order="surface_order",
        compress_late=None,
        accel="auto",
        log=None,
    ):
        """Compute the peak size of combined intermediate tensors when a
        compressed contraction is performed with maximum bond size ``chi``,
        ordered by ``order``. This is the practical space complexity if one is
        not swapping intermediates in and out of memory.
        """
        P = self.compressed_contract_stats(
            chi=chi,
            order=order,
            compress_late=compress_late,
        ).peak_size

        if log is not None:
            P = math.log(P, log)

        return P

    def contraction_width_compressed(
        self, chi=None, order="surface_order", compress_late=None, log=2
    ):
        """Compute log2 of the maximum sized tensor produced when a compressed
        contraction is performed with maximum bond size ``chi``, ordered by
        ``order``.
        """
        return self.max_size_compressed(chi, order, compress_late, log=log)

    def _update_tracked(self, node):
        if self._track_flops:
            self._flops += self.get_flops(node)
        if self._track_write:
            self._write += self.get_size(node)
        if self._track_size:
            self._sizes.add(self.get_size(node))

    def contract_nodes_pair(
        self,
        x,
        y,
        legs=None,
        cost=None,
        size=None,
        check=False,
    ):
        """Contract node ``x`` with node ``y`` in the tree to create a new
        parent node, which is returned.

        Parameters
        ----------
        x : frozenset[int]
            The first node to contract.
        y : frozenset[int]
            The second node to contract.
        legs : dict[str, int], optional
            The effective 'legs' of the new node if already known. If not
            given, this is computed from the inputs of ``x`` and ``y``.
        cost : int, optional
            The cost of the contraction if already known. If not given, this is
            computed from the inputs of ``x`` and ``y``.
        size : int, optional
            The size of the new node if already known. If not given, this is
            computed from the inputs of ``x`` and ``y``.
        check : bool, optional
            Whether to check the inputs are valid.

        Returns
        -------
        parent : frozenset[int]
            The new parent node of ``x`` and ``y``.
        """
        parent = x.union(y)

        # make sure info entries exist for all (default dict)
        for node in (x, y, parent):
            self._add_node(node, check=check)

        # enforce left ordering of 'heaviest' subtrees
        nx, ny = len(x), len(y)
        if nx == ny:
            # deterministically break ties
            sortx = -min(x)
            sorty = -min(y)
        else:
            sortx = nx
            sorty = ny

        if sortx > sorty:
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

        # pre-computed information
        if legs is not None:
            self.info[parent]["legs"] = legs
        if cost is not None:
            self.info[parent]["flops"] = cost
        if size is not None:
            self.info[parent]["size"] = size

        self._update_tracked(parent)

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

        from .interface import find_path

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
        path_inputs = [tuple(self.get_legs(x)) for x in nodes]
        path_output = tuple(self.get_legs(grandparent))

        path = find_path(
            path_inputs,
            path_output,
            self.size_dict,
            optimize=optimize,
            **(extra_opts or {}),
        )

        # now we have path create the nodes in between
        temp_nodes = list(nodes)
        for p in path:
            to_contract = [temp_nodes.pop(i) for i in sorted(p, reverse=True)]
            temp_nodes.append(self.contract_nodes(to_contract, check=check))

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

    def _traverse_dfs(self):
        """Traverse the tree in a depth first, non-recursive, order."""
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

    def _traverse_ordered(self, order):
        """Traverse the tree in the order that minimizes ``order(node)``, but
        still constrained to produce children before parents.
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
        order : None, "dfs", or callable, optional
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
        if self.N == 1:
            return

        if order is None:
            order = self.get_default_order()

        if order == "dfs":
            yield from self._traverse_dfs()
        else:
            yield from self._traverse_ordered(order=order)

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

    def get_subtree(self, node, size, search="bfs", seed=None):
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

        seed : None, int or random.Random, optional
            Random number generator seed, if ``search`` is 'random'.

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

        if search == "random":
            rng = get_rng(seed)
        else:
            rng = None
            if search == "bfs":
                i = 0
            elif search == "dfs":
                i = -1

        while (len(queue) + len(real_leaves) < size) and queue:
            if rng is not None:
                i = rng.randint(0, len(queue) - 1)

            p = queue.pop(i)
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

    def remove_ind(self, ind, project=None, inplace=False):
        """Remove (i.e. by default slice) index ``ind`` from this contraction
        tree, taking care to update all relevant information about each node.
        """
        tree = self if inplace else self.copy()

        if ind in tree.sliced_inds:
            raise ValueError(f"Index {ind} already sliced.")

        # make sure all flops and size information has been populated
        tree.contract_stats()

        d = tree.size_dict[ind]
        if project is None:
            # we are slicing the index
            si = SliceInfo(ind not in tree.output, ind, d, None)
            tree.multiplicity = tree.multiplicity * d
        else:
            si = SliceInfo(ind not in tree.output, ind, 1, project)

        # update the ordered slice information dictionary, but maintain the
        # order such that output sliced indices always appear first ->
        # enforced by the dataclass SliceInfo ordering
        tree.sliced_inds = {
            si.ind: si for si in sorted((*tree.sliced_inds.values(), si))
        }

        for node, node_info in tree.info.items():
            if len(node) == 1:
                # handle leaves separately
                i = node_get_single_el(node)
                term = tree.inputs[i]
                if ind in term:
                    # n.b. leaves don't contribute to size, flops or write
                    # simply recalculate all information, incl. preprocessing
                    tree._remove_node(node)
                    tree.sliced_inputs = tree.sliced_inputs | frozenset([i])
            else:
                involved = tree.get_involved(node)
                if ind not in involved:
                    # if ind doesn't feature in this node (contraction)
                    # -> nothing to do
                    continue

                # else update all the relevant information about this node
                # -> flops changes for all involved indices
                node_info["involved"] = legs_without(involved, ind)
                old_flops = tree.get_flops(node)
                new_flops = old_flops // d
                node_info["flops"] = new_flops
                tree._flops += new_flops - old_flops

                # -> size and write only changes for node legs (output) indices
                legs = tree.get_legs(node)
                if ind in legs:
                    node_info["legs"] = legs_without(legs, ind)
                    old_size = tree.get_size(node)
                    tree._sizes.discard(old_size)
                    new_size = old_size // d
                    tree._sizes.add(new_size)
                    node_info["size"] = new_size
                    tree._write += new_size - old_size

                # delete info we can't change
                for k in (
                    "inds",
                    "einsum_eq",
                    "can_dot",
                    "tensordot_axes",
                    "tensordot_perm",
                ):
                    tree.info[node].pop(k, None)

        tree.already_optimized.clear()
        tree.contraction_cores.clear()

        return tree

    remove_ind_ = functools.partialmethod(remove_ind, inplace=True)

    def restore_ind(self, ind, inplace=False):
        """Restore (unslice or un-project) index ``ind`` to this contraction
        tree, taking care to update all relevant information about each node.

        Parameters
        ----------
        ind : str
            The index to restore.
        inplace : bool, optional
            Whether to perform the restoration inplace or not.

        Returns
        -------
        ContractionTree
        """
        tree = self if inplace else self.copy()

        # pop sliced index info
        si = tree.sliced_inds.pop(ind)

        # make sure all flops and size information has been populated
        tree.contract_stats()
        tree.multiplicity //= si.size

        # handle inputs
        for i, term in enumerate(tree.inputs):
            # this is the original term with all indices
            if ind in term:
                tree._remove_node(node_from_single(i))
                if all(ix not in tree.sliced_inds for ix in term):
                    # mark this input as not sliced
                    tree.sliced_inputs = tree.sliced_inputs - frozenset([i])

        # delete and re-add dependent intermediates
        for p, l, r in tree.traverse():
            if ind in tree.get_legs(l) or ind in tree.get_legs(r):
                tree._remove_node(p)
                tree.contract_nodes_pair(l, r)

        # reset caches
        tree.already_optimized.clear()
        tree.contraction_cores.clear()

        return tree

    restore_ind_ = functools.partialmethod(restore_ind, inplace=True)

    def unslice_rand(self, seed=None, inplace=False):
        """Unslice (restore) a random index from this contraction tree.

        Parameters
        ----------
        seed : None, int or random.Random, optional
            Random number generator seed.
        inplace : bool, optional
            Whether to perform the unslicing inplace or not.

        Returns
        -------
        ContractionTree
        """
        rng = get_rng(seed)
        ix = rng.choice(tuple(self.sliced_inds))
        return self.restore_ind(ix, inplace=inplace)

    unslice_rand_ = functools.partialmethod(unslice_rand, inplace=True)

    def unslice_all(self, inplace=False):
        """Unslice (restore) all sliced indices from this contraction tree.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the unslicing inplace or not.

        Returns
        -------
        ContractionTree
        """
        tree = self if inplace else self.copy()

        for ind in tuple(tree.sliced_inds):
            tree.restore_ind_(ind)

        return tree

    unslice_all_ = functools.partialmethod(unslice_all, inplace=True)

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
        minimize=None,
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
        tree.contract_stats()

        if minimize is None:
            minimize = self.get_default_objective()
        scorer = get_score_fn(minimize)

        if optimize is None:
            from .pathfinders.path_basic import OptimalOptimizer

            opt = OptimalOptimizer(
                minimize=scorer.get_dynamic_programming_minimize()
            )
        else:
            opt = optimize

        node_cost = getattr(scorer, "cost_local_tree_node", lambda _: 2)

        # different caches as we might want to reconfigure one before other
        tree.already_optimized.setdefault(minimize, set())
        already_optimized = tree.already_optimized[minimize]

        if select == "random":
            rng = get_rng(seed)
        else:
            if select == "max":
                i = 0
            elif select == "min":
                i = -1
            rng = None

        candidates, weights = tree.calc_subtree_candidates(
            pwr=weight_pwr, what=weight_what
        )

        if progbar:
            import tqdm

            pbar = tqdm.tqdm()
            pbar.set_description(_describe_tree(tree), refresh=False)

        r = 0
        try:
            while candidates and r < maxiter:
                if rng is not None:
                    (i,) = rng.choices(range(len(candidates)), weights=weights)

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
                current_cost = node_cost(tree, sub_root)
                for node in sub_branches:
                    if minimize == "size":
                        current_cost = max(current_cost, node_cost(tree, node))
                    else:
                        current_cost += node_cost(tree, node)
                    tree._remove_node(node)

                # make the optimizer more efficient by supplying accurate cap
                opt.cost_cap = max(2, current_cost)

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
        minimize=None,
        seed=None,
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
        seed : None, int or random.Random, optional
            A random seed to use.
        progbar : bool, optional
            Whether to show live progress.
        inplace : bool, optional
            Whether to perform the subtree reconfiguration inplace.

        Returns
        -------
        ContractionTree
        """
        tree = self if inplace else self.copy()

        # some of these might be unpicklable
        tree.contraction_cores.clear()

        # candidate trees
        num_keep = max(1, int(num_trees * restart_fraction))

        # how to rank the trees
        if minimize is None:
            minimize = self.get_default_objective()
        score = get_score_fn(minimize)

        rng = get_rng(seed)

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
                        "subtree_search": rng.choice(subtree_search),
                        "select": rng.choice(subtree_select),
                        "weight_pwr": rng.choice(subtree_weight_pwr),
                        "weight_what": rng.choice(subtree_weight_what),
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

    simulated_anneal = simulated_anneal_tree
    simulated_anneal_ = functools.partialmethod(simulated_anneal, inplace=True)
    parallel_temper = parallel_temper_tree
    parallel_temper_ = functools.partialmethod(parallel_temper, inplace=True)

    def slice(
        self,
        target_size=None,
        target_overhead=None,
        target_slices=None,
        temperature=0.01,
        minimize=None,
        allow_outer=True,
        max_repeats=16,
        reslice=False,
        seed=None,
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
            terminate after this is breached. This is on top of the current
            number of slices.
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
        reslice : bool, optional
            Whether to reslice the tree, i.e. first remove all currently
            sliced indices and start the search again. Generally any 'good'
            sliced indices will be easily found again.
        seed : None, int or random.Random, optional
            A random seed or generator to use for the search.
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

        if minimize is None:
            minimize = self.get_default_objective()

        tree = self if inplace else self.copy()

        if reslice:
            if target_slices is not None:
                target_slices *= tree.nslices
            tree.unslice_all_()

        sf = SliceFinder(
            tree,
            target_size=target_size,
            target_overhead=target_overhead,
            target_slices=target_slices,
            temperature=temperature,
            minimize=minimize,
            allow_outer=allow_outer,
            seed=seed,
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
        minimize=None,
        allow_outer=True,
        max_repeats=16,
        reslice=False,
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

        if minimize is None:
            minimize = self.get_default_objective()
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
                    reslice=reslice,
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
        reslice=False,
        minimize=None,
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

        # some of these might be unpicklable
        tree.contraction_cores.clear()

        # candidate trees
        num_keep = max(1, int(num_trees * restart_fraction))

        # how to rank the trees
        if minimize is None:
            minimize = self.get_default_objective()
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
                        "reslice": reslice,
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
                        groupby(lambda r: r["sliced_ind_set"], res).values()
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
        minimize=None,
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
        from .experimental.path_compressed_branchbound import (
            CompressedExhaustive,
        )

        if minimize is None:
            minimize = self.get_default_objective()

        if max_nodes == "auto":
            if max_time is None:
                max_nodes = max(10_000, self.N**2)
            else:
                max_nodes = float("inf")

        opt = CompressedExhaustive(
            minimize=minimize,
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
            objective=minimize,
        )
        if inplace:
            self.set_state_from(rtree)
            rtree = self

        rtree.contraction_cores.clear()
        return rtree

    compressed_reconfigure_ = functools.partialmethod(
        compressed_reconfigure, inplace=True
    )

    def windowed_reconfigure(
        self,
        minimize=None,
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

        if minimize is None:
            minimize = self.get_default_objective()

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
            objective=minimize,
        )

        if inplace:
            self.set_state_from(rtree)
            rtree = self

        rtree.contraction_cores.clear()
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
        """Generate a standard path (with linear recycled ids) from the
        contraction tree.

        Parameters
        ----------
        order : None, "dfs", or callable, optional
            How to order the contractions within the tree. If a callable is
            given (which should take a node as its argument), try to contract
            nodes that minimize this function first.

        Returns
        -------
        path: tuple[tuple[int, int]]
        """
        from bisect import bisect_left

        ssa = self.N
        ssas = list(range(ssa))
        node_to_ssa = dict(zip(self.gen_leaves(), ssas))
        path = []

        for parent, left, right in self.traverse(order=order):
            # map nodes to ssas
            lssa = node_to_ssa[left]
            rssa = node_to_ssa[right]
            # map ssas to linear indices, using bisection
            i, j = sorted((bisect_left(ssas, lssa), bisect_left(ssas, rssa)))
            # 'contract' nodes
            ssas.pop(j)
            ssas.pop(i)
            path.append((i, j))
            ssas.append(ssa)
            # update mapping
            node_to_ssa[parent] = ssa
            ssa += 1

        return tuple(path)

    path = deprecated(get_path, "path", "get_path")

    def get_numpy_path(self, order=None):
        """Generate a path compatible with the `optimize` kwarg of
        `numpy.einsum`.
        """
        return ["einsum_path", *self.get_path(order=order)]

    def get_ssa_path(self, order=None):
        """Generate a single static assignment path from the contraction tree.

        Parameters
        ----------
        order : None, "dfs", or callable, optional
            How to order the contractions within the tree. If a callable is
            given (which should take a node as its argument), try to contract
            nodes that minimize this function first.

        Returns
        -------
        ssa_path: tuple[tuple[int, int]]
        """
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
            involved = self.get_involved(node)
            legs = self.get_legs(node)
            removed = [ix for ix in involved if ix not in legs]
            for ix in removed:
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

        for _, l, r in self.descend():
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

        if self.has_preprocessing():
            for pi, eq in self.preprocessing.items():
                print(f"{GREY}preprocessing {pi}: {RESET}{eq}")
            print()

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

            pa = (
                "".join(
                    PINK + f"{{{ix}}}"
                    if (ix in l_legs) and (ix in r_legs)
                    else GREEN + f"({ix})"
                    if ix in r_legs
                    else BLUE + ix
                    for ix in p_inds
                )
                .replace(f"){GREEN}(", "")
                .replace(f"}}{PINK}{{", "")
            )
            la = (
                "".join(
                    PINK + f"{{{ix}}}"
                    if (ix in p_legs) and (ix in r_legs)
                    else RED + f"[{ix}]"
                    if ix in r_legs
                    else BLUE + ix
                    for ix in l_inds
                )
                .replace(f"]{RED}[", "")
                .replace(f"}}{PINK}{{", "")
            )
            ra = (
                "".join(
                    PINK + f"{{{ix}}}"
                    if (ix in p_legs) and (ix in l_legs)
                    else RED + f"[{ix}]"
                    if ix in l_legs
                    else GREEN + ix
                    for ix in r_inds
                )
                .replace(f"]{RED}[", "")
                .replace(f"}}{PINK}{{", "")
            )

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

    def get_contractor(
        self,
        order=None,
        prefer_einsum=False,
        strip_exponent=False,
        check_zero=False,
        implementation=None,
        autojit=False,
        progbar=False,
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
            If ``True``, the function will eagerly strip the exponent (in
            log10) from intermediate tensors to control numerical problems from
            leaving the range of the datatype. This method then returns the
            scaled 'mantissa' output array and the exponent separately.
        check_zero : bool, optional
            If ``True``, when ``strip_exponent=True``, explicitly check for
            zero-valued intermediates that would otherwise produce ``nan``,
            instead terminating early if encountered and returning
            ``(0.0, 0.0)``.
        implementation : str or tuple[callable, callable], optional
            What library to use to actually perform the contractions. Options
            are:

            - None: let cotengra choose.
            - "autoray": dispatch with autoray, using the ``tensordot`` and
              ``einsum`` implementation of the backend.
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
        progbar : bool, optional
            Whether to show progress through the contraction by default.

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
            check_zero,
            implementation,
            progbar,
        )
        try:
            fn = self.contraction_cores[key]
        except KeyError:
            fn = self.contraction_cores[key] = make_contractor(
                tree=self,
                order=order,
                prefer_einsum=prefer_einsum,
                strip_exponent=strip_exponent,
                check_zero=check_zero,
                implementation=implementation,
                autojit=autojit,
                progbar=progbar,
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
        autojit="auto",
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
        autojit : "auto" or bool, optional
            Whether to use ``autoray.autojit`` to jit compile the expression.
            If "auto", then let ``cotengra`` choose.
        progbar : bool, optional
            Show progress through the contraction.
        """
        if autojit == "auto":
            # choose for the user
            autojit = backend == "jax"

        fn = self.get_contractor(
            order=order,
            prefer_einsum=prefer_einsum,
            strip_exponent=strip_exponent is not False,
            implementation=implementation,
            autojit=autojit,
            check_zero=check_zero,
            progbar=progbar,
        )
        return fn(*arrays, backend=backend)

    def slice_key(self, i, strides=None):
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
        if strides is None:
            strides = get_slice_strides(self.sliced_inds)

        key = {}
        for (ind, info), stride in zip(self.sliced_inds.items(), strides):
            if info.project is None:
                key[ind] = i // stride
                i %= stride
            else:
                # size is 1 and i doesn't change
                key[ind] = info.project

        return key

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
            return functools.reduce(add_maybe_exponent_stripped, slices)

        # first we sum over non-output sliced indices
        chunks = {}
        for i, s in enumerate(slices):
            key_slice = self.slice_key(i)
            key = tuple(key_slice[ix] for ix in output_pos)
            try:
                chunks[key] = add_maybe_exponent_stripped(chunks[key], s)
            except KeyError:
                chunks[key] = s

        if isinstance(next(iter(chunks.values())), tuple):
            # have stripped exponents, need to scale to largest
            emax = max(v[1] for v in chunks.values())
            chunks = {
                k: mi * 10 ** (ei - emax) for k, (mi, ei) in chunks.items()
            }
        else:
            emax = None

        # then we stack these summed chunks over output sliced indices
        def recursively_stack_chunks(loc, remaining):
            if not remaining:
                return chunks[loc]
            arrays = [
                recursively_stack_chunks(loc + (d,), remaining[1:])
                for d in self.sliced_inds[remaining[0]].sliced_range
            ]
            axes = output_pos[remaining[0]] - len(loc)
            return do("stack", arrays, axes, like=backend)

        result = recursively_stack_chunks((), tuple(output_pos))

        if emax is not None:
            # strip_exponent was True, return the exponent separately
            return result, emax

        return result

    def gen_output_chunks(
        self, arrays, with_key=False, progbar=False, **contract_opts
    ):
        """Generate each output chunk of the contraction - i.e. take care of
        summing internally sliced indices only first. This assumes that the
        ``sliced_inds`` are sorted by whether they appear in the output or not
        (the default order). Useful for performing some kind of reduction over
        the final tensor object like  ``fn(x).sum()`` without constructing the
        entire thing.

        Parameters
        ----------
        arrays : sequence of array
            The arrays to contract.
        with_key : bool, optional
            Whether to yield the output index configuration key along with the
            chunk.
        progbar : bool, optional
            Show progress through the contraction chunks.

        Yields
        ------
        chunk : array
            A chunk of the contracted result.
        key : dict[str, int]
            The value each sliced output index takes for this chunk.
        """
        # consecutive slices of size ``stepsize`` all belong to the same output
        # block because the sliced indices are sorted output first
        stepsize = prod(
            si.size for si in self.sliced_inds.values() if si.inner
        )

        if progbar:
            import tqdm

            it = tqdm.trange(self.nslices // stepsize)
        else:
            it = range(self.nslices // stepsize)

        for o in it:
            chunk = self.contract_slice(arrays, o * stepsize, **contract_opts)

            if with_key:
                output_key = {
                    ix: x
                    for ix, x in self.slice_key(o * stepsize).items()
                    if ix in self.output
                }

            for j in range(1, stepsize):
                i = o * stepsize + j
                chunk = chunk + self.contract_slice(arrays, i, **contract_opts)

            if with_key:
                yield chunk, output_key
            else:
                yield chunk

    def contract(
        self,
        arrays,
        order=None,
        prefer_einsum=False,
        strip_exponent=False,
        check_zero=False,
        backend=None,
        implementation=None,
        autojit="auto",
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
            instead terminating early if encountered and returning
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

        return self.gather_slices(slices, backend=backend, progbar=progbar)

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

    def benchmark(
        self,
        dtype,
        max_time=60,
        min_reps=3,
        max_reps=100,
        warmup=True,
        **contract_opts,
    ):
        """Benchmark the contraction of this tree.

        Parameters
        ----------
        dtype : {"float32", "float64", "complex64", "complex128"}
            The datatype to use.
        max_time : float, optional
            The maximum time to spend benchmarking in seconds.
        min_reps : int, optional
            The minimum number of repetitions to perform, regardless of time.
        max_reps : int, optional
            The maximum number of repetitions to perform, regardless of time.
        warmup : bool or int, optional
            Whether to perform a warmup run before the benchmark. If an int,
            the number of warmup runs to perform.
        contract_opts
            Supplied to :meth:`~cotengra.ContractionTree.contract_slice`.

        Returns
        -------
        dict
            A dictionary of benchmarking results. The keys are:

            - "time_per_slice" : float
                The average time to contract a single slice.
            - "est_time_total" : float
                The estimated total time to contract all slices.
            - "est_gigaflops" : float
                The estimated gigaflops of the contraction.

        See Also
        --------
        contract_slice
        """
        import time

        from .utils import make_arrays_from_inputs

        arrays = make_arrays_from_inputs(
            self.inputs, self.size_dict, dtype=dtype
        )

        for i in range(int(warmup)):
            self.contract_slice(arrays, i % self.nslices, **contract_opts)

        t0 = time.time()
        ti = t0
        i = 0
        while (ti - t0 < max_time) or (i < min_reps):
            self.contract_slice(arrays, i % self.nslices, **contract_opts)
            ti = time.time()
            i += 1
            if i >= max_reps:
                break

        time_per_slice = (ti - t0) / i
        est_time_total = time_per_slice * self.nslices
        est_gigaflops = self.total_flops(dtype=dtype) / (1e9 * est_time_total)

        return {
            "time_per_slice": time_per_slice,
            "est_time_total": est_time_total,
            "est_gigaflops": est_gigaflops,
        }

    plot_ring = plot_tree_ring
    plot_tent = plot_tree_tent
    plot_span = plot_tree_span
    plot_flat = plot_tree_flat
    plot_circuit = plot_tree_circuit
    plot_rubberband = plot_tree_rubberband
    plot_contractions = plot_contractions
    plot_contractions_alt = plot_contractions_alt

    @functools.wraps(plot_hypergraph)
    def plot_hypergraph(self, **kwargs):
        hg = self.get_hypergraph(accel=False)
        hg.plot(**kwargs)

    def describe(self, info="normal", join=" "):
        """Return a string describing the contraction tree."""
        self.contract_stats()
        if info == "normal":
            return join.join(
                (
                    f"log10[FLOPs]={self.total_flops(log=10):.2f}",
                    f"log2[SIZE]={self.max_size(log=2):.2f}",
                )
            )

        elif info == "full":
            s = [
                f"log10[FLOPS]={self.total_flops(log=10):.2f}",
                f"log10[COMBO]={self.combo_cost(log=10):.2f}",
                f"log2[SIZE]={self.max_size(log=2):.2f}",
                f"log2[PEAK]={self.peak_size(log=2):.2f}",
            ]
            if self.sliced_inds:
                s.append(f"NSLICES={self.multiplicity:.2f}")
            return join.join(s)

        elif info == "concise":
            s = [
                f"F={self.total_flops(log=10):.2f}",
                f"C={self.combo_cost(log=10):.2f}",
                f"S={self.max_size(log=2):.2f}",
                f"P={self.peak_size(log=2):.2f}",
            ]
            if self.sliced_inds:
                s.append(f"$={self.multiplicity:.2f}")
            return join.join(s)

    def __repr__(self):
        if self.is_complete():
            return f"<{self.__class__.__name__}(N={self.N})>"
        else:
            s = "<{}(N={}, branches={}, complete={})>"
            return s.format(
                self.__class__.__name__,
                self.N,
                len(self.children),
                self.is_complete(),
            )

    def __str__(self):
        if not self.is_complete():
            return self.__repr__()
        else:
            d = self.describe("concise", join=", ")
            return f"<{self.__class__.__name__}(N={self.N}, {d})>"


def _reconfigure_tree(tree, *args, **kwargs):
    return tree.subtree_reconfigure(*args, **kwargs)


def _slice_and_reconfigure_tree(tree, *args, **kwargs):
    return tree.slice_and_reconfigure(*args, **kwargs)


def _get_tree_info(tree):
    stats = tree.contract_stats()
    stats["sliced_ind_set"] = frozenset(tree.sliced_inds)
    return stats


def _describe_tree(tree, info="normal"):
    return tree.describe(info=info)


class ContractionTreeCompressed(ContractionTree):
    """A contraction tree for compressed contractions. Currently the only
    difference is that this defaults to the 'surface' traversal ordering.
    """

    def set_state_from(self, other):
        super().set_state_from(other)
        self.set_surface_order_from_path(other.get_ssa_path())

    @classmethod
    def from_path(
        cls,
        inputs,
        output,
        size_dict,
        *,
        path=None,
        ssa_path=None,
        autocomplete="auto",
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
                "Exactly one of ``path`` or ``ssa_path`` must be supplied."
            )

        if path is not None:
            from .pathfinders.path_basic import linear_to_ssa

            ssa_path = linear_to_ssa(path)

        tree = cls(inputs, output, size_dict, **kwargs)
        terms = list(tree.gen_leaves())

        for p in ssa_path:
            merge = [terms[i] for i in p]
            terms.append(tree.contract_nodes(merge, check=check))

        tree.set_surface_order_from_path(ssa_path)

        if (len(tree.children) < tree.N - 1) and autocomplete:
            if autocomplete == "auto":
                # warn that we are completing
                warnings.warn(
                    "Path was not complete - contracting all remaining. "
                    "You can silence this warning with `autocomplete=True`."
                    "Or produce an incomplete tree with `autocomplete=False`."
                )

            tree.autocomplete(optimize="greedy-compressed")

        return tree

    def get_default_order(self):
        return "surface_order"

    def get_default_objective(self):
        if self._default_objective is None:
            self._default_objective = get_score_fn("peak-compressed")
        return self._default_objective

    def get_default_chi(self):
        objective = self.get_default_objective()
        try:
            chi = objective.chi
        except AttributeError:
            chi = "auto"

        if chi == "auto":
            chi = max(self.size_dict.values()) ** 2

        return chi

    def get_default_compress_late(self):
        objective = self.get_default_objective()
        try:
            return objective.compress_late
        except AttributeError:
            return False

    total_flops = ContractionTree.total_flops_compressed
    total_write = ContractionTree.total_write_compressed
    combo_cost = ContractionTree.combo_cost_compressed
    total_cost = ContractionTree.total_cost_compressed
    max_size = ContractionTree.max_size_compressed
    peak_size = ContractionTree.peak_size_compressed
    contraction_cost = ContractionTree.contraction_cost_compressed
    contraction_width = ContractionTree.contraction_width_compressed

    total_flops_exact = ContractionTree.total_flops
    total_write_exact = ContractionTree.total_write
    combo_cost_exact = ContractionTree.combo_cost
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

    def simulated_anneal(
        self,
        minimize=None,
        tfinal=0.0001,
        tstart=0.01,
        tsteps=50,
        numiter=50,
        seed=None,
        inplace=False,
        progbar=False,
        **kwargs,
    ):
        """Perform simulated annealing refinement of this *compressed*
        contraction tree.
        """
        from .pathfinders.path_compressed import WindowedOptimizer

        if minimize is None:
            minimize = self.get_default_objective()

        wo = WindowedOptimizer(
            self.inputs,
            self.output,
            self.size_dict,
            minimize=minimize,
            ssa_path=self.get_ssa_path(),
            seed=seed,
        )

        wo.simulated_anneal(
            tfinal=tfinal,
            tstart=tstart,
            tsteps=tsteps,
            numiter=numiter,
            progbar=progbar,
            **kwargs,
        )
        ssa_path = wo.get_ssa_path()

        rtree = self.__class__.from_path(
            self.inputs,
            self.output,
            self.size_dict,
            ssa_path=ssa_path,
            objective=minimize,
        )

        if inplace:
            self.set_state_from(rtree)
            rtree = self

        rtree.contraction_cores.clear()
        return rtree

    simulated_anneal_ = functools.partialmethod(simulated_anneal, inplace=True)


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
        sub_optimize="greedy",
        super_optimize="auto-hq",
        check=False,
        seed=None,
        **partition_opts,
    ):
        tree = ContractionTree(inputs, output, size_dict, track_childless=True)

        rng = get_rng(seed)
        rand_size_dict = jitter_dict(size_dict, random_strength, rng)

        dynamic_imbalance = ("imbalance" in partition_opts) and (
            "imbalance_decay" in partition_opts
        )
        if dynamic_imbalance:
            imbalance = partition_opts.pop("imbalance")
            imbalance_decay = partition_opts.pop("imbalance_decay")
        else:
            imbalance = imbalance_decay = None

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
            inputs = tuple(map(tuple, tree.node_to_terms(subgraph)))
            output = tuple(tree.get_legs(tree_node))
            membership = self.partition_fn(
                inputs,
                output,
                rand_size_dict,
                parts=parts_s,
                seed=rng,
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
        seed=None,
        **partition_opts,
    ):
        tree = ContractionTree(inputs, output, size_dict, track_childless=True)
        rand_size_dict = jitter_dict(size_dict, random_strength, seed)
        leaves = tuple(tree.gen_leaves())
        for node in leaves:
            tree._add_node(node, check=check)
        output = tuple(tree.output)

        while len(leaves) > groupsize:
            parts = max(2, len(leaves) // groupsize)

            inputs = [tuple(tree.get_legs(node)) for node in leaves]
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


def jitter(x, strength, rng):
    return x * (1 + strength * rng.expovariate(1.0))


def jitter_dict(d, strength, seed=None):
    rng = get_rng(seed)
    return {k: jitter(v, strength, rng) for k, v in d.items()}


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
