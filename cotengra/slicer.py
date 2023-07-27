import random
import collections
from math import log

from .oe import compute_size_by_dict, flop_count
from .core import ContractionTree
from .utils import MaxCounter, oset
from .scoring import get_score_fn
from .plot import plot_slicings, plot_slicings_alt


class ContractionCosts:
    """A simplified struct for tracking the contraction costs of a path only.

    Parameters
    ----------
    contractions : sequence of Contraction
        The set of individual operations that make up a full contraction path.
    size_dict : dict[str, int]
        The sizes of the indices involved.
    nslices : int, optional
        For keeping track of the 'multiplicity' of this set of contractions if
        part of a sliced contration where indices have been removed.
    """

    __slots__ = (
        "size_dict",
        "contractions",
        "nslices",
        "original_flops",
        "_flops",
        "_sizes",
        "_flop_reductions",
        "_write_reductions",
        "_where",
    )

    def __init__(
        self,
        contractions,
        size_dict,
        nslices=1,
        original_flops=None,
    ):
        self.size_dict = dict(size_dict)
        self.contractions = tuple(contractions)

        self._flops = 0
        self._sizes = MaxCounter()
        self._flop_reductions = collections.defaultdict(lambda: 0)
        self._write_reductions = collections.defaultdict(lambda: 0)
        self._where = collections.defaultdict(set)

        for i, c in enumerate(self.contractions):
            self._flops += c["flops"]
            self._sizes.add(c["size"])

            for ix in c["involved"].union(c["legs"]):
                d = self.size_dict[ix]
                self._flop_reductions[ix] += int((1 - 1 / d) * c["flops"])
                self._where[ix].add(i)
                if ix in c["legs"]:
                    self._write_reductions[ix] += int((1 - 1 / d) * c["size"])

        self.nslices = nslices
        if original_flops is None:
            original_flops = self._flops
        self.original_flops = original_flops

    def _set_state_from(self, other):
        """Copy all internal structure from another ``ContractionCosts``."""
        self.size_dict = other.size_dict.copy()
        self.contractions = tuple(c.copy() for c in other.contractions)
        self.nslices = other.nslices
        self.original_flops = other.original_flops
        self._flops = other._flops
        self._sizes = other._sizes.copy()
        self._flop_reductions = other._flop_reductions.copy()
        self._write_reductions = other._write_reductions.copy()
        self._where = other._where.copy()

    def copy(self):
        """Get a copy of this ``ContractionCosts``."""
        new = object.__new__(ContractionCosts)
        new._set_state_from(self)
        return new

    @property
    def size(self):
        return self._sizes.max()

    @property
    def flops(self):
        return self._flops

    @property
    def total_flops(self):
        return self.nslices * self.flops

    @property
    def overhead(self):
        return self.total_flops / self.original_flops

    @classmethod
    def from_info(cls, info, **kwargs):
        """Generate a set of contraction costs from a ``PathInfo`` object."""
        cs = []
        size_dict = info.size_dict

        # add all the input 'contractions'
        for term in info.input_subscripts.split(","):
            cs.append(
                {
                    "involved": oset(),
                    "legs": oset(term),
                    "size": compute_size_by_dict(term, size_dict),
                    "flops": 0,
                }
            )

        for c in info.contraction_list:
            eq = c[2]
            lhs, rhs = eq.split("->")
            legs = oset(rhs)
            involved = oset.union(*map(oset, lhs.split(",")))

            cs.append(
                {
                    "involved": involved,
                    "legs": legs,
                    "size": compute_size_by_dict(legs, size_dict),
                    "flops": flop_count(involved, c[1], 2, size_dict),
                }
            )

        return cls(cs, size_dict)

    @classmethod
    def from_contraction_tree(cls, contraction_tree, **kwargs):
        """Generate a set of contraction costs from a ``ContractionTree``
        object.
        """
        size_dict = contraction_tree.size_dict
        cs = (
            {
                "involved": oset(contraction_tree.get_involved(node)),
                "legs": oset(contraction_tree.get_legs(node)),
                "size": contraction_tree.get_size(node),
                "flops": contraction_tree.get_flops(node),
            }
            for node in contraction_tree.info
        )
        return cls(cs, size_dict, **kwargs)

    def remove(self, ix, inplace=False):
        """ """
        cost = self if inplace else self.copy()

        d = cost.size_dict[ix]
        cost.nslices *= d
        ix_s = oset([ix])

        for i in cost._where[ix]:
            c = cost.contractions[i]

            # update the potential flops reductions of other inds
            for oix in c["involved"]:
                di = cost.size_dict[oix]
                cost._flop_reductions[oix] -= int(
                    (1 - 1 / di) * c["flops"] * (1 - 1 / d)
                )

            # update the actual flops reduction
            old_flops = c["flops"]
            new_flops = old_flops // d
            cost._flops += new_flops - old_flops
            c["flops"] = new_flops
            c["involved"] = c["involved"].difference(ix_s)

            # update the tensor sizes
            if ix in c["legs"]:
                # update the potential size reductions of other inds
                for oix in c["legs"]:
                    di = cost.size_dict[oix]
                    cost._write_reductions[oix] -= int(
                        (1 - 1 / di) * c["size"] * (1 - 1 / d)
                    )

                old_size = c["size"]
                new_size = old_size // d
                cost._sizes.discard(old_size)
                cost._sizes.add(new_size)
                c["size"] = new_size
                c["legs"] = c["legs"].difference(ix_s)

        del cost.size_dict[ix]
        del cost._flop_reductions[ix]
        del cost._write_reductions[ix]

        return cost

    def __repr__(self):
        s = (
            "<ContractionCosts(flops={:.3e}, size={:.3e}, "
            "nslices={:.3e}, overhead={:.3f})>"
        )
        return s.format(
            self.total_flops, self.size, self.nslices, self.overhead
        )


class SliceFinder:
    """An object to help find the best indices to slice over in order to reduce
    the memory footprint of a contraction as much as possible whilst
    introducing as little extra overhead. It searches for and stores
    ``ContractionCosts``.

    Parameters
    ----------
    tree_or_info : ContractionTree or PathInfo
        Object describing the target full contraction to slice, generated for
        example from a call to :func:`~opt_einsum.contract_path`.
    target_size : int, optional
        The target number of entries in the largest tensor of the sliced
        contraction. The search algorithm will terminate after this is reached.
    target_slices : int, optional
        The target or minimum number of 'slices' to consider - individual
        contractions after slicing indices. The search algorithm will terminate
        after this is breached.
    target_overhead : float, optional
        The target increase in total number of floating point operations.
        For example, a value of  ``2.0`` will terminate the search
        just before the cost of computing all the slices individually breaches
        twice that of computing the original contraction all at once.
    temperature : float, optional
        When sampling combinations of indices, how far to randomly stray from
        what looks like the best (local) choice.
    """

    def __init__(
        self,
        tree_or_info,
        target_size=None,
        target_overhead=None,
        target_slices=None,
        temperature=0.01,
        minimize="flops",
        allow_outer=True,
    ):
        if all(
            t is None for t in (target_size, target_overhead, target_slices)
        ):
            raise ValueError(
                "You need to specify at least one of `target_size`, "
                "`target_overhead` or `target_slices`."
            )

        self.info = tree_or_info

        # the unsliced cost
        if isinstance(tree_or_info, ContractionTree):
            self.cost0 = ContractionCosts.from_contraction_tree(tree_or_info)
            self.forbidden = set(tree_or_info.output)
        else:
            # assume ``opt_einsum.PathInfo``
            self.cost0 = ContractionCosts.from_info(tree_or_info)
            self.forbidden = set(tree_or_info.output_subscript)

        if allow_outer == "only":
            # invert so only outer indices are allowed
            self.forbidden = set(self.cost0.size_dict) - self.forbidden
        elif allow_outer:  # is True
            # no restrictions
            self.forbidden = ()

        # the cache of possible slicings
        self.costs = {frozenset(): self.cost0}

        # algorithmic parameters
        self.temperature = temperature

        # search criteria
        self.target_size = target_size
        self.target_overhead = target_overhead
        self.target_slices = target_slices

        self.minimize = get_score_fn(minimize)

    def _maybe_default(self, attr, value):
        if value is None:
            return getattr(self, attr)
        return value

    def best(
        self,
        k=None,
        target_size=None,
        target_overhead=None,
        target_slices=None,
    ):
        """Return the best contraction slicing, subject to target filters."""
        target_size = self._maybe_default("target_size", target_size)
        target_overhead = self._maybe_default(
            "target_overhead", target_overhead
        )
        target_slices = self._maybe_default("target_slices", target_slices)

        size_specified = target_size is not None
        overhead_specified = target_overhead is not None
        slices_specified = target_slices is not None

        valid = filter(
            lambda x: (
                (not size_specified or (x[1].size <= target_size))
                and (
                    not overhead_specified
                    or (x[1].overhead <= target_overhead)
                )
                and (not slices_specified or (x[1].nslices >= target_slices))
            ),
            self.costs.items(),
        )

        if size_specified or slices_specified:
            # sort primarily by overall flops
            def best_scorer(x):
                return (x[1].total_flops, x[1].nslices, x[1].size)

        else:  # only overhead_specified
            # sort by size of contractions achieved
            def best_scorer(x):
                return (x[1].size, x[1].total_flops, x[1].nslices)

        if k is None:
            return min(valid, key=best_scorer)

        return sorted(valid, key=best_scorer)[:k]

    def trial(
        self,
        target_size=None,
        target_overhead=None,
        target_slices=None,
        temperature=None,
    ):
        """A single slicing attempt, greedily select indices from the popular
        pool, subject to the score function, terminating when any of the
        target criteria are met.
        """
        # optionally override some defaults
        temperature = self._maybe_default("temperature", temperature)
        target_size = self._maybe_default("target_size", target_size)
        target_overhead = self._maybe_default(
            "target_overhead", target_overhead
        )
        target_slices = self._maybe_default("target_slices", target_slices)

        size_specified = target_size is not None
        overhead_specified = target_overhead is not None
        slices_specified = target_slices is not None

        # hashable set of indices we are slicing
        ix_sl = frozenset()
        cost = self.costs[ix_sl]

        already_satisfied = (
            (size_specified and (cost.size <= target_size))
            or (overhead_specified and (cost.overhead > target_overhead))
            or (slices_specified and (cost.nslices >= target_slices))
        )

        while not already_satisfied:
            ix = max(
                cost.size_dict,
                key=lambda ix:
                # the base score
                self.minimize.score_slice_index(cost, ix) -
                # a smudge that replicates boltzmann sampling
                temperature * log(-log(random.random())) -
                # penalize forbidden (outer) indices
                (0 if ix not in self.forbidden else float("inf")),
            )
            if ix in self.forbidden:
                raise RuntimeError("Ran out of valid indices to slice.")

            next_ix_sl = ix_sl | frozenset([ix])

            # cache sliced contraction costs
            try:
                next_cost = self.costs[next_ix_sl]
            except KeyError:
                next_cost = self.costs[next_ix_sl] = cost.remove(ix)

            # check if we are about to break the flops limit
            if overhead_specified and (next_cost.overhead > target_overhead):
                break

            # accept the index
            ix_sl = next_ix_sl
            cost = next_cost

            # check if we are about to generate too many slices
            if slices_specified and (cost.nslices >= target_slices):
                break

            # check if we have reached the desired memory target
            if size_specified and (cost.size <= target_size):
                break

        return cost

    def search(
        self,
        max_repeats=16,
        temperature=None,
        target_size=None,
        target_overhead=None,
        target_slices=None,
    ):
        """Repeat trial several times and return the best found so far."""
        for _ in range(max_repeats):
            self.trial(
                target_overhead=target_overhead,
                target_slices=target_slices,
                target_size=target_size,
                temperature=temperature,
            )

        return self.best(
            target_overhead=target_overhead,
            target_slices=target_slices,
            target_size=target_size,
        )

    plot_slicings = plot_slicings
    plot_slicings_alt = plot_slicings_alt


def create_size_dict(inputs, arrays):
    size_dict = {}
    for term, array in zip(inputs, arrays):
        for ix, d in zip(term, array.shape):
            size_dict[ix] = max(size_dict.get(ix, 1), d)
    return size_dict
