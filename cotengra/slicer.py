import math
import random
import operator
import functools
import collections

from opt_einsum import contract_expression, contract_path
from opt_einsum.contract import PathInfo
from opt_einsum.helpers import compute_size_by_dict, flop_count


Contraction = collections.namedtuple(
    'Contraction', ['involved', 'legs', 'size', 'flops']
)


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

    __slots__ = ['size_dict', 'nslices', 'contractions',
                 '_size', '_flops', '_scores', 'original_flops']

    def __init__(self, contractions, size_dict, nslices=1, original_flops=None):
        self.size_dict = size_dict
        self.nslices = nslices
        self.contractions = tuple(sorted(contractions, key=lambda c: -c.size))
        self._size = max(c.size for c in self.contractions)
        self._flops = sum(c.flops for c in self.contractions)
        self._scores = {}
        if original_flops is None:
            original_flops = self._flops
        self.original_flops = original_flops

    @property
    def size(self):
        return self._size

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
    def from_pathinfo(cls, pathinfo, **kwargs):
        """Generate a set of contraction costs from a ``PathInfo`` object.
        """
        cs = []
        size_dict = pathinfo.size_dict.copy()

        # add all the input 'contractions'
        for term in pathinfo.input_subscripts.split(','):
            cs.append(Contraction(
                involved=frozenset(),
                legs=frozenset(term),
                size=compute_size_by_dict(term, size_dict),
                flops=0,
            ))

        for c in pathinfo.contraction_list:
            eq = c[2]
            lhs, rhs = eq.split('->')
            legs = frozenset(rhs)
            involved = frozenset.union(*map(frozenset, lhs.split(',')))

            cs.append(Contraction(
                involved=involved,
                legs=legs,
                size=compute_size_by_dict(legs, size_dict),
                flops=flop_count(involved, c[1], 2, size_dict),
            ))

        return cls(cs, size_dict)

    @classmethod
    def from_contraction_tree(cls, contraction_tree, **kwargs):
        """Generate a set of contraction costs from a ``ContractionTree``
        object.
        """
        size_dict = contraction_tree.size_dict.copy()
        cs = (
            Contraction(involved=contraction_tree.get_involved(node),
                        legs=contraction_tree.get_legs(node),
                        size=contraction_tree.get_size(node),
                        flops=contraction_tree.get_flops(node))
            for node in contraction_tree.info
        )
        return cls(cs, size_dict, **kwargs)

    def score(self, ix):
        """A heuristic score for judging whether to remove index ``ix``.
        """
        try:
            return self._scores[ix]
        except KeyError:
            pass

        flops_reduction = 0.0
        size_reduction = 0.0
        d = self.size_dict[ix]

        for c in self.contractions:
            if ix in c.involved:
                flops_reduction += (1 - 1 / d) * c.flops
            if ix in c.legs:
                size_reduction += (1 - 1 / d) * c.size

        flops_increase = d * (1 - flops_reduction / self.flops)

        score = self._scores[ix] = (-size_reduction, flops_increase)
        return score

    def remove(self, ix):
        """Create a new set of ``ContractionCosts`` with ``ix`` removed.
        """
        new_cs = []
        d = self.size_dict[ix]

        for c in self.contractions:
            if ix in c.involved:
                new_flops = c.flops / d
                new_size = c.size
                if ix in c.legs:
                    new_size /= d
                fi = frozenset((ix,))
                involved = c.involved - fi
                legs = c.legs - fi
                new_cs.append(Contraction(involved, legs, new_size, new_flops))
            else:
                new_cs.append(c)

        return ContractionCosts(new_cs, self.size_dict,
                                nslices=d * self.nslices,
                                original_flops=self.original_flops)

    def __repr__(self):
        s = "<ContractionCosts(flops={:.3e}, size={:.3e}, nslices={:.3e})>"
        return s.format(self.flops, self.size, self.nslices)


class SliceFinder:
    """An object to help find the best indices to slice over in order to reduce
    the memory footprint of a contraction as much as possible whilst
    introducing as little extra overhead. It searches for and stores
    ``ContractionCosts``.

    Parameters
    ----------
    pathinfo : PathInfo
        Object describing the target full contraction to slice, generated for
        example from a call to :func:`~opt_einsum.contract_path`.
    target_size : int, optional
        The target number of entries in the largest tensor of the sliced
        contraction. The search algorithm will terminate after this is reached.
    target_slices : int, optional
        The target or maxmimum number of 'slices' to consider - individual
        contractions after slicing indices. The search algorithm will terminate
        if this is breached.
    target_flops : float, optional
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
        pathinfo,
        target_size=None,
        target_flops=None,
        target_slices=None,
        temperature=0.01,
        allow_outer=False,
        max_candidates='auto',
    ):
        if all(t is None for t in (target_size, target_flops, target_slices)):
            raise ValueError(
                "You need to specify at least one of `target_size`, "
                "`target_flops` or `target_slices`.")

        self.pathinfo = pathinfo

        # the unsliced cost
        if isinstance(pathinfo, PathInfo):
            self.cost0 = ContractionCosts.from_pathinfo(pathinfo)
            self.forbidden = set(pathinfo.output_subscript)
        else:
            self.cost0 = ContractionCosts.from_contraction_tree(pathinfo)
            self.forbidden = pathinfo.output

        if allow_outer:
            self.forbidden = {}

        # the cache of possible slicings
        self.costs = {frozenset(): self.cost0}

        # roughly rank indices according to how often and how 'big' they appear
        popularity_score = collections.defaultdict(lambda: 0)
        for c in self.cost0.contractions:
            for ix in c.legs:
                if ix not in self.forbidden:
                    popularity_score[ix] -= math.log2(c.size)
        self.ranked_indices = sorted(
            popularity_score, key=popularity_score.__getitem__,
        )

        # algorithmic parameters
        self.temperature = temperature
        if max_candidates == 'auto':
            max_candidates = max(16, int(0.1 * len(self.ranked_indices)))
        self.max_candidates = int(max_candidates)

        # search criteria
        self.target_size = target_size
        self.target_flops = target_flops
        self.target_slices = target_slices

    def _maybe_default(self, attr, value):
        if value is None:
            return getattr(self, attr)
        return value

    def best(self, target_size=None, target_flops=None, target_slices=None):
        """Return the best contraction slicing, subject to target filters.
        """
        target_size = self._maybe_default('target_size', target_size)
        target_flops = self._maybe_default('target_flops', target_flops)
        target_slices = self._maybe_default('target_slices', target_slices)

        valid = filter(
            lambda x: (
                ((target_size is None) or (x[1].size <= target_size)) and
                ((target_flops is None) or
                 (x[1].total_flops / self.cost0.flops <= target_flops)) and
                ((target_slices is None) or (x[1].nslices <= target_slices))
            ),
            self.costs.items()
        )

        return min(
            valid,
            key=lambda x: (x[1].total_flops, x[1].nslices, x[1].size)
        )

    def trial(self, target_size=None, target_flops=None, target_slices=None,
              max_candidates=None, temperature=None):
        """A single slicing attempt, greedily select indices from the popular
        pool, subject to the score function, terminating when any of the
        target criteria are met.
        """
        # optionally override some defaults
        temperature = self._maybe_default('temperature', temperature)
        target_size = self._maybe_default('target_size', target_size)
        target_flops = self._maybe_default('target_flops', target_flops)
        target_slices = self._maybe_default('target_slices', target_slices)
        max_candidates = self._maybe_default('max_candidates', max_candidates)

        # hashable set of indices we are slicing
        fsix = frozenset()
        inds_to_check = set(self.ranked_indices[:max_candidates])
        inds_backup = iter(self.ranked_indices[max_candidates:])
        cost = cost0 = self.costs[fsix]
        d = 1

        already_satisfied = (
            ((target_size is not None) and (cost.size <= target_size)) or
            (target_slices == 1)
        )

        while not already_satisfied and inds_to_check:
            ix = min(
                inds_to_check,
                key=lambda ix:
                (1 + temperature * random.expovariate(1.0)) * cost.score(ix)[0]
            )
            next_fsix = fsix | frozenset([ix])

            # cache sliced contraction costs
            try:
                next_cost = self.costs[next_fsix]
            except KeyError:
                next_cost = self.costs[next_fsix] = cost.remove(ix)

            next_d = d * next_cost.size_dict[ix]
            next_flops = next_d * next_cost.flops / cost0.flops

            # check if we are about to break the flops limit
            if (target_flops is not None) and (next_flops > target_flops):
                break

            # check if we are about to generate too many slices
            if (target_slices is not None) and (next_d > target_slices):
                break

            # accept the index
            inds_to_check.remove(ix)
            d = next_d
            fsix = next_fsix
            cost = next_cost

            # check if we have reached the desired memory target
            if (target_size is not None) and (cost.size <= target_size):
                break

            # add more indices to check
            try:
                inds_to_check.add(next(inds_backup))
            except StopIteration:
                pass

        return cost

    def search(self, max_repeats=16, temperature=None, max_candidates=None,
               target_size=None, target_flops=None, target_slices=None):
        """Repeat trial several times and return the best found so far.
        """
        for _ in range(max_repeats):
            self.trial(target_flops=target_flops, target_slices=target_slices,
                       target_size=target_size, temperature=temperature,
                       max_candidates=max_candidates)

        return self.best(target_flops=target_flops,
                         target_slices=target_slices, target_size=target_size)

    def SlicedContractor(self, arrays, target_size=None, target_flops=None,
                         target_slices=None, **kwargs):
        """Generate a sliced contraction using the best indices found by this
        `SliceFinder` and by default the original contraction path as well.
        """
        sliced = self.best(
            target_size=target_size, target_flops=target_flops,
            target_slices=target_slices
        )[0]

        return SlicedContractor.from_pathinfo(
            pathinfo=self.pathinfo, arrays=arrays, sliced=sliced, **kwargs
        )


def create_size_dict(inputs, arrays):
    size_dict = {}
    for term, array in zip(inputs, arrays):
        for ix, d in zip(term, array.shape):
            size_dict[ix] = max(size_dict.get(ix, 1), d)
    return size_dict


def prod(it):
    """Compute the product of sequence of numbers ``it``.
    """
    x = 1
    for i in it:
        x *= i
    return x


def dynal(x, bases):
    """Represent the integer ``x`` with respect to the 'dynamical' ``bases``.
    Gives a way to reliably enumerate and 'de-enumerate' the combination of
    all different index values.

    Examples
    --------

        >>> dynal(9, [2, 2, 2, 2])  # binary
        [1, 0, 0, 1]

        >>> dynal(123, [10, 10, 10])  # decimal
        [1, 2, 3]

        >>> # arbitrary
        >>> bases = [2, 5, 7, 3, 8, 7, 20, 4]
        >>> for i in range(301742, 301752):
        ...     print(dynal(i, bases))
        [0, 3, 1, 1, 2, 5, 15, 2]
        [0, 3, 1, 1, 2, 5, 15, 3]
        [0, 3, 1, 1, 2, 5, 16, 0]
        [0, 3, 1, 1, 2, 5, 16, 1]
        [0, 3, 1, 1, 2, 5, 16, 2]
        [0, 3, 1, 1, 2, 5, 16, 3]
        [0, 3, 1, 1, 2, 5, 17, 0]
        [0, 3, 1, 1, 2, 5, 17, 1]
        [0, 3, 1, 1, 2, 5, 17, 2]
        [0, 3, 1, 1, 2, 5, 17, 3]

    """
    bs_szs = [prod(bases[i + 1:]) for i in range(len(bases))]
    dx = []
    for b in bs_szs:
        div = x // b
        dx.append(div)
        x -= div * b
    return dx


class SlicedContractor:
    """A contraction where certain indices are explicitly summed over,
    corresponding to taking different 'slices' of the input arrays, each of
    which can be contracted independently with *hopefully* a lower memory
    requirement. The recommended way of instantiating this is from a
    directly from ``SliceFinder`` which already.

    Parameters
    ----------
    eq : str
        The overall contraction to perform.
    arrays : sequence of array
        The arrays to contract.
    sliced : sequence of str
        Which indices in ``eq`` to slice over.
    optimize : str or path or PathOptimizer, optional
        How to optimize the sliced contraction path - the contraction with
        ``sliced`` indices removed. If these ``sliced`` indices were found
        automatically is it generally best to supply the full path they were
        found with respect to rather than trying to re-optimize the path.
    size_dict : dict[str, int], optional
        If already known, the sizes of each index.
    """

    def __init__(
        self,
        eq,
        arrays,
        sliced,
        optimize='auto',
        size_dict=None,
    ):
        # basic info
        self.inputs = eq.split('->')[0].split(',')
        self.arrays = tuple(arrays)
        self.sliced = tuple(sorted(sliced, key=eq.index))
        if size_dict is None:
            size_dict = create_size_dict(self.inputs, self.arrays)

        # find which arrays are going to be sliced or not
        self.constant, self.changing = [], []
        for i, term in enumerate(self.inputs):
            if any(ix in self.sliced for ix in term):
                self.changing.append(i)
            else:
                self.constant.append(i)

        # information about the contraction of a single slice
        self.eq_sliced = "".join(c for c in eq if c not in sliced)
        self.sliced_sizes = tuple(size_dict[i] for i in self.sliced)
        self.nslices = compute_size_by_dict(self.sliced, size_dict)
        self.shapes_sliced = tuple(
            tuple(size_dict[i] for i in term)
            for term in self.eq_sliced.split('->')[0].split(',')
        )
        self.path, self.pathinfo_sliced = contract_path(
            self.eq_sliced, *self.shapes_sliced, shapes=True, optimize=optimize
        )

        # generate the contraction expression
        self._expr = contract_expression(
            self.eq_sliced, *self.shapes_sliced, optimize=self.path
        )

    @classmethod
    def from_pathinfo(cls, pathinfo, arrays, sliced, optimize=None, **kwargs):
        """Creat a `SlicedContractor` directly from a `PathInfo` object.
        """
        # by default inherit the pathinfo's path
        if optimize is None:
            optimize = pathinfo.path

        return cls(eq=pathinfo.eq, arrays=arrays, sliced=sliced,
                   optimize=optimize, size_dict=pathinfo.size_dict, **kwargs)

    @property
    def individual_flops(self):
        """FLOP cost of a single contraction slice.
        """
        return self.pathinfo_sliced.opt_cost

    @property
    def total_flops(self):
        """FLOP cost of performing all sliced contractions.
        """
        return self.individual_flops * self.nslices

    @property
    def max_size(self):
        """The largest size tensor produced in an individual contraction.
        """
        return self.pathinfo_sliced.largest_intermediate

    def get_sliced_arrays(self, i):
        """Generate the tuple of array inputs corresponding to slice ``i``.
        """
        temp_arrays = list(self.arrays)

        # e.g. {'a': 2, 'd': 7, 'z': 0}
        locations = dict(zip(self.sliced, dynal(i, self.sliced_sizes)))

        for i in self.changing:
            # the indexing object, e.g. [:, :, 7, :, 2, :, :, 0]
            selector = tuple(
                locations.get(ix, slice(None)) for ix in self.inputs[i]
            )
            # re-insert the sliced array
            temp_arrays[i] = temp_arrays[i][selector]

        return tuple(temp_arrays)

    def contract_slice(self, i, **kwargs):
        """Contraction of just slice ``i``.
        """
        arrays = self.get_sliced_arrays(i)
        return self._expr(*arrays, **kwargs)

    def contract_all(self, **kwargs):
        """Contract (and sum) all slices at once.
        """
        return functools.reduce(
            operator.add,
            (self.contract_slice(i, **kwargs) for i in range(self.nslices))
        )

    def get_dask_chunked(self, **kwargs):
        """
        """
        import dask.array as da

        return tuple(
            da.from_array(x, chunks=tuple(
                1 if ix in self.sliced else None
                for ix in term
            ), **kwargs)
            for term, x in zip(self.inputs, self.arrays)
        )

    def get_mars_chunked(self, **kwargs):
        """
        """
        import mars.tensor as mt

        return tuple(
            mt.tensor(x, chunk_size=tuple(
                1 if ix in self.sliced else max(x.shape)
                for ix in term
            ), **kwargs)
            for term, x in zip(self.inputs, self.arrays)
        )
