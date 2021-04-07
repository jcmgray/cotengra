import operator
import functools
import itertools
import collections

try:
    from cytoolz import groupby, interleave, unique
except ImportError:
    from toolz import groupby, interleave, unique


__all__ = ('groupby', 'interleave', 'unique')


def prod(it):
    """Compute the product of sequence of numbers ``it``.
    """
    x = 1
    for i in it:
        x *= i
    return x


def dynary(x, bases):
    """Represent the integer ``x`` with respect to the 'dynamical' ``bases``.
    Gives a way to reliably enumerate and 'de-enumerate' the combination of
    all different index values.

    Examples
    --------

        >>> dynary(9, [2, 2, 2, 2])  # binary
        [1, 0, 0, 1]

        >>> dynary(123, [10, 10, 10])  # decimal
        [1, 2, 3]

        >>> # arbitrary
        >>> bases = [2, 5, 7, 3, 8, 7, 20, 4]
        >>> for i in range(301742, 301752):
        ...     print(dynary(i, bases))
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


class oset:
    """An ordered set which stores elements as the keys of dict (ordered as of
    python 3.6). 'A few times' slower than using a set directly for small
    sizes, but makes everything deterministic.
    """

    __slots__ = ('_d',)

    def __init__(self, it=()):
        self._d = dict.fromkeys(it)

    @classmethod
    def _from_dict(cls, d):
        obj = object.__new__(oset)
        obj._d = d
        return obj

    @classmethod
    def from_dict(cls, d):
        """Public method makes sure to copy incoming dictionary.
        """
        return oset._from_dict(d.copy())

    def copy(self):
        return oset.from_dict(self._d)

    def add(self, k):
        self._d[k] = None

    def discard(self, k):
        self._d.pop(k, None)

    def remove(self, k):
        self._d.pop(k)

    def clear(self):
        self._d.clear()

    def update(self, *others):
        for o in others:
            self._d.update(o._d)

    def union(self, *others):
        u = self.copy()
        u.update(*others)
        return u

    def intersection_update(self, *others):
        if len(others) > 1:
            si = set.intersection(*(set(o._d) for o in others))
        else:
            si = others[0]._d
        self._d = {k: None for k in self._d if k in si}

    def intersection(self, *others):
        n_others = len(others)
        if n_others == 0:
            return self.copy()
        elif n_others == 1:
            si = others[0]._d
        else:
            si = set.intersection(*(set(o._d) for o in others))
        return oset._from_dict({k: None for k in self._d if k in si})

    def difference_update(self, *others):
        if len(others) > 1:
            su = set.union(*(set(o._d) for o in others))
        else:
            su = others[0]._d
        self._d = {k: None for k in self._d if k not in su}

    def difference(self, *others):
        if len(others) > 1:
            su = set.union(*(set(o._d) for o in others))
        else:
            su = others[0]._d
        return oset._from_dict({k: None for k in self._d if k not in su})

    def symmetric_difference(self, other):
        return oset._from_dict({
            k: None for k in itertools.chain(self._d, other._d)
            if (k not in self._d) or (k not in other)
        })

    def __eq__(self, other):
        if isinstance(other, oset):
            return self._d == other._d
        return False

    def __or__(self, other):
        return self.union(other)

    def __ior__(self, other):
        self.update(other)
        return self

    def __and__(self, other):
        return self.intersection(other)

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __sub__(self, other):
        return self.difference(other)

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def __len__(self):
        return self._d.__len__()

    def __iter__(self):
        return self._d.__iter__()

    def __contains__(self, x):
        return self._d.__contains__(x)

    def __repr__(self):
        return f"oset({list(self._d)})"


class MaxCounter:
    """Simple class to keep track of the maximum in a likely changing
    sequence of elements.

    Parameters
    ----------
    it : None or sequence of hashable, optional
        The initial items to add.

    Examples
    --------

        >>> mc = MaxCounter([1, 2, 3, 3])
        >>> mc.max()
        3

        >>> mc.discard(3)
        >>> mc.max()
        3

        >>> mc.discard(3)
        >>> mc.max()
        2

        >>> mc.add(10)
        >>> mc.max()
        10

    """

    __slots__ = ('_c', '_max_element')

    def __init__(self, it=None):
        self._c = collections.Counter(it)
        if it is None:
            self._max_element = -float('inf')
        else:
            self._max_element = max(self._c)

    def copy(self):
        new = object.__new__(MaxCounter)
        new._max_element = self._max_element
        new._c = self._c.copy()
        return new

    def discard(self, x):
        """Discard element ``x`` and possibly update the maximum.
        """
        cnt = self._c[x]
        if cnt <= 1:
            del self._c[x]
            if x == self._max_element:
                # only need to update the max if ``x``
                # was the last maximum sized element
                try:
                    self._max_element = max(self._c)
                except ValueError:
                    self._max_element = -float('inf')
        else:
            self._c[x] = cnt - 1

    def add(self, x):
        """Add element ``x`` and possibly update the maximum.
        """
        self._c[x] += 1
        self._max_element = max(self._max_element, x)

    def max(self):
        """The maximum element in this list.
        """
        return self._max_element


class BitSet:

    __slots__ = ('members', 'map', 'size', 'infimum', 'supremum')

    def __init__(self, it):
        self.members = tuple(unique(it))
        self.map = {m: i for i, m in enumerate(self.members)}
        self.size = len(self.members)
        self.supremum = self.fromint(2**self.size - 1)
        self.infimum = self.fromint(0)

    def asint(self, elem):
        return 1 << self.map[elem]

    def fromint(self, n):
        return BitMembers.fromint(self, n)

    def frommembers(self, it=()):
        return BitMembers.frommembers(self, it)

    __call__ = frommembers


class BitMembers:

    __slots__ = ('i', 'bitset')

    @classmethod
    def fromint(cls, bitset, n):
        self = object.__new__(cls)
        self.bitset = bitset
        self.i = n
        return self

    @classmethod
    def frommembers(cls, bitset, it=()):
        self = object.__new__(cls)
        self.bitset = bitset
        self.i = functools.reduce(operator.or_, map(self.bitset.asint, it), 0)
        return self

    def __int__(self):
        return self.i

    __hash__ = __int__

    def __eq__(self, other):
        return (self.bitset is other.bitset) and (self.i == other.i)

    def __len__(self):
        return f"{self.i:b}".count("1")

    def __iter__(self):
        return (
            x for b, x in zip(bin(self.i)[:1:-1], self.bitset.members)
            if b == "1"
        )

    def add(self, elem):
        self.i |= self.bitset.asint(elem)

    def clear(self):
        self.i = 0

    def copy(self):
        return self.bitset.fromint(self.i)

    def __bool__(self):
        return self.i != 0

    def __contains__(self, elem):
        return self.i & self.bitset.asint(elem)

    def discard(self, elem):
        self.i &= self.bitset.supremum.i - self.bitset.asint(elem)

    def remove(self, elem):
        if elem not in self:
            raise KeyError
        self.discard(elem)

    def difference_update(self, *others):
        for other in others:
            self.i &= ~other.i

    __isub__ = difference_update

    def difference(self, *others):
        bm = self.copy()
        bm.difference_update(*others)
        return bm

    __sub__ = difference

    def intersection_update(self, *others):
        for other in others:
            self.i &= other.i

    __iand__ = intersection_update

    def intersection(self, *others):
        bm = self.copy()
        bm.intersection_update(*others)
        return bm

    __and__ = intersection

    def isdisjoint(self, other):
        return not self.i & other.i

    def issubset(self, other):
        return self.i & other.i == self.i

    def issuperset(self, other):
        return self.i | other.i == self.i

    def symmetric_difference_update(self, other):
        self.i ^= other.i

    __ixor__ = symmetric_difference_update

    def symmetric_difference(self, other):
        return self.bitset.fromint(self.i ^ other.i)

    __xor__ = symmetric_difference

    def update(self, *others):
        for other in others:
            self.i |= other.i

    __ior__ = update

    def union(self, *others):
        bm = self.copy()
        bm.update(*others)
        return bm

    __or__ = union


try:
    # accelerated bitmember iteration
    from cotengra.cotengra import indexes

    def fast_bitmember_iter(self):
        return map(self.bitset.members.__getitem__, indexes(f"{self.i:b}"))

    BitMembers.__iter__ = fast_bitmember_iter

except ImportError:
    pass


def rand_equation(
    n, reg,
    n_out=0,
    n_hyper_in=0,
    n_hyper_out=0,
    d_min=2,
    d_max=3,
    seed=None
):
    """A more advanced version of ``opt_einsum.helpers.rand_equation`` that
    can also generate both inner and outer hyper-edges. Mostly useful for
    generating test instances covering all edge cases.

    Parameters
    ----------
    n : int
        The number of tensors.
    reg : int
        The average number of indices per tensor if no hyper-edges, i.e.
        total number of inds ``= n * reg // 2``.
    n_out : int, optional
        The number of output indices.
    n_hyper_in : int, optional
        The number of inner hyper-indices.
    n_hyper_out : int, optional
        The number of outer hyper-indices.
    d_min : int, optional
        The minimum dimension size.
    d_max : int, optional
        The maximum dimension size.
    seed : None or int, optional
        Seed for ``np.random`` for repeatibility.

    Returns
    -------
    inputs : list[list[str]]
    output : list[str]
    shapes : list[tuple[int]]
    size_dict : dict[str, int]
    """
    import numpy as np
    import opt_einsum as oe

    if seed is not None:
        np.random.seed(seed)

    num_inds = max((n * reg) // 2, n_hyper_out + n_hyper_in + n_out)
    size_dict = {oe.get_symbol(i): np.random.randint(d_min, d_max + 1)
                 for i in range(num_inds)}

    inds = iter(size_dict)
    inputs = [[] for _ in range(n)]
    output = []

    for _ in range(n_hyper_out):
        ind = next(inds)
        output.append(ind)
        s = np.random.randint(3, n + 1)
        where = np.random.choice(np.arange(n), size=s, replace=False)
        for i in where:
            inputs[i].append(ind)

    for _ in range(n_hyper_in):
        ind = next(inds)
        s = np.random.randint(3, n + 1)
        where = np.random.choice(np.arange(n), size=s, replace=False)
        for i in where:
            inputs[i].append(ind)

    for _ in range(n_out):
        ind = next(inds)
        output.append(ind)
        where = np.random.choice(np.arange(n), size=2, replace=False)
        for i in where:
            inputs[i].append(ind)

    for ind in inds:
        where = np.random.choice(np.arange(n), size=2, replace=False)
        for i in where:
            inputs[i].append(ind)

    shapes = [tuple(size_dict[ix] for ix in term) for term in inputs]

    output = list(np.random.permutation(output))

    return inputs, output, shapes, size_dict
