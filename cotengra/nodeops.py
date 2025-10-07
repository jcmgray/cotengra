from functools import reduce
from operator import or_


NODE_TYPE = (
    "frozenset[int]"
    # "BitSetInt"
    # "intbitset"
)


class BitSetInt(int):
    """A bitset for keeping track of dense sets of integers. Subclass of
    ``int`` that overrides ``-``,  ``&`` and  ``|`` and provides ``__len__``
    and ``__iter__``. As of python 3.8 doesn't seem much better than using
    frozenset[int] (slower and limited memory savings).

    Parameters
    ----------
    it : Iterable[int] or int
        If sequence of ``int``, treat these as the set members. If raw ``int``
        interpret as bitset directly.
    """

    __slots__ = ()

    def __new__(cls, it=0):
        if isinstance(it, int):
            i = it
        else:
            i = reduce(or_, (1 << i for i in it), 0)
        return super(cls, cls).__new__(cls, i)

    def __hash__(self):
        return self ^ self.bit_length()

    @classmethod
    def infimum(cls):
        return super(cls, cls).__new__(cls, 0)

    @classmethod
    def supremum(cls, size):
        return super(cls, cls).__new__(cls, (1 << size) - 1)

    def __len__(self):
        return self.bit_count()

    def iter_sparse(self):
        x = self
        while x:
            i = x.bit_length() - 1
            yield i
            x ^= 1 << i

    def iter_dense(self):
        return (x for x, b in enumerate(bin(self)[:1:-1]) if b == "1")

    def iter_numpy(self):
        import numpy as np

        bits = np.frombuffer(
            self.to_bytes((self.bit_length() + 7) // 8, "little"),
            dtype=np.uint8,
        )
        bits = np.unpackbits(bits, bitorder="little")
        return np.flatnonzero(bits)

    def __iter__(self):
        raise

        n = self.bit_length()
        k = self.bit_count()

        if n == k:
            # contains all elements 0..n-1
            return iter(range(n))
        if k == 1:
            # contains single element, at location of highest bit
            return iter((self.bit_length() - 1,))
        if k < 512:
            # sparse enough to be worth iterating sparsely
            # XXX: better fit is k ~< 18 * n**0.25
            return self.iter_sparse()

        try:
            return self.iter_numpy()
        except ImportError:
            return self.iter_dense()

    def __contains__(self, i):
        return bool(self & (1 << i))

    def __sub__(self, i):
        return BitSetInt(int.__and__(self, ~i))

    def difference(self, i):
        return self - i

    def __and__(self, i):
        return BitSetInt(int.__and__(self, i))

    def intersection(self, i):
        return self & i

    def __or__(self, i):
        return BitSetInt(int.__or__(self, i))

    def union(*it):
        return BitSetInt(reduce(int.__or__, it))

    def issubset(self, other):
        """Check if this bitset is a subset of another bitset.

        x.issubset(y) returns True if all bits set in x are also set in y.
        Equivalent to: x & y == x
        """
        return (self & other) == self

    def __repr__(self):
        return f"BitSetInt({list(self.iter_dense())})"


if NODE_TYPE == "frozenset[int]":
    node_type = frozenset
    node_size = len
    node_from_seq = frozenset

    def node_from_single(x):
        return frozenset((x,))

    def node_supremum(size):
        return frozenset(range(size))

    def node_get_single_el(node):
        """Assuming ``node`` has one element, return it."""
        return next(iter(node))

    def node_tie_breaker(x):
        return -min(x)

    def is_valid_node(node):
        """Check ``node`` is of type ``frozenset[int]``."""
        try:
            if not isinstance(node, frozenset):
                return False
            el = next(iter(node))
            if not isinstance(el, int):
                return False
            return True
        except TypeError:
            return False

    def node_union(x, y):
        return x | y

    def node_union_it(bs):
        """Non-variadic version of various set type unions."""
        b0, *bs = bs
        return b0.union(*bs)

    def node_issubset(x, y):
        return x.issubset(y)


elif NODE_TYPE == "BitSetInt":
    # the set of functions needed to use BitSetInt as contraction tree nodes

    node_type = BitSetInt
    node_size = BitSetInt.__len__
    node_from_seq = BitSetInt

    def node_from_single(x):
        return BitSetInt(1 << x)

    def node_supremum(size):
        return BitSetInt.supremum(size)

    def node_get_single_el(node):
        return node.bit_length() - 1

    def node_tie_breaker(x):
        return x

    def is_valid_node(node):
        try:
            return isinstance(node, BitSetInt)
        except TypeError:
            return False

    def node_union(x, y):
        return BitSetInt(int.__or__(x, y))

    def node_union_it(bs):
        b0, *bs = bs
        return b0.union(*bs)

    def node_issubset(x, y):
        return (x & y) == x


elif NODE_TYPE == "intbitset":
    from intbitset import intbitset

    node_type = intbitset
    node_size = len
    node_from_seq = intbitset

    # intbitset could be great, but hash collisions make use in dicts v v slow

    def node_from_single(x):
        return intbitset((x,))

    def node_supremum(size):
        return intbitset(range(size))

    def node_get_single_el(node):
        return next(iter(node))

    def is_valid_node(node):
        try:
            return isinstance(node, intbitset)
        except TypeError:
            return False

    def node_union(x, y):
        return x.union(y)

    def node_union_it(bs):
        """Non-variadic version of various set type unions."""
        b0, *bs = bs
        return b0.union(*bs)

    def node_issubset(x, y):
        return x.issubset(y)
