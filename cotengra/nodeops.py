from functools import reduce
from operator import or_


class NodeOpsFrozenset:
    """Namespace for interacting with frozenset[int] as contraction tree nodes."""

    __slots__ = ()

    def copy(self):
        return self

    node_type = frozenset
    """The type used for nodes, to check isinstance with."""

    node_size = len
    """Function to get the size of a node."""

    node_from_seq = frozenset
    """From a sequence of integers, make a node (``frozenset``)."""

    @staticmethod
    def node_from_single(x):
        """From single integer ``x``, make a node (``frozenset``)."""
        return frozenset((x,))

    @staticmethod
    def node_supremum(size):
        """Return the node containing all elements up to ``size``."""
        return frozenset(range(size))

    @staticmethod
    def node_get_single_el(node):
        """Assuming ``node`` has one element, return it."""
        return next(iter(node))

    @staticmethod
    def node_tie_breaker(x):
        """A tie breaker function for ordering nodes when they are the same
        size. Returns the negative of the smallest element in the node.
        Nodes with larger `node_tie_breaker` values get contracted first.
        """
        return -min(x)

    @staticmethod
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

    @staticmethod
    def node_union(x, y):
        """Return the node given by the union of two nodes."""
        return x | y

    @staticmethod
    def node_union_it(bs):
        """Return the node given by the union of an iterable of nodes."""
        b0, *bs = bs
        return b0.union(*bs)

    @staticmethod
    def node_issubset(x, y):
        """Check if node ``x`` is a subset of node ``y``."""
        return x.issubset(y)


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


class NodeOpsBitSetInt:
    """Namespace for interacting with BitSetInt as contraction tree nodes."""

    __slots__ = ()

    def copy(self):
        return self

    node_type = BitSetInt
    node_size = BitSetInt.__len__
    node_from_seq = BitSetInt

    @staticmethod
    def node_from_single(x):
        return BitSetInt(1 << x)

    @staticmethod
    def node_supremum(size):
        return BitSetInt.supremum(size)

    @staticmethod
    def node_get_single_el(node):
        return node.bit_length() - 1

    @staticmethod
    def node_tie_breaker(x):
        return -x

    @staticmethod
    def is_valid_node(node):
        try:
            return isinstance(node, BitSetInt)
        except TypeError:
            return False

    @staticmethod
    def node_union(x, y):
        return BitSetInt(int.__or__(x, y))

    @staticmethod
    def node_union_it(bs):
        b0, *bs = bs
        return b0.union(*bs)

    @staticmethod
    def node_issubset(x, y):
        return (x & y) == x


nodeops_frozenset = NodeOpsFrozenset()
nodeops_bitsetint = NodeOpsBitSetInt()


def get_nodeops(node_type_str: str):
    """Get the node operations namespace for the given node type.

    Parameters
    ----------
    node_type_str : str
        The node type as a string. One of "frozenset[int]", "BitSetInt".

    Returns
    -------
    NodeOpsFrozenset or NodeOpsBitSetInt
        The corresponding node operations namespace.
    """
    if node_type_str == "frozenset[int]":
        return nodeops_frozenset
    elif node_type_str == "BitSetInt":
        return nodeops_bitsetint
    else:
        raise ValueError(f"Unknown node type: {node_type_str}")
