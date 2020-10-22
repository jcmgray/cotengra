import collections


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
