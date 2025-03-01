import collections
import hashlib
import pickle
import threading

from .oe import PathOptimizer
from .utils import DiskDict


def sortedtuple(x):
    return tuple(sorted(x))


def make_hashable(x):
    """Make ``x`` hashable by recursively turning list into tuples and dicts
    into sorted tuples of key-value pairs.
    """
    if isinstance(x, list):
        return tuple(map(make_hashable, x))
    if isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    return x


def hash_contraction_a(inputs, output, size_dict):
    if not isinstance(next(iter(size_dict.values()), 1), int):
        # hashing e.g. numpy int won't match!
        size_dict = {k: int(v) for k, v in size_dict.items()}

    return hashlib.sha1(
        pickle.dumps(
            (
                tuple(map(sortedtuple, inputs)),
                sortedtuple(output),
                sortedtuple(size_dict.items()),
            )
        )
    ).hexdigest()


def hash_contraction_b(inputs, output, size_dict):
    # label each index as the sorted tuple of nodes it is incident to
    edges = collections.defaultdict(list)
    for ix in output:
        edges[ix].append(-1)
    for i, term in enumerate(inputs):
        for ix in term:
            edges[ix].append(i)

    # then sort edges by each's incidence nodes
    canonical_edges = sortedtuple(map(sortedtuple, edges.values()))

    return hashlib.sha1(
        pickle.dumps((canonical_edges, sortedtuple(size_dict.items())))
    ).hexdigest()


def hash_contraction(inputs, output, size_dict, method="a"):
    """Compute a hash for a particular contraction geometry."""
    if method == "a":
        return hash_contraction_a(inputs, output, size_dict)
    elif method == "b":
        return hash_contraction_b(inputs, output, size_dict)
    else:
        raise ValueError("Unknown hash method: {}".format(method))


class ReusableOptimizer(PathOptimizer):
    """Mixin class for optimizers that can be reused, caching the paths
    and other relevant information for reconstructing the full tree.

    The following methods should be implemented in the subclass:

        _get_path_relevant_opts(self)
        _get_suboptimizer(self)
        _deconstruct_tree(self, opt, tree)
        _reconstruct_tree(self, inputs, output, size_dict, con)

    Parameters
    ----------
    directory : None, True, or str, optional
        If specified use this directory as a persistent cache. If ``True`` auto
        generate a directory in the current working directory based on the
        options which are most likely to affect the path (see
        `ReusableHyperOptimizer._get_path_relevant_opts`).
    overwrite : bool or 'improved', optional
        If ``True``, the optimizer will always run, overwriting old results in
        the cache. This can be used to update paths without deleting the whole
        cache. If ``'improved'`` then only overwrite if the new path is better.
    hash_method : {'a', 'b', ...}, optional
        The method used to hash the contraction tree. The default, ``'a'``, is
        faster hashwise but doesn't recognize when indices are permuted.
    cache_only : bool, optional
        If ``True``, the optimizer will only use the cache, and will raise
        ``KeyError`` if a contraction is not found.
    directory_split : "auto" or bool, optional
        If specified, the hash will be split into two parts, the first part
        will be used as a subdirectory, and the second part will be used as the
        filename. This is useful for avoiding a very large flat diretory. If
        "auto" it will check the current cache if any and guess from that.
    opt_kwargs
        Supplied to ``self._get_suboptimizer(self)``.
    """

    def __init__(
        self,
        *,
        directory=None,
        overwrite=False,
        hash_method="a",
        cache_only=False,
        directory_split="auto",
        **opt_kwargs,
    ):
        self._suboptimizers = {}
        self._suboptimizer_kwargs = opt_kwargs
        if directory is True:
            # automatically generate the directory
            directory = f"ctg_cache/opts{self.auto_hash_path_relevant_opts()}"
        self._cache = DiskDict(directory)
        self.overwrite = overwrite
        self._hash_method = hash_method
        self.cache_only = cache_only

        if directory_split == "auto":
            # peak at cache and see if it has a subdirectory structure
            path = self._cache._path
            if (path is not None) and path.exists():
                anysub = next(path.glob("*"), None)
                if anysub is not None:
                    directory_split = anysub.is_dir()
                else:
                    # default to True
                    directory_split = True
            else:
                # default to True
                directory_split = True

        self.directory_split = directory_split

    @property
    def last_opt(self):
        return self._suboptimizers.get(threading.get_ident(), None)

    def _get_path_relevant_opts(self):
        """We only want to hash on options that affect the contraction, not
        things like `progbar`.
        """
        raise NotImplementedError

    def auto_hash_path_relevant_opts(self):
        """Automatically hash the path relevant options used to create the
        optimizer.
        """
        key = tuple(
            (key, make_hashable(self._suboptimizer_kwargs.get(key, default)))
            for key, default in self._get_path_relevant_opts()
        )
        return hashlib.sha1(pickle.dumps(key)).hexdigest()

    def hash_query(self, inputs, output, size_dict):
        """Hash the contraction specification, returning this and whether the
        contraction is already present as a tuple.
        """
        h = hash_contraction(inputs, output, size_dict, self._hash_method)

        if self.directory_split:
            # use first part of the hash (256 options) as sub directory
            h = (h[:2], h[2:])

        missing = h not in self._cache
        return h, missing

    @property
    def minimize(self):
        if self.last_opt is not None:
            return self.last_opt.minimize
        else:
            return self._suboptimizer_kwargs.get("minimize", "flops")

    def update_from_tree(self, tree, overwrite="improved"):
        """Explicitly add the contraction that ``tree`` represents into the
        cache. For example, if you have manually improved it via reconfing.
        If ``overwrite=False`` and the contracton is present already then do
        nothing. If ``overwrite='improved'`` then only overwrite if the new
        path is better. If ``overwrite=True`` then always overwrite.

        Parameters
        ----------
        tree : ContractionTree
            The tree to add to the cache.
        overwrite : bool or "improved", optional
            If ``True`` always overwrite, if ``False`` only overwrite if the
            contraction is missing, if ``'improved'`` only overwrite if the new
            path is better (the default). Note that the comparison of scores
            is based on default objective of the tree.
        """
        h, missing = self.hash_query(tree.inputs, tree.output, tree.size_dict)

        new_con = {
            "path": tree.get_path(),
            "score": tree.get_score(),
            "sliced_inds": tuple(tree.sliced_inds),
        }

        if missing:
            # write to the cache
            self._cache[h] = new_con
        elif overwrite:
            if overwrite == "improved":
                old_con = self._cache[h]
                if new_con["score"] < old_con["score"]:
                    # overwrite only if we have a better score
                    self._cache[h] = new_con
            else:
                # overwrite regardless of score
                self._cache[h] = new_con

    def _run_optimizer(self, inputs, output, size_dict):
        opt = self.suboptimizer(**self._suboptimizer_kwargs)
        tree = opt.search(inputs, output, size_dict)
        thrid = threading.get_ident()
        self._suboptimizers[thrid] = opt
        return {
            "path": tree.get_path(),
            "score": tree.get_score(),
            # dont' need to store all slice info, just which indices
            "sliced_inds": tuple(tree.sliced_inds),
        }

    def _maybe_run_optimizer(self, inputs, output, size_dict):
        h, missing = self.hash_query(inputs, output, size_dict)

        should_run = missing or self.overwrite
        if should_run:
            if self.cache_only:
                raise KeyError("Contraction missing from cache.")

            con = self._run_optimizer(inputs, output, size_dict)

            if (self.overwrite == "improved") and (not missing):
                # only overwrite if the new path is better
                old_con = self._cache[h]
                if con["score"] < old_con["score"]:
                    # replace the old path
                    self._cache[h] = con
                else:
                    # use the old path
                    con = old_con
                    # need flag that we can't use the last run
                    should_run = False
            else:
                # write to the cache
                self._cache[h] = con
        else:
            # just retrieve from the cache
            con = self._cache[h]

        return should_run, con

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        _, con = self._maybe_run_optimizer(inputs, output, size_dict)
        return con["path"]

    def _get_suboptimizer(self):
        raise NotImplementedError

    def _deconstruct_tree(self, opt, tree):
        raise NotImplementedError

    def _run_optimizer(self, inputs, output, size_dict):
        opt = self._get_suboptimizer()
        tree = opt.search(inputs, output, size_dict)
        thrid = threading.get_ident()
        self._suboptimizers[thrid] = opt
        return self._deconstruct_tree(opt, tree)

    def _reconstruct_tree(self, inputs, output, size_dict, con):
        raise NotImplementedError

    def search(self, inputs, output, size_dict):
        searched, con = self._maybe_run_optimizer(inputs, output, size_dict)
        if searched:
            # already have the tree to return
            return self.last_opt.tree

        # else need to *reconstruct* the tree from the more compact path
        return self._reconstruct_tree(inputs, output, size_dict, con)

    def cleanup(self):
        self._cache.cleanup()
