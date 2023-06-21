import math
import heapq
import itertools
import collections
from time import sleep

import tqdm

from ..oe import ssa_to_linear
from ..scoring import get_score_fn
from ..core import (
    get_hypergraph,
    ContractionTreeCompressed,
)
from ..plot import use_neutral_style, show_and_close
from ..utils import GumbelBatchedGenerator


class CompressedExhaustive:
    """Exhaustively search all possible contraction orders for a given
    compressed bond dimension, using several forms of pruning and local
    hueristics to accelerate the search.

    Parameters
    ----------
    chi : int
        The max bond dimension of the compressed hypergraph.
    max_nodes : int, optional
        Set the maximum number of contraction steps to consider.
    max_time : float, optional
        Set the maximum time to spend on the search.
    local_score : callable, optional
        A function that assigns a score to a potential contraction, with a
        lower score giving more priority to explore that contraction earlier.
        It should have signature::

            local_score(step : int, tracker: CompressedStatsTracker) -> float

        where ``step`` is the number of steps so far, ``tracker`` is a
        `CompressedStatsTracker` object that tracks various properties
        like flops and peak size.
    exploration_power : float, optional
        If not ``0.0``, the inverse power to which the step is raised in the
        default local score function. Higher values favor exploring more
        promising branches early on - at the cost of increased memory. Ignored
        if ``local_score`` is supplied.
    best_score : float, optional
        Manually specify an upper bound for best score found so far.
    progbar : bool, optional
        If ``True``, display a progress bar.
    """

    def __init__(
        self,
        chi,
        minimize="peak",
        compress_late=True,
        max_nodes=float("inf"),
        max_time=None,
        local_score=None,
        exploration_power=0.0,
        best_score=None,
        progbar=False,
    ):
        self.chi = chi

        if isinstance(minimize, str) and "compressed" not in minimize:
            minimize = minimize + "-compressed"

        self.minimize = get_score_fn(minimize)
        self.compress_late = compress_late
        if best_score is None:
            self.best_score = float("inf")
        else:
            self.best_score = abs(best_score)
        self.best_ssa_path = None
        self.counter = self.queue = self.cands = self.seen = None
        self.max_nodes = max_nodes
        self.max_time = max_time
        self.exploration_power = exploration_power
        self.allow = None

        if local_score is None:
            if exploration_power <= 0:

                def local_score(step, tracker):
                    """Default ordering is depth first greedy search based on
                    memory removed.
                    """
                    return -step, tracker.size_change

            else:

                def local_score(step, tracker):
                    """Use the actual score to order search, but modified by
                    how 'complete' the contraction is to favor finishing.
                    """
                    return tracker.score / (step + 1) ** (
                        1 / self.exploration_power
                    )

        self.local_score = local_score
        self.progbar = progbar

    def setup(self, inputs, output, size_dict):
        """Set-up the optimizer with a specific contraction."""
        if self.counter is None:
            hg = get_hypergraph(
                inputs,
                output,
                size_dict,
                accel=False,
            )
            # maps node integer to subgraph
            tree_map = {i: frozenset([i]) for i in hg.nodes}
            # keeps track of scores on the fly
            tracker0 = self.minimize.get_compressed_stats_tracker(hg)
            ssa_path0 = ()

            # the actual queue is a heap so need to reference candidates by int
            self.counter = itertools.count()
            c = next(self.counter)

            # our initial search space is the full graph
            self.root = (hg, tree_map, ssa_path0, tracker0)
            self.cands = {c: self.root}
            self.queue = [(self.local_score(0, tracker0), c)]

            self.seen = {}
            self.priority_queue = []

    def expand_node(
        self, i, j, hg, tree_map, ssa_path, tracker, high_priority=False
    ):
        """Given a current contraction node, expand it by contracting nodes
        ``i`` and ``j``.

        Parameters
        ----------
        i, j : int
            The nodes to contract.
        hg : Hypergraph
            The hypergraph to contract.
        ssa_path : list
            The contraction path so far.
        tracker : CompressedStatsTracker
            Scoring object that tracks costs of current compressed contraction.
        high_priority : bool, optional
            If True, the contraction will be assessed before any other normal
            contractions.

        Returns
        -------
        int or None
            The contraction index, or None if the contraction is guaranteed to
            be worse than another contraction.
        """
        ti = tree_map[i]
        tj = tree_map[j]
        tij = ti | tj
        if (self.allow is not None) and (tij not in self.allow):
            # search is restricted to a subset of pairs excluding tij
            return

        # fork this node
        hg = hg.copy()
        tracker = tracker.copy()

        # simulate a contraction step while tracking costs
        tracker.update_pre_step()

        if self.compress_late:
            tracker.update_pre_compress(hg, i, j)
            # compress late - just before contraction
            hg.compress(self.chi, hg.get_node(i) + hg.get_node(j))
            tracker.update_post_compress(hg, i, j)

        tracker.update_pre_contract(hg, i, j)
        ij = hg.contract(i, j)
        tracker.update_post_contract(hg, ij)

        if not self.compress_late:
            tracker.update_pre_compress(hg, ij)
            # compress early - immediately after contraction
            hg.compress(self.chi, hg.get_node(ij))
            tracker.update_post_compress(hg, ij)

        tracker.update_post_step()

        if tracker.score >= self.best_score:
            # already worse than best score seen -> drop
            return

        tree_map_next = tree_map.copy()
        del tree_map_next[i]
        del tree_map_next[j]
        tree_map_next[ij] = tij

        # uniquely identify partially contracted graph
        graph_key = hash(frozenset(tree_map_next.values()))
        if tracker.score >= self.seen.get(graph_key, float("inf")):
            # already reached this exact point with an equal or better score
            return
        # record new or better score
        self.seen[graph_key] = tracker.score

        # construct the next candidate and add to queue
        new_ssa_path = ssa_path + ((j, i) if j < i else (i, j),)
        c = next(self.counter)
        self.cands[c] = (
            hg,
            tree_map_next,
            new_ssa_path,
            tracker,
        )

        if not high_priority:
            # this is used to determine priority within the queue
            step = len(new_ssa_path)
            priority = self.local_score(step, tracker)
            heapq.heappush(self.queue, (priority, c))
        else:
            self.priority_queue.append(c)

        return c

    def _update_progbar(self, pbar, c):
        if self.progbar:
            pbar.update()
            pbar.set_description(
                f"[{c}] "
                f"cands:{len(self.cands)} "
                f"best:{self.best_score:.2f}",
                refresh=False
            )

    def run(self, inputs, output, size_dict):
        self.setup(inputs, output, size_dict)

        if self.progbar:
            pbar = tqdm.tqdm()
        else:
            pbar = None

        if self.max_time is not None:
            import time

            time0 = time.time()

            def should_stop(c):
                return (time.time() - time0 >= self.max_time) or (
                    self.best_ssa_path and c and (c > self.max_nodes)
                )

        else:

            def should_stop(c):
                return self.best_ssa_path and c and (c > self.max_nodes)

        def edge_sort(edge_nodes):
            edge, nodes = edge_nodes
            return edge
            # return sum(hg.node_size(n) for n in nodes), edge

        try:
            while self.cands:
                if self.priority_queue:
                    c = self.priority_queue.pop()
                else:
                    # get candidate with the best rank
                    _, c = heapq.heappop(self.queue)

                hg, tree_map, ssa_path, tracker = self.cands.pop(c)

                # check if full contraction
                if hg.get_num_nodes() == 1:
                    # ignore unless beats best so far
                    if tracker.score < self.best_score:
                        self.best_score = tracker.score
                        self.best_ssa_path = ssa_path
                        self._update_progbar(pbar, c)
                    continue

                # check next candidate contractions
                for _, nodes in sorted(hg.edges.items(), key=edge_sort):
                    if len(nodes) != 2:
                        continue
                    c = self.expand_node(
                        *nodes, hg, tree_map, ssa_path, tracker
                    )

                if should_stop(c):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            if self.progbar:
                pbar.close()

    @property
    def ssa_path(self):
        return self.best_ssa_path

    @property
    def path(self):
        return ssa_to_linear(self.ssa_path)

    def explore_path(self, path, high_priority=True, restrict=False):
        """Explicitly supply a path to be added to the search space, by default
        it is added to the priority queue and will be processed first.

        Parameters
        ----------
        path : sequence[tuple[int]]
            A contraction path to explore.
        high_priority : bool, optional
            If ``True``, the path will be assessed before anything else,
            regardless of cost - the default.
        restrict : bool, optional
            If ``True``, only allow contractions in this path, so only the
            order will be optimized.
        """
        # convert to ssa_path
        hg, tree_map, ssa_path, tracker = self.root

        if restrict and self.allow is None:
            self.allow = set()

        ssas = list(range(hg.get_num_nodes()))
        ssa = ssas[-1]
        for pi, pj in path:
            i, j = map(ssas.pop, sorted((pi, pj), reverse=True))
            ssa += 1
            ij = ssa
            if restrict:
                self.allow.add(tree_map[i] | tree_map[j])
            ssas.append(ij)
            c = self.expand_node(
                i,
                j,
                hg,
                tree_map,
                ssa_path,
                tracker,
                high_priority=high_priority,
            )
            if c is None:
                return

            # descend to the next contraction in the path
            hg, tree_map, ssa_path, tracker = self.cands[c]

    def search(self, inputs, output, size_dict):
        """Run and return the best ``ContractionTreeCompressed``."""
        self.run(inputs, output, size_dict)
        return ContractionTreeCompressed.from_path(
            inputs,
            output,
            size_dict,
            ssa_path=self.ssa_path,
        )

    def __call__(self, inputs, output, size_dict):
        """Run and return the best ``path``."""
        self.run(inputs, output, size_dict)
        return self.path


def do_reconfigure(tree, time, chi):
    tree.compressed_reconfigure_(
        chi, progbar=False, max_time=time, order_only=True
    )
    tree.compressed_reconfigure_(
        chi, progbar=False, max_time=time, order_only=False
    )
    new = math.log2(tree.peak_size_compressed(chi))
    return tree, new


class CompressedTreeRefiner:
    def __init__(
        self,
        trees,
        copt,
        chi,
        max_refine_time=8,
        executor=None,
        pre_dispatch=8,
        progbar=False,
        plot=False,
    ):
        self.copt = copt
        self.chi = chi
        self.scores = []
        self.trees = trees
        self.times = collections.defaultdict(lambda: 2)
        self.max_refine_time = max_refine_time
        self.finished_scores = []
        self.futures = []
        self.executor = executor
        self.pre_dispatch = pre_dispatch
        self.plot = plot
        self.progbar = progbar
        for key, tree in trees.items():
            self._check_score(key, tree)

    def _check_score(self, key, tree, score=None):
        if self.times[key] <= self.max_refine_time:
            if score is None:
                score = math.log2(tree.peak_size_compressed(self.chi))
            heapq.heappush(self.scores, (-score, key))
            self.trees[key] = tree
        else:
            self.finished_scores.append(score)

    def _get_next_tree(self):
        score, key = heapq.heappop(self.scores)
        tree = self.trees[key]
        time = self.times[key]
        return tree, key, time, abs(score)

    def _get_next_result_seq(self):
        tree, key, time, old = self._get_next_tree()
        tree, new = do_reconfigure(tree, time, self.chi)
        return tree, key, time, old, new

    def _get_next_result_par(self, max_futures):
        while self.scores and (
            len(self.futures) < min(self.pre_dispatch, max_futures)
        ):
            tree, key, time, old = self._get_next_tree()
            f = self.executor.submit(do_reconfigure, tree, time, self.chi)
            self.futures.append((f, key, time, old))

        while self.futures:
            for i in range(len(self.futures)):
                f, key, time, old = self.futures[i]
                if f.done():
                    del self.futures[i]
                    tree, new = f.result()
                    return tree, key, time, old, new
            sleep(1e-3)

    def _process_result(self, tree, key, time, old, new):
        if old == new:
            self.times[key] *= 2
        else:
            self.copt.update_from_tree(tree)
            self.times[key] = max(2, self.times[key] // 2)
        self._check_score(key, tree, new)

    def refine(self, num_its=None, bins=30):
        if num_its is None:
            num_its = len(self.trees)

        old_scores = [-x[0] for x in self.scores]

        if self.progbar:
            import tqdm

            its = tqdm.trange(num_its, desc="Refining...")
        else:
            its = range(num_its)

        for i in its:
            if not (self.scores or self.futures):
                # everything finished
                break

            if self.executor is None:
                tree, key, time, old, new = self._get_next_result_seq()
            else:
                tree, key, time, old, new = self._get_next_result_par(
                    num_its - i
                )

            self._process_result(tree, key, time, old, new)
            if self.progbar:
                its.set_description(f"worst: {self.scores[0]}", refresh=False)

        if self.plot:
            import matplotlib.pyplot as plt

            new_scores = [-x[0] for x in self.scores]
            _, bins, _ = plt.hist(old_scores, bins=bins, alpha=0.8)
            plt.hist(new_scores, bins=bins, color="orange", alpha=0.8)
            plt.hist(self.finished_scores, bins=bins, color="red", alpha=0.8)


class MiniTree:
    """A very minimal tree data structure for tracking possible contractions
    within a window only.
    """

    __slots__ = ('children', 'parents', 'leaves', 'candidates')

    def __init__(self):
        self.children = {}
        self.parents = {}
        # use these as ordered sets
        self.leaves = {}
        self.candidates = {}

    def copy(self):
        m = object.__new__(MiniTree)
        m.children = self.children.copy()
        m.parents = self.parents.copy()
        m.leaves = self.leaves.copy()
        m.candidates = self.candidates.copy()
        return m

    def add(self, p, l, r):
        self.children[p] = (l, r)
        self.parents[l] = p
        self.parents[r] = p
        lleaf = l not in self.children
        rleaf = r not in self.children
        if lleaf:
            # add to leaf set
            self.leaves[l] = None
        if rleaf:
            # add to leaf set
            self.leaves[r] = None
        if lleaf and rleaf:
            # both can be contracted
            self.candidates[p] = None

    def contract(self, p):
        self.candidates.pop(p)
        l, r = self.children.pop(p)
        self.parents.pop(l)
        self.parents.pop(r)
        self.leaves[p] = None
        try:
            pp = self.parents[p]
        except KeyError:
            # one of possibly many root nodes
            return l, r
        psibling = next(i for i in self.children[pp] if i != p)
        if psibling in self.leaves:
            self.candidates[pp] = None
        return l, r

    def __repr__(self):
        return (
            f"MiniTree("
            f"children={len(self.children)}, "
            f"parents={len(self.parents)}, "
            f"leaves={len(self.leaves)}, "
            f"candidates={len(self.candidates)}"
            ")"
        )


class EmptyMiniTree:

    __slots__ = ("candidates",)

    def __init__(self, hgi, hgf):
        roots = {}

        # find nodes in end but not start hypergraph
        for p in hgf.nodes:
            if p not in hgi.nodes:
                roots[p] = []

        # find nodes in start but not end hypergraph
        # and group them according to root
        for l in hgi.nodes:
            if l not in hgf.nodes:
                for p in roots:
                    # is subset of p
                    if l & p == l:
                        roots[p].append(l)
                        # don't need to check other roots
                        break

        # collect possible candidates
        self.candidates = {}
        # self.siblings = {}
        for p, sg in roots.items():
            # within each subgraph add potential contractions
            if len(sg) == 2:
                # only one ordering possible
                l, r = sg
                self.candidates[p] = (l, r)
            else:
                # add all connected combinations
                for l, r in itertools.combinations(sg, 2):
                    el = hgi.get_node(l)
                    er = hgi.get_node(r)
                    if not set(el).isdisjoint(er):
                        # they share an edge
                        p = l | r
                        self.candidates[p] = (l, r)

    def copy(self):
        new = object.__new__(self.__class__)
        new.candidates = self.candidates.copy()
        return new

    def contract(self, p):
        l, r  = self.candidates.pop(p)

        # check other contractions to see if they contained l or r
        for po, (lo, ro) in tuple(self.candidates.items()):
            if lo in (l, r):
                # replace the old left with p
                del self.candidates[po]
                self.candidates[po | p] = (p, ro)
            elif ro in (l, r):
                # replace the old right with p
                del self.candidates[po]
                self.candidates[po | p] = (lo, p)

        return l, r


class Node:
    """A possible intermediate contraction state.
    """

    __slots__ = ("hg", "plr", "chi", "tracker", "compress_late")

    def __init__(self, hg, plr, chi, tracker, compress_late=False):
        self.hg = hg
        self.plr = plr
        self.chi = chi
        self.tracker = tracker
        self.compress_late = compress_late

    @classmethod
    def first(cls, inputs, output, size_dict, minimize):
        hg = get_hypergraph(
            # use bit encoding
            inputs={
                1 << i: term
                for i, term in enumerate(inputs)
            },
            output=output,
            size_dict=size_dict,
            # can't use bit encoding in rust
            accel=False,
        )
        plr = None

        minimize = get_score_fn(minimize)

        if minimize.chi == "auto":
            chi = max(size_dict.values())**2
        else:
            chi = minimize.chi

        return cls(
            hg=hg,
            plr=plr,
            chi=chi,
            tracker=minimize.get_compressed_stats_tracker(hg),
            compress_late=minimize.compress_late,
        )

    def next(self, p, l, r):
        hg = self.hg.copy()
        tracker = self.tracker.copy()

        # simulate a contraction step while tracking costs
        tracker.update_pre_step()

        if self.compress_late:
            tracker.update_pre_compress(hg, l, r)
            # compress late - just before contraction
            hg.compress(self.chi, hg.get_node(l))
            hg.compress(self.chi, hg.get_node(r))
            tracker.update_post_compress(hg, l, r)

        tracker.update_pre_contract(hg, l, r)
        hg.contract(l, r, node=p)
        tracker.update_post_contract(hg, p)

        if not self.compress_late:
            tracker.update_pre_compress(hg, p)
            # compress early - immediately after contraction
            hg.compress(self.chi, hg.get_node(p))
            tracker.update_post_compress(hg, p)

        tracker.update_post_step()

        return self.__class__(
            hg=hg,
            plr=(p, l, r),
            chi=self.chi,
            tracker=tracker,
            compress_late=self.compress_late,
        )

    def graph_key(self):
        return frozenset(self.hg.nodes)

    def __repr__(self):
        return (
            f"Node("
            f"num_nodes={self.hg.num_nodes}, "
            f"tracker={self.tracker}, "
            ")"
        )


def ssa_path_to_bit_path(path):
    N = len(path) + 1
    bitpath = []
    ssa_to_bit = {i: 1 << i for i in range(N)}
    for c, (si, sj) in enumerate(path):
        ni = ssa_to_bit[si]
        nj = ssa_to_bit[sj]
        nij = ni | nj
        ssa_to_bit[len(ssa_to_bit)] = nij
        bitpath.append((nij, ni, nj))

    return tuple(bitpath)


def bit_path_to_ssa_path(bitpath):
    N = len(bitpath) + 1
    bit_to_ssa = {1 << i: i for i in range(N)}
    path = []
    for nij, ni, nj in bitpath:
        path.append((bit_to_ssa[ni], bit_to_ssa[nj]))
        bit_to_ssa[nij] = len(bit_to_ssa)
    return tuple(path)



class WindowedOptimizer:

    def __init__(
        self,
        inputs,
        output,
        size_dict,
        minimize,
        ssa_path,
        seed=None,
    ):
        import numpy as np
        bitpath = ssa_path_to_bit_path(ssa_path)
        self.nodes = {
            0: Node.first(inputs, output, size_dict, minimize)
        }
        for c, (nij, ni, nj) in enumerate(bitpath):
            self.nodes[c + 1] = self.nodes[c].next(nij, ni, nj)
        self.rng = np.random.default_rng(seed)
        self.gumbel = GumbelBatchedGenerator(self.rng)

    @property
    def tracker(self):
        return self.nodes[len(self.nodes) - 1].tracker

    @show_and_close
    @use_neutral_style
    def plot_size_footprint(self, figsize=(8, 3)):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        cs = range(len(self.nodes))
        xs0 = [self.nodes[c].tracker.total_size_post_contract for c in cs]
        xs1 = [self.nodes[c].tracker.total_size for c in cs]
        xs2 = [self.nodes[c].tracker.contracted_size for c in cs]
        ax.plot(cs, xs0, label="total size contracted", zorder=4)
        ax.plot(cs, xs1, label="total size compressed", zorder=3)
        ax.plot(cs, xs2, label="single size", zorder=2)
        ax.legend()
        ax.set_yscale('log')
        return fig, ax

    def optimize_window(
        self,
        ci,
        cf,
        order_only=False,
        max_window_tries=1000,
        score_temperature=0.0,
        queue_temperature=1.0,
        scorer=None,
        queue_scorer=None,
    ):
        if scorer is None:

            def scorer(nodes, T=0.0):
                """This is the score we want to minimize overall for the
                window.
                """
                return (
                    nodes[-1].tracker.score - T * self.gumbel(),
                    # secondarily try and minimize combo cost
                    nodes[-1].tracker.combo_score,
                )

        if queue_scorer is None:

            def queue_scorer(nodes, T):
                """This is the score we use to control queue exploration, which
                is called on potentially only a partial window.
                """
                return (
                    # prioritize more complete pats
                    -len(nodes),
                    nodes[-1].tracker.score - T * self.gumbel()
                )

        if order_only:
            subtree = MiniTree()
            for c in range(ci + 1, cf):
                node = self.nodes[c]
                p, l, r = node.plr
                subtree.add(p, l, r)
        else:
            subtree = EmptyMiniTree(self.nodes[ci].hg, self.nodes[cf - 1].hg)

        counter = itertools.count()
        best_score = scorer([self.nodes[c] for c in range(ci, cf)])
        q = next(counter)
        queue = [(0, q)]
        cands = {q: (subtree, (self.nodes[ci],))}
        tries = 0

        while queue and tries < max_window_tries:

            # get the next candidate contraction path
            _, q = heapq.heappop(queue)
            this_subtree, this_subnodes = cands.pop(q)

            # check the next possible contractions
            for p in this_subtree.candidates:

                next_subtree = this_subtree.copy()
                l, r = next_subtree.contract(p)
                next_subnodes = (
                    this_subnodes + (this_subnodes[-1].next(p, l, r),)
                )
                score = scorer(next_subnodes, score_temperature)

                if score >= best_score:
                    # know no improvement, count as try and terminate
                    tries += 1
                elif next_subtree.candidates:
                    # still more to go -> add to queue
                    q = next(counter)
                    qs = queue_scorer(next_subnodes, queue_temperature)
                    heapq.heappush(queue, (qs, q))
                    cands[q] = (next_subtree, next_subnodes)
                else:
                    # finished the local contraction with good score, check it
                    for c, node in enumerate(next_subnodes[1:], ci + 1):
                        # replace the nodes in the window
                        self.nodes[c] = node
                    best_score = score
                    tries += 1

        # update the later trackers for global score changes
        for c in range(cf, len(self.nodes)):
            self.nodes[c].tracker.update_score(self.nodes[c - 1].tracker)

    def refine(
        self,
        window_size=20,
        max_iterations=100,
        order_only=False,
        max_window_tries=1000,
        score_temperature=0.01,
        queue_temperature=1.0,
        scorer=None,
        queue_scorer=None,
        progbar=False,
        **kwargs,
    ):
        import numpy as np

        wl = window_size // 2
        wr = window_size - wl

        its = range(max_iterations)
        if progbar:
            from tqdm import tqdm
            its = tqdm(its)

        cs = np.array(list(self.nodes.keys()))
        for _ in its:
            p = np.array(
                [n.tracker.total_size for n in self.nodes.values()],
                dtype='float'
            )
            p /= p.sum()
            wc = self.rng.choice(cs, p=p)

            # window can't extend beyond edges
            wc = min(max(wl, wc), len(self.nodes) - wr)
            self.optimize_window(
                wc - wl, wc + wr,
                order_only=order_only,
                max_window_tries=max_window_tries,
                score_temperature=score_temperature,
                queue_temperature=queue_temperature,
                scorer=scorer,
                queue_scorer=queue_scorer,
                **kwargs
            )

            if progbar:
                its.set_description(f"{self.tracker}", refresh=False)

    def get_ssa_path(self):
        bitpath = [
            self.nodes[c].plr
            for c in range(1, len(self.nodes))
        ]
        return bit_path_to_ssa_path(bitpath)
