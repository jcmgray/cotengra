"""Compressed contraction tree finding routines."""

import heapq
import itertools

from ..core import get_hypergraph
from ..plot import show_and_close, use_neutral_style
from ..scoring import get_score_fn
from ..utils import GumbelBatchedGenerator, get_rng


class MiniTree:
    """A very minimal tree data structure for tracking possible contractions
    within a window only.
    """

    __slots__ = ("children", "parents", "leaves", "candidates")

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
        l, r = self.candidates.pop(p)

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
    """A possible intermediate contraction state."""

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
            inputs={1 << i: term for i, term in enumerate(inputs)},
            output=output,
            size_dict=size_dict,
            # can't use bit encoding in rust
            accel=False,
        )
        plr = None

        minimize = get_score_fn(minimize)

        if minimize.chi == "auto":
            chi = max(size_dict.values()) ** 2
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
    for si, sj in path:
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
    """ """

    def __init__(
        self,
        inputs,
        output,
        size_dict,
        minimize,
        ssa_path,
        seed=None,
    ):
        bitpath = ssa_path_to_bit_path(ssa_path)
        self.nodes = {0: Node.first(inputs, output, size_dict, minimize)}
        for c, (nij, ni, nj) in enumerate(bitpath):
            self.nodes[c + 1] = self.nodes[c].next(nij, ni, nj)
        self.rng = get_rng(seed)
        self.gumbel = GumbelBatchedGenerator(self.rng)

    @property
    def tracker(self):
        return self.nodes[len(self.nodes) - 1].tracker

    @show_and_close
    @use_neutral_style
    def plot_size_footprint(self, figsize=(8, 3)):
        import matplotlib.pyplot as plt
        import math

        fig, ax = plt.subplots(figsize=figsize)
        cs = range(len(self.nodes))
        xs0 = [
            math.log2(max(1, self.nodes[c].tracker.total_size_post_contract))
            for c in cs
        ]
        xs1 = [math.log2(max(1, self.nodes[c].tracker.total_size)) for c in cs]
        xs2 = [
            math.log2(max(1, self.nodes[c].tracker.contracted_size))
            for c in cs
        ]
        ax.plot(cs, xs0, label="total size contracted", zorder=4)
        ax.plot(cs, xs1, label="total size compressed", zorder=3)
        ax.plot(cs, xs2, label="single size", zorder=2)
        ax.legend()
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
                    nodes[-1].tracker.score - T * self.gumbel(),
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
                next_subnodes = this_subnodes + (
                    this_subnodes[-1].next(p, l, r),
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
        wl = window_size // 2
        wr = window_size - wl

        its = range(max_iterations)
        if progbar:
            import tqdm

            its = tqdm.tqdm(its)

        cs = list(self.nodes.keys())
        for _ in its:
            p = [n.tracker.total_size for n in self.nodes.values()]
            wc = self.rng.choices(cs, weights=p)[0]

            # window can't extend beyond edges
            wc = min(max(wl, wc), len(self.nodes) - wr)
            self.optimize_window(
                wc - wl,
                wc + wr,
                order_only=order_only,
                max_window_tries=max_window_tries,
                score_temperature=score_temperature,
                queue_temperature=queue_temperature,
                scorer=scorer,
                queue_scorer=queue_scorer,
                **kwargs,
            )

            if progbar:
                its.set_description(f"{self.tracker}", refresh=False)

    def simulated_anneal(
        self,
        tfinal=0.0001,
        tstart=0.01,
        tsteps=50,
        numiter=50,
        select="descend",
        target_size=None,
        slice_mode=None,
        progbar=False,
    ):
        import math
        from .path_simulated_annealing import linspace_generator

        if progbar:
            import tqdm

            pbar = tqdm.tqdm(total=numiter * tsteps)
        else:
            pbar = None

        N = len(self.nodes)

        if select == "descend":
            ns = range(N - 2, 0, -1)
        elif select == "ascend":
            ns = range(1, N - 1)
        elif select in ("random", "bounce"):
            ns = list(range(1, N - 1))
        else:
            raise ValueError(f"Unknown select mode: {select}")

        try:
            for temp in linspace_generator(tstart, tfinal, tsteps, log=True):
                for _ in range(numiter):

                    if select == "random":
                        self.rng.shuffle(ns)
                    elif select == "bounce":
                        ns.reverse()

                    for n in ns:

                        node_0 = self.nodes[n - 1]
                        node_1 = self.nodes[n]
                        node_2 = self.nodes[n + 1]

                        pa, la, ra = node_1.plr
                        pb, lb, rb = node_2.plr

                        if (pa == lb) or (pa == rb):
                            # dependent contractions
                            if pa == lb:
                                a, b, c = la, ra, rb
                            else:  # pa == rb
                                a, b, c = la, ra, lb

                            if self.rng.choice([0, 1]) == 0:
                                # propose ((AC)B)
                                x = a | c
                                node_p1 = node_0.next(x, a, c)
                                node_p2 = node_p1.next(pb, x, b)
                            else:
                                # propose (A(BC))
                                x = b | c
                                node_p1 = node_0.next(x, b, c)
                                node_p2 = node_p1.next(pb, x, a)
                        else:
                            # parallel contractions (AB)(CD)
                            node_p1 = node_0.next(pb, lb, rb)
                            node_p2 = node_p1.next(pa, la, ra)

                        current_score = max(
                            node_1.tracker.score, node_2.tracker.score
                        )
                        proposed_score = max(
                            node_p1.tracker.score, node_p2.tracker.score
                        )

                        dE = proposed_score - current_score
                        accept = (dE <= 0) or (
                            math.log(self.rng.random()) < -dE / temp
                        )

                        if accept:
                            self.nodes[n] = node_p1
                            self.nodes[n + 1] = node_p2

                            # # need to update any global score trackers
                            # for c in range(n + 2, len(self.nodes)):
                            #     self.nodes[c].tracker.update_score(
                            #         self.nodes[c - 1].tracker
                            #     )

                    for c in range(1, len(self.nodes)):
                        self.nodes[c].tracker.update_score(
                            self.nodes[c - 1].tracker
                        )

                    if progbar:
                        pbar.update()
                        pbar.set_description(
                            f"T={temp:.3g} {self.tracker.describe()}"
                        )

        except KeyboardInterrupt:
            pass
        finally:
            if pbar:
                pbar.close()

    def get_ssa_path(self):
        bitpath = [self.nodes[c].plr for c in range(1, len(self.nodes))]
        return bit_path_to_ssa_path(bitpath)
