import math

from .core import ContractionTree, cached_node_property, node_get_single_el


class ContractionTreeMulti(ContractionTree):
    def __init__(
        self,
        inputs,
        output,
        size_dict,
        sliced_inds,
        objective,
        track_cache=False,
    ):
        super().__init__(inputs, output, size_dict, objective=objective)
        self.sliced_inds = {ix: None for ix in sliced_inds}

        self._track_cache = track_cache
        if track_cache:
            self._cache_est = 0

    def set_state_from(self, other):
        super().set_state_from(other)
        self._track_cache = other._track_cache
        if other._track_cache:
            self._cache_est = other._cache_est

    def _remove_node(self, node):
        if self._track_cache:
            self._cache_est -= self.get_cache_contrib(node)
        super()._remove_node(node)

    def _update_tracked(self, node):
        if self._track_cache:
            self._cache_est += self.get_cache_contrib(node)
        super()._update_tracked(node)

    @cached_node_property("node_var_inds")
    def get_node_var_inds(self, node):
        """Get the set of variable indices that a node depends on."""
        if len(node) == 1:
            i = node_get_single_el(node)
            term = self.inputs[i]
            return {ix: None for ix in term if ix in self.sliced_inds}

        try:
            l, r = self.children[node]
            return self.get_node_var_inds(l) | self.get_node_var_inds(r)
        except KeyError:
            return {
                ix: None
                for term in self.node_to_terms(node)
                for ix in term
                if ix in self.sliced_inds
            }

    @cached_node_property("node_is_bright")
    def get_node_is_bright(self, node):
        """Get whether a node is 'bright', i.e. contains a different set of
        variable indices to either of its children, if a node is not bright
        then its children never have to be stored in the cache.
        """
        if len(node) == 1:
            i = node_get_single_el(node)
            term = self.inputs[i]
            return any(ix in self.sliced_inds for ix in term)

        l, r = self.children[node]
        return (self.get_node_var_inds(node) != self.get_node_var_inds(l)) or (
            self.get_node_var_inds(node) != self.get_node_var_inds(r)
        )

    @cached_node_property("node_mult")
    def get_node_mult(self, node):
        """Get the estimated 'multiplicity' of a node, i.e. the number of times
        it will have to be recomputed for different index configurations.
        """
        return self.get_default_objective().estimate_node_mult(self, node)

    def get_node_cache_mult(self, node, sliced_ind_ordering):
        """Get the estimated 'cache multiplicity' of a node, i.e. the total
        number of versions with different index configurations that must be
        stored simultaneously in the cache.
        """
        return self.get_default_objective().estimate_node_cache_mult(
            self, node, sliced_ind_ordering
        )

    # @cached_node_property("multi_flops")
    def get_flops(self, node):
        """The the estimated total cost of computing a node for all index
        configurations.
        """
        return super().get_flops(node) * self.get_node_mult(node)

    @cached_node_property("cache_contrib")
    def get_cache_contrib(self, node):
        l, r = self.children[node]
        lr_peak = 0
        if self.get_node_is_bright(l):
            lr_peak += self.get_size(l)
        if self.get_node_is_bright(r):
            lr_peak += self.get_size(r) * self.get_node_mult(r)

        rl_peak = 0
        if self.get_node_is_bright(r):
            rl_peak += self.get_size(r)
        if self.get_node_is_bright(l):
            rl_peak += self.get_size(l) * self.get_node_mult(l)

        if lr_peak < rl_peak:
            return lr_peak
        else:
            self.children[node] = (r, l)
            return rl_peak

    def peak_size(self, log=None):
        if not self._track_cache:
            self._cache_est = 0
            for (
                node,
                _,
                _,
            ) in self.traverse():
                self._cache_est += self.get_cache_contrib(node)
            self._track_cache = True
        peak = self._cache_est

        if log is not None:
            peak = math.log(peak, log)

        return peak

    def reorder_contractions_for_peak_est(self):
        """Reorder the contractions to try and reduce the peak memory usage."""
        swapped = False

        for p, l, r in self.descend():
            lr_peak = 0
            if self.get_node_is_bright(l):
                lr_peak += self.get_size(l)
            if self.get_node_is_bright(r):
                lr_peak += self.get_size(r) * self.get_node_mult(r)

            rl_peak = 0
            if self.get_node_is_bright(r):
                rl_peak += self.get_size(r)
            if self.get_node_is_bright(l):
                rl_peak += self.get_size(l) * self.get_node_mult(l)

            if rl_peak < lr_peak:
                self.children[p] = (r, l)
                swapped = True

        return swapped

    def reorder_sliced_inds(self):
        """ """
        sliced_ind_ordering = dict()

        for node, _, _ in self.traverse():
            sliced_ind_ordering.update(self.get_node_var_inds(node))

        self.sliced_inds = {ix: None for ix in sliced_ind_ordering}

    def exact_multi_stats(self, configs):
        # ragged list of lists (configs and contractions)
        cons = []

        # build this for efficiency
        plr = tuple(self.traverse())

        def to_key(node, config):
            subconfig = tuple(
                map(config.__getitem__, self.get_node_var_inds(node))
            )
            return hash((node, subconfig))

        # iterate forward, recording only when we first need to produce a 'parent'
        seen = set()
        for config in configs:
            cons_i = []
            for p, l, r in plr:
                pkey = to_key(p, config)
                first = pkey not in seen
                if first:
                    seen.add(pkey)
                    cons_i.append(
                        {
                            "p": p,
                            "l": l,
                            "r": r,
                            "pkey": pkey,
                            "lkey": to_key(l, config),
                            "rkey": to_key(r, config),
                        }
                    )
            cons.append(cons_i)
        del seen

        # iterate backward, checking the last
        # time a 'child' is seen -> can delete
        deleted = set()
        for cons_i in reversed(cons):
            for con in cons_i:
                rkey = con["rkey"]
                rdel = rkey not in deleted
                if rdel:
                    deleted.add(rkey)
                con["rdel"] = rdel

                lkey = con["lkey"]
                ldel = lkey not in deleted
                if ldel:
                    deleted.add(lkey)
                con["ldel"] = ldel
        del deleted

        # iterate forward again if we want to compute flops and memory usage:
        # not needed if we already know these & just want to contract
        flops = 0
        mems = []
        mem_current = 0
        mem_peak = 0
        mem_write = 0

        for cons_i in cons:
            for con in cons_i:
                p = con["p"]
                flops += super().get_flops(p)
                psize = self.get_size(p)
                mem_current += psize
                mem_write += psize

                mems.append(mem_current)
                mem_peak = max(mem_peak, mem_current)

                l, r = con["l"], con["r"]
                if con["ldel"] and len(l) > 1:
                    mem_current -= self.get_size(l)
                if con["rdel"] and len(r) > 1:
                    mem_current -= self.get_size(r)

            # final output of each config is always deletable
            mem_current -= self.get_size(p)

        return {
            "flops": flops,
            "write": mem_write,
            "size": self.max_size(),
            "peak": mem_peak,
        }
