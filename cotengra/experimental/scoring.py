class CompressedTracedObjective(Objective):
    def __init__(self, chi, compress_late=False, r=1):
        self.chi = chi
        self.compress_late = compress_late
        self.r = r

    def trace(self, trial):
        import autoray as ar
        import quimb.tensor as qtn
        from autoray.experimental.complexity_tracing import compute_cost

        tree = trial["tree"]

        inputs = tree.inputs
        output = tree.output
        size_dict = tree.size_dict

        tn = qtn.TensorNetwork()
        for term in inputs:
            shape = [size_dict[ix] for ix in term]
            data = ar.lazy.Variable(shape, backend="numpy")
            tn |= qtn.Tensor(data=data, inds=term)

        Z = tn.contract_compressed(
            max_bond=self.chi,
            cutoff=0.0,
            optimize=tree.get_path(),
            canonize_distance=self.r,
            canonize_after_distance=0,
            compress_opts=dict(
                mode="virtual-tree",
            ),
            compress_late=self.compress_late,
            output_inds=output,
        )

        size = 0
        peak = Z.history_peak_size()
        write = 0
        cost = compute_cost(Z)
        for node in Z:
            size = max(size, node.size)
            write += node.size

        trial["flops"] = cost
        trial["write"] = write
        trial["size"] = size

        return size, peak, write, cost


class CompressedSizeTracedObjective(CompressedTracedObjective):
    def __init__(self, secondary_weight=1e-3, **kwargs):
        self.secondary_weight = secondary_weight
        super().__init__(**kwargs)

    def __call__(self, trial):
        size, _, write, cost = self.trace(trial)
        return (
            math.log2(size)
            + self.secondary_weight * math.log2(cost)
            + self.secondary_weight * math.log2(write)
        )


class CompressedPeakTracedObjective(CompressedTracedObjective):
    def __init__(self, secondary_weight=1e-3, **kwargs):
        self.secondary_weight = secondary_weight
        super().__init__(**kwargs)

    def __call__(self, trial):
        _, peak, write, cost = self.trace(trial)
        return (
            math.log2(peak)
            + self.secondary_weight * math.log2(cost)
            + self.secondary_weight * math.log2(write)
        )


class CompressedFlopsTracedObjective(CompressedTracedObjective):
    def __init__(self, secondary_weight=1e-3, **kwargs):
        self.secondary_weight = secondary_weight
        super().__init__(**kwargs)

    def __call__(self, trial):
        _, peak, write, cost = self.trace(trial)
        return (
            math.log2(cost)
            + self.secondary_weight * math.log2(peak)
            + self.secondary_weight * math.log2(write)
        )


class CompressedComboTracedObjective(CompressedTracedObjective):
    def __init__(self, factor=DEFAULT_COMBO_FACTOR, **kwargs):
        self.factor = factor
        super().__init__(**kwargs)

    def __call__(self, trial):
        _, peak, write, cost = self.trace(trial)
        return math.log2(peak + self.factor * write + self.factor * cost)
