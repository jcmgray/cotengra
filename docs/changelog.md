# Changelog

## v0.5.5 (unreleased)

**Enhancements**

- [`HyperOptimizer`](cotengra.HyperOptimizer): by default simply warn if an
  individual trial fails, rather than raising an exception. This is to ensure
  rare failures do not spoil an entire optimization run. The behavior can
  be controlled with the `on_trial_error` argument.

**Bug fixes**

- fixed bug in greedy optimizer that produced negative scores and otherwise
  inaccurate scores.
- fixed bug for contraction with many inputs and also preprocessing steps


## v0.5.4 (2023-10-17)

**Bug fixes**

- the `auto` and `auto-hq` optimizers are now safe to run under multi-threading.

## v0.5.3 (2023-10-16)

- [``einsum``](cotengra.einsum), [`einsum_tree`](cotengra.einsum_tree)
  and [`einsum_expression`](cotengra.einsum_expression): add support for all
  numpy input formats, including interleaved indices and ellipses.
- remove some hidden `opt_einsum` dependence (via a `PathOptimizer` method)


## v0.5.2 (2023-10-13)

- add [``ctg.array_contract_path``](cotengra.array_contract_path) for returning
  the raw contraction path only, with caching. Add caching to
  [``array_contract_expression``](cotengra.array_contract_expression) and
  related functions too.
- fix [`tree.get_eq()`](cotengra.ContractionTree.get_eq) when the ``inputs``
  are a tuple of `Sequence[str]` rather than a `str`.


## v0.5.1 (2023-10-3)

- add [tree.contraction_scaling](cotengra.ContractionTree.contraction_scaling)
- add [get_symbol_map](cotengra.get_symbol_map)


## v0.5.0 (2023-09-26)

- add [`einsum`](cotengra.einsum)
- add [`einsum_tree`](cotengra.einsum_tree)
- add [`einsum_expression`](cotengra.einsum_expression)
- add [`array_contract`](cotengra.array_contract)
- add [`array_contract_tree`](cotengra.array_contract_tree)
- add [`array_contract_expression`](cotengra.array_contract_expression)
- add [`AutoOptimizer`](cotengra.AutoOptimizer)
- add [`AutoHQOptimizer`](cotengra.AutoHQOptimizer)
- remove most hard dependencies (`numpy`, `opt_einsum`)
- update [`tree.plot_contractions`](cotengra.plot.plot_contractions)
