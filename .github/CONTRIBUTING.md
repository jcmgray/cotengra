# Contributing

Contributions to `cotengra` in the form of
[pull requests](https://github.com/jcmgray/cotengra/pulls) are very welcome.
Opening an [issue](https://github.com/jcmgray/cotengra/issues) first can be
useful for larger changes, design questions, or work that might affect public
APIs.

If this is your first time contributing on GitHub, the following guide may be
useful:

- [GitHub - Creating a pull request](https://help.github.com/articles/creating-a-pull-request/)


## AI Policy

Please treat the [numpy AI policy](https://numpy.org/devdocs/dev/ai_policy.html) as a rough guide.


## Development Setup

`cotengra` uses [pixi](https://pixi.sh) for development environments and
predefined project tasks. The environments and commands are defined in
[`pyproject.toml`](../pyproject.toml).

From a fresh clone:

```bash
git clone https://github.com/jcmgray/cotengra.git
cd cotengra
pixi install
```


## Common Commands

Run the full test suite with coverage (matches CI):

```bash
pixi run -e testpymid test
```

Run a focused test via the `pytest` task (arguments after `--` are forwarded
to pytest):

```bash
pixi run pytest -- tests/test_tree.py
pixi run pytest -- tests/test_tree.py::test_contraction_tree_equivalency -v
pixi run pytest -- "tests/test_tree.py::test_contraction_tree_equivalency[frozenset-int]" -v
```

Run the full suite in a specific environment:

```bash
pixi run -e testpyold test
pixi run -e testpymid test
pixi run -e testpynew test
pixi run -e testjax test
pixi run -e testtorch test
pixi run -e testtensorflow test
```

Tests that should not run in CI should be marked with `@pytest.mark.localonly`;
the `test` task filters them out via `-m "not localonly"`. For a quick local
check that *includes* `localonly` tests, prefer the `pytest` task (no marker
filter).

The cross-backend contraction checks live in `tests/test_backends.py` and have
a dedicated task:

```bash
pixi run -e testpymid test-backends
```

To test cotengra with a minimal dependency installation (without `kahypar`,
`cotengrust`, `opt_einsum`, ...), use the `testminimal` environment:

```bash
pixi run -e testminimal test
```

Format and lint:

```bash
pixi run lint
pixi run format
pixi run format-all   # also runs `squeaky` on notebooks
```

Build and serve the docs:

```bash
pixi run docs
pixi run docs-clean
pixi run docs-serve
```

More developer details are in the
[development guide](https://cotengra.readthedocs.io/en/latest/develop.html).


## Contribution Checklist

- [ ] Tests have been added for new functionality. Tests that depend on
      optional packages (`kahypar`, `optuna`, `cmaes`, `cotengrust`, ...)
      should use `pytest.importorskip("...")` so the suite still runs in
      the minimal environment.
- [ ] Tests requiring local-only resources are marked with
      `@pytest.mark.localonly` so the CI `test` task skips them.
- [ ] Public functions have
      [NumPy-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
- [ ] New public API is exported from `cotengra/__init__.py` and added to
      `"__all__"` if appropriate.
- [ ] New functionality is documented in `docs/` or demonstrated with an
      example notebook when appropriate.
- [ ] User-facing changes are noted in `docs/changelog.md`.
- [ ] Experimental / unstable features go under `cotengra/experimental/`.
      That path is explicitly omitted from coverage and is not part of
      the stable API surface.
- [ ] Formatting and lint checks pass with `pixi run lint`.
