# Developer Notes


## Contributing

Contributions to `cotengra` are very welcome, whether they are bug reports,
documentation fixes, examples, tests, or new features. If you are planning a
larger change, opening an issue first is often the easiest way to check the
approach before spending too much time on implementation.

Things to check if new functionality is added:

1. Ensure functions are unit tested. Tests that depend on optional packages
   (`kahypar`, `optuna`, `cmaes`, `cotengrust`, ...) should
   `pytest.importorskip("...")` so the suite still runs in the minimal
   environment.
2. Mark tests that require local-only resources with `@pytest.mark.localonly`.
   The CI `test` task filters them out via `-m "not localonly"`.
3. Ensure functions have
   [NumPy-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
4. Ensure code is formatted and linted with `pixi run lint`.
5. Add to `cotengra/__init__.py` and `"__all__"` if appropriate.
6. Add to changelog and elsewhere in docs.
7. Experimental / unstable features go under `cotengra/experimental/`. That
   path is explicitly omitted from coverage.


### AI Policy

Please treat the [numpy AI policy](https://numpy.org/devdocs/dev/ai_policy.html) as a rough guide.


## Development Setup

`cotengra` uses [pixi](https://pixi.sh) to manage development environments and
reproducible tasks. The environments and tasks are defined in `pyproject.toml`,
which is the source of truth for the commands below.

After cloning the repository, install the pixi environments from the project
root:

```bash
git clone https://github.com/jcmgray/cotengra.git
cd cotengra
pixi install
```

You can then run project tasks with `pixi run ...`. For example, to run a short
python command inside the default test environment:

```bash
pixi run -e testpymid python -c "import cotengra; print(cotengra.__version__)"
```


## Running the Tests

Testing `cotengra` is handled by pixi tasks. The most common commands are:

```bash
pixi run -e testpymid test    # full suite with coverage, matches CI
```

The `test` task expands to:

```bash
pytest tests/ \
    --cov=cotengra \
    --cov-report=xml \
    --verbose \
    --durations=10 \
    -m "not localonly"
```

For a narrower check, use the `pytest` task (which runs in the `testpymid`
environment, no marker filter) and forward arguments after `--`:

```bash
pixi run pytest -- tests/test_tree.py
pixi run pytest -- tests/test_tree.py::test_contraction_tree_equivalency -v
pixi run pytest -- "tests/test_tree.py::test_contraction_tree_equivalency[frozenset-int]" -v
```

To run the full suite in a specific environment, use `-e`:

```bash
pixi run -e testpyold test
pixi run -e testpymid test
pixi run -e testpynew test
pixi run -e testjax test
pixi run -e testtorch test
pixi run -e testtensorflow test
```

To test the minimal dependency installation, use the `testminimal` environment
(it omits the `full` feature, so `kahypar`, `cotengrust`, `opt_einsum`, ... are
not installed):

```bash
pixi run -e testminimal test
```

The cross-backend contraction checks live in `tests/test_backends.py` and have
a dedicated task:

```bash
pixi run -e testpymid test-backends
```


## Formatting the Code

`cotengra` uses [`ruff`](https://docs.astral.sh/ruff/) to format imports and
code style. Use the predefined pixi tasks rather than running the tools
directly:

```bash
pixi run lint
pixi run format
```

The `format-all` task also runs notebook cleanup with `squeaky`:

```bash
pixi run format-all
```


## Building the docs locally

The documentation dependencies are managed by pixi. To build, clean, and serve
the docs locally, use:

```bash
pixi run docs
pixi run docs-clean
pixi run docs-serve
```

The local server hosts the built docs at `http://localhost:8000/`. The
generated HTML is in `docs/_build/html/`.

On ReadTheDocs, the build is driven by `.readthedocs.yml` and uses the
dedicated `readthedocs` pixi task.


## Minting a release

`cotengra` uses [`hatch-vcs`](https://github.com/ofek/hatch-vcs) to derive the
version from git tags, and
[GitHub Actions](https://github.com/jcmgray/cotengra/actions)
to publish to [PyPI](https://pypi.org/project/cotengra/). To mint a new
release:

1. Make sure all the
   [tests are passing on CI](https://github.com/jcmgray/cotengra/actions/workflows/tests.yml).
2. `git tag` the release with the next `vX.Y.Z`.
3. Push the tag to GitHub: `git push --tags`. The release workflow will
   build the sdist and wheel and upload them to the
   [PyPI **test** server](https://test.pypi.org/project/cotengra/).
4. If the test-pypi build looks good, create a GitHub release from the
   tag. Publishing the release triggers the same workflow to upload to
   the [PyPI **production** server](https://pypi.org/project/cotengra/).
5. The [`conda-forge/cotengra-feedstock`](https://github.com/conda-forge/cotengra-feedstock)
   repo should automatically pick up the new PyPI release and build a
   new [conda package](https://anaconda.org/conda-forge/cotengra); the
   recipe should only need to be manually updated if there are, for
   example, new dependencies.

Alternate manual release steps (after tagging):

1. Remove any old builds: `rm -rf dist/*`
2. Build the sdist and wheel: `python -m build`
3. Upload using twine: `twine upload dist/*`
