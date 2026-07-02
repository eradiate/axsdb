# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Project

AxsDB is the reader and query interface for the absorption coefficient
databases used by the Eradiate radiative transfer model. It supports two
spectral representations: monochromatic and CKD (correlated-k distribution).
The package uses a `src/` layout and is managed with `uv`.

## Commands

Dependencies are managed with `uv`, so there is no separate virtual
environment to activate — prefix commands with `uv run` and `uv` will resolve
the environment automatically.

```shell
uv sync --group dev              # install all dev dependencies (test+docs+benchmark)

uv run pytest                    # run the full test suite (also runs doctests)
uv run pytest tests/test_core.py # run a single test file
uv run pytest tests/test_core.py::test_name  # run a single test
uv run pytest --cov=src          # run tests with coverage
uv run task test-cov-report      # coverage with HTML report (taskipy)

uv run task docs                 # build the Sphinx docs (docs/_build/html)
uv run task docs-serve           # live-reload docs server
uv run task benchmark            # run the pytest-benchmark suite (benchmarks/)

uv run ruff check .              # lint
uv run ruff format .             # format
pre-commit run --all-files       # ruff, taplo (TOML), nbstripout, uv-export
```

The pytest configuration lives in `pyproject.toml` under
`[tool.pytest.ini_options]`. Note that `testpaths` covers `tests/`, `src/` and
`docs/`, and doctest execution (`--doctest-plus`, `--doctest-glob='*.rst'`) is
enabled globally — this means the docstring examples embedded in
`src/axsdb/*.py` are collected and run as part of the test suite, so keep
them accurate and runnable when editing docstrings.

CI, defined in `.github/workflows/ci.yml`, runs the full test matrix
(Python 3.9–3.14 across Linux, macOS and Windows) via
`uv run coverage run -m pytest`, then combines the per-OS coverage data in a
follow-up job. The path mapping used to combine coverage collected on
different OSes is configured in `[tool.coverage.paths]` in `pyproject.toml` —
keep it in sync if the CI runner's checkout paths ever change.

## Architecture

Everything lives in `src/axsdb/`, a small, flat module layout:

- **`core.py`** holds the core abstraction. `AbsorptionDatabase` is an
  `attrs` class that holds a file index (`_index`, sorted by ascending
  wavelength) and a spectral coverage table (`_spectral_coverage`). Both are
  built lazily from the NetCDF files found in a directory and then cached to
  `index.csv`/`spectral.csv` alongside the data, so subsequent loads skip the
  expensive rebuild (see `from_directory`; `fix=True` regenerates missing
  index files on the fly). `MonoAbsorptionDatabase` and
  `CKDAbsorptionDatabase` subclass it and each implement `_make_index` plus
  `eval_sigma_a_mono`/`eval_sigma_a_ckd` — the CKD variant additionally
  carries a `g` quadrature-point dimension. Spectral lookup
  (`lookup_filenames`/`lookup_datasets`) maps a wavelength or wavenumber
  query onto the file(s) that cover it, using a precomputed digitize-able
  mesh stored in `_chunks`. Loaded dataset objects are LRU-cached
  (`cachetools`, default size 8), and can be opened either lazily or eagerly
  (`xr.open_dataset` vs. `xr.load_dataset`) via the `lazy` flag.

- **`error.py`** implements configurable error handling for interpolation
  out-of-bounds and missing-coordinate conditions.
  `ErrorHandlingConfiguration` holds one `ErrorHandlingPolicy` per coordinate
  kind (`x` for species concentration, `p` for pressure, `t` for
  temperature); each policy has `missing`/`scalar` actions
  (ignore/warn/raise) plus an asymmetric `bounds` pair of `BoundsPolicy`
  objects, each with its own action, mode (fill/clamp) and fill value. All of
  this funnels through `.convert()` classmethods that accept partial dicts,
  bare strings, numbers, or already-built instances interchangeably — this
  is what lets a caller pass something as terse as `bounds="clamp"` or as
  detailed as `{"lower": {...}, "upper": {...}}`. A process-global default
  configuration is read and written via `get_error_handling_config()` and
  `set_error_handling_config()`; the built-in default, defined at the bottom
  of `error.py`, ignores pressure/temperature bound errors (physically,
  out-of-bounds usually just means high altitude, where absorption is
  negligible anyway) while raising on missing or out-of-bounds species
  concentrations.

- **`interpolation.py`** provides Numba-accelerated multi-dimensional
  interpolation over the thermophysical profile dimensions (pressure,
  temperature, mole fractions). It's used by
  `AbsorptionDatabase._interp_thermophysical` to interpolate a data array
  read from a database file onto the thermophysical profile supplied by the
  caller.

- **`factory.py`** provides `AbsorptionDatabaseFactory`, a simple name →
  `RegistryEntry` registry (name, database class, a path or a
  path-resolving callable, and default constructor kwargs). This lets a
  caller such as Eradiate register known databases once at startup and then
  instantiate them by name via `.create()`, without hardcoding filesystem
  paths throughout the codebase.

- **`units.py`** provides Pint unit-registry access and quantity-conversion
  helpers used throughout the package for wavelength/wavenumber and
  dimensionless conversions. Databases may be indexed in either length or
  inverse-length units; the unit is auto-detected from each NetCDF file's
  `w` coordinate.

- **`cli.py`** wraps a `typer` CLI, exposed as the `axsdb` entry point. It
  currently has a single command, `axsdb check <path> -m {mono,ckd}
  [--fix]`, which simply runs `from_directory` for its validation side
  effects (building or fixing missing index files).

**Data model.** A database directory holds one or more NetCDF data files,
plus an `index.csv` (per-file spectral bounds), a `spectral.csv` (unrolled
per-point spectral coverage) and an optional `metadata.json`. Monochromatic
files have a single `w` (spectral coordinate) dimension; CKD files
additionally carry a `g` (quadrature point) dimension and a `wbounds`
variable giving spectral bin edges. Both `_make_index` implementations
detect whether `w` is stored in wavelength or wavenumber units and normalize
the index to hold both.

Known open items are tracked in `TODO.md`: a review of the exception
hierarchy, missing integration tests for per-bound error handling, and a
handful of interpolation performance ideas (parallel dimension groups, Numba
parallel gufuncs, more aggressive compiled-function caching, additional
interpolation methods).
