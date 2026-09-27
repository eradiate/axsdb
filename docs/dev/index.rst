Developer guide
===============

This guide is for contributors of all sorts — developers, maintainers, users
reporting issues — with or without AI programming assistance.

AxsDB is the reader and query interface for the absorption coefficient
databases used by the `Eradiate <https://eradiate.eu>`__ radiative transfer
model. Its main consumer is Eradiate: a change in behaviour here is a change in
Eradiate's computed radiances, whether or not Eradiate's own code moved.

.. _dev-ai_programming:

AI programming
--------------

AI coding assistants (*e.g.* Claude Code, Codex, Cursor) are allowed in this
project, but only as a tool under human supervision. All changes are authored by
a human responsible for them regardless of how they were produced; "the agent
wrote it" is not a defence for a regression.

Project context for agents
    ``AGENTS.md`` at the repository root is the source of truth for conventions,
    layout and gotchas, and is loaded automatically by AI tools (``CLAUDE.md``
    imports it and adds Claude-specific instructions). Keep it in sync with the
    code: when a convention changes, update ``AGENTS.md`` and this guide in the
    same change.

Attribution
    Commits are authored by the human running the session. An assistant never
    appears as author, committer or co-author, and commit messages carry no
    attribution trailer — no ``Co-Authored-By``, no session URL. Pull request
    descriptions the same. This is stated in ``AGENTS.md``. An agent whose own
    harness instructs it to append a trailer must not.

Identifiers are verified, not recalled
    No variable name, attribute, function, unit or reference enters the code on
    recall alone. Each is checked against the primary source or against a run of
    the software. This applies to humans and agents alike — an agent producing a
    plausible-looking dataset variable name or unit string from memory is the
    failure mode this rule exists for.

Licensing caution
    This is an LGPLv3 project. Do not paste in code of unknown or incompatible
    provenance; an agent suggesting a verbatim block from elsewhere is a
    licensing risk.

Tooling guardrails
    Rely on pre-commit and CI as the source of truth for formatting and lint,
    not on an agent's — or your own — claim that a file is clean.

.. _dev-general_guidelines:

General guidelines
------------------

These apply to everyone, whether or not AI tools are used.

Verification expectations
    All changes must pass ``uv run pytest``. The suite collects ``tests/``,
    ``src/`` and ``docs/``: docstring examples and reStructuredText code blocks
    are executed, so a renamed argument or a changed repr breaks the build
    rather than rotting quietly in the documentation.

Numerical claims name their reference
    A test asserting numerical correctness (an interpolated absorption
    coefficient, a spectral lookup result) states what it is checked against.
    A number that only reproduces the current implementation protects nothing.

Simplicity
    The smallest thing that works, first. No interface with one implementation,
    no configuration for a value that never changes, no abstraction for a need
    that has not appeared.

Open items
    Known open work is tracked in ``TODO.md``.

.. _dev-high_risk:

High-risk areas
---------------

The areas below are where a change can look correct, pass review and still be
wrong. What breaks is a number or a meaning (*e.g.* an absorption coefficient
interpolated at the wrong wavelength, a bound violation that is now silently
filled) rather than the structure of the code. Nothing crashes, so the error
shows up neither in the diff nor in a traceback: only in the output, and only if
someone checks it against a reference.

Spectral units
    A database may store its ``w`` coordinate as wavelength or as wavenumber;
    the unit is detected from each file, and both ``_make_index``
    implementations normalize the index to hold both. Spectral lookup
    (``lookup_filenames``) must give the same file whichever unit the query is
    expressed in. Wavenumber order is the reverse of wavelength order: any code
    that assumes a sort direction must say which one.

Cached index files
    ``index.csv`` and ``spectral.csv`` are written next to the data and reused
    on later loads without being checked against the NetCDF files. A change to
    what ``_make_index`` or ``_make_spectral_coverage`` writes does not reach
    existing databases until their index files are regenerated
    (``axsdb check <path> -m {mono,ckd} --fix`` rebuilds missing ones only).

Default error handling
    The built-in configuration at the bottom of ``error.py`` ignores pressure
    and temperature out-of-bounds conditions and raises on missing or
    out-of-bounds species concentrations. Eradiate relies on it. Changing it
    changes results silently for every caller that does not pass its own
    configuration, so it is a behaviour change to announce in the changelog,
    not a refactor.

Bounds handling in the interpolation kernels
    The Numba gufuncs in ``math.py`` encode the bounds mode as integer
    constants (``_BOUNDS_FILL``, ``_BOUNDS_CLAMP``, ``_BOUNDS_RAISE``). The
    mapping from :class:`~axsdb.BoundsMode` to these constants, and the
    per-bound (lower/upper) split, are where a fill can turn into a clamp
    without any test failing unless one covers that exact case.

CKD quadrature dimension
    CKD data carries a ``g`` dimension and a ``wbounds`` variable giving bin
    edges. A lookup that works on bin centres instead of bin edges, or that
    drops ``g`` ordering, returns plausible values for the wrong bin or
    quadrature point.

Dataset cache lifetime
    Opened datasets are held in an LRU cache (default size 8) that closes
    datasets on eviction (``ClosingLRUCache`` in ``util.py``). In lazy mode, a
    reference to an evicted dataset refers to a closed file.

Development environment
-----------------------

The project is managed with `uv <https://docs.astral.sh/uv/>`__, and tasks are
defined for `Poe the Poet <https://poethepoet.natn.io/>`__::

    git clone https://github.com/eradiate/axsdb
    cd axsdb
    uv sync --all-groups --all-extras
    uv run pytest

The supported Python range is 3.9 through 3.14, and 3.9 is the floor the code is
written against. Intel Mac users need an extra constraint on Numba, see
:doc:`installation`.

Dependency groups
^^^^^^^^^^^^^^^^^

``pyproject.toml`` declares the following dependency groups:

``test``
    pytest and its plugins, coverage, rich.

``docs``
    Sphinx, the Shibuya theme and extensions.

``benchmark``
    pytest-benchmark and pyinstrument.

``lint``
    Poe the Poet and prek; all the lint CI job installs.

``dev``
    ``test``, ``docs``, ``benchmark`` and ``lint``, plus Ruff and
    ``sp-repo-review``.

``interactive``
    JupyterLab and Matplotlib, for exploratory work.

Task cheatsheet
^^^^^^^^^^^^^^^

All tasks run as ``uv run poe <task>``. ``uv run poe`` with no argument lists
them.

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Task
      - What it does
    * - ``test``
      - Run the suite over ``tests/``, ``src/`` and ``docs/``; benchmarks
        excluded. Extra arguments pass through to ``pytest``.
    * - ``test-cov``
      - Run the suite under coverage and print the report.
    * - ``test-cov-report``
      - Same, and write an HTML report to ``reports/coverage/html``.
    * - ``bench``
      - Run the benchmarks in ``benchmarks/``.
    * - ``profile``
      - Profile with pyinstrument and write ``reports/profile.html``.
    * - ``lint``
      - Run every pre-commit hook over all files (prek).
    * - ``repo-review``
      - ``sp-repo-review`` packaging and hygiene checks.
    * - ``docs``
      - Build the HTML docs into ``docs/_build/html``.
    * - ``docs-serve``
      - Live-serve the docs with sphinx-autobuild.
    * - ``docs-clean``
      - Remove the ``docs/_build/`` tree.
    * - ``docs-lock``
      - Regenerate ``docs/requirements.txt`` from ``uv.lock``.

Architecture
------------

The library indexes a directory of NetCDF absorption data files, maps a
spectral query onto the files that cover it, and interpolates the absorption
coefficient onto a thermophysical profile (pressure, temperature, mole
fractions) supplied by the caller. xarray is used throughout for the data;
Numba provides the interpolation kernels.

Package layout
^^^^^^^^^^^^^^

Python, src layout, one flat package; ``axsdb.testing`` is the only
subpackage.

``core.py``
    :class:`~axsdb.AbsorptionDatabase` and its two subclasses,
    :class:`~axsdb.MonoAbsorptionDatabase` and
    :class:`~axsdb.CKDAbsorptionDatabase`: index construction and caching,
    spectral lookup, dataset cache, and evaluation of the absorption
    coefficient.

``error.py``
    :class:`~axsdb.ErrorHandlingConfiguration`, :class:`~axsdb.ErrorHandlingPolicy`,
    :class:`~axsdb.BoundsPolicy` and the enums they are built from, plus the
    process-global default configuration
    (:func:`~axsdb.get_error_handling_config`,
    :func:`~axsdb.set_error_handling_config`).

``interpolation.py``
    Multi-dimensional interpolation of a data array over the thermophysical
    dimensions, built on the kernels in ``math.py``.

``math.py``
    Numba gufuncs for 1D linear interpolation with bounds handling.

``factory.py``
    :class:`~axsdb.AbsorptionDatabaseFactory`: a name-to-database registry so
    that a caller registers known databases once and instantiates them by name.

``units.py``
    Pint registry access and quantity conversion helpers.

``cli.py``
    The ``axsdb`` command-line entry point (``axsdb check``).

``util.py``, ``typing.py``
    Generic helpers (``ClosingLRUCache``) and type aliases.

``testing/fixtures.py``
    pytest fixtures shared by the test suite.

:doc:`../formats` specifies the on-disk database layout.

Module layout and public API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything in ``axsdb.__all__`` is public; every underscore name and every
module path is internal and may change without notice, with two exceptions:
``axsdb.units``, which is part of ``__all__``, and ``axsdb.interpolation``,
whose ``interp_dataarray`` is documented for direct use.

Conventions
-----------

Docstrings
    `Numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__ style.
    In-code documentation is the primary documentation surface; the user
    manual stays separate from the API reference. Comments describe current
    behaviour, not development history, and do not cite a measured timing —
    performance numbers belong to the machine they were taken on, and the
    benchmark suite is there to re-take them.

Type hints
    Type hints in every signature. Ruff enforces the modern annotation style
    (see ``ruff.toml``). Two rules it cannot check: ``X | Y`` is only safe
    inside annotations (with ``from __future__ import annotations``), and
    raises ``TypeError`` on Python 3.9 anywhere evaluated at runtime — use
    ``typing.Union``/``typing.Optional`` there; test functions carry no type
    annotations.

Units
    Units are managed with Pint through ``axsdb.units``. Quantities crossing the
    public API carry their units.

Testing
-------

Layout
^^^^^^

Tests live in ``tests/``, roughly one module per source module; shared fixtures
are in ``src/axsdb/testing/fixtures.py`` and small test databases in
``tests/data/``. The pytest configuration is in ``pyproject.toml``, under
``[tool.pytest.ini_options]``. In particular:

- tests are collected from ``tests``, ``src`` and ``docs``;
- warnings are errors;
- ``xfail_strict`` is on;
- ``--doctest-plus`` plus ``--doctest-glob='*.rst'`` turn docstring examples
  and the reStructuredText documentation into tests. The root ``conftest.py``
  injects ``np``, ``xr``, ``axsdb``, the main classes and a ``pprint`` helper
  into the doctest namespace so examples need no import preamble.

Obvious things are not tested. Fixtures and parametrization keep the suite
concise. A test is motivated by the outcome it protects, not by a requirement it
satisfies.

Benchmarks
^^^^^^^^^^

Benchmarks live in ``benchmarks/``, use
`pytest-benchmark <https://pytest-benchmark.readthedocs.io/>`__, and are
configured by their own ``pytest.ini`` (``bench_`` and ``Bench`` prefixes).
They are excluded from the default suite and run with ``uv run poe bench``. See
:doc:`benchmarking`.

Documentation
-------------

Built with Sphinx and the `Shibuya <https://shibuya.lepture.com/>`__ theme,
deployed on Read the Docs. Markdown sources are supported through MyST-Parser;
``docs/changelog.rst`` includes ``CHANGELOG.md``.

Read the Docs does not use uv, so ``.readthedocs.yml`` installs from
``docs/requirements.txt``, exported from ``uv.lock`` by the ``uv-export``
pre-commit hook (or by ``uv run poe docs-lock``). Commit it whenever the
lock file changes.

Conventions and tooling
-----------------------

Code style
^^^^^^^^^^

Python code is formatted and linted with `Ruff <https://docs.astral.sh/ruff/>`__.

pre-commit
^^^^^^^^^^

Hooks are defined in ``.pre-commit-config.yaml``: Ruff, taplo (TOML
formatting), nbstripout, ``uv-export`` and zizmor (GitHub Actions audit). They
run with `prek <https://prek.j178.dev/>`__ (``uv run poe lint`` runs them over
all files).

Packaging and repository hygiene
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``uv run poe repo-review`` runs `sp-repo-review
<https://learn.scientific-python.org/development/guides/repo-review/>`__.
Exceptions are permitted; each is recorded with the check ID it waives and the
reason, in ``[tool.repo-review.ignore]``.

Licensing
^^^^^^^^^

The project is licensed under ``LGPL-3.0-or-later`` (``LICENSE`` at the
repository root).

Continuous integration
^^^^^^^^^^^^^^^^^^^^^^

Three workflows:

``test.yml``
    The suite across Python 3.9 to 3.14 on Linux, macOS and Windows, on pushes
    to ``main``, on pull requests, and on manual dispatch. A follow-up job
    combines per-OS coverage data; the path mapping it relies on is in
    ``[tool.coverage.paths]``.

``lint.yml``
    The pre-commit hooks, in the ``lint`` dependency group only.

``release.yml``
    Builds and publishes to PyPI when a tag is pushed. The version lives in
    ``pyproject.toml`` and is read back at runtime through
    ``importlib.metadata``; the workflow fails if the tag and the version
    disagree. See :doc:`release`.
