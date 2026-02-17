Getting started
===============

Installation
------------

Required dependencies:

* Python 3.9 or later

Install from PyPI in your virtual environment:

.. code:: shell

    pip install axsdb
    # or, with uv
    uv pip install axsdb

To install from source for development:

.. code:: shell

    git clone https://github.com/eradiate/axsdb.git
    cd axsdb
    uv sync --dev --all-extras

Verify your installation:

.. code:: shell

    python -c "import axsdb; print(axsdb.__version__)"

Quick start
-----------

AxsDB reads absorption coefficient databases stored as directories of NetCDF
files. Two spectral representations are supported:

* **Monochromatic** (:class:`.MonoAbsorptionDatabase`): dense wavelength grids for
  monochromatic calculations.
* **Correlated-k distribution** (:class:`.CKDAbsorptionDatabase`): spectral bin and *g*-coordinate indexed data
  for efficient band calculations.

See the :doc:`formats` page for details on the database directory structure.

Loading a database
^^^^^^^^^^^^^^^^^^

Open a database by pointing to its root directory:

.. code:: python

    from axsdb import MonoAbsorptionDatabase

    db = MonoAbsorptionDatabase.from_directory("path/to/database")

By default, data files are loaded eagerly. For large databases, lazy loading
avoids reading all files upfront:

.. code:: python

    db = MonoAbsorptionDatabase.from_directory("path/to/database", lazy=True)

If index files (``index.csv``, ``spectral.csv``) are missing, they are
automatically generated from the NetCDF data files on first load.

AxsDB also exposes a factory class (:class:`.AbsorptionDatabaseFactory`)
that can map string identifiers to database directory paths. See the API
documentation for a usage example.

Evaluating absorption coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation requires a thermophysical profile provided as an
``xarray.Dataset`` with pressure, temperature, and species mole fraction
coordinates:

.. code:: python

    import xarray as xr
    from axsdb import MonoAbsorptionDatabase
    from axsdb.units import get_unit_registry

    ureg = get_unit_registry()

    db = MonoAbsorptionDatabase.from_directory("path/to/database")
    thermoprops = xr.load_dataset("path/to/thermoprops.nc")

    # Evaluate the monochromatic absorption coefficient
    sigma_a = db.eval_sigma_a_mono(
        w=550.0 * ureg.nm,
        thermoprops=thermoprops,
    )

Such profiles can be created with the `Joseki <https://github.com/rayference/joseki>`__ library,

For CKD databases, use :meth:`.CKDAbsorptionDatabase.eval_sigma_a_ckd`,
which takes an additional ``g`` parameter for the *g*-point:

.. code:: python

    from axsdb import CKDAbsorptionDatabase

    db = CKDAbsorptionDatabase.from_directory("path/to/ckd_database")

    sigma_a = db.eval_sigma_a_ckd(
        w=550.0 * ureg.nm,
        g=0.5,
        thermoprops=thermoprops,
    )

Error handling
^^^^^^^^^^^^^^

AxsDB provides configurable error handling for out-of-bounds interpolation
on each thermophysical coordinate (pressure, temperature, mole fractions).
You can control behaviour per coordinate using
:class:`.ErrorHandlingConfiguration`:

.. code:: python

    from axsdb import ErrorHandlingConfiguration

    config = ErrorHandlingConfiguration.convert({
        "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    })

    sigma_a = db.eval_sigma_a_mono(
        w=550.0 * ureg.nm,
        thermoprops=thermoprops,
        error_handling_config=config,
    )

See the :doc:`error_handling` page for a full description of available
policies and modes.

Command-line interface
----------------------

AxsDB ships a CLI tool for database validation. Check a database for
integrity:

.. code:: shell

    axsdb check path/to/database -m mono

Use the ``--fix`` flag to automatically generate missing index files:

.. code:: shell

    axsdb check path/to/database -m ckd --fix

Use ``--log-level DEBUG`` for detailed output during validation.
