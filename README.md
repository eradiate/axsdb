# AxsDB — The Eradiate Absorption Cross-section Database Interface

[![PyPI version](https://img.shields.io/pypi/v/axsdb?color=blue)](https://pypi.org/project/axsdb)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/eradiate/axsdb/ci.yml?branch=main)](https://github.com/eradiate/axsdb/actions/workflows/ci.yml)
[![Documentation Status](https://img.shields.io/readthedocs/axsdb)](https://axsdb.readthedocs.io)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This library provides an interface to read and query the absorption databases
of the [Eradiate radiative transfer model](https://eradiate.eu).

## Features

- **Monochromatic and CKD databases**: Read and evaluate absorption coefficients
  for both monochromatic and correlated-k distribution spectral representations.
- **Fast interpolation**: Numba-accelerated multi-dimensional interpolation of
  thermophysical profiles (pressure, temperature, mole fractions).
- **Configurable error handling**: Fine-grained, per-coordinate control over
  out-of-bounds behaviour (clamp, fill, raise, warn).
- **Efficient I/O**: Lazy or eager loading of NetCDF data files, with LRU
  caching for repeated lookups.
- **xarray and Pint integration**: Works natively with xarray datasets and
  Pint quantities.
- **CLI**: Validate database integrity from the command line with `axsdb check`.

## Installation

Python 3.9 or later is required.

```shell
pip install axsdb
```

## Quick start

```python
import xarray as xr
from axsdb import MonoAbsorptionDatabase
from axsdb.units import get_unit_registry

ureg = get_unit_registry()

# Open a database directory
db = MonoAbsorptionDatabase.from_directory("path/to/database")

# Load a thermophysical profile
thermoprops = xr.load_dataset("path/to/thermoprops.nc")

# Evaluate the absorption coefficient at a given wavelength
sigma_a = db.eval_sigma_a_mono(
    w=550.0 * ureg.nm,
    thermoprops=thermoprops,
)
```

For CKD databases, use `CKDAbsorptionDatabase` and `eval_sigma_a_ckd` (which
takes an additional `g` parameter for the quadrature point).

## CLI

AxsDB ships a command-line tool to validate databases:

```shell
# Check a monochromatic database
axsdb check path/to/database -m mono

# Check and fix missing index files
axsdb check path/to/database -m ckd --fix
```

## Documentation

Full documentation is available at
[axsdb.readthedocs.io](https://axsdb.readthedocs.io).

## License

AxsDB is distributed under the terms of the
[GNU Lesser General Public License v3.0](https://choosealicense.com/licenses/lgpl-3.0/).
