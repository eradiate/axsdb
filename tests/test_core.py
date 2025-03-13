"""
Tests for the radprops._absorption module.
"""

import numpy as np
import pytest
from rich.pretty import pprint
from eradiate_absdb import (
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)
from eradiate_absdb.units import ureg


@pytest.fixture
def absdb_ckd(shared_datadir):
    _db = CKDAbsorptionDatabase.from_directory(
        shared_datadir / "nanockd_v1", lazy=False, fix=False
    )
    yield _db
    _db.cache_clear()


def test_ckd_construct(shared_datadir, absdb_ckd):
    # Default monotropa settings use eager data loading
    assert absdb_ckd.lazy is False

    # Additionally, test the dict converter
    db = MonoAbsorptionDatabase.from_dict(
        {
            "construct": "from_directory",
            "dir_path": shared_datadir / "nanockd_v1",
            "lazy": True,
        }
    )
    assert db.lazy is True


@pytest.mark.parametrize(
    "w, expected",
    [
        ({"wl": 350.0}, ["nanockd_v1-345_355.nc"]),
        ({"wl": 350.0 * ureg.nm}, ["nanockd_v1-345_355.nc"]),
        ({"wl": 0.35 * ureg.micron}, ["nanockd_v1-345_355.nc"]),
        ({"wl": [350.0, 350.0]}, ["nanockd_v1-345_355.nc"] * 2),
    ],
    ids=[
        "wl_scalar_unitless",
        "wl_scalar_nm",
        "wl_scalar_micron",
        "wl_array_unitless",
    ],
)
def test_ckd_filename_lookup(absdb_ckd, w, expected):
    assert absdb_ckd.lookup_filenames(**w) == expected


@pytest.mark.parametrize("wg", [([550.0] * ureg.nm, 0.5)])
def test_eval(
    absdb_ckd, thermoprops_us_standard, absorption_database_error_handler_config, wg
):
    sigma_a = absdb_ckd.eval_sigma_a_ckd(
        *wg,
        thermoprops_us_standard,
        ErrorHandlingConfiguration.convert(absorption_database_error_handler_config),
    )

    # sigma_a should have a shape of (w, z)
    z = thermoprops_us_standard.z.values
    assert sigma_a.values.shape == (wg[0].size, z.size)


def test_cache_clear(absdb_ckd):
    # Make a query to ensure that the cache is filling up
    absdb_ckd.load_dataset("nanockd_v1-345_355.nc")
    assert absdb_ckd._cache.currsize > 0
    # Clear the cache: it should be empty after that
    absdb_ckd.cache_clear()
    assert absdb_ckd._cache.currsize == 0


def test_cache_reset(absdb_ckd):
    absdb_ckd.cache_reset(2)
    assert absdb_ckd._cache.currsize == 0
    assert absdb_ckd._cache.maxsize == 2
    absdb_ckd.cache_reset(8)
    assert absdb_ckd._cache.currsize == 0
    assert absdb_ckd._cache.maxsize == 8
