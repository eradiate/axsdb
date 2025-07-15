import numpy as np
import xarray as xr
import pytest

import axsdb
from axsdb import (
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)
from axsdb.units import ureg


@pytest.fixture
def absorption_database_error_handler_config():
    """
    Error handler configuration for absorption coefficient interpolation.

    Notes
    -----
    This configuration is chosen to ignore all interpolation issues (except
    bounds error along the mole fraction dimension) because warnings are
    captured by pytest which will raise.
    Ignoring the bounds on pressure and temperature is safe because
    out-of-bounds values usually correspond to locations in the atmosphere
    that are so high that the contribution to the absorption coefficient
    are negligible at these heights.
    The bounds error for the 'x' (mole fraction) coordinate is considered
    fatal.
    """
    return {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    }


@pytest.fixture
def thermoprops_us_standard(shared_datadir):
    """
    This dataset is created with the following command:

    .. code:: python

        joseki.make(
            identifier="afgl_1986-us_standard",
            z=np.linspace(0.0, 120.0, 121) * ureg.km,
            additional_molecules=False,
        )
    """
    yield xr.load_dataset(shared_datadir / "afgl_1986-us_standard.nc")


def _absdb(mode, path):
    if mode == "mono":
        return MonoAbsorptionDatabase.from_directory(
            path / "nanomono_v1", lazy=True, fix=False
        )
    elif mode == "ckd":
        return CKDAbsorptionDatabase.from_directory(
            path / "nanockd_v1", lazy=False, fix=False
        )
    else:
        raise RuntimeError


@pytest.fixture
def absdb(shared_datadir, request):
    mode = request.param
    _db = _absdb(mode, shared_datadir)
    yield _db
    _db.cache_clear()


@pytest.fixture
def absdb_mono(shared_datadir):
    _db = _absdb("mono", shared_datadir)
    yield _db
    _db.cache_clear()


@pytest.fixture
def absdb_ckd(shared_datadir):
    _db = _absdb("ckd", shared_datadir)
    yield _db
    _db.cache_clear()


def test_mono_construct(shared_datadir):
    # The dict converter accepts kwargs and can be used to override defaults
    db = MonoAbsorptionDatabase.from_dict(
        {
            "construct": "from_directory",
            "dir_path": shared_datadir / "nanockd_v1",
            "lazy": False,
        }
    )
    assert db.lazy is False


@pytest.mark.parametrize(
    "w",
    [
        [350.0] * ureg.nm,
        np.linspace(349.0, 351.0, 3) * ureg.nm,
    ],
    ids=["scalar", "vector"],
)
def test_mono_eval(
    absdb_mono, thermoprops_us_standard, absorption_database_error_handler_config, w
):
    sigma_a = absdb_mono.eval_sigma_a_mono(
        w,
        thermoprops_us_standard,
        ErrorHandlingConfiguration.convert(absorption_database_error_handler_config),
    )

    # sigma_a should have a shape of (w, z)
    z = thermoprops_us_standard.z.values
    assert sigma_a.values.shape == (w.size, z.size)


def test_ckd_construct(shared_datadir):
    # Additionally, test the dict converter
    db = CKDAbsorptionDatabase.from_dict(
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


@pytest.mark.parametrize("wg", [([350.0] * ureg.nm, 0.5)])
def test_ckd_eval(
    absdb_ckd, thermoprops_us_standard, absorption_database_error_handler_config, wg
):
    sigma_a = absdb_ckd.eval_sigma_a_ckd(
        *wg,
        thermoprops=thermoprops_us_standard,
        error_handling_config=ErrorHandlingConfiguration.convert(
            absorption_database_error_handler_config
        ),
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


@pytest.mark.parametrize("absdb", ["mono", "ckd"], indirect=True)
def test_error_handling(absdb, thermoprops_us_standard):
    # The default error handling config is the global one
    assert absdb.error_handling_config is axsdb.get_error_handling_config()

    # Valid dicts are successfully converted
    absdb.error_handling_config = {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    }
    assert absdb.error_handling_config is not axsdb.get_error_handling_config()
    assert absdb.error_handling_config == axsdb.get_error_handling_config()

    # Invalid dicts cannot be converted
    with pytest.raises(ValueError):
        absdb.error_handling_config = {"wrong": "value"}

    # Check error handling config override

    absdb.error_handling_config = {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "raise"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "raise"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    }
    with pytest.raises(ValueError):
        if isinstance(absdb, MonoAbsorptionDatabase):
            absdb.eval_sigma_a_mono(
                w=350.0 * ureg.nm, thermoprops=thermoprops_us_standard
            )
        elif isinstance(absdb, CKDAbsorptionDatabase):
            absdb.eval_sigma_a_ckd(
                w=350.0 * ureg.nm, g=0.5, thermoprops=thermoprops_us_standard
            )
        else:
            assert False, "unhandled case"
