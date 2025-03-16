import numpy as np
import xarray as xr
import pytest
from eradiate_absdb import (
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)
from eradiate_absdb.units import ureg


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


@pytest.fixture
def absdb_mono(shared_datadir):
    _db = MonoAbsorptionDatabase.from_directory(
        shared_datadir / "nanomono_v1", lazy=True, fix=False
    )
    yield _db
    _db.cache_clear()


@pytest.fixture
def absdb_ckd(shared_datadir):
    _db = CKDAbsorptionDatabase.from_directory(
        shared_datadir / "nanockd_v1", lazy=False, fix=False
    )
    yield _db
    _db.cache_clear()


def test_mono_construct(shared_datadir, absdb_mono):
    # Default gecko settings use lazy data loading
    assert absdb_mono.lazy is True

    # The dict converter accepts kwargs and can be used to override defaults
    absdb_mono = MonoAbsorptionDatabase.from_dict(
        {
            "construct": "from_directory",
            "dir_path": shared_datadir / "nanockd_v1",
            "lazy": False,
        }
    )
    assert absdb_mono.lazy is False


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


def test_ckd_construct(shared_datadir, absdb_ckd):
    # Default monotropa settings use eager data loading
    assert absdb_ckd.lazy is False

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
