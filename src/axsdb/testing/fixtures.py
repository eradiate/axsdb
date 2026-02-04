import pytest
import xarray as xr

from ..core import CKDAbsorptionDatabase, MonoAbsorptionDatabase


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
    This dataset is created with the following command::

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
