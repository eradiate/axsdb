import numpy as np
import pytest

import axsdb
from axsdb import (
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)
from axsdb.error import ErrorHandlingAction, InterpolationError
from axsdb.testing.fixtures import *  # noqa: F403
from axsdb.units import get_unit_registry

ureg = get_unit_registry()


class TestFromDirectory:
    def test_path_exists(self, shared_datadir):
        assert (
            MonoAbsorptionDatabase.from_directory(shared_datadir / "nanomono_v1")
            is not None
        )

        assert (
            CKDAbsorptionDatabase.from_directory(shared_datadir / "nanockd_v1")
            is not None
        )

    def test_path_doesnt_exist(self):
        with pytest.raises(NotADirectoryError, match="doesnt_exist"):
            CKDAbsorptionDatabase.from_directory("doesnt_exist")


class TestMonoAbsorptionDatabase:
    def test_construct(self, shared_datadir):
        # The dict converter accepts kwargs and can be used to override defaults
        db = MonoAbsorptionDatabase.from_dict(
            {
                "construct": "from_directory",
                "dir_path": shared_datadir / "nanomono_v1",
                "lazy": False,
                "error_handling_config": {"t": {"missing": "warn"}},
            }
        )
        assert db.lazy is False
        assert db.error_handling_config.t.missing is ErrorHandlingAction.WARN

    @pytest.mark.parametrize(
        "w",
        [
            [350.0] * ureg.nm,
            np.linspace(349.0, 351.0, 3) * ureg.nm,
        ],
        ids=["scalar", "vector"],
    )
    def test_eval(
        self,
        absdb_mono,
        thermoprops_us_standard,
        absorption_database_error_handler_config,
        w,
    ):
        sigma_a = absdb_mono.eval_sigma_a_mono(
            w,
            thermoprops_us_standard,
            ErrorHandlingConfiguration.convert(
                absorption_database_error_handler_config
            ),
        )

        # sigma_a should have a shape of (w, z)
        z = thermoprops_us_standard.z.values
        assert sigma_a.values.shape == (w.size, z.size)


class TestCKDAbsorptionDatabase:
    def test_ckd_construct(self, shared_datadir):
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
    def test_filename_lookup(self, absdb_ckd, w, expected):
        assert absdb_ckd.lookup_filenames(**w) == expected

    @pytest.mark.parametrize("wg", [([350.0] * ureg.nm, 0.5)])
    def test_eval(
        self,
        absdb_ckd,
        thermoprops_us_standard,
        absorption_database_error_handler_config,
        wg,
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

        # Regression test: the result must carry a "w" coordinate labeling
        # the matched grid wavelength, like the pre-refactor
        # .sel(method="nearest")-based implementation did.
        assert "w" in sigma_a.coords
        np.testing.assert_allclose(sigma_a.coords["w"].values, [349.9286])

    def test_interp_thermophysical_raw_matches(
        self,
        absdb_ckd,
        thermoprops_us_standard,
        absorption_database_error_handler_config,
    ):
        # _interp_thermophysical_raw (numpy in/out) must produce the exact
        # same result as _interp_thermophysical (DataArray in/out) for the
        # same inputs.
        error_handling_config = ErrorHandlingConfiguration.convert(
            absorption_database_error_handler_config
        )
        ds = absdb_ckd.load_dataset("nanockd_v1-345_355.nc")
        da = ds["sigma_a"].sel(w=350.0, method="nearest")

        expected, expected_x_ds = absdb_ckd._interp_thermophysical(
            ds, da, thermoprops_us_standard, error_handling_config
        )

        data, dims, out_coords, x_ds = absdb_ckd._interp_thermophysical_raw(
            ds,
            da.values,
            list(da.dims),
            da.coords,
            thermoprops_us_standard,
            error_handling_config,
        )

        assert x_ds == expected_x_ds
        assert dims == list(expected.dims)
        np.testing.assert_array_equal(data, expected.values)
        assert set(out_coords) == set(expected.coords)


def test_cache_clear(absdb_ckd):
    # Make a query to ensure that the cache is filling up
    absdb_ckd.load_dataset("nanockd_v1-345_355.nc")
    assert absdb_ckd._fname_dataset_cache.currsize > 0
    # Clear the cache: it should be empty after that
    absdb_ckd.cache_clear()
    assert absdb_ckd._fname_dataset_cache.currsize == 0


def test_cache_reset(absdb_ckd):
    absdb_ckd.cache_reset(2)
    assert absdb_ckd._fname_dataset_cache.currsize == 0
    assert absdb_ckd._fname_dataset_cache.maxsize == 2
    absdb_ckd.cache_reset(8)
    assert absdb_ckd._fname_dataset_cache.currsize == 0
    assert absdb_ckd._fname_dataset_cache.maxsize == 8


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
    with pytest.raises(InterpolationError, match="Out-of-bounds"):
        if isinstance(absdb, MonoAbsorptionDatabase):
            absdb.eval_sigma_a_mono(
                w=350.0 * ureg.nm, thermoprops=thermoprops_us_standard
            )
        elif isinstance(absdb, CKDAbsorptionDatabase):
            absdb.eval_sigma_a_ckd(
                w=350.0 * ureg.nm, g=0.5, thermoprops=thermoprops_us_standard
            )
        else:
            raise AssertionError("unhandled case")


def _eval(absdb, thermoprops, config):
    """Helper to evaluate absorption with a given error handling config."""
    ehc = ErrorHandlingConfiguration.convert(config)
    if isinstance(absdb, MonoAbsorptionDatabase):
        return absdb.eval_sigma_a_mono(
            w=350.0 * ureg.nm,
            thermoprops=thermoprops,
            error_handling_config=ehc,
        )
    elif isinstance(absdb, CKDAbsorptionDatabase):
        return absdb.eval_sigma_a_ckd(
            w=350.0 * ureg.nm,
            g=0.5,
            thermoprops=thermoprops,
            error_handling_config=ehc,
        )
    else:
        raise RuntimeError("unhandled case")


@pytest.mark.parametrize("absdb", ["mono", "ckd"], indirect=True)
def test_bounds_policies(absdb, thermoprops_us_standard):
    """
    Test OOB handling policies:

    * RAISE raises an exception;
    * WARN emits a warning and proceeds anyway;
    * IGNORE proceeds silently.
    """

    config_raise = {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "raise"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "raise"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    }
    config_warn = {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "warn"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "warn"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "warn"},
    }
    config_ignore = {
        "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "ignore"},
    }

    with pytest.raises(InterpolationError, match="Out-of-bounds"):
        _eval(absdb, thermoprops_us_standard, config_raise)

    with pytest.warns(UserWarning, match="Out-of-bounds"):
        result_warn = _eval(absdb, thermoprops_us_standard, config_warn)

    result_ignore = _eval(absdb, thermoprops_us_standard, config_ignore)

    np.testing.assert_array_equal(result_ignore.values, result_warn.values)


@pytest.mark.parametrize("absdb", ["mono", "ckd"], indirect=True)
def test_bounds_clamp_mode(absdb, thermoprops_us_standard):
    """
    Regression test: a bounds policy configured with mode="clamp" must
    actually clamp out-of-bounds queries to the grid bounds, rather than
    silently behaving like "fill" (or crashing). ``BoundsPolicy.mode`` is
    stored as a ``BoundsMode`` enum internally, which must be converted to
    its string value before reaching the interpolation layer.
    """
    fill_sentinel = -12345.0
    config_clamp = {
        "p": {"missing": "raise", "scalar": "raise", "bounds": {"mode": "clamp"}},
        "t": {"missing": "raise", "scalar": "raise", "bounds": {"mode": "clamp"}},
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": {"mode": "clamp"}},
    }
    config_fill = {
        "p": {
            "missing": "raise",
            "scalar": "raise",
            "bounds": {"mode": "fill", "fill_value": fill_sentinel},
        },
        "t": {
            "missing": "raise",
            "scalar": "raise",
            "bounds": {"mode": "fill", "fill_value": fill_sentinel},
        },
        "x": {
            "missing": "ignore",
            "scalar": "ignore",
            "bounds": {"mode": "fill", "fill_value": fill_sentinel},
        },
    }

    result_clamp = _eval(absdb, thermoprops_us_standard, config_clamp)
    result_fill = _eval(absdb, thermoprops_us_standard, config_fill)

    # Sanity check: this profile does have out-of-bounds altitudes against
    # the tiny test databases, otherwise this test would vacuously pass.
    assert np.any(result_fill.values == fill_sentinel)
    # Clamping must never leak the raw fill sentinel.
    assert not np.any(result_clamp.values == fill_sentinel)
