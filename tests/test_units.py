import pytest
import numpy as np
import pint
import xarray as xr
from axsdb import units


def test_unit_registry():
    ureg = pint.UnitRegistry()
    units.set_unit_registry(ureg)
    assert units.get_unit_registry() is ureg

    units.set_unit_registry(None)
    assert units.get_unit_registry() is pint.get_application_registry()


def test_ensure_units():
    ureg = units.get_unit_registry()

    result = units.ensure_units(1.0, default_units=ureg.m)
    assert result == 1.0 * ureg.m

    result = units.ensure_units(1.0 * ureg.km, default_units=ureg.m, convert=True)
    assert result.m == 1000.0
    assert result.u == ureg.m


def test_xarray_to_quantity():
    da = xr.DataArray(np.ones((3,)))

    with pytest.raises(
        ValueError, match="this DataArray has no 'units' attribute field"
    ):
        units.xarray_to_quantity(da)

    da.attrs.update({"units": "m"})
    ureg = units.get_unit_registry()
    np.testing.assert_array_equal(
        units.xarray_to_quantity(da), [1.0, 1.0, 1.0] * ureg.m
    )
