"""
Unit handling components, based on the `Pint <https://github.com/hgrecco/pint>`__
library.

.. note::
    By default,
    `Pint's application registry <https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html#having-a-shared-registry>`__
    is used.
"""

from __future__ import annotations

from typing import Any
import xarray as xr
import pint

# Internal unit registry. If None, use application registry
_ureg: pint.UnitRegistry | None = None


def set_unit_registry(ureg: pint.UnitRegistry) -> None:
    """
    Set internal unit registry.

    Parameters
    ----------
    ureg : pint.UnitRegistry
        New default unit registry.
    """
    global _ureg
    _ureg = ureg


def get_unit_registry() -> pint.UnitRegistry:
    """
    Access the internal unit registry. By default, the Pint application registry
    is returned.
    """
    global _ureg
    return _ureg if _ureg is not None else pint.get_application_registry()


def ensure_units(
    value: Any, default_units: pint.Unit, convert: bool = False
) -> pint.Quantity:
    """
    Ensure that a value is wrapped in a Pint quantity container.

    Parameters
    ----------
    value
        Checked value.

    default_units : pint.Unit
        Units to use to initialize the :class:`pint.Quantity` if ``value`` is
        not a :class:`pint.Quantity`.

    convert : bool, default: False
        If ``True``, ``value`` will also be converted to ``default_units`` if it
        is a :class:`pint.Quantity`.

    Returns
    -------
    Converted ``value``.
    """
    if isinstance(value, pint.Quantity):
        if convert:
            return value.to(default_units)
        else:
            return value
    else:
        return value * default_units


def xarray_to_quantity(da: xr.DataArray) -> pint.Quantity:
    """
    Converts a :class:`~xarray.DataArray` to a :class:`~pint.Quantity`.
    The array's ``attrs`` metadata mapping must contain a ``units`` field.

    Parameters
    ----------
    da : DataArray
        :class:`~xarray.DataArray` instance which will be converted.

    Returns
    -------
    quantity
        The corresponding Pint quantity.

    Raises
    ------
    ValueError
        If array attributes do not contain a ``units`` field.

    Notes
    -----
    This function can also be used on coordinate variables.
    """
    try:
        units = da.attrs["units"]
    except KeyError as e:
        raise ValueError("this DataArray has no 'units' attribute field") from e

    ureg = get_unit_registry()
    return ureg.Quantity(da.values, units)
