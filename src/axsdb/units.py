from __future__ import annotations

from typing import Any
import xarray as xr
import pint

#: An alias to Pint's application registry.
ureg = pint.get_application_registry()


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
        If ``True``, ``value`` will also be converted to ``default_units`` if it is a
        :class:`pint.Quantity`.

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

    return ureg.Quantity(da.values, units)
