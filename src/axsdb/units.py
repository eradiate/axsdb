"""
Unit handling components, based on the `Pint <https://github.com/hgrecco/pint>`__
library.

.. note::
    By default,
    `Pint's application registry <https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html#having-a-shared-registry>`__
    is used.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pint
import xarray as xr

# Internal unit registry. If None, use application registry
_ureg: pint.UnitRegistry | None = None


def set_unit_registry(ureg: pint.UnitRegistry | None) -> None:
    """
    Set internal unit registry.

    Parameters
    ----------
    ureg : pint.UnitRegistry or None
        New default unit registry. Set to ``None`` to reset to the default
        (the application registry).
    """
    global _ureg
    _ureg = ureg
    _parse_units_cached.cache_clear()


@cache
def _parse_units_cached(unit_str: str) -> pint.Quantity:
    return get_unit_registry()(unit_str)


def parse_units(value: str | pint.Quantity) -> pint.Quantity:
    """
    Parse a unit string into a unit-only :class:`pint.Quantity` (magnitude
    1), caching the result. This matches what ``get_unit_registry()(unit_str)``
    returns, but avoids re-parsing the same string repeatedly.

    Pint's unit-string parsing (tokenizing, name/symbol lookup) is
    comparatively expensive, so repeatedly parsing the same string (e.g. a
    NetCDF variable's ``units`` attribute, read on every call in a hot loop)
    is wasteful. This caches the result globally, keyed on the string.

    .. warning::
        This cache is only invalidated by :func:`set_unit_registry`. If you
        swap the active registry via Pint's own
        :func:`pint.set_application_registry` instead of
        :func:`set_unit_registry`, this cache keeps returning quantities
        bound to the previous registry. Always call :func:`set_unit_registry`
        when changing the registry used with this package.

    Parameters
    ----------
    value : str or pint.Quantity
        Unit string to parse, or an already-parsed unit quantity (returned
        unchanged).

    Returns
    -------
    pint.Quantity
        The parsed unit quantity.
    """
    if isinstance(value, pint.Quantity):
        return value
    return _parse_units_cached(value)


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
