"""Pandera xarray schemas for axsdb dataset formats."""

from __future__ import annotations

import numpy as np
import pandera.xarray as pa
import xarray as xr
from pandera import check, dataframe_check
from pandera.typing.xarray import Coordinate


class AcMonov1Dataset(pa.DatasetModel):
    """Pandera schema for Ac-Mono v1 absorption coefficient datasets.

    Validates the structure, dtypes, and physical value ranges of NetCDF files
    storing monochromatic absorption coefficients as a function of wavelength,
    pressure, temperature, and species mole fractions, as produced by the
    shicho tool for use in the Eradiate radiative transfer model.

    Notes
    -----
    Species mole fraction coordinates (``x_<species>``, e.g. ``x_O3``,
    ``x_H2O``) vary by dataset and cannot be declared as fixed fields.
    Their presence and correct association with ``sigma_a`` are enforced by
    custom checks.
    """

    # --- Coordinates ---

    p: Coordinate[np.float32] = pa.Field(
        ge=0.0,
        title="Air pressure [Pa]",
        description="Pressure grid points.",
    )
    t: Coordinate[np.float32] = pa.Field(
        ge=0.0,
        title="Air temperature [K]",
        description="Temperature grid points.",
    )
    w: Coordinate[np.float64] = pa.Field(
        ge=0.0,
        title="Wavelength [nm]",
        description="Monochromatic wavelength grid.",
    )

    # --- Data variables ---

    sigma_a: np.float32 = pa.Field(
        ge=0.0,
        nullable=False,
        title="Volume absorption coefficient [km^-1]",
        description=(
            "Absorption coefficient as a function of species mole fraction, "
            "pressure, temperature, and wavelength."
        ),
    )

    # --- Custom checks ---

    @check("sigma_a")
    @classmethod
    def sigma_a_trailing_dims(cls, da: xr.DataArray) -> bool:
        """Trailing dimensions of sigma_a must be (p, t, w)."""
        return da.dims[-3:] == ("p", "t", "w")

    @check("sigma_a")
    @classmethod
    def sigma_a_has_species_dim(cls, da: xr.DataArray) -> bool:
        """sigma_a must have at least one leading x_<species> dimension."""
        return any(d.startswith("x_") for d in da.dims[:-3])

    @dataframe_check
    @classmethod
    def has_species_coordinates(cls, ds: xr.Dataset) -> bool:
        """Dataset must contain at least one x_<species> mole fraction coordinate."""
        return any(str(c).startswith("x_") for c in ds.coords)

    class Config:
        name = "AcMonov1Dataset"
        strict = False  # allow extra data variables
        strict_coords = False  # allow x_<species> coordinates


class AcCKDv1Dataset(pa.DatasetModel):
    """Pandera schema for Ac-CKD v1 absorption coefficient datasets.

    Validates the structure, dtypes, and physical value ranges of NetCDF files
    storing band-averaged absorption coefficients with correlated-k (CKD)
    quadrature points, as produced by the sabaki tool for use in the Eradiate
    radiative transfer model.

    Notes
    -----
    Species mole fraction coordinates (``x_<species>``, e.g. ``x_O3``,
    ``x_H2O``) vary by dataset and cannot be declared as fixed fields.
    Their presence and correct association with ``sigma_a`` are enforced by
    custom checks.
    """

    # --- Coordinates ---

    p: Coordinate[np.float32] = pa.Field(
        ge=0.0,
        title="Air pressure [Pa]",
        description="Pressure grid points.",
    )
    t: Coordinate[np.float32] = pa.Field(
        ge=0.0,
        title="Air temperature [K]",
        description="Temperature grid points.",
    )
    w: Coordinate[np.float32] = pa.Field(
        ge=0.0,
        title="Central wavelength [nm]",
        description="Spectral bin central wavelength.",
    )
    g: Coordinate[np.float32] = pa.Field(
        ge=0.0,
        le=1.0,
        title="g-point quantile",
        description="Cumulative probability quantile for CKD quadrature, range [0, 1].",
    )
    ng: Coordinate[np.int64] = pa.Field(
        ge=1,
        title="Quadrature point count",
        description="Number of Gauss-Legendre quadrature g-points.",
    )
    wbv: Coordinate[str] = pa.Field(
        isin=["lower", "upper"],
        title="Wavelength bound label",
        description='String label for spectral bin bounds: "lower" or "upper".',
    )

    # --- Data variables ---

    sigma_a: np.float64 = pa.Field(
        ge=0.0,
        nullable=False,
        title="Volume absorption coefficient [km^-1]",
        description=(
            "Absorption coefficient as a function of species mole fraction, "
            "pressure, temperature, wavelength, and g-point."
        ),
    )
    wbounds: np.float64 = pa.Field(
        ge=0.0,
        nullable=False,
        dims=("wbv", "w"),
        title="Wavelength bounds [nm]",
        description="Lower and upper wavelength boundaries of each spectral bin.",
    )
    error: np.float64 = pa.Field(
        ge=0.0,
        nullable=False,
        dims=("w", "ng"),
        title="Relative transmittance error [1]",
        description=(
            "Relative error in transmittance for the US Standard atmosphere, "
            "tabulated against the number of Gauss-Legendre quadrature points."
        ),
    )

    # --- Custom checks ---

    @check("sigma_a")
    @classmethod
    def sigma_a_trailing_dims(cls, da: xr.DataArray) -> bool:
        """Trailing dimensions of sigma_a must be (p, t, w, g)."""
        return da.dims[-4:] == ("p", "t", "w", "g")

    @check("sigma_a")
    @classmethod
    def sigma_a_has_species_dim(cls, da: xr.DataArray) -> bool:
        """sigma_a must have at least one leading x_<species> dimension."""
        return any(d.startswith("x_") for d in da.dims[:-4])

    @dataframe_check
    @classmethod
    def has_species_coordinates(cls, ds: xr.Dataset) -> bool:
        """Dataset must contain at least one x_<species> mole fraction coordinate."""
        return any(str(c).startswith("x_") for c in ds.coords)

    class Config:
        name = "AcCKDv1Dataset"
        strict = False  # allow extra data variables
        strict_coords = False  # allow x_<species> coordinates
