"""
Realistic thermophysical profile interpolation.

This benchmark mimics the actual AxsDB use case: interpolating absorption
coefficients from a (x_CO2, x_H2O, x_O3, p, t, g) grid to a vertical profile of
z altitude levels.
"""

import numpy as np
import xarray as xr
import pytest
from axsdb.interpolation import interp_dataarray
from scipy.special import roots_sh_legendre


Z_LEVELS = [121, 1201, 12001]


@pytest.fixture(scope="module", params=Z_LEVELS)
def setup(request):
    rng = np.random.default_rng(seed=42)
    z_levels = request.param

    # Create synthetic thermoprops dataset
    thermoprops = xr.Dataset(
        {
            "t": ("z", np.linspace(210, 330, z_levels)),
            "p": ("z", np.logspace(np.log10(200), np.log10(50000), z_levels)),
            "x_H2O": (
                "z",
                np.logspace(np.log10(1.0e-6), np.log10(9.0e-5), z_levels),
            ),
            "x_O3": (
                "z",
                np.logspace(np.log10(5.0e-10), np.log10(7.84e-6), z_levels),
            ),
            "x_CO2": ("z", np.linspace(3.5e-5, 3.3e-4, z_levels)),
        },
        coords={"z": np.linspace(0, 120, z_levels)},
    )

    # Create synthetic sigma_a DataArray
    da = xr.DataArray(
        rng.uniform(0, 1, (2, 4, 2, 16, 8, 32)),
        dims=["x_CO2", "x_H2O", "x_O3", "p", "t", "g"],
        coords={
            "x_CO2": [0.0, 5.0e-4],
            "x_H2O": [0.0, 1.0e-6, 1.0e-5, 1.0e-4],
            "x_O3": [0.0, 1.0e-5],
            "p": np.logspace(2, 5, 16),
            "t": np.linspace(200, 340, 8),
            "g": np.linspace(0, 1, 32),
        },
    )

    # Build interpolation coordinate list
    dims_coords = [
        ("g", roots_sh_legendre(16)[0]),
        ("t", thermoprops["t"]),
        ("p", thermoprops["p"]),
        ("x_H2O", thermoprops["x_H2O"]),
        ("x_O3", thermoprops["x_O3"]),
        ("x_CO2", thermoprops["x_CO2"]),
    ]

    interp_kwargs = {
        "method": "linear",
        "assume_sorted": True,
        "kwargs": {"bounds_error": False, "fill_value": None},
    }

    yield {
        "thermoprops": thermoprops,
        "da": da,
        "dims_coords": dims_coords,
        "interp_kwargs": interp_kwargs,
    }


class BenchmarkInterpDataArrayThermophysical:
    def time_custom(self, da, dims_coords):
        coords_dict = {}
        for dim, coords in dims_coords:
            if dim in da.dims:
                if isinstance(coords, xr.DataArray):
                    coords_dict[dim] = coords
                else:
                    coords_dict[dim] = np.asarray(coords)

        interp_dataarray(da, coords_dict, bounds="clamp")

    def benchmark_time_custom(self, setup, benchmark):
        benchmark(self.time_custom, setup["da"], setup["dims_coords"])

    def time_xarray_sequential(self, da, dims_coords, interp_kwargs):
        result = da
        for dim, coords in dims_coords:
            if dim in result.dims:
                result = result.interp({dim: coords}, **interp_kwargs)

    def benchmark_time_xarray_sequential(self, setup, benchmark):
        benchmark(
            self.time_xarray_sequential,
            setup["da"],
            setup["dims_coords"],
            setup["interp_kwargs"],
        )

    def time_xarray_multi(self, da, dims_coords, interp_kwargs):
        da.interp(dict(dims_coords), **interp_kwargs)

    def benchmark_time_xarray_multi(self, setup, benchmark):
        benchmark(
            self.time_xarray_multi,
            setup["da"],
            setup["dims_coords"],
            setup["interp_kwargs"],
        )
