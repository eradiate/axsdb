"""
Benchmark comparing xarray interpolation methods with our custom implementation.

This demonstrates the performance regression in xarray 2025+ and how our
numba-based implementation addresses it.
"""

import time

import numpy as np
import xarray as xr

from scipy.special import roots_sh_legendre
from axsdb.interpolation import interp_dataarray

NRUNS = 5
NWARMUP = 1

print(f"xarray version: {xr.__version__}")

# Create synthetic thermoprops dataset (atmospheric profile with many altitude levels)
z_levels = 12001
thermoprops = xr.Dataset(
    {
        # Ranges kept strictly within da's coordinate grids so that the
        # xarray (extrapolating) vs custom (clamping) comparison is valid.
        "t": ("z", np.linspace(210, 330, z_levels)),  # Temperature (K), grid [200, 340]
        "p": (
            "z",
            np.logspace(np.log10(200), np.log10(50000), z_levels),
        ),  # Pressure (Pa), grid [100, 1e5]
        "x_H2O": (
            "z",
            np.logspace(np.log10(1.0e-6), np.log10(9.0e-5), z_levels),
        ),  # H2O, grid [0, 1e-4]
        "x_O3": (
            "z",
            np.logspace(np.log10(5.0e-10), np.log10(7.84e-6), z_levels),
        ),  # O3, grid [0, 1e-5]
        "x_CO2": ("z", np.linspace(3.5e-5, 3.3e-4, z_levels)),  # CO2, grid [0, 5e-4]
    },
    coords={"z": np.linspace(0, 120, z_levels)},  # Altitude (km)
)

# Create synthetic sigma_a data array (absorption coefficient from CKD database)
# Dimensions: (x_CO2, x_H2O, x_O3, p, t, g)
da = xr.DataArray(
    np.random.rand(2, 4, 2, 16, 8, 32),  # Random absorption coefficients
    dims=["x_CO2", "x_H2O", "x_O3", "p", "t", "g"],
    coords={
        "x_CO2": [0.0, 5.0e-4],
        "x_H2O": [0.0, 1.0e-6, 1.0e-5, 1.0e-4],
        "x_O3": [0.0, 1.0e-5],
        "p": np.logspace(2, 5, 16),  # 100 to 100,000 Pa
        "t": np.linspace(200, 340, 8),  # 200 to 340 K
        "g": np.linspace(0, 1, 32),  # g-points for quadrature
    },
)

# Build interpolation coordinate list
dims_coords = [
    ("g", roots_sh_legendre(16)[0]),
    ("t", thermoprops["t"]),
    ("p", thermoprops["p"]),
    *[
        (species_dim, thermoprops[species_dim])
        for species_dim in ["x_H2O", "x_O3", "x_CO2"]
    ],
]

interp_kwargs = {
    "method": "linear",
    "assume_sorted": True,
    "kwargs": {"bounds_error": False, "fill_value": None},
}


def time_function(func, n_warmup=NWARMUP, n_runs=NRUNS):
    """Time a function with warmup runs."""
    for _ in range(n_warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return min(times), np.mean(times), max(times)


def interp_xarray_multi():
    """xarray multi-dimension interpolation (all at once)."""
    return da.interp(dict(dims_coords), **interp_kwargs)


def interp_xarray_seq():
    """xarray sequential interpolation (one dimension at a time)."""
    result = da
    for dim, coords in dims_coords:
        if dim in result.dims:
            result = result.interp({dim: coords}, **interp_kwargs)
    return result


def interp_custom():
    """Our custom numba-based interpolation."""
    # Build coords dict, preserving DataArrays so shared dimensions (e.g. z)
    # are respected during interpolation.
    coords_dict = {}
    for dim, coords in dims_coords:
        if dim in da.dims:
            if isinstance(coords, xr.DataArray):
                coords_dict[dim] = coords
            else:
                coords_dict[dim] = np.asarray(coords)

    return interp_dataarray(
        da,
        coords_dict,
        bounds="clamp",  # Clamp out-of-bounds; xarray extrapolates with fill_value=None
    )


label_func = [
    ("multi", interp_xarray_multi),
    ("seq", interp_xarray_seq),
    ("custom", interp_custom),
]


print("\n" + "=" * 70)
print("BENCHMARK: xarray interpolation performance")
print("=" * 70)


print(f"\nInput shape: {da.shape}")
print(f"Input dims: {da.dims}")
print(f"Interpolation dimensions: {[d for d, _ in dims_coords]}")

# Check output shapes
print("\n--- Output shapes ---")
results = {}
for label, func in label_func:
    print(f"{label + ':':<7} ", end="", flush=True)
    results[label] = result = func()
    print(result.shape)

# Verify numerical agreement between sequential methods.
# Only compare where xarray did not extrapolate (finite in both).
# xarray uses fill_value=None (extrapolate) while custom uses clamp, so
# out-of-bounds points will legitimately differ.
if results["seq"] is not None and results["custom"] is not None:
    try:
        xr_vals = results["seq"].values
        cu_vals = results["custom"].values
        mask = np.isfinite(xr_vals) & np.isfinite(cu_vals)
        if mask.any():
            max_diff = float(np.max(np.abs(xr_vals[mask] - cu_vals[mask])))
            print(f"\nMax difference (seq vs custom, in-bounds): {max_diff:.2e}")
        else:
            print("\nNo in-bounds points to compare.")
    except Exception as e:
        print(f"\nCouldn't compare results: {e}")

# Run benchmarks
print(f"\n--- Timing ({NRUNS} runs, {NWARMUP} warmup) ---")

timings = {}

for label, func in label_func:
    print(f"{label + ':':<7} ", end="", flush=True)
    timings[label] = (t_min, t_mean, t_max) = time_function(func)
    print(
        f"min={t_min * 1000:.1f}ms, mean={t_mean * 1000:.1f}ms, max={t_max * 1000:.1f}ms"
    )

# Summary
print("\n--- Speedup Summary ---")
print(f"custom vs seq: {timings['seq'][0] / timings['custom'][0]:.1f}x faster")
print(f"custom vs multi: {timings['multi'][0] / timings['custom'][0]:.1f}x faster")

print("\n" + "=" * 70)
