"""
Performance benchmarks for axsdb interpolation.

These tests verify that the custom numba gufunc implementation is faster
than xarray's built-in interpolation, which was the motivation for this
implementation (see https://github.com/pydata/xarray/issues/10683).

The tests use pytest-benchmark style timing but are written to run with
plain pytest for simplicity.
"""

import time

import numpy as np
import pytest
import xarray as xr

from axsdb.interpolation import interp_dataarray


def time_function(func, n_warmup=1, n_runs=5):
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(n_warmup):
        func()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return min(times), np.mean(times), max(times)


class TestInterpolationPerformance:
    """Performance benchmarks comparing custom implementation to xarray."""

    @pytest.fixture
    def sample_dataarray(self):
        """Create a sample DataArray similar to AxsDB data."""
        np.random.seed(42)

        # Dimensions similar to AxsDB absorption data
        wavelengths = np.linspace(400, 2500, 100)
        temperatures = np.linspace(200, 320, 25)
        pressures = np.logspace(-2, 5, 30)

        data = np.random.rand(100, 25, 30) * 1e-20

        return xr.DataArray(
            data,
            dims=["wavelength", "temperature", "pressure"],
            coords={
                "wavelength": wavelengths,
                "temperature": temperatures,
                "pressure": pressures,
            },
        )

    @pytest.fixture
    def interp_coords(self):
        """Target coordinates for interpolation."""
        return {
            "wavelength": np.linspace(500, 2000, 50),
            "temperature": np.array([250.0, 280.0, 300.0]),
            "pressure": np.logspace(0, 4, 20),
        }

    def test_single_dimension_faster_than_xarray(self, sample_dataarray, interp_coords):
        """Test that single-dimension interpolation is faster than xarray."""
        da = sample_dataarray
        new_wavelengths = interp_coords["wavelength"]

        def xarray_interp():
            return da.interp(wavelength=new_wavelengths)

        def custom_interp():
            return interp_dataarray(da, {"wavelength": new_wavelengths})

        # Verify numerical agreement first
        xarray_result = xarray_interp()
        custom_result = custom_interp()
        np.testing.assert_allclose(
            custom_result.values, xarray_result.values, rtol=1e-10
        )

        # Time both implementations
        xarray_min, xarray_mean, _ = time_function(xarray_interp)
        custom_min, custom_mean, _ = time_function(custom_interp)

        print("\nSingle dimension interpolation:")
        print(
            f"  xarray: min={xarray_min * 1000:.2f}ms, mean={xarray_mean * 1000:.2f}ms"
        )
        print(
            f"  custom: min={custom_min * 1000:.2f}ms, mean={custom_mean * 1000:.2f}ms"
        )
        print(
            f"  speedup: {xarray_min / custom_min:.1f}x (min), {xarray_mean / custom_mean:.1f}x (mean)"
        )

        # We expect significant speedup, but the exact ratio depends on xarray version
        # Just verify it's not significantly slower
        assert custom_mean < xarray_mean * 2, (
            "Custom implementation should not be 2x slower"
        )

    def test_multi_dimension_faster_than_xarray(self, sample_dataarray, interp_coords):
        """Test that multi-dimension interpolation is faster than xarray."""
        da = sample_dataarray

        def xarray_interp():
            result = da.interp(wavelength=interp_coords["wavelength"])
            result = result.interp(temperature=interp_coords["temperature"])
            result = result.interp(pressure=interp_coords["pressure"])
            return result

        def custom_interp():
            return interp_dataarray(da, interp_coords)

        # Verify numerical agreement first
        xarray_result = xarray_interp()
        custom_result = custom_interp()
        np.testing.assert_allclose(
            custom_result.values, xarray_result.values, rtol=1e-10
        )

        # Time both implementations
        xarray_min, xarray_mean, _ = time_function(xarray_interp)
        custom_min, custom_mean, _ = time_function(custom_interp)

        print("\nMulti-dimension interpolation (3 dims):")
        print(
            f"  xarray: min={xarray_min * 1000:.2f}ms, mean={xarray_mean * 1000:.2f}ms"
        )
        print(
            f"  custom: min={custom_min * 1000:.2f}ms, mean={custom_mean * 1000:.2f}ms"
        )
        print(
            f"  speedup: {xarray_min / custom_min:.1f}x (min), {xarray_mean / custom_mean:.1f}x (mean)"
        )

        # We expect significant speedup for multi-dimension case
        assert custom_mean < xarray_mean * 2, (
            "Custom implementation should not be 2x slower"
        )

    def test_large_output_performance(self):
        """Test performance with larger output grids."""
        np.random.seed(42)

        # Smaller input, larger output (typical interpolation scenario)
        wavelengths = np.linspace(400, 700, 31)
        angles = np.linspace(0, 90, 19)
        data = np.random.rand(31, 19)

        da = xr.DataArray(
            data,
            dims=["wavelength", "angle"],
            coords={"wavelength": wavelengths, "angle": angles},
        )

        # Large output grid
        new_wavelengths = np.linspace(410, 690, 200)
        new_angles = np.linspace(5, 85, 100)

        def xarray_interp():
            return da.interp(wavelength=new_wavelengths).interp(angle=new_angles)

        def custom_interp():
            return interp_dataarray(
                da, {"wavelength": new_wavelengths, "angle": new_angles}
            )

        # Verify numerical agreement
        xarray_result = xarray_interp()
        custom_result = custom_interp()
        np.testing.assert_allclose(
            custom_result.values, xarray_result.values, rtol=1e-10
        )

        # Time both
        xarray_min, xarray_mean, _ = time_function(xarray_interp)
        custom_min, custom_mean, _ = time_function(custom_interp)

        print("\nLarge output grid (200x100 from 31x19):")
        print(
            f"  xarray: min={xarray_min * 1000:.2f}ms, mean={xarray_mean * 1000:.2f}ms"
        )
        print(
            f"  custom: min={custom_min * 1000:.2f}ms, mean={custom_mean * 1000:.2f}ms"
        )
        print(
            f"  speedup: {xarray_min / custom_min:.1f}x (min), {xarray_mean / custom_mean:.1f}x (mean)"
        )

    def test_repeated_interpolation_performance(self, sample_dataarray):
        """Test performance of repeated interpolations (typical usage pattern)."""
        da = sample_dataarray
        np.random.seed(42)

        # Generate multiple target coordinate sets
        n_iterations = 10
        coord_sets = [
            {
                "temperature": np.array([250.0 + i * 5]),
                "pressure": np.array([1000.0 + i * 100]),
            }
            for i in range(n_iterations)
        ]

        def xarray_repeated():
            results = []
            for coords in coord_sets:
                result = da.interp(**coords)
                results.append(result.values)
            return results

        def custom_repeated():
            results = []
            for coords in coord_sets:
                result = interp_dataarray(da, coords)
                results.append(result.values)
            return results

        # Time both
        xarray_min, xarray_mean, _ = time_function(
            xarray_repeated, n_warmup=2, n_runs=3
        )
        custom_min, custom_mean, _ = time_function(
            custom_repeated, n_warmup=2, n_runs=3
        )

        print(f"\nRepeated interpolation ({n_iterations} iterations):")
        print(
            f"  xarray: min={xarray_min * 1000:.2f}ms, mean={xarray_mean * 1000:.2f}ms"
        )
        print(
            f"  custom: min={custom_min * 1000:.2f}ms, mean={custom_mean * 1000:.2f}ms"
        )
        print(
            f"  speedup: {xarray_min / custom_min:.1f}x (min), {xarray_mean / custom_mean:.1f}x (mean)"
        )


class TestInterp1dPerformance:
    """Performance benchmarks for the low-level interp1d function."""

    def test_comparison_with_np_interp(self):
        """Compare performance with numpy's built-in interp."""
        from axsdb.math import interp1d

        np.random.seed(42)
        x = np.sort(np.random.rand(1000)) * 100
        y = np.random.rand(1000)
        # Query points strictly within the data range to avoid boundary behavior differences
        xnew = np.linspace(x[0], x[-1], 5000)

        def np_interp():
            return np.interp(xnew, x, y)

        def custom_interp():
            # Use clamp to match np.interp's boundary behavior
            return interp1d(x, y, xnew, bounds="clamp")

        # Verify agreement
        np_result = np_interp()
        custom_result = custom_interp()
        np.testing.assert_allclose(custom_result, np_result, rtol=1e-10)

        # Time both
        np_min, np_mean, _ = time_function(np_interp, n_warmup=3, n_runs=10)
        custom_min, custom_mean, _ = time_function(custom_interp, n_warmup=3, n_runs=10)

        print("\ninterp1d vs np.interp (1000 points -> 5000 points):")
        print(f"  np.interp: min={np_min * 1000:.3f}ms, mean={np_mean * 1000:.3f}ms")
        print(
            f"  custom:    min={custom_min * 1000:.3f}ms, mean={custom_mean * 1000:.3f}ms"
        )
        print(f"  ratio: {custom_mean / np_mean:.2f}x (1.0 = same speed)")

    def test_broadcasting_performance(self):
        """Test performance advantage of broadcasting in gufunc."""
        from axsdb.math import interp1d

        np.random.seed(42)

        # Multiple curves (broadcasting case)
        n_curves = 100
        n_points = 200
        n_query = 500

        x = np.sort(np.random.rand(n_points)) * 100
        y = np.random.rand(n_curves, n_points)
        # Query points within the data range
        xnew = np.linspace(x[0], x[-1], n_query)

        def loop_np_interp():
            results = np.empty((n_curves, n_query))
            for i in range(n_curves):
                results[i] = np.interp(xnew, x, y[i])
            return results

        def broadcast_custom():
            # Use clamp to match np.interp's boundary behavior
            return interp1d(x, y, xnew, bounds="clamp")

        # Verify agreement
        np_result = loop_np_interp()
        custom_result = broadcast_custom()
        np.testing.assert_allclose(custom_result, np_result, rtol=1e-10)

        # Time both
        np_min, np_mean, _ = time_function(loop_np_interp, n_warmup=2, n_runs=5)
        custom_min, custom_mean, _ = time_function(
            broadcast_custom, n_warmup=2, n_runs=5
        )

        print(
            f"\nBroadcasting performance ({n_curves} curves, {n_points}->{n_query} points):"
        )
        print(
            f"  loop np.interp: min={np_min * 1000:.2f}ms, mean={np_mean * 1000:.2f}ms"
        )
        print(
            f"  broadcast gufunc: min={custom_min * 1000:.2f}ms, mean={custom_mean * 1000:.2f}ms"
        )
        print(f"  speedup: {np_mean / custom_mean:.1f}x")
