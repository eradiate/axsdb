"""
Benchmark comparing custom Numba gufunc vs scipy alternatives for 1D interpolation.

This benchmark reproduces the analysis documented in
docs/interpolation-implementation.md that justifies the custom gufunc
implementation for 1D linear interpolation.

Run correctness tests:
    uv run pytest tests/benchmarks/test_scipy_alternatives.py -v

Run performance benchmarks (standalone script):
    uv run python tests/benchmarks/test_scipy_alternatives.py
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator, interpn

from axsdb.math import interp1d


class TestScipyAlternatives:
    """Compare custom gufunc correctness against scipy alternatives."""

    @pytest.fixture
    def single_curve_data(self):
        """Single curve interpolation data."""
        np.random.seed(42)
        n_grid = 100
        n_query = 1000

        x = np.linspace(0, 10, n_grid)
        y = np.random.randn(n_grid)
        xnew = np.sort(np.random.uniform(0, 10, n_query))

        return x, y, xnew

    @pytest.fixture
    def multiple_curves_data(self):
        """Multiple curves for batched interpolation."""
        np.random.seed(42)
        n_grid = 50
        n_query = 100
        n_batch = 50

        x = np.linspace(0, 10, n_grid)
        y = np.random.randn(n_batch, n_grid)
        xnew = np.sort(np.random.uniform(0, 10, n_query))

        return x, y, xnew

    @pytest.fixture
    def atmospheric_profile_data(self):
        """Realistic atmospheric profile interpolation."""
        np.random.seed(42)
        n_p = 50
        n_t = 30
        n_z = 121

        p_grid = np.logspace(-3, 3, n_p)
        sigma_a = np.random.lognormal(0, 1, (n_t, n_p))
        p_query = np.logspace(-3, 3, n_z)

        return p_grid, sigma_a, p_query

    @pytest.fixture
    def high_dimensional_data(self):
        """High-dimensional broadcasting case."""
        np.random.seed(42)
        n_w = 5
        n_t = 10
        n_x = 8
        n_z = 121
        n_p = 30

        # Data shape: (w, t, x, z, p) - p on last axis for gufunc
        data_5d = np.random.randn(n_w, n_t, n_x, n_z, n_p)
        p_grid = np.linspace(0, 100, n_p)
        p_query = np.linspace(0, 100, 15)

        return p_grid, data_5d, p_query

    def test_correctness_single_curve(self, single_curve_data):
        """Verify results match between custom gufunc and scipy.interpn."""
        x, y, xnew = single_curve_data
        xi = xnew.reshape(-1, 1)

        result_scipy = interpn(
            (x,), y, xi, method="linear", bounds_error=False, fill_value=np.nan
        )
        result_gufunc = interp1d(x, y, xnew, bounds="fill", fill_value=np.nan)

        np.testing.assert_allclose(result_scipy, result_gufunc, rtol=1e-10, atol=1e-10)

    def test_correctness_multiple_curves(self, multiple_curves_data):
        """Verify results match for batched interpolation."""
        x, y, xnew = multiple_curves_data
        xi = xnew.reshape(-1, 1)
        n_batch = y.shape[0]

        results_scipy = np.empty((n_batch, len(xnew)))
        for i in range(n_batch):
            results_scipy[i] = interpn(
                (x,), y[i], xi, method="linear", bounds_error=False, fill_value=np.nan
            )

        results_gufunc = interp1d(x, y, xnew, bounds="fill", fill_value=np.nan)

        np.testing.assert_allclose(
            results_scipy, results_gufunc, rtol=1e-10, atol=1e-10
        )

    def test_correctness_high_dimensional(self, high_dimensional_data):
        """Verify results match for high-dimensional broadcasting."""
        p_grid, data_5d, p_query = high_dimensional_data
        n_w, n_t, n_x, n_z, _ = data_5d.shape
        xi = p_query.reshape(-1, 1)

        results_scipy = np.empty((n_w, n_t, n_x, n_z, len(p_query)))
        for iw in range(n_w):
            for it in range(n_t):
                for ix in range(n_x):
                    for iz in range(n_z):
                        results_scipy[iw, it, ix, iz, :] = interpn(
                            (p_grid,),
                            data_5d[iw, it, ix, iz, :],
                            xi,
                            method="linear",
                            bounds_error=False,
                            fill_value=np.nan,
                        )

        results_gufunc = interp1d(p_grid, data_5d, p_query, bounds="fill")

        np.testing.assert_allclose(
            results_scipy, results_gufunc, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_out_of_bounds_handling(self):
        """Test different out-of-bounds modes."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])
        xnew_oob = np.array([-1.0, 1.5, 5.0])

        # Fill mode
        result_fill = interp1d(
            x, y, xnew_oob, bounds="fill", fill_value=(-999.0, 999.0)
        )
        expected_fill = np.array([-999.0, 2.5, 999.0])
        np.testing.assert_allclose(result_fill, expected_fill)

        # Clamp mode
        result_clamp = interp1d(x, y, xnew_oob, bounds="clamp")
        expected_clamp = np.array([0.0, 2.5, 9.0])
        np.testing.assert_allclose(result_clamp, expected_clamp)

        # Raise mode
        with pytest.raises(ValueError, match="Query points out of bounds"):
            interp1d(x, y, xnew_oob, bounds="raise")

    def test_regular_grid_interpolator_comparison(self, single_curve_data):
        """Compare with RegularGridInterpolator (for documentation)."""
        x, y, xnew = single_curve_data
        xi = xnew.reshape(-1, 1)

        # RGI requires creating interpolator object
        rgi = RegularGridInterpolator(
            (x,), y, method="linear", bounds_error=False, fill_value=np.nan
        )
        result_rgi = rgi(xi)

        result_gufunc = interp1d(x, y, xnew, bounds="fill", fill_value=np.nan)

        np.testing.assert_allclose(result_rgi, result_gufunc, rtol=1e-10, atol=1e-10)


class TestNumpyInterpComparison:
    """Test that custom gufunc matches numpy.interp for clamp mode."""

    def test_matches_numpy_interp(self):
        """Verify custom gufunc with clamp mode matches numpy.interp."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.random.randn(50)
        xnew = np.array([-1.0, 0.0, 5.0, 10.0, 11.0])

        result_numpy = np.interp(xnew, x, y)
        result_gufunc = interp1d(x, y, xnew, bounds="clamp")

        np.testing.assert_allclose(result_numpy, result_gufunc, rtol=1e-14)

    def test_batched_comparison(self):
        """Verify batched interpolation matches numpy.interp."""
        np.random.seed(42)
        n_batch = 20
        x = np.linspace(0, 10, 50)
        y = np.random.randn(n_batch, 50)
        xnew = np.sort(np.random.uniform(-1, 11, 100))

        # Manual loop with numpy.interp
        results_numpy = np.empty((n_batch, len(xnew)))
        for i in range(n_batch):
            results_numpy[i] = np.interp(xnew, x, y[i])

        # Custom gufunc
        results_gufunc = interp1d(x, y, xnew, bounds="clamp")

        np.testing.assert_allclose(
            results_numpy, results_gufunc, rtol=1e-12, atol=1e-15
        )


# =============================================================================
# Standalone performance benchmark script
# =============================================================================


def run_benchmark(name, func, n_iters=5, warmup=1):
    """Time a function over multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)

    return np.median(times)


def main():
    """Run standalone performance benchmarks."""
    print("=" * 80)
    print("scipy alternatives benchmark")
    print("=" * 80)
    print()

    # Test 1: Single curve
    print("Test 1: Single curve (100 grid → 1000 query)")
    print("-" * 80)
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.random.randn(100)
    xnew = np.sort(np.random.uniform(0, 10, 1000))
    xi = xnew.reshape(-1, 1)

    t_scipy = run_benchmark(
        "scipy.interpn",
        lambda: interpn(
            (x,), y, xi, method="linear", bounds_error=False, fill_value=np.nan
        ),
    )
    t_gufunc = run_benchmark(
        "custom gufunc", lambda: interp1d(x, y, xnew, bounds="fill", fill_value=np.nan)
    )
    t_numpy = run_benchmark("numpy.interp", lambda: np.interp(xnew, x, y))

    print(f"  scipy.interpn:  {t_scipy * 1e3:6.3f} ms")
    print(f"  custom gufunc:  {t_gufunc * 1e3:6.3f} ms")
    print(f"  numpy.interp:   {t_numpy * 1e3:6.3f} ms")
    print(f"  Speedup (scipy): {t_scipy / t_gufunc:.1f}x")
    print()

    # Test 2: Multiple curves
    print("Test 2: Multiple curves (50 × 50 grid → 100 query)")
    print("-" * 80)
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = np.random.randn(50, 50)
    xnew = np.sort(np.random.uniform(0, 10, 100))
    xi = xnew.reshape(-1, 1)

    def scipy_loop():
        results = np.empty((50, 100))
        for i in range(50):
            results[i] = interpn(
                (x,), y[i], xi, method="linear", bounds_error=False, fill_value=np.nan
            )
        return results

    def numpy_loop():
        results = np.empty((50, 100))
        for i in range(50):
            results[i] = np.interp(xnew, x, y[i])
        return results

    t_scipy = run_benchmark("scipy.interpn loop", scipy_loop)
    t_gufunc = run_benchmark(
        "custom gufunc", lambda: interp1d(x, y, xnew, bounds="fill", fill_value=np.nan)
    )
    t_numpy = run_benchmark("numpy.interp loop", numpy_loop)

    print(f"  scipy.interpn:  {t_scipy * 1e3:6.3f} ms")
    print(f"  custom gufunc:  {t_gufunc * 1e3:6.3f} ms")
    print(f"  numpy.interp:   {t_numpy * 1e3:6.3f} ms")
    print(f"  Speedup (scipy): {t_scipy / t_gufunc:.1f}x")
    print()

    # Test 3: Atmospheric profile
    print("Test 3: Atmospheric profile (30 × 50 grid → 121 query)")
    print("-" * 80)
    np.random.seed(42)
    p_grid = np.logspace(-3, 3, 50)
    sigma_a = np.random.lognormal(0, 1, (30, 50))
    p_query = np.logspace(-3, 3, 121)
    xi = p_query.reshape(-1, 1)

    def scipy_atmo():
        results = np.empty((30, 121))
        for i in range(30):
            results[i] = interpn(
                (p_grid,),
                sigma_a[i],
                xi,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        return results

    t_scipy = run_benchmark("scipy.interpn loop", scipy_atmo)
    t_gufunc = run_benchmark(
        "custom gufunc", lambda: interp1d(p_grid, sigma_a, p_query, bounds="clamp")
    )

    print(f"  scipy.interpn:  {t_scipy * 1e3:6.3f} ms")
    print(f"  custom gufunc:  {t_gufunc * 1e3:6.3f} ms")
    print(f"  Speedup: {t_scipy / t_gufunc:.1f}x")
    print()

    # Test 4: High-dimensional broadcasting
    print("Test 4: High-dimensional broadcasting (5×10×8×121×30 → 15)")
    print("-" * 80)
    np.random.seed(42)
    n_w, n_t, n_x, n_z, n_p = 5, 10, 8, 121, 30
    data_5d = np.random.randn(n_w, n_t, n_x, n_z, n_p)
    p_grid = np.linspace(0, 100, n_p)
    p_query = np.linspace(0, 100, 15)
    xi = p_query.reshape(-1, 1)

    def scipy_5d():
        results = np.empty((n_w, n_t, n_x, n_z, 15))
        for iw in range(n_w):
            for it in range(n_t):
                for ix in range(n_x):
                    for iz in range(n_z):
                        results[iw, it, ix, iz, :] = interpn(
                            (p_grid,),
                            data_5d[iw, it, ix, iz, :],
                            xi,
                            method="linear",
                            bounds_error=False,
                            fill_value=np.nan,
                        )
        return results

    t_scipy = run_benchmark("scipy.interpn nested loops", scipy_5d, n_iters=5, warmup=1)
    t_gufunc = run_benchmark(
        "custom gufunc", lambda: interp1d(p_grid, data_5d, p_query, bounds="fill")
    )

    print(f"  scipy.interpn:  {t_scipy * 1e3:7.1f} ms")
    print(f"  custom gufunc:  {t_gufunc * 1e3:7.3f} ms")
    print(f"  Speedup: {t_scipy / t_gufunc:.0f}x")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("The custom Numba gufunc provides:")
    print("  • 4-20x speedup over scipy.interpn for single/multiple curves")
    print("  • 300+x speedup for high-dimensional broadcasting")
    print("  • Equivalent to numpy.interp for single curves")
    print("  • Native broadcasting that scipy.interpn cannot provide")
    print()
    print("The hybrid architecture (custom gufunc for 1D, scipy.interpn for multi-D)")
    print("maximizes performance across all use cases.")


if __name__ == "__main__":
    main()
