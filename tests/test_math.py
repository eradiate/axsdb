"""
Tests for axsdb.math module.
"""

import numpy as np
import pytest

from axsdb.math import interp1d, lerp, lerp_indices


class TestInterp1dBasic:
    """Basic interpolation tests."""

    def test_simple_linear(self):
        """Test simple linear interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        xnew = np.array([0.5, 1.5, 2.5])

        result = interp1d(x, y, xnew)

        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_quadratic_data(self):
        """Test interpolation on quadratic data."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x**2
        xnew = np.array([0.5, 1.5, 2.5, 3.5])

        result = interp1d(x, y, xnew)

        # Linear interpolation of quadratic, so not exact
        expected = np.array([0.5, 2.5, 6.5, 12.5])
        np.testing.assert_allclose(result, expected)

    def test_exact_match(self):
        """Test that exact data points are returned without numerical error."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        xnew = np.array([0.0, 1.0, 2.0, 3.0])

        result = interp1d(x, y, xnew)

        np.testing.assert_array_equal(result, y)

    def test_comparison_with_np_interp(self):
        """Test that results match np.interp for in-bounds points."""
        np.random.seed(42)
        x = np.sort(np.random.rand(20)) * 10
        y = np.random.rand(20) * 100
        xnew = np.linspace(x[0], x[-1], 50)

        result = interp1d(x, y, xnew)
        expected = np.interp(xnew, x, y)

        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_single_query_point(self):
        """Test with a single query point."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        xnew = np.array([0.5])

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result, [0.5])


class TestInterp1dBounds:
    """Tests for bounds handling modes."""

    def test_fill_default_nan(self):
        """Test default fill mode with NaN."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, 0.5, 3.0])

        result = interp1d(x, y, xnew, bounds="fill")

        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 15.0)
        assert np.isnan(result[2])

    def test_fill_scalar_value(self):
        """Test fill mode with a scalar fill value."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, 0.5, 3.0])

        result = interp1d(x, y, xnew, bounds="fill", fill_value=-999.0)

        np.testing.assert_allclose(result, [-999.0, 15.0, -999.0])

    def test_fill_tuple_value(self):
        """Test fill mode with different lower/upper fill values."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, 0.5, 3.0])

        result = interp1d(x, y, xnew, bounds="fill", fill_value=(-100.0, 100.0))

        np.testing.assert_allclose(result, [-100.0, 15.0, 100.0])

    def test_clamp_lower(self):
        """Test clamp mode for below-range points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, -0.5])

        result = interp1d(x, y, xnew, bounds="clamp")

        np.testing.assert_array_equal(result, [10.0, 10.0])

    def test_clamp_upper(self):
        """Test clamp mode for above-range points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([2.5, 3.0])

        result = interp1d(x, y, xnew, bounds="clamp")

        np.testing.assert_array_equal(result, [30.0, 30.0])

    def test_clamp_both(self):
        """Test clamp mode for both out-of-bounds directions."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, 1.0, 3.0])

        result = interp1d(x, y, xnew, bounds="clamp")

        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_raise_in_bounds(self):
        """Test raise mode with all points in bounds (should not raise)."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        result = interp1d(x, y, xnew, bounds="raise")

        np.testing.assert_allclose(result, [10.0, 15.0, 20.0, 25.0, 30.0])

    def test_raise_below(self):
        """Test raise mode raises for below-range points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-0.5, 0.5])

        with pytest.raises(ValueError, match="out of bounds"):
            interp1d(x, y, xnew, bounds="raise")

    def test_raise_above(self):
        """Test raise mode raises for above-range points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([0.5, 2.5])

        with pytest.raises(ValueError, match="out of bounds"):
            interp1d(x, y, xnew, bounds="raise")

    def test_raise_error_message_contains_delta(self):
        """Test that raise mode error message contains bound delta info."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-0.5, 2.5])

        with pytest.raises(ValueError) as exc_info:
            interp1d(x, y, xnew, bounds="raise")

        error_msg = str(exc_info.value)
        assert "Below lower bound" in error_msg or "Above upper bound" in error_msg

    def test_invalid_bounds_mode(self):
        """Test that invalid bounds mode raises ValueError."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([0.5])

        with pytest.raises(ValueError, match="Invalid bounds mode"):
            interp1d(x, y, xnew, bounds="invalid")  # type: ignore

    def test_invalid_fill_value_tuple_length(self):
        """Test that fill_value tuple with wrong length raises ValueError."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([0.5])

        with pytest.raises(ValueError, match="exactly 2 elements"):
            interp1d(x, y, xnew, fill_value=(1.0, 2.0, 3.0))  # type: ignore


class TestInterp1dNaN:
    """Tests for NaN handling."""

    def test_nan_in_query_points(self):
        """Test that NaN in query points produces NaN output."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([0.5, np.nan, 1.5])

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result[0], 15.0)
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[2], 25.0)

    def test_nan_passthrough_all_modes(self):
        """Test NaN passthrough works in all bounds modes."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([np.nan])

        for mode in ["fill", "clamp", "raise"]:
            result = interp1d(x, y, xnew, bounds=mode)
            assert np.isnan(result[0])

    def test_raise_with_nan_only(self):
        """Test that raise mode doesn't raise when only NaN query points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([np.nan, np.nan])

        # Should not raise
        result = interp1d(x, y, xnew, bounds="raise")
        assert np.all(np.isnan(result))


class TestInterp1dBroadcasting:
    """Tests for multidimensional broadcasting."""

    def test_2d_y_values(self):
        """Test broadcasting with 2D y values (multiple curves)."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array(
            [
                [0.0, 1.0, 2.0],  # Linear
                [0.0, 1.0, 4.0],  # Quadratic
            ]
        )
        xnew = np.array([0.5, 1.5])

        result = interp1d(x, y, xnew)

        expected = np.array(
            [
                [0.5, 1.5],
                [0.5, 2.5],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_3d_y_values(self):
        """Test broadcasting with 3D y values (typical AxsDB use case)."""
        x = np.array([0.0, 1.0, 2.0])
        # Shape: (2, 3, 3) - 2 wavelengths, 3 angles, 3 x-points
        y = np.arange(18).reshape(2, 3, 3).astype(float)
        xnew = np.array([0.5, 1.5])

        result = interp1d(x, y, xnew)

        # Result shape should be (2, 3, 2)
        assert result.shape == (2, 3, 2)

        # Check specific values
        # y[0, 0, :] = [0, 1, 2], interpolated at [0.5, 1.5] -> [0.5, 1.5]
        np.testing.assert_allclose(result[0, 0, :], [0.5, 1.5])

    def test_broadcasting_x_and_y(self):
        """Test with different x arrays per curve."""
        # Shape: (2, 3) - 2 curves with different x coordinates
        x = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0],
            ]
        )
        y = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0],
            ]
        )
        xnew = np.array(
            [
                [0.5, 1.5],
                [1.0, 3.0],
            ]
        )

        result = interp1d(x, y, xnew)

        expected = np.array(
            [
                [0.5, 1.5],
                [1.0, 3.0],
            ]
        )
        np.testing.assert_allclose(result, expected)


class TestInterp1dDtype:
    """Tests for dtype handling."""

    def test_float32_preservation(self):
        """Test that float32 dtype is used when inputs are float32."""
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        xnew = np.array([0.5, 1.5], dtype=np.float32)

        result = interp1d(x, y, xnew)

        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [0.5, 1.5], rtol=1e-6)

    def test_float64_default(self):
        """Test that float64 is used by default."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        xnew = np.array([0.5, 1.5])

        result = interp1d(x, y, xnew)

        assert result.dtype == np.float64

    def test_int_to_float_conversion(self):
        """Test that integer inputs are converted to float."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        xnew = np.array([0, 1])

        result = interp1d(x, y, xnew)

        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_mixed_dtype_promotion(self):
        """Test that mixed dtypes are promoted correctly."""
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        xnew = np.array([0.5, 1.5], dtype=np.float32)

        result = interp1d(x, y, xnew)

        # Should promote to float64 (highest precision)
        assert result.dtype == np.float64


class TestInterp1dEdgeCases:
    """Tests for edge cases."""

    def test_two_points(self):
        """Test interpolation with minimum number of points."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 10.0])
        xnew = np.array([0.0, 0.5, 1.0])

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result, [0.0, 5.0, 10.0])

    def test_many_query_points(self):
        """Test with many query points."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        xnew = np.linspace(0, 1, 10000)

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result, xnew, rtol=1e-14)

    def test_closely_spaced_points(self):
        """Test with closely spaced x coordinates."""
        x = np.array([0.0, 1e-10, 2e-10])
        y = np.array([0.0, 1.0, 2.0])
        xnew = np.array([0.5e-10, 1.5e-10])

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result, [0.5, 1.5], rtol=1e-6)

    def test_large_values(self):
        """Test with large coordinate values."""
        x = np.array([1e10, 2e10, 3e10])
        y = np.array([1.0, 2.0, 3.0])
        xnew = np.array([1.5e10, 2.5e10])

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result, [1.5, 2.5])

    def test_negative_values(self):
        """Test with negative coordinate values."""
        x = np.array([-3.0, -2.0, -1.0, 0.0])
        y = np.array([9.0, 4.0, 1.0, 0.0])
        xnew = np.array([-2.5, -1.5, -0.5])

        result = interp1d(x, y, xnew)

        np.testing.assert_allclose(result, [6.5, 2.5, 0.5])


class TestPrecomputeLerpIndices:
    """Tests for precompute_lerp_indices function."""

    def test_basic_indices_and_weights(self):
        """Test basic index and weight computation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        xnew = np.array([0.5, 1.5, 2.5])

        indices, weights = lerp_indices(x, xnew)

        # Check indices: 0.5 is in bin 0, 1.5 in bin 1, 2.5 in bin 2
        np.testing.assert_array_equal(indices, [0, 1, 2])

        # Check weights: all should be 0.5 (midpoint)
        np.testing.assert_allclose(weights, [0.5, 0.5, 0.5])

    def test_exact_grid_points(self):
        """Test precompute with exact grid points."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([0.0, 1.0, 2.0])

        indices, weights = lerp_indices(x, xnew)

        # Exact points: weight should be 0.0 or 1.0
        # xnew[0]=0.0 -> bin 0, weight 0.0
        # xnew[1]=1.0 -> bin 1, weight 0.0
        # xnew[2]=2.0 -> bin 1, weight 1.0
        expected_indices = [0, 1, 1]
        expected_weights = [0.0, 0.0, 1.0]

        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(weights, expected_weights)

    def test_bounds_fill(self):
        """Test bounds='fill' marks out-of-bounds with NaN weights."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([-1.0, 0.5, 3.0])

        indices, weights = lerp_indices(x, xnew, bounds="fill")

        # Out-of-bounds points should have NaN weights
        assert np.isnan(weights[0])
        np.testing.assert_allclose(weights[1], 0.5)
        assert np.isnan(weights[2])

        # Indices should be clamped to valid range [0, n-2]
        assert 0 <= indices[0] <= 1
        assert 0 <= indices[2] <= 1

    def test_bounds_clamp_lower(self):
        """Test bounds='clamp' for below-range points."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([-1.0, -0.5])

        indices, weights = lerp_indices(x, xnew, bounds="clamp")

        # Below range should clamp to bin 0 with weight 0.0
        np.testing.assert_array_equal(indices, [0, 0])
        np.testing.assert_array_equal(weights, [0.0, 0.0])

    def test_bounds_clamp_upper(self):
        """Test bounds='clamp' for above-range points."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([2.5, 3.0])

        indices, weights = lerp_indices(x, xnew, bounds="clamp")

        # Above range should clamp to last bin with weight 1.0
        np.testing.assert_array_equal(indices, [1, 1])
        np.testing.assert_array_equal(weights, [1.0, 1.0])

    def test_bounds_raise(self):
        """Test bounds='raise' raises for out-of-bounds points."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([-0.5, 0.5, 2.5])

        with pytest.raises(ValueError, match="out of bounds"):
            lerp_indices(x, xnew, bounds="raise")

    def test_bounds_raise_in_bounds(self):
        """Test bounds='raise' succeeds when all points in bounds."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        # Should not raise
        indices, weights = lerp_indices(x, xnew, bounds="raise")

        assert len(indices) == len(xnew)
        assert len(weights) == len(xnew)

    def test_single_query_point(self):
        """Test with single query point."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.array([0.5])

        indices, weights = lerp_indices(x, xnew)

        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_allclose(weights, [0.5])

    def test_many_query_points(self):
        """Test with many query points."""
        x = np.array([0.0, 1.0, 2.0])
        xnew = np.linspace(0.0, 2.0, 1000)

        indices, weights = lerp_indices(x, xnew)

        assert len(indices) == 1000
        assert len(weights) == 1000
        assert np.all((indices >= 0) & (indices <= 1))
        assert np.all((weights >= 0.0) & (weights <= 1.0))


class TestLerpPrecomputed:
    """Tests for lerp function."""

    def test_basic_interpolation(self):
        """Test basic lerp with precomputed indices."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        xnew = np.array([0.5, 1.5, 2.5])

        indices, weights = lerp_indices(x, xnew)
        result = lerp(y, indices, weights)

        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_comparison_with_interp1d(self):
        """Test that lerp matches interp1d results."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x**2
        xnew = np.array([0.5, 1.5, 2.5, 3.5])

        # Using interp1d
        result_interp = interp1d(x, y, xnew, bounds="clamp")

        # Using precompute + lerp
        indices, weights = lerp_indices(x, xnew, bounds="clamp")
        result_lerp = lerp(y, indices, weights)

        np.testing.assert_allclose(result_lerp, result_interp)

    def test_nan_weights_propagate(self):
        """Test that NaN weights (from bounds='fill') produce NaN output."""
        y = np.array([0.0, 1.0, 2.0])
        indices = np.array([0.0, 0.0, 1.0])
        weights = np.array([np.nan, 0.5, np.nan])  # OOB marked as NaN

        result = lerp(y, indices, weights)

        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 0.5)
        assert np.isnan(result[2])

    def test_broadcasting_2d_y(self):
        """Test broadcasting with 2D y values."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array(
            [
                [0.0, 1.0, 2.0],  # Linear
                [0.0, 1.0, 4.0],  # Quadratic
            ]
        )
        xnew = np.array([0.5, 1.5])

        indices, weights = lerp_indices(x, xnew)
        result = lerp(y, indices, weights)

        expected = np.array(
            [
                [0.5, 1.5],
                [0.5, 2.5],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_broadcasting_3d_y(self):
        """Test broadcasting with 3D y values."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.arange(18).reshape(2, 3, 3).astype(float)
        xnew = np.array([0.5, 1.5])

        indices, weights = lerp_indices(x, xnew)
        result = lerp(y, indices, weights)

        # Result shape should be (2, 3, 2)
        assert result.shape == (2, 3, 2)

        # Check specific values
        # y[0, 0, :] = [0, 1, 2], interpolated at [0.5, 1.5] -> [0.5, 1.5]
        np.testing.assert_allclose(result[0, 0, :], [0.5, 1.5])

    def test_exact_grid_points(self):
        """Test with exact grid points (weight 0.0 or 1.0)."""
        y = np.array([10.0, 20.0, 30.0])

        # xnew = [0.0, 1.0, 2.0] would give these indices/weights
        indices = np.array([0.0, 1.0, 1.0])
        weights = np.array([0.0, 0.0, 1.0])

        result = lerp(y, indices, weights)

        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])


class TestPrecomputeLerpWorkflow:
    """Tests for the combined precompute + lerp workflow."""

    def test_reuse_indices_multiple_y(self):
        """Test reusing precomputed indices for multiple y arrays."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        xnew = np.array([0.5, 1.5, 2.5])

        # Precompute once
        indices, weights = lerp_indices(x, xnew)

        # Apply to multiple y arrays
        y1 = np.array([0.0, 1.0, 2.0, 3.0])
        y2 = np.array([0.0, 2.0, 4.0, 6.0])
        y3 = np.array([0.0, 1.0, 4.0, 9.0])

        result1 = lerp(y1, indices, weights)
        result2 = lerp(y2, indices, weights)
        result3 = lerp(y3, indices, weights)

        np.testing.assert_allclose(result1, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(result2, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(result3, [0.5, 2.5, 6.5])

    def test_workflow_matches_interp1d_fill(self):
        """Test that workflow matches interp1d with bounds='fill'."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, 0.5, 1.5, 3.0])

        # Using interp1d
        result_interp = interp1d(x, y, xnew, bounds="fill", fill_value=-999.0)

        # Using workflow (note: precompute uses NaN, need to replace)
        indices, weights = lerp_indices(x, xnew, bounds="fill")
        result_lerp = lerp(y, indices, weights)
        result_lerp = np.where(np.isnan(result_lerp), -999.0, result_lerp)

        np.testing.assert_allclose(result_lerp, result_interp)

    def test_workflow_matches_interp1d_clamp(self):
        """Test that workflow matches interp1d with bounds='clamp'."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        xnew = np.array([-1.0, 0.5, 1.5, 3.0])

        # Using interp1d
        result_interp = interp1d(x, y, xnew, bounds="clamp")

        # Using workflow
        indices, weights = lerp_indices(x, xnew, bounds="clamp")
        result_lerp = lerp(y, indices, weights)

        np.testing.assert_allclose(result_lerp, result_interp)

    def test_atmospheric_profile_use_case(self):
        """Test realistic use case: interpolating atmospheric profiles."""
        # Pressure grid (high to low altitude)
        p_grid = np.array([1000.0, 800.0, 600.0, 400.0, 200.0])  # hPa

        # Target pressure levels
        p_target = np.array([900.0, 700.0, 500.0, 300.0])  # hPa

        # Multiple atmospheric variables at grid points
        temperature = np.array([288.0, 280.0, 270.0, 255.0, 220.0])  # K
        humidity = np.array([0.015, 0.010, 0.005, 0.002, 0.0001])  # kg/kg
        ozone = np.array([30.0, 40.0, 60.0, 80.0, 100.0])  # DU

        # Precompute once for all variables
        indices, weights = lerp_indices(p_grid, p_target, bounds="clamp")

        # Interpolate all variables
        temp_interp = lerp(temperature, indices, weights)
        humid_interp = lerp(humidity, indices, weights)
        ozone_interp = lerp(ozone, indices, weights)

        # Verify shapes
        assert temp_interp.shape == p_target.shape
        assert humid_interp.shape == p_target.shape
        assert ozone_interp.shape == p_target.shape

        # Verify values are reasonable (between grid bounds)
        assert np.all(temp_interp >= 220.0)
        assert np.all(temp_interp <= 288.0)
        assert np.all(humid_interp >= 0.0001)
        assert np.all(humid_interp <= 0.015)

    def test_batched_profiles(self):
        """Test with batched atmospheric profiles (3D use case)."""
        # Dimensions: (wavelength, angle, altitude)
        x = np.array([0.0, 10.0, 20.0, 30.0])  # altitude (km)
        y = np.random.rand(5, 8, 4)  # (5 wavelengths, 8 angles, 4 altitudes)
        xnew = np.array([5.0, 15.0, 25.0])  # target altitudes

        # Precompute
        indices, weights = lerp_indices(x, xnew)

        # Interpolate batched data
        result = lerp(y, indices, weights)

        # Check output shape
        assert result.shape == (5, 8, 3)

        # Verify no NaN values for in-bounds interpolation
        assert not np.any(np.isnan(result))
