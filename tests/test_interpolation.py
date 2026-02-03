"""
Tests for axsdb.interpolation module.
"""

import numpy as np
import pytest
import xarray as xr

from axsdb.interpolation import interp_dataarray


class TestInterpDataarrayBasic:
    """Basic interpolation tests."""

    def test_single_dimension(self):
        """Test interpolation along a single dimension."""
        da = xr.DataArray(
            np.array([0.0, 1.0, 4.0, 9.0]),
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0, 3.0]},
        )

        result = interp_dataarray(da, {"x": np.array([0.5, 1.5, 2.5])})

        expected = np.array([0.5, 2.5, 6.5])
        np.testing.assert_allclose(result.values, expected)
        assert result.dims == ("x",)
        np.testing.assert_array_equal(result.coords["x"].values, [0.5, 1.5, 2.5])

    def test_multiple_dimensions_sequential(self):
        """Test sequential interpolation across multiple dimensions."""
        # Create 2D data
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.outer(x, y)  # Shape: (3, 4)

        da = xr.DataArray(
            data,
            dims=["x", "y"],
            coords={"x": x, "y": y},
        )

        result = interp_dataarray(
            da,
            {
                "x": np.array([0.5, 1.5]),
                "y": np.array([0.5, 1.5, 2.5]),
            },
        )

        assert result.shape == (2, 3)
        assert result.dims == ("x", "y")

        # Check expected values (outer product interpolated)
        # At x=0.5, y=0.5: should be 0.5 * 0.5 = 0.25
        np.testing.assert_allclose(result.values[0, 0], 0.25)
        # At x=1.5, y=2.5: should be 1.5 * 2.5 = 3.75
        np.testing.assert_allclose(result.values[1, 2], 3.75)

    def test_preserves_name(self):
        """Test that DataArray name is preserved."""
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
            name="my_variable",
        )

        result = interp_dataarray(da, {"x": np.array([0.5])})

        assert result.name == "my_variable"

    def test_preserves_attrs(self):
        """Test that DataArray attributes are preserved."""
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
            attrs={"units": "meters", "description": "test data"},
        )

        result = interp_dataarray(da, {"x": np.array([0.5])})

        assert result.attrs == {"units": "meters", "description": "test data"}

    def test_preserves_non_interpolated_coords(self):
        """Test that non-interpolated coordinates are preserved."""
        da = xr.DataArray(
            np.arange(12).reshape(3, 4).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            },
        )

        # Only interpolate x, keep y
        result = interp_dataarray(da, {"x": np.array([0.5, 1.5])})

        assert "y" in result.coords
        np.testing.assert_array_equal(result.coords["y"].values, [0.0, 1.0, 2.0, 3.0])

    def test_scalar_interpolation(self):
        """Test interpolation to a scalar value removes the dimension."""
        da = xr.DataArray(
            np.arange(12).reshape(3, 4).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            },
        )

        result = interp_dataarray(da, {"x": 0.5})

        assert result.dims == ("y",)
        assert result.shape == (4,)


class TestInterpDataarrayBoundsDict:
    """Tests for dict-based bounds configuration."""

    def test_different_bounds_per_dimension(self):
        """Test different bounds modes for different dimensions."""
        da = xr.DataArray(
            np.arange(12).reshape(3, 4).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            },
        )

        # x uses fill (default NaN), y uses clamp
        result = interp_dataarray(
            da,
            {
                "x": np.array([-0.5, 0.5]),  # -0.5 is out of bounds
                "y": np.array([-0.5, 1.5]),  # -0.5 is out of bounds
            },
            bounds={"x": "fill", "y": "clamp"},
        )

        # x=-0.5 should produce NaN (fill mode)
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[0, 1])

        # y=-0.5 with clamp should use y=0 value
        # At x=0.5, y clamped to 0: interpolated x gives (0+4)/2 = 2, clamped y=0
        assert not np.isnan(result.values[1, 0])

    def test_bounds_raise_with_dimension_context(self):
        """Test that raise mode includes dimension in error message."""
        da = xr.DataArray(
            np.arange(12).reshape(3, 4).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            },
        )

        with pytest.raises(ValueError) as exc_info:
            interp_dataarray(
                da,
                {"y": np.array([5.0])},  # Out of bounds
                bounds={"y": "raise"},
            )

        assert "'y'" in str(exc_info.value)

    def test_missing_bounds_defaults_to_fill(self):
        """Test that missing dimension in bounds dict defaults to fill."""
        da = xr.DataArray(
            np.arange(12).reshape(3, 4).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            },
        )

        # Only specify bounds for x, y should default to fill
        result = interp_dataarray(
            da,
            {
                "x": np.array([0.5]),
                "y": np.array([-0.5]),  # Out of bounds
            },
            bounds={"x": "clamp"},
        )

        # y should use fill (NaN by default)
        assert np.isnan(result.values[0, 0])


class TestInterpDataarrayFillValueDict:
    """Tests for dict-based fill_value configuration."""

    def test_different_fill_values_per_dimension(self):
        """Test different fill values for different dimensions."""
        da = xr.DataArray(
            np.arange(12).reshape(3, 4).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0, 3.0],
            },
        )

        result = interp_dataarray(
            da,
            {
                "x": np.array([-0.5]),  # Out of bounds
            },
            bounds="fill",
            fill_value={"x": -999.0},
        )

        # Result shape is (1, 4) - x is interpolated to 1 point, y remains 4 points
        # Dimension order is preserved as ["x", "y"]
        assert result.shape == (1, 4)
        assert result.dims == ("x", "y")
        np.testing.assert_array_equal(result.values, [[-999.0, -999.0, -999.0, -999.0]])

    def test_tuple_fill_value_per_dimension(self):
        """Test tuple (lower, upper) fill values per dimension."""
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
        )

        result = interp_dataarray(
            da,
            {"x": np.array([-1.0, 0.5, 3.0])},
            bounds="fill",
            fill_value={"x": (-100.0, 100.0)},
        )

        np.testing.assert_allclose(result.values, [-100.0, 1.5, 100.0])

    def test_missing_fill_value_defaults_to_nan(self):
        """Test that missing dimension in fill_value dict defaults to NaN."""
        da = xr.DataArray(
            np.arange(6).reshape(2, 3).astype(float),
            dims=["x", "y"],
            coords={
                "x": [0.0, 1.0],
                "y": [0.0, 1.0, 2.0],
            },
        )

        result = interp_dataarray(
            da,
            {
                "x": np.array([-0.5]),  # Out of bounds
            },
            bounds="fill",
            fill_value={"y": -999.0},  # Only specify y, not x
        )

        # x should use NaN (default)
        assert np.all(np.isnan(result.values))


class TestInterpDataarrayErrors:
    """Tests for error handling."""

    def test_missing_dimension_error(self):
        """Test that missing dimension raises informative error."""
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
        )

        with pytest.raises(ValueError) as exc_info:
            interp_dataarray(da, {"z": np.array([0.5])})

        assert "'z'" in str(exc_info.value)
        assert "not found" in str(exc_info.value)
        assert "['x']" in str(exc_info.value)


class TestInterpDataarrayRealWorldScenario:
    """Tests mimicking real-world AxsDB usage patterns."""

    def test_spectral_angle_interpolation(self):
        """Test interpolation pattern similar to AxsDB spectral/angle data."""
        # Create data similar to absorption coefficient arrays
        wavelengths = np.linspace(400, 700, 31)  # 31 wavelengths
        angles = np.linspace(0, 90, 19)  # 19 angles
        temperatures = np.linspace(200, 300, 11)  # 11 temperatures

        # Random data with shape (wavelength, angle, temperature)
        np.random.seed(42)
        data = np.random.rand(31, 19, 11) * 1e-20

        da = xr.DataArray(
            data,
            dims=["wavelength", "angle", "temperature"],
            coords={
                "wavelength": wavelengths,
                "angle": angles,
                "temperature": temperatures,
            },
            attrs={"units": "m^2"},
        )

        # Interpolate to new grid (typical AxsDB operation)
        new_wavelengths = np.array([450.0, 550.0, 650.0])
        new_angles = np.array([30.0, 60.0])
        new_temps = np.array([250.0, 275.0])

        result = interp_dataarray(
            da,
            {
                "wavelength": new_wavelengths,
                "angle": new_angles,
                "temperature": new_temps,
            },
        )

        assert result.shape == (3, 2, 2)
        assert result.dims == ("wavelength", "angle", "temperature")
        assert result.attrs["units"] == "m^2"

        # Verify coordinates
        np.testing.assert_array_equal(
            result.coords["wavelength"].values, new_wavelengths
        )
        np.testing.assert_array_equal(result.coords["angle"].values, new_angles)
        np.testing.assert_array_equal(result.coords["temperature"].values, new_temps)

    def test_strict_wavelength_permissive_angles(self):
        """Test strict validation for wavelength, permissive for angles."""
        wavelengths = np.linspace(400, 700, 31)
        angles = np.linspace(0, 90, 19)

        np.random.seed(42)
        data = np.random.rand(31, 19)

        da = xr.DataArray(
            data,
            dims=["wavelength", "angle"],
            coords={
                "wavelength": wavelengths,
                "angle": angles,
            },
        )

        # Wavelength in range, angle out of range (should be clamped)
        result = interp_dataarray(
            da,
            {
                "wavelength": np.array([450.0, 550.0]),
                "angle": np.array([-10.0, 45.0, 100.0]),  # -10 and 100 out of range
            },
            bounds={
                "wavelength": "raise",
                "angle": "clamp",
            },
        )

        assert result.shape == (2, 3)
        # Verify clamped angle values work (no NaN)
        assert not np.any(np.isnan(result.values))

        # Now test that out-of-range wavelength raises
        with pytest.raises(ValueError) as exc_info:
            interp_dataarray(
                da,
                {
                    "wavelength": np.array([350.0]),  # Out of range
                    "angle": np.array([45.0]),
                },
                bounds={"wavelength": "raise", "angle": "clamp"},
            )

        assert "'wavelength'" in str(exc_info.value)


class TestInterpDataarraySharedDimension:
    """Tests for shared-dimension (pointwise) interpolation path."""

    def test_shared_dimension_basic(self):
        """Test interpolation with DataArray coords sharing a dimension."""
        # Create data with dimensions (x, y, z)
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 10.0, 20.0, 30.0])

        # Data: simple pattern for verification
        data = np.zeros((3, 2, 4))
        for i in range(3):
            for j in range(2):
                for k in range(4):
                    data[i, j, k] = i * 100 + j * 10 + k

        da = xr.DataArray(
            data,
            dims=["x", "y", "z"],
            coords={"x": x, "y": y, "z": z},
        )

        # Interpolate x using coordinates that vary with z
        # This triggers shared-dimension path since new_coords shares 'z' with da
        x_new_values = np.array([0.5, 0.5, 0.5, 0.5])  # Shape (4,) - varies with z
        x_new = xr.DataArray(
            x_new_values,
            dims=["z"],
            coords={"z": z},
        )

        result = interp_dataarray(da, {"x": x_new})

        # Result should have dims (y, z) - x is removed
        assert "x" not in result.dims
        assert result.shape == (2, 4)
        # Verify interpolation worked (x=0.5 should give average of x=0 and x=1)
        # At y=0, z=0: average of 0 and 100 = 50
        np.testing.assert_allclose(result.values[0, 0], 50.0)

    def test_shared_dimension_with_clamp(self):
        """Test shared-dimension path with clamp bounds mode."""
        # Create 3D data
        t = np.array([200.0, 250.0, 300.0])
        p = np.array([1000.0, 500.0, 100.0])
        z = np.array([0.0, 10.0, 20.0])

        data = np.random.rand(3, 3, 3)
        da = xr.DataArray(
            data,
            dims=["t", "p", "z"],
            coords={"t": t, "p": p, "z": z},
        )

        # Interpolate t to values that vary with z, some out of bounds
        t_new_values = np.array([180.0, 225.0, 320.0])  # Below, in, above range
        t_new = xr.DataArray(
            t_new_values,
            dims=["z"],
            coords={"z": z},
        )

        result = interp_dataarray(da, {"t": t_new}, bounds={"t": "clamp"})

        # Should not have NaN (clamped)
        assert not np.any(np.isnan(result.values))
        assert result.dims == ("p", "z")

    def test_shared_dimension_with_fill(self):
        """Test shared-dimension path with fill bounds mode."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])

        data = np.arange(12).reshape(3, 4).astype(float)
        da = xr.DataArray(
            data,
            dims=["x", "y"],
            coords={"x": x, "y": y},
        )

        # Interpolate x with coords that vary with y, including OOB
        x_new_values = np.array([-0.5, 0.5, 1.5, 3.0])  # First and last OOB
        x_new = xr.DataArray(
            x_new_values,
            dims=["y"],
            coords={"y": y},
        )

        result = interp_dataarray(
            da, {"x": x_new}, bounds={"x": "fill"}, fill_value={"x": -999.0}
        )

        # First and last y-slices should have fill value
        assert result.values[0] == -999.0
        assert result.values[-1] == -999.0
        # Middle values should be interpolated
        assert result.values[1] != -999.0
        assert result.values[2] != -999.0

    def test_shared_dimension_tuple_fill_values(self):
        """Test shared-dimension path with different lower/upper fill values."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])

        data = np.arange(9).reshape(3, 3).astype(float)
        da = xr.DataArray(
            data,
            dims=["x", "y"],
            coords={"x": x, "y": y},
        )

        # OOB on both sides
        x_new_values = np.array([-1.0, 1.0, 3.0])
        x_new = xr.DataArray(
            x_new_values,
            dims=["y"],
            coords={"y": y},
        )

        result = interp_dataarray(
            da,
            {"x": x_new},
            bounds={"x": "fill"},
            fill_value={"x": (-100.0, 100.0)},
        )

        # Check fill values
        assert result.values[0] == -100.0  # Below lower bound
        assert result.values[-1] == 100.0  # Above upper bound
        # Middle should be actual value
        assert result.values[1] != -100.0 and result.values[1] != 100.0


class TestInterpDataarrayEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_point_dimension(self):
        """Test interpolation on a dimension with only one point."""
        da = xr.DataArray(
            np.array([[1.0], [2.0]]),
            dims=["x", "y"],
            coords={"x": [0.0, 1.0], "y": [0.0]},  # y has single point
        )

        # Interpolate x dimension to an array
        result = interp_dataarray(da, {"x": np.array([0.5])})

        assert result.shape == (1, 1)
        assert result.dims == ("x", "y")
        # Should interpolate between 1.0 and 2.0 -> 1.5
        np.testing.assert_allclose(result.values, [[1.5]])

    def test_many_dimensions(self):
        """Test interpolation with 5+ dimensions."""
        dims = ["a", "b", "c", "d", "e"]
        shape = (3, 4, 5, 2, 3)
        coords = {
            "a": np.linspace(0, 1, 3),
            "b": np.linspace(0, 1, 4),
            "c": np.linspace(0, 1, 5),
            "d": np.linspace(0, 1, 2),
            "e": np.linspace(0, 1, 3),
        }

        data = np.random.rand(*shape)
        da = xr.DataArray(data, dims=dims, coords=coords)

        # Interpolate multiple dimensions
        # Note: scalar interpolation (e=0.5) keeps the dimension with size 1
        result = interp_dataarray(
            da,
            {
                "a": np.array([0.25, 0.75]),
                "c": np.array([0.2, 0.4, 0.6]),
                "e": 0.5,  # Scalar - removes dimension
            },
        )

        assert result.shape == (2, 4, 3, 2)
        assert result.dims == ("a", "b", "c", "d")

    def test_multiple_scalar_interpolations(self):
        """Test removing multiple dimensions via scalar interpolation."""
        da = xr.DataArray(
            np.arange(24).reshape(2, 3, 4).astype(float),
            dims=["x", "y", "z"],
            coords={
                "x": [0.0, 1.0],
                "y": [0.0, 1.0, 2.0],
                "z": [0.0, 1.0, 2.0, 3.0],
            },
        )

        # Interpolate to scalars on x and z
        result = interp_dataarray(da, {"x": 0.5, "z": 1.5})

        # Only y dimension should remain
        assert result.dims == ("y",)
        assert result.shape == (3,)

    def test_dimension_order_independence(self):
        """Test that coord order doesn't affect final result."""
        da = xr.DataArray(
            np.arange(24).reshape(2, 3, 4).astype(float),
            dims=["x", "y", "z"],
            coords={
                "x": [0.0, 1.0],
                "y": [0.0, 1.0, 2.0],
                "z": [0.0, 1.0, 2.0, 3.0],
            },
        )

        # Two different orders
        result1 = interp_dataarray(
            da,
            {"x": np.array([0.5]), "y": np.array([0.5, 1.5]), "z": np.array([1.5])},
        )

        result2 = interp_dataarray(
            da,
            {"z": np.array([1.5]), "y": np.array([0.5, 1.5]), "x": np.array([0.5])},
        )

        # Results should be identical (though dims might be reordered)
        # Transpose result2 to match result1's dimension order
        result2_reordered = result2.transpose(*result1.dims)
        np.testing.assert_allclose(result1.values, result2_reordered.values)

    def test_nan_in_input_data(self):
        """Test that NaN in input data propagates correctly."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])

        da = xr.DataArray(
            data,
            dims=["x", "y"],
            coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        )

        result = interp_dataarray(da, {"x": np.array([0.5, 1.5])})

        # NaN should propagate through interpolation
        # At x=0.5, y=1: should interpolate between 2.0 and NaN -> NaN
        # At x=1.5, y=1: should interpolate between NaN and 8.0 -> NaN
        assert np.isnan(result.values[0, 1])
        assert np.isnan(result.values[1, 1])

        # Non-NaN values should interpolate normally
        assert not np.isnan(result.values[0, 0])  # x=0.5, y=0
        assert not np.isnan(result.values[0, 2])  # x=0.5, y=2

    def test_all_dimensions_interpolated(self):
        """Test interpolating all dimensions (complete replacement)."""
        da = xr.DataArray(
            np.arange(6).reshape(2, 3).astype(float),
            dims=["x", "y"],
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
        )

        result = interp_dataarray(
            da,
            {
                "x": np.array([0.25, 0.75]),
                "y": np.array([0.5, 1.5]),
            },
        )

        assert result.shape == (2, 2)
        assert result.dims == ("x", "y")
        # Check that interpolation worked
        assert not np.any(np.isnan(result.values))


class TestInterpDataarrayFastPath:
    """Tests for scipy.interpn fast path with uniform bounds modes."""

    def test_fast_path_uniform_fill(self):
        """Test that fast path activates with uniform fill bounds mode."""
        # Create 3D data (t, p, x) where t and p will both map to z
        t_grid = np.array([200.0, 250.0, 300.0])
        p_grid = np.array([100.0, 500.0, 1000.0])
        x_grid = np.array([10.0, 20.0, 30.0])

        data = np.random.rand(len(t_grid), len(p_grid), len(x_grid))
        da = xr.DataArray(
            data,
            dims=["t", "p", "x"],
            coords={"t": t_grid, "p": p_grid, "x": x_grid},
        )

        # Both t and p map to z (shared destination)
        z_coords = np.array([100.0, 200.0, 300.0])
        t_new = xr.DataArray([225.0, 275.0, 225.0], dims=["z"], coords={"z": z_coords})
        p_new = xr.DataArray([300.0, 700.0, 800.0], dims=["z"], coords={"z": z_coords})

        # All dims use "fill" mode → should use fast path
        result = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds="fill",
        )

        # Verify shape and dimensions
        assert result.shape == (len(z_coords), len(x_grid))
        assert result.dims == ("z", "x")
        assert "z" in result.coords
        np.testing.assert_array_equal(result.coords["z"].values, z_coords)

    def test_fast_path_different_fill_values(self):
        """Test fast path with uniform bounds mode but different fill values."""
        # Create 3D data
        t_grid = np.array([200.0, 250.0, 300.0])
        p_grid = np.array([100.0, 500.0, 1000.0])
        x_grid = np.array([10.0, 20.0])

        data = np.random.rand(len(t_grid), len(p_grid), len(x_grid))
        da = xr.DataArray(
            data,
            dims=["t", "p", "x"],
            coords={"t": t_grid, "p": p_grid, "x": x_grid},
        )

        # Create out-of-bounds query points
        z_coords = np.array([100.0, 200.0])
        t_new = xr.DataArray(
            [150.0, 250.0], dims=["z"], coords={"z": z_coords}
        )  # 150 is OOB low
        p_new = xr.DataArray(
            [300.0, 1100.0], dims=["z"], coords={"z": z_coords}
        )  # 1100 is OOB high

        # Both use "fill" mode but different fill values
        result = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds="fill",
            fill_value={"t": -100.0, "p": -200.0},
        )

        # First z point: t is OOB low, p is in bounds
        assert result.values[0, 0] == -100.0

        # Second z point: t is in bounds, p is OOB high
        assert result.values[1, 0] == -200.0

    def test_fallback_mixed_bounds(self):
        """Test that mixed bounds modes trigger sequential fallback."""
        # Create 3D data
        t_grid = np.array([200.0, 250.0, 300.0])
        p_grid = np.array([100.0, 500.0, 1000.0])
        x_grid = np.array([10.0, 20.0])

        data = np.random.rand(len(t_grid), len(p_grid), len(x_grid))
        da = xr.DataArray(
            data,
            dims=["t", "p", "x"],
            coords={"t": t_grid, "p": p_grid, "x": x_grid},
        )

        # Create query points
        z_coords = np.array([100.0, 200.0])
        t_new = xr.DataArray([225.0, 275.0], dims=["z"], coords={"z": z_coords})
        p_new = xr.DataArray([300.0, 700.0], dims=["z"], coords={"z": z_coords})

        # t uses "fill", p uses "clamp" → fallback to sequential
        result_mixed = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds={"t": "fill", "p": "clamp"},
        )

        # Compare with uniform bounds (should give same result for in-bounds points)
        result_uniform = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds="fill",
        )

        # Results should be very close (within numerical tolerance)
        np.testing.assert_allclose(
            result_mixed.values, result_uniform.values, rtol=1e-10
        )

    def test_fast_path_asymmetric_fill(self):
        """Test fast path with asymmetric (lower, upper) fill values."""
        # Create 2D data
        t_grid = np.array([200.0, 250.0, 300.0])
        p_grid = np.array([100.0, 500.0, 1000.0])

        data = np.random.rand(len(t_grid), len(p_grid))
        da = xr.DataArray(
            data,
            dims=["t", "p"],
            coords={"t": t_grid, "p": p_grid},
        )

        # Create query points with OOB on both sides
        z_coords = np.array([0.0, 100.0, 200.0, 300.0])
        t_new = xr.DataArray(
            [150.0, 225.0, 275.0, 350.0], dims=["z"], coords={"z": z_coords}
        )  # 150 low, 350 high
        p_new = xr.DataArray(
            [50.0, 300.0, 700.0, 1100.0], dims=["z"], coords={"z": z_coords}
        )  # 50 low, 1100 high

        # Use tuple (fill_lo, fill_hi) for fill_value
        result = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds="fill",
            fill_value={
                "t": (-100.0, 100.0),
                "p": (-200.0, 200.0),
            },
        )

        # First point: both dimensions OOB low
        assert result.values[0] == -200.0  # p applied last

        # Last point: both dimensions OOB high
        assert result.values[3] == 200.0  # p applied last

    def test_fast_path_clamp_mode(self):
        """Test fast path with uniform clamp bounds mode."""
        # Create 2D data
        t_grid = np.array([200.0, 250.0, 300.0])
        p_grid = np.array([100.0, 500.0, 1000.0])

        data = np.arange(len(t_grid) * len(p_grid)).reshape(len(t_grid), len(p_grid))
        da = xr.DataArray(
            data,
            dims=["t", "p"],
            coords={"t": t_grid, "p": p_grid},
        )

        # Create OOB query points
        z_coords = np.array([100.0, 200.0])
        t_new = xr.DataArray(
            [150.0, 350.0], dims=["z"], coords={"z": z_coords}
        )  # Both OOB
        p_new = xr.DataArray([300.0, 700.0], dims=["z"], coords={"z": z_coords})

        # Both use "clamp" mode
        result = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds="clamp",
        )

        # Verify no NaN values (clamping should prevent them)
        assert not np.any(np.isnan(result.values))

        # Verify shape
        assert result.shape == (len(z_coords),)

    def test_fast_path_raise_mode(self):
        """Test fast path with uniform raise bounds mode."""
        # Create 2D data
        t_grid = np.array([200.0, 250.0, 300.0])
        p_grid = np.array([100.0, 500.0, 1000.0])

        data = np.random.rand(len(t_grid), len(p_grid))
        da = xr.DataArray(
            data,
            dims=["t", "p"],
            coords={"t": t_grid, "p": p_grid},
        )

        # Create in-bounds query points (should succeed)
        z_coords = np.array([100.0, 200.0])
        t_new = xr.DataArray([225.0, 275.0], dims=["z"], coords={"z": z_coords})
        p_new = xr.DataArray([300.0, 700.0], dims=["z"], coords={"z": z_coords})

        # Should not raise with in-bounds points
        result = interp_dataarray(
            da,
            {"t": t_new, "p": p_new},
            bounds="raise",
        )
        assert result.shape == (len(z_coords),)

        # Create OOB query points (should raise)
        t_oob = xr.DataArray(
            [150.0, 275.0], dims=["z"], coords={"z": z_coords}
        )  # 150 is OOB

        with pytest.raises(ValueError, match="Interpolation error on dimension 't'"):
            interp_dataarray(
                da,
                {"t": t_oob, "p": p_new},
                bounds="raise",
            )
