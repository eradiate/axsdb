"""
Fast xarray interpolation.

This module provides high-performance interpolation functions for xarray
DataArrays that bypass xarray's built-in interpolation. This is motivated
by performance regressions in recent xarray versions (see
https://github.com/pydata/xarray/issues/10683).
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import numpy as np
import xarray as xr
from scipy.interpolate import interpn

from .math import interp1d, lerp, lerp_indices


def _apply_fill_values(
    data: np.ndarray,
    new_coords_arr: np.ndarray,
    old_coords_arr: np.ndarray,
    dim_fill_value: float | tuple[float, float],
) -> None:
    """
    Replace NaN values with specified fill values for out-of-bounds points.

    Modifies data in-place. Assumes NaN has already been set for OOB points
    by the interpolation functions.

    Parameters
    ----------
    data : ndarray
        Data array with NaN for out-of-bounds points. Modified in-place.

    new_coords_arr : ndarray
        Query coordinates

    old_coords_arr : ndarray
        Original grid coordinates

    dim_fill_value : float or tuple
        Fill value(s) to use. If tuple, (``fill_lower``, ``fill_upper``).
    """
    oob = (new_coords_arr < old_coords_arr[0]) | (new_coords_arr > old_coords_arr[-1])
    if not oob.any():
        return

    if isinstance(dim_fill_value, tuple):
        fill_lo, fill_hi = dim_fill_value
    else:
        fill_lo = fill_hi = dim_fill_value

    lo = new_coords_arr < old_coords_arr[0]
    hi = new_coords_arr > old_coords_arr[-1]

    # Apply fill values along the last axis
    data[..., lo] = fill_lo
    data[..., hi] = fill_hi


def _decompose_interp_groups(
    interp_specs: list[dict],
) -> list[list[dict]]:
    """
    Decompose interpolation specs into groups based on destination dimensions.

    Groups specs so that dimensions with independent destinations are in
    separate groups, while dimensions sharing a destination are grouped together.
    This enables using scipy.interpn for multi-dimensional interpolation when
    multiple source dimensions map to the same destination dimension.

    Parameters
    ----------
    interp_specs : list of dict
        List of interpolation specs, each with 'dim', 'arr', 'new_dims', etc.

    Returns
    -------
    list of list of dict
        Groups of specs. Each group can be processed together.
    """
    if not interp_specs:
        return []

    groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_dest_dims: set = set()

    for spec in interp_specs:
        spec_dest_dims = set(spec["new_dims"]) if spec["new_dims"] else {spec["dim"]}

        if not current_group:
            # First spec starts a new group
            current_group = [spec]
            current_dest_dims = spec_dest_dims
        elif spec_dest_dims & current_dest_dims:
            # Shares destination with current group - add to it
            current_group.append(spec)
            current_dest_dims |= spec_dest_dims
        else:
            # Independent of current group - yield current and start new
            groups.append(current_group)
            current_group = [spec]
            current_dest_dims = spec_dest_dims

    if current_group:
        groups.append(current_group)

    return groups


def _check_uniform_bounds(
    group: list[dict],
    bounds_dict: dict[Hashable, Literal["fill", "clamp", "raise"]],
) -> tuple[bool, Literal["fill", "clamp", "raise"] | None]:
    """
    Check if all dimensions in a group have uniform bounds mode.

    Parameters
    ----------
    group : list of dict
        Group of interpolation specs.
    bounds_dict : dict
        Mapping from dimension names to bounds modes.

    Returns
    -------
    is_uniform : bool
        True if all dimensions have the same bounds mode.

    uniform_mode : {"fill", "clamp", "raise"} or None
        The shared bounds mode, or None if not uniform.
    """
    if len(group) == 0:
        return False, None

    # Extract bounds mode for first spec
    first_dim = group[0]["dim"]
    first_mode = bounds_dict.get(first_dim, "fill")

    # Check if all other specs have same bounds mode
    for spec in group[1:]:
        dim = spec["dim"]
        mode = bounds_dict.get(dim, "fill")
        if mode != first_mode:
            return False, None

    return True, first_mode


def _should_use_fast_path(
    group: list[dict],
    dims: list[Hashable],
    bounds_dict: dict[Hashable, Literal["fill", "clamp", "raise"]],
) -> bool:
    """
    Determine if group qualifies for scipy.interpn fast path.

    Parameters
    ----------
    group : list of dict
        Group of interpolation specs.
    dims : list
        Current dimension names in the data.
    bounds_dict : dict
        Mapping from dimension names to bounds modes.

    Returns
    -------
    bool
        True if the group qualifies for fast path processing.
    """
    # Must have multiple specs in group
    if len(group) <= 1:
        return False

    # All specs must have same non-empty destination dims
    if not group[0]["new_dims"]:
        return False

    dest_dims = group[0]["new_dims"]
    if not all(spec["new_dims"] == dest_dims for spec in group):
        return False

    # Destination dims must not already exist (else need pointwise path)
    if set(dest_dims) & set(dims):
        return False

    # No scalar specs allowed
    if any(spec["is_scalar"] for spec in group):
        return False

    # All dimensions must have uniform bounds mode
    is_uniform, _ = _check_uniform_bounds(group, bounds_dict)
    if not is_uniform:
        return False

    return True


def _interp_group_with_interpn(
    data: np.ndarray,
    dims: list[Hashable],
    da: xr.DataArray,
    group: list[dict],
    bounds_mode: Literal["fill", "clamp", "raise"],
    fill_value_dict: dict[Hashable, float | tuple[float, float]],
) -> tuple[np.ndarray, list[Hashable]]:
    """
    Interpolate multiple dimensions at once using scipy.interpn.

    This is used when multiple source dimensions share a common destination
    dimension (e.g., t, p, x_H2O all mapping to z) and have uniform bounds
    mode. Using interpn is much faster than sequential 1D interpolation.

    Parameters
    ----------
    data : np.ndarray
        Current data array.
    dims : list
        Current dimension names corresponding to data axes.
    da : xr.DataArray
        Original DataArray (for coordinate grids).
    group : list of dict
        Group of specs to interpolate together.
    bounds_mode : {"fill", "clamp", "raise"}
        Uniform bounds mode for all dimensions in group.
    fill_value_dict : dict
        Per-dimension fill values (used for "fill" mode).

    Returns
    -------
    data : np.ndarray
        Interpolated data.
    dims : list
        Updated dimension names.
    """
    # Identify source dims and destination dims
    src_dims = [spec["dim"] for spec in group]
    dest_dims = group[0]["new_dims"]  # All specs in group share destination dims

    # Build grid points tuple for interpn (order matters!)
    grid_points = tuple(da.coords[dim].values for dim in src_dims)

    # Build query points array: shape (dest_size, n_src_dims)
    query_arrays = [spec["arr"] for spec in group]
    xi = np.stack(query_arrays, axis=-1)

    # Find axes of source dims in current data layout
    src_axes = [dims.index(dim) for dim in src_dims]
    other_axes = [i for i in range(len(dims)) if i not in src_axes]
    other_dims = [dims[i] for i in other_axes]

    # Transpose: src_dims first, then other dims (batch dims at the end for interpn)
    perm = src_axes + other_axes
    data_transposed = data.transpose(perm)

    # Handle bounds mode uniformly
    if bounds_mode == "raise":
        # Pre-validate all dimensions before interpolation
        for i, (dim, grid) in enumerate(zip(src_dims, grid_points)):
            xi_dim = xi[..., i]
            xi_min, xi_max = xi_dim.min(), xi_dim.max()
            grid_min, grid_max = grid[0], grid[-1]

            if xi_min < grid_min or xi_max > grid_max:
                raise ValueError(
                    f"Interpolation error on dimension {dim!r}: "
                    f"Query points out of bounds.\n"
                    f"  Grid range: [{grid_min}, {grid_max}]\n"
                    f"  Query range: [{xi_min}, {xi_max}]\n"
                    f"  Bounds mode: raise"
                )
        # All checks passed - call interpn with bounds_error=False
        result = interpn(
            grid_points,
            data_transposed,
            xi,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    elif bounds_mode == "clamp":
        # Clamp query points to grid bounds
        xi_clamped = xi.copy()
        for i, grid in enumerate(grid_points):
            xi_clamped[..., i] = np.clip(xi_clamped[..., i], grid[0], grid[-1])
        result = interpn(
            grid_points,
            data_transposed,
            xi_clamped,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    else:  # bounds_mode == "fill"
        # Call interpn with bounds_error=False to get NaN for OOB
        result = interpn(
            grid_points,
            data_transposed,
            xi,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Post-process per-dimension fill values
        # For each dimension, replace NaN where that dim was OOB
        for i, (dim, grid) in enumerate(zip(src_dims, grid_points)):
            dim_fill_value = fill_value_dict.get(dim, np.nan)

            # Skip if fill value is already NaN (no-op)
            if isinstance(dim_fill_value, tuple):
                fill_lo, fill_hi = dim_fill_value
            else:
                fill_lo = fill_hi = dim_fill_value

            if not (np.isnan(fill_lo) and np.isnan(fill_hi)):
                # Identify OOB points for this dimension
                xi_dim = xi[..., i]
                grid_min, grid_max = grid[0], grid[-1]

                # Apply fill values
                # Note: result shape is (dest_size, *batch_dims)
                oob_lo = xi_dim < grid_min
                oob_hi = xi_dim > grid_max

                # Broadcast OOB mask to result shape
                # xi_dim shape: (dest_size,), result shape: (dest_size, *batch)
                if oob_lo.any():
                    result[oob_lo, ...] = fill_lo
                if oob_hi.any():
                    result[oob_hi, ...] = fill_hi

    # Update dims: replace src_dims with dest_dims
    new_dims = list(dest_dims) + other_dims

    return result, new_dims


def interp_dataarray(
    da: xr.DataArray,
    coords: dict[Hashable, float | np.ndarray | xr.DataArray],
    bounds: Literal["fill", "clamp", "raise"]
    | dict[Hashable, Literal["fill", "clamp", "raise"]] = "fill",
    fill_value: float
    | tuple[float, float]
    | dict[Hashable, float | tuple[float, float]] = np.nan,
) -> xr.DataArray:
    """
    Fast linear interpolation for xarray DataArrays.

    This function provides a high-performance alternative to xarray's
    ``.interp()`` method for linear interpolation, with the possibility to
    control out-of-bound handling behaviour per dimension.

    Parameters
    ----------
    da : DataArray
        The input DataArray to interpolate.

    coords : dict
        Mapping from dimension names to new coordinate values. Each value
        can be a scalar, numpy array, or xarray DataArray. Dimensions are
        interpolated sequentially in the order given.

    bounds : {"fill", "clamp", "raise"} or dict, default: "fill"
        How to handle out-of-bounds query points. Can be a single value
        applied to all dimensions, or a dict mapping dimension names to
        bounds modes. Missing dimensions default to ``"fill"``.

        * "fill": Use ``fill_value`` for points outside the data range.
        * "clamp": Use the nearest boundary value.
        * "raise": Raise a ValueError if any query point is out of bounds.

    fill_value : float or tuple or dict, default: np.nan
        Value(s) to use for out-of-bounds points when ``bounds="fill"``.
        Can be:

        * a single float (used for both lower and upper bounds, all dims);
        * a 2-tuple (``fill_lower``, ``fill_upper``) for all dimensions;
        * a dict mapping dimension names to floats or 2-tuples.

        Missing dimensions default to ``np.nan``.

    Returns
    -------
    DataArray
        Interpolated DataArray with the new coordinates. Preserves the
        original DataArray's name and attributes.

    Raises
    ------
    ValueError
        If a dimension in ``coords`` does not exist in the DataArray.
        If ``bounds="raise"`` and any query point is out of bounds.
        The error message includes the dimension name for easier debugging.

    Notes
    -----
    The function assumes coordinates are sorted in ascending order along
    each interpolation dimension. Results are undefined if this assumption
    is violated.

    Interpolation is performed sequentially across dimensions in the order
    given in ``coords``. This is equivalent to chained ``.interp()`` calls
    but significantly faster due to the optimized gufunc implementation.

    Examples
    --------
    Basic multi-dimensional interpolation:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from axsdb.interpolation import interp_dataarray
    >>>
    >>> # Create sample data
    >>> da = xr.DataArray(
    ...     np.random.rand(10, 20, 30),
    ...     dims=["wavelength", "angle", "time"],
    ...     coords={
    ...         "wavelength": np.linspace(400, 700, 10),
    ...         "angle": np.linspace(0, 90, 20),
    ...         "time": np.linspace(0, 1, 30),
    ...     },
    ... )
    >>>
    >>> # Interpolate to new coordinates
    >>> result = interp_dataarray(
    ...     da,
    ...     {
    ...         "wavelength": np.array([450.0, 550.0, 650.0]),
    ...         "angle": np.array([30.0, 60.0]),
    ...     },
    ... )
    >>> result.shape
    (3, 2, 30)

    Different bounds handling per dimension:

    >>> result = interp_dataarray(
    ...     da,
    ...     {"wavelength": new_wavelengths, "angle": new_angles},
    ...     bounds={"wavelength": "raise", "angle": "clamp"},
    ... )

    Custom fill values per dimension:

    >>> result = interp_dataarray(
    ...     da,
    ...     {"wavelength": new_wavelengths, "angle": new_angles},
    ...     bounds="fill",
    ...     fill_value={"wavelength": 0.0, "angle": (-1.0, 1.0)},
    ... )
    """
    # Normalize bounds to dict format
    if isinstance(bounds, str):
        bounds_dict: dict[Hashable, Literal["fill", "clamp", "raise"]] = {
            dim: bounds for dim in coords
        }
    else:
        bounds_dict = dict(bounds)
        for dim in coords:
            if dim not in bounds_dict:
                bounds_dict[dim] = "fill"

    # Normalize fill_value to dict format
    if isinstance(fill_value, (int, float)) or (
        isinstance(fill_value, tuple) and len(fill_value) == 2
    ):
        fill_value_dict: dict[Hashable, float | tuple[float, float]] = {
            dim: fill_value for dim in coords
        }
    else:
        fill_value_dict = dict(fill_value)  # type: ignore[arg-type]
        for dim in coords:
            if dim not in fill_value_dict:
                fill_value_dict[dim] = np.nan

    # --- Pre-process all interpolation targets once ---
    # For each dimension we resolve the new coordinates to a plain numpy
    # array and record the metadata needed for the final DataArray wrap.
    interp_specs: list[dict] = []
    for dim, new_coords in coords.items():
        if dim not in da.dims:
            raise ValueError(
                f"Dimension {dim!r} not found in DataArray. "
                f"Available dimensions: {list(da.dims)}"
            )
        if isinstance(new_coords, xr.DataArray):
            spec = {
                "dim": dim,
                "arr": new_coords.values,
                "new_dims": new_coords.dims,
                "new_coords": {
                    k: v for k, v in new_coords.coords.items() if k in new_coords.dims
                },
                "is_scalar": False,
            }
        elif np.isscalar(new_coords):
            spec = {
                "dim": dim,
                "arr": np.array([new_coords]),
                "new_dims": (),
                "new_coords": {},
                "is_scalar": True,
            }
        else:
            arr = np.asarray(new_coords)
            spec = {
                "dim": dim,
                "arr": arr,
                "new_dims": (),
                "new_coords": {},
                "is_scalar": arr.ndim == 0,
            }
            if spec["is_scalar"]:
                spec["arr"] = arr.reshape(1)
        interp_specs.append(spec)

    # --- Reorder: shrinking dimensions before expanding ones ---
    # Interpolating a dimension that expands (query size > grid size) while
    # other spectator dimensions are still large multiplies work
    # unnecessarily. Processing shrinking dims first keeps the intermediate
    # array small until the expansion happens on a much smaller array.
    #
    # Within expanding dimensions, process LARGER grids first. This minimizes
    # the intermediate array size when introducing a new dimension (e.g. z).
    # The first expander creates (total / grid_size) * query_size elements;
    # larger grid_size means smaller intermediate. Subsequent expanders
    # share the new dimension and reduce the array; processing larger grids
    # first means fewer lerp operations overall.
    def _interp_sort_key(spec: dict) -> tuple:
        grid_size = da.sizes[spec["dim"]]
        query_size = len(spec["arr"])
        # (0, ...) = shrink/same -> first; (1, ...) = expand -> last
        # Within each group, larger grid_size first (hence negative)
        if query_size <= grid_size:
            return (0, -grid_size)
        else:
            return (1, -grid_size)

    interp_specs.sort(key=_interp_sort_key)

    # --- Decompose into groups for potential fast path ---
    # Group specs by destination dimensions. When multiple source dims map to
    # the same destination (e.g. t, p, x_H2O all -> z), processing them together
    # with scipy.interpn is much faster than sequential 1D interpolation.
    interp_groups = _decompose_interp_groups(interp_specs)

    # --- Main interpolation loop on raw numpy arrays ---
    # `dims` tracks the current logical dimension order.
    # `data` is the raw ndarray; axes correspond 1-to-1 with `dims`.
    dims: list[Hashable] = list(da.dims)
    data: np.ndarray = da.values

    for group in interp_groups:
        # Check if this group can use the fast interpn path:
        # - Multiple specs in the group
        # - All specs have the same non-empty new_dims (shared destination)
        # - The destination dims don't already exist in the current array
        #   (if they do, we need the pointwise/shared path)
        # - All dimensions have uniform bounds mode
        use_interpn = _should_use_fast_path(group, dims, bounds_dict)

        if use_interpn:
            # Fast path: multi-dimensional interpn
            is_uniform, uniform_mode = _check_uniform_bounds(group, bounds_dict)
            data, dims = _interp_group_with_interpn(
                data, dims, da, group, uniform_mode, fill_value_dict
            )
            continue

        # Sequential path: process each spec one at a time
        for spec in group:
            dim = spec["dim"]
            new_coords_arr: np.ndarray = spec["arr"]
            new_coords_dims: tuple = spec["new_dims"]
            is_scalar: bool = spec["is_scalar"]

            dim_bounds = bounds_dict.get(dim, "fill")
            dim_fill_value = fill_value_dict.get(dim, np.nan)

            dim_axis = dims.index(dim)
            old_coords_arr = da.coords[dim].values

            # Detect shared-dimension case: new_coords has a dim that already
            # exists in the current working set.
            shared_dims = set(new_coords_dims) & set(dims) if new_coords_dims else set()

            if shared_dims:
                # --- Shared-dimension (pointwise) path ---
                # This path handles cases where the new coordinates share a dimension
                # with the existing data (e.g. interpolating t->z when z already exists).
                #
                # Strategy: Precompute indices/weights once for the shared dimension,
                # then apply them pointwise to each slice. This avoids redundant
                # binary searches across the shared dimension.
                #
                # Limitation: Only single shared dimension is currently supported.
                # Multiple shared dims would require more complex bookkeeping.
                if len(shared_dims) > 1:
                    raise NotImplementedError(
                        f"Multiple shared dimensions not supported: {shared_dims}. "
                        f"Interpolating dimension {dim!r} with coordinates that share "
                        f"multiple dimensions with the current data is not implemented."
                    )

                shared_dim = next(iter(shared_dims))
                shared_axis = dims.index(shared_dim)
                shared_size = data.shape[shared_axis]

                # Transpose to (..., shared_dim, dim)
                other_axes = [
                    i for i in range(len(dims)) if i != dim_axis and i != shared_axis
                ]
                perm = other_axes + [shared_axis, dim_axis]
                data = data.transpose(perm)

                # Precompute bin indices and weights once for all shared-dim
                # query points against the (small, uniform) grid. This avoids
                # repeating the binary search for every slice along the leading
                # batch dimensions.
                try:
                    indices, weights = lerp_indices(
                        old_coords_arr, new_coords_arr, bounds=dim_bounds
                    )
                except ValueError as e:
                    x_min, x_max = old_coords_arr[0], old_coords_arr[-1]
                    q_min, q_max = new_coords_arr.min(), new_coords_arr.max()
                    raise ValueError(
                        f"Interpolation error on dimension {dim!r}: {e}\n"
                        f"  Grid range: {[x_min, x_max] =}\n"
                        f"  Query range: {[q_min, q_max] = }\n"
                        f"  Bounds mode: {dim_bounds = }"
                    ) from e

                # indices/weights are (shared_size,). Reshape to
                # (shared_size, 1) so the gufunc signature (n),(m),(m)->(m)
                # treats the last axis as the core (m=1 query per z-slice)
                # and broadcasts over all leading batch dims.
                indices_bc = indices.reshape((shared_size, 1))
                weights_bc = weights.reshape((shared_size, 1))

                data = lerp(data, indices_bc, weights_bc)

                # Handle fill values for bounds="fill" (lerp propagates NaN
                # for OOB weights; replace with the requested fill value).
                if dim_bounds == "fill":
                    # Note: data layout is (..., shared_size, 1); OOB mask is
                    # over the shared_size axis (second-to-last).
                    # Need special indexing for this layout
                    oob = (new_coords_arr < old_coords_arr[0]) | (
                        new_coords_arr > old_coords_arr[-1]
                    )
                    if oob.any():
                        if isinstance(dim_fill_value, tuple):
                            fill_lo, fill_hi = dim_fill_value
                        else:
                            fill_lo = fill_hi = dim_fill_value
                        lo = new_coords_arr < old_coords_arr[0]
                        hi = new_coords_arr > old_coords_arr[-1]
                        # data layout: (..., shared_size, 1)
                        data[..., lo, :] = fill_lo
                        data[..., hi, :] = fill_hi

                # Squeeze the trailing length-1 query axis
                data = data[..., 0]

                # Update dims: remove dim, keep shared_dim in its original
                # relative position among surviving dims.
                dims = [d for d in dims if d != dim]
                # data is now in layout (other_dims..., shared_dim); transpose
                # back so shared_dim is where it belongs.
                # Build the permutation that undoes the earlier reorder.
                # Current logical order after removal: other_dims + [shared_dim]
                # Target order: dims (which preserved original relative order).
                current_order = [d for d in dims if d != shared_dim] + [shared_dim]
                if current_order != dims:
                    perm = [current_order.index(d) for d in dims]
                    data = data.transpose(perm)

            else:
                # --- Standard path (no shared dims) ---
                # Move dim to last axis for the gufunc
                if dim_axis != len(dims) - 1:
                    perm = list(range(len(dims)))
                    perm.remove(dim_axis)
                    perm.append(dim_axis)
                    data = data.transpose(perm)

                # When new_coords is a uniform 1-D array (same query points for
                # every slice along the leading dims), precompute the bin indices
                # and weights once and use the search-free lerp gufunc. This
                # avoids redundant binary searches across spectator dimensions.
                use_precomputed = new_coords_arr.ndim == 1 and data.ndim > 1

                try:
                    if use_precomputed:
                        indices, weights = lerp_indices(
                            old_coords_arr, new_coords_arr, bounds=dim_bounds
                        )
                        data = lerp(data, indices, weights)
                        # precompute_lerp_indices marks OOB weights as NaN so
                        # lerp propagates NaN. Replace with the requested fill
                        # value for bounds="fill".
                        if dim_bounds == "fill":
                            _apply_fill_values(
                                data, new_coords_arr, old_coords_arr, dim_fill_value
                            )
                    else:
                        old_bc = np.broadcast_to(
                            old_coords_arr,
                            data.shape[:-1] + (len(old_coords_arr),),
                        )
                        new_bc = (
                            new_coords_arr.reshape(1)
                            if new_coords_arr.ndim == 0
                            else new_coords_arr
                        )
                        data = interp1d(
                            old_bc,
                            data,
                            new_bc,
                            bounds=dim_bounds,
                            fill_value=dim_fill_value,
                        )
                except ValueError as e:
                    x_min, x_max = old_coords_arr[0], old_coords_arr[-1]
                    q_min = (
                        new_coords_arr.min()
                        if new_coords_arr.size > 0
                        else float("nan")
                    )
                    q_max = (
                        new_coords_arr.max()
                        if new_coords_arr.size > 0
                        else float("nan")
                    )
                    raise ValueError(
                        f"Interpolation error on dimension {dim!r}: {e}\n"
                        f"  Grid range: [{x_min}, {x_max}]\n"
                        f"  Query range: [{q_min}, {q_max}]\n"
                        f"  Bounds mode: {dim_bounds}"
                    ) from e

                if is_scalar:
                    # Scalar query: squeeze the last axis, remove dim entirely
                    data = data[..., 0]
                    # dims without dim, preserving order (dim was moved to end
                    # for the gufunc but we track logical order separately)
                    dims = [d for d in dims if d != dim]
                elif new_coords_dims:
                    # Dimension replaced: swap dim for new_coords_dims at dim_axis.
                    # data layout is (...without dim..., len(new_coords_arr))
                    # which means new dim is at the end. We want it at dim_axis.
                    dims = [d for d in dims if d != dim]
                    # Insert new dims at the original position
                    for i, nd in enumerate(new_coords_dims):
                        dims.insert(dim_axis + i, nd)
                    # Move the last axis (new dim) to dim_axis
                    n = len(dims)
                    perm = list(range(n - 1))
                    perm.insert(dim_axis, n - 1)
                    data = data.transpose(perm)
                else:
                    # Same dim, resized in place. data has dim at the end;
                    # move it back to dim_axis.
                    if dim_axis != len(dims) - 1:
                        n = len(dims)
                        perm = list(range(n - 1))
                        perm.insert(dim_axis, n - 1)
                        data = data.transpose(perm)
                    # dims unchanged (same names, same order)

    # --- Wrap back into a DataArray ---
    # Collect coordinates for the output: keep original coords whose dims
    # are all still present, then add any new coords from interp targets.
    out_coords: dict = {}
    for coord_name, coord_val in da.coords.items():
        if all(d in dims for d in coord_val.dims):
            out_coords[coord_name] = coord_val

    for spec in interp_specs:
        dim = spec["dim"]
        if spec["is_scalar"]:
            continue
        if spec["new_dims"]:
            # Coordinates from the replacement DataArray
            out_coords.update(spec["new_coords"])
        else:
            # Plain array: attach as a coordinate on its own dim
            out_coords[dim] = (dim, spec["arr"])

    return xr.DataArray(
        data, dims=dims, coords=out_coords, name=da.name, attrs=da.attrs
    )
