"""
Fast interpolation with Numba.

This module provides high-performance interpolation functions implemented
using Numba's ``guvectorize`` decorator. These functions are designed to replace
xarray's interpolation for specific use cases where performance is critical.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np
from numba import guvectorize


# Bounds mode constants (used internally by gufunc)
_BOUNDS_FILL = 0
_BOUNDS_CLAMP = 1
_BOUNDS_RAISE = 2


def _make_interp1d_gufunc():
    """
    Create the Numba gufunc for 1D linear interpolation.

    Returns a gufunc with signature ``(n),(n),(m),(),(),()->(m)`` that performs
    linear interpolation.

    The function is created at module load time to ensure JIT compilation
    happens only once.
    """

    @guvectorize(
        [
            "void(float32[:], float32[:], float32[:], int64, float32, float32, float32[:])",
            "void(float64[:], float64[:], float64[:], int64, float64, float64, float64[:])",
        ],
        "(n),(n),(m),(),(),()->(m)",
        nopython=True,
        cache=True,
    )
    def _interp1d_gufunc_impl(x, y, xnew, bounds_mode, fill_lower, fill_upper, out):
        """
        Low-level gufunc for 1D linear interpolation.

        Parameters
        ----------
        x : ndarray
            X coordinates of the data points (must be sorted in ascending order).
            Shape (n,).

        y : ndarray
            Y coordinates of the data points.
            Shape (n,).

        xnew : ndarray
            X coordinates at which to evaluate the interpolation.
            Shape (m,).


        bounds_mode : int
            Bounds handling mode:

            * 0 (fill): use ``fill_lower``/``fill_upper`` for out-of-bounds
              points;
            * 1 (clamp): use nearest boundary value;
            * 2 (raise): mark out-of-bounds with NaN for later validation.

        fill_lower : float
            Fill value for points below ``x[0]`` (only used when bounds_mode=0).

        fill_upper : float
            Fill value for points above ``x[-1]`` (only used when bounds_mode=0).

        out : ndarray
            Output array for interpolated values.
            Shape (m,).
        """
        n = len(x)
        m = len(xnew)

        x_min = x[0]
        x_max = x[n - 1]

        for i in range(m):
            xi = xnew[i]

            # Handle NaN in query point
            if np.isnan(xi):
                out[i] = np.nan
                continue

            # Handle out-of-bounds: below minimum
            if xi < x_min:
                if bounds_mode == _BOUNDS_FILL:
                    out[i] = fill_lower
                elif bounds_mode == _BOUNDS_CLAMP:
                    out[i] = y[0]
                else:  # bounds_mode == _BOUNDS_RAISE
                    out[i] = np.nan  # Mark for validation in wrapper
                continue

            # Handle out-of-bounds: above maximum
            if xi > x_max:
                if bounds_mode == _BOUNDS_FILL:
                    out[i] = fill_upper
                elif bounds_mode == _BOUNDS_CLAMP:
                    out[i] = y[n - 1]
                else:  # bounds_mode == _BOUNDS_RAISE
                    out[i] = np.nan  # Mark for validation in wrapper
                continue

            # Binary search to find the interval [x[left], x[right]]
            left = 0
            right = n - 1

            while right - left > 1:
                mid = (left + right) // 2
                if x[mid] <= xi:
                    left = mid
                else:
                    right = mid

            # Handle exact match at boundary to avoid numerical issues
            if xi == x[left]:
                out[i] = y[left]
                continue
            if xi == x[right]:
                out[i] = y[right]
                continue

            # Linear interpolation: y0 + t * (y1 - y0) where t = (xi - x0) / (x1 - x0)
            x0 = x[left]
            x1 = x[right]
            y0 = y[left]
            y1 = y[right]

            t = (xi - x0) / (x1 - x0)
            out[i] = y0 + t * (y1 - y0)

    return _interp1d_gufunc_impl


def _make_lerp_gufunc():
    """
    Create the numba gufunc for lerp with precomputed indices and weights.

    Returns a gufunc with signature ``(n),(m),(m)->(m)``.  The search step
    is skipped entirely; the caller must supply the left-index and weight
    arrays produced by :func:`lerp_indices`.
    """

    @guvectorize(
        [
            "void(float32[:], float32[:], float32[:], float32[:])",
            "void(float64[:], float64[:], float64[:], float64[:])",
        ],
        "(n),(m),(m)->(m)",
        nopython=True,
        cache=True,
    )
    def _lerp_gufunc_impl(y, indices, weights, out):
        m = len(indices)
        for i in range(m):
            left = int(indices[i])
            t = weights[i]
            out[i] = y[left] + t * (y[left + 1] - y[left])

    return _lerp_gufunc_impl


# Create gufuncs at module load time
_interp1d_gufunc = _make_interp1d_gufunc()
_lerp_gufunc = _make_lerp_gufunc()


def interp1d(
    x: np.ndarray,
    y: np.ndarray,
    xnew: np.ndarray,
    bounds: Literal["fill", "clamp", "raise"] = "fill",
    fill_value: Union[float, Tuple[float, float]] = np.nan,
) -> np.ndarray:
    """
    Fast 1D linear interpolation.

    This function provides high-performance linear interpolation that
    broadcasts over leading dimensions. It powers a drop-in replacement for
    cases where xarray's interpolation is too slow.

    Parameters
    ----------
    x : array-like
        X coordinates of the data points. Must be sorted in ascending order
        along the last axis. Results are undefined for unsorted x.
        Shape (..., n).

    y : array-like
        Y coordinates of the data points.
        Shape (..., n).

    xnew : array-like
        X coordinates at which to evaluate the interpolation.
        Shape (..., m).

    bounds : {"fill", "clamp", "raise"}, default: "fill"
        How to handle out-of-bounds query points:

        * ``"fill"``: use ``fill_value`` for points outside the data range.
        * ``"clamp"``: use the nearest boundary value (``y[0]`` or ``y[-1]``).
        * ``"raise"``: raise a ValueError if any query point is out of bounds.

    fill_value : float or tuple of (float, float), default: np.nan
        Value(s) to use for out-of-bounds points when ``bounds="fill"``:

        * if a single float, use for both lower and upper bounds;
        * if a 2-tuple, use (``fill_lower``, ``fill_upper``).

    Returns
    -------
    ndarray
        Interpolated values at the query points. The output shape is
        determined by numpy broadcasting rules applied to x, y, and xnew.
        Shape (..., m).

    Raises
    ------
    ValueError
        * If ``bounds="raise"`` and any query point is outside the data range.
        * If ``bounds`` is not one of "fill", "clamp", or "raise".
        * If ``fill_value`` is a tuple with length != 2.

    Notes
    -----
    * The implementation uses a Numba gufunc with signature ``(n),(n),(m)->(m)``
      for the core interpolation, enabling efficient broadcasting over arbitrary
      leading dimensions.
    * The function assumes ``x`` is sorted in ascending order along the last
      axis. Results are undefined if this assumption is violated.
    * NaN values in ``xnew`` are passed through (output will be NaN).

    Examples
    --------
    Basic interpolation:

    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = np.array([0.0, 1.0, 4.0, 9.0])
    >>> xnew = np.array([0.5, 1.5, 2.5])
    >>> interp1d(x, y, xnew)
    array([0.5, 2.5, 6.5])

    With fill values for out-of-bounds:

    >>> xnew = np.array([-1.0, 1.5, 5.0])
    >>> interp1d(x, y, xnew, bounds="fill", fill_value=(-999.0, 999.0))
    array([-999. ,    2.5,  999. ])

    Clamping to boundary values:

    >>> interp1d(x, y, xnew, bounds="clamp")
    array([0. , 2.5, 9. ])

    Broadcasting over multiple curves:

    >>> x = np.array([0.0, 1.0, 2.0])
    >>> y = np.array([[0.0, 1.0, 2.0],    # Linear
    ...               [0.0, 1.0, 4.0]])   # Quadratic
    >>> xnew = np.array([0.5, 1.5])
    >>> interp1d(x, y, xnew)
    array([[0.5, 1.5],
           [0.5, 2.5]])
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    xnew = np.asarray(xnew)

    # Validate bounds mode
    if bounds == "fill":
        bounds_mode = _BOUNDS_FILL
    elif bounds == "clamp":
        bounds_mode = _BOUNDS_CLAMP
    elif bounds == "raise":
        bounds_mode = _BOUNDS_RAISE
    else:
        raise ValueError(
            f"Invalid bounds mode: {bounds!r}. Must be one of 'fill', 'clamp', 'raise'."
        )

    # Parse fill_value
    if isinstance(fill_value, tuple):
        if len(fill_value) != 2:
            raise ValueError(
                f"fill_value tuple must have exactly 2 elements, got {len(fill_value)}"
            )
        fill_lower, fill_upper = fill_value
    else:
        fill_lower = fill_upper = fill_value

    # Ensure float dtype (convert integers to float64)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float64)
    if not np.issubdtype(xnew.dtype, np.floating):
        xnew = xnew.astype(np.float64)

    # Promote to common dtype
    common_dtype = np.result_type(x, y, xnew)
    if x.dtype != common_dtype:
        x = x.astype(common_dtype)
    if y.dtype != common_dtype:
        y = y.astype(common_dtype)
    if xnew.dtype != common_dtype:
        xnew = xnew.astype(common_dtype)

    # Convert fill values to the common dtype
    fill_lower = common_dtype.type(fill_lower)
    fill_upper = common_dtype.type(fill_upper)

    # Pre-validate bounds="raise" mode
    if bounds == "raise":
        # Get valid (non-NaN) query points
        xnew_flat = xnew.ravel()
        xnew_valid = xnew_flat[~np.isnan(xnew_flat)]

        if xnew_valid.size > 0:
            # Get overall bounds of x (simplest and most robust approach)
            x_min_val = np.min(x)
            x_max_val = np.max(x)

            # Check for violations
            min_query = np.min(xnew_valid)
            max_query = np.max(xnew_valid)

            below = min_query < x_min_val
            above = max_query > x_max_val

            if below or above:
                # Build error message
                msg_parts = ["Query points out of bounds."]
                if below:
                    delta_low = x_min_val - min_query
                    msg_parts.append(f"Below lower bound by up to {delta_low:.6g}")
                if above:
                    delta_high = max_query - x_max_val
                    msg_parts.append(f"Above upper bound by up to {delta_high:.6g}")
                raise ValueError(" ".join(msg_parts))

    # Call the gufunc
    result = _interp1d_gufunc(x, y, xnew, np.int64(bounds_mode), fill_lower, fill_upper)

    return result


def lerp_indices(
    x: np.ndarray,
    xnew: np.ndarray,
    bounds: Literal["fill", "clamp", "raise"] = "fill",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute left-indices and interpolation weights for linear interpolation.

    When the same query points ``xnew`` will be applied to many different
    ``y`` arrays sharing the same ``x`` grid, it is far cheaper to run the
    binary search once here and then call :func:`lerp` for each ``y``.
    That function skips the search entirely and executes only the
    ``y[left] + t*(y[left+1] - y[left])`` step.

    Parameters
    ----------
    x : ndarray
        Sorted coordinate grid (1-D).
        Shape (n,).

    xnew : ndarray
        Query points (1-D).
        Shape (m,).

    bounds : {"fill", "clamp", "raise"}, default: "fill"
        Out-of-bounds handling, same semantics as :func:`interp1d`.

        * ``"fill"``: out-of-bounds indices are set to 0 with weight NaN so that
          :func:`lerp` will produce NaN there.  The caller can replace those
          NaNs after the fact if a different fill value is needed.
        * ``"clamp"``: out-of-bounds queries are clamped to the nearest boundary
          index with weight 0 (reproducing ``y[0]`` or ``y[-1]``).
        * ``"raise"``: raises immediately if any query is out of bounds.

    Returns
    -------
    indices : ndarray
        Left-bin indices as floats (required by the gufunc signature).
        Shape (m,), dtype float64.

    weights : ndarray
        Fractional position within each bin: ``t = (xnew - x[i]) / (x[i+1] - x[i])``.
        Shape (m,), dtype float64.

    Raises
    ------
    ValueError
        If ``bounds="raise"`` and any query point is outside ``[x[0], x[-1]]``.
    """
    x = np.asarray(x, dtype=np.float64)
    xnew = np.asarray(xnew, dtype=np.float64)
    n = len(x)

    # searchsorted gives insertion point; left-bin index = insertion - 1
    raw = np.searchsorted(x, xnew, side="right") - 1  # shape (m,)

    if bounds == "raise":
        # Check for out-of-bounds points
        x_min_val = x[0]
        x_max_val = x[-1]
        min_query = xnew.min()
        max_query = xnew.max()

        below = min_query < x_min_val
        above = max_query > x_max_val

        if below or above:
            # Build informative error message
            msg_parts = ["Query points out of bounds."]
            if below:
                delta_low = x_min_val - min_query
                msg_parts.append(f"Below lower bound by up to {delta_low:.6g}")
            if above:
                delta_high = max_query - x_max_val
                msg_parts.append(f"Above upper bound by up to {delta_high:.6g}")
            raise ValueError(" ".join(msg_parts))

    # Clamp indices to valid bin range [0, n-2]
    indices = np.clip(raw, 0, n - 2)

    # Compute weights
    x0 = x[indices]
    x1 = x[indices + 1]
    weights = (xnew - x0) / (x1 - x0)

    if bounds == "clamp":
        # For clamping, we need special handling for boundary points:
        # - Points below x[0]: index=0, weight=0  -> y[0] + 0*(y[1]-y[0]) = y[0]
        # - Points above x[-1]: index=n-2, weight=1 -> y[n-2] + 1*(y[n-1]-y[n-2]) = y[n-1]
        # We use <= and >= (not < and >) to avoid numerical issues with exact
        # boundary matches where floating-point arithmetic might produce tiny
        # non-zero weights.
        weights = np.where(xnew <= x[0], 0.0, weights)
        weights = np.where(xnew >= x[-1], 1.0, weights)
    elif bounds == "fill":
        # Mark out-of-bounds with NaN weight so lerp produces NaN
        oob = (xnew < x[0]) | (xnew > x[-1])
        weights = np.where(oob, np.nan, weights)

    return indices.astype(np.float64), weights


def lerp(y: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Linear interpolation using precomputed indices and weights.

    This is the fast inner loop for the case where many ``y`` arrays share
    the same ``x`` grid and query points. The binary search is done once
    via :func:`lerp_indices`; this function executes only the linear
    combination ``y[i] + t * (y[i+1] - y[i])``.

    Parameters
    ----------
    y : ndarray
        Data values.  The last axis must correspond to the ``x`` grid used
        in :func:`lerp_indices`.  Broadcasting over leading dimensions is
        handled by the underlying gufunc.
        Shape (..., n).

    indices : ndarray
        Left-bin indices from :func:`lerp_indices`.
        **IMPORTANT**: Indices must be in the range ``[0, n-2]`` where
        ``n = y.shape[-1]``. This invariant is enforced by :func:`lerp_indices`
        but is not validated here for performance reasons.
        Shape (m,).

    weights : ndarray
        Interpolation weights from :func:`lerp_indices`.
        Shape (m,).

    Returns
    -------
    ndarray
        Interpolated values. NaN weights (from ``bounds="fill"``) propagate
        as NaN in the output.
        Shape (..., m).

    Notes
    -----
    This function does not perform bounds checking on ``indices`` for
    performance. The caller must ensure indices are valid (in ``[0, n-2]``).
    Using :func:`lerp_indices` guarantees this invariant.
    """
    y = np.asarray(y, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return _lerp_gufunc(y, indices, weights)
