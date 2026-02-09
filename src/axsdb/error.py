from __future__ import annotations

import enum
import warnings
from collections.abc import Mapping

import attrs

# ------------------------------------------------------------------------------
#                                   Exceptions
# ------------------------------------------------------------------------------


class DataError(Exception):
    """Raised when encountering issues with data."""

    pass


class InterpolationError(Exception):
    """Raised when encountering errors during interpolation."""

    pass


# ------------------------------------------------------------------------------
#                           Error handling components
# ------------------------------------------------------------------------------


class ErrorHandlingAction(enum.Enum):
    """
    Error handling action descriptors.
    """

    IGNORE = "ignore"  #: Ignore the error.
    RAISE = "raise"  #: Raise the error.
    WARN = "warn"  #: Emit a warning.


class BoundsMode(enum.Enum):
    """
    Bounds policy mode descriptors.
    """

    FILL = "fill"  #: Use fill_value for out-of-bounds points.
    CLAMP = "clamp"  #: Clamp to nearest boundary value.


@attrs.define
class BoundsPolicy:
    """
    Policy for handling out-of-bounds values on one side (lower or upper).

    Parameters
    ----------
    action : ErrorHandlingAction, default: IGNORE
        Whether to check/warn/raise on out-of-bounds values.

    mode : BoundsPolicyMode, default: FILL
        How to handle out-of-bounds values during interpolation.

        * FILL: Use fill_value for out-of-bounds points.
        * CLAMP: Clamp to nearest boundary value.

    fill_value : float or None, default: 0.0
        Value to use when mode=FILL. None means NaN.
    """

    action: ErrorHandlingAction = attrs.field(
        default=ErrorHandlingAction.IGNORE, repr=lambda x: f"<{x.name}>"
    )
    mode: BoundsMode = attrs.field(
        default=BoundsMode.FILL, repr=lambda x: f"<{x.name}>"
    )
    fill_value: float | None = 0.0

    @classmethod
    def convert(cls, value):
        """
        Convert various inputs to BoundsPolicy.

        Parameters
        ----------
        value
            Value to convert. Can be:

            * a :class:`.BoundsPolicy` instance (returned as-is).
            * a string ``"ignore"``, ``"warn"``, or ``"raise"`` (sets action).
            * a string ``"fill"`` or ``"clamp"`` (sets mode).
            * a numeric value (sets ``fill_value``).
            * ``None`` (returns default policy).
            * a dict with keys: ``action``, ``mode``, ``fill_value``.

        Returns
        -------
        BoundsPolicy

        Examples
        --------
        >>> BoundsPolicy.convert("ignore")
        BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.0)
        >>> BoundsPolicy.convert("clamp")
        BoundsPolicy(action=<IGNORE>, mode=<CLAMP>, fill_value=0.0)
        >>> BoundsPolicy.convert(0.5)
        BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.5)
        >>> BoundsPolicy.convert({"action": "warn", "fill_value": 1.0})
        BoundsPolicy(action=<WARN>, mode=<FILL>, fill_value=1.0)
        >>> BoundsPolicy.convert(None)
        BoundsPolicy(action=<IGNORE>, mode=<FILL>, fill_value=0.0)
        """
        if isinstance(value, cls):
            return value

        if value is None:  # default
            return cls()

        if isinstance(value, str):
            # Check if it's an action
            if value in ("ignore", "warn", "raise"):
                return cls(action=ErrorHandlingAction(value))
            # Otherwise assume it's a mode
            elif value in ("fill", "clamp"):
                return cls(mode=BoundsMode(value))
            else:
                raise ValueError(
                    f"String value '{value}' must be an action "
                    "('ignore', 'warn', 'raise') or mode ('fill', 'clamp')"
                )

        if isinstance(value, (int, float)):  # fill_value
            return cls(fill_value=float(value))

        if isinstance(value, Mapping):  # full control
            kwargs = {}
            if "action" in value:
                kwargs["action"] = ErrorHandlingAction(value["action"])
            if "mode" in value:
                mode_val = value["mode"]
                if isinstance(mode_val, str):
                    kwargs["mode"] = BoundsMode(mode_val)
                else:
                    kwargs["mode"] = mode_val
            if "fill_value" in value:
                kwargs["fill_value"] = value["fill_value"]
            return cls(**kwargs)

        return value


def _convert_bounds(value) -> tuple[BoundsPolicy, BoundsPolicy]:
    """
    Convert various inputs to a tuple of two BoundsPolicy instances.

    Parameters
    ----------
    value
        Value to convert. Can be:

        * tuple of 2 :class:`.BoundsPolicy` instances (returned as-is);
        * tuple of 2 numbers (interpreted as fill values for lower/upper);
        * dict with "lower" and/or "upper" keys;
        * single value (string, number, dict, :class:`.BoundsPolicy`): symmetric.

    Returns
    -------
    tuple of BoundsPolicy
        (lower_policy, upper_policy)
    """

    if isinstance(value, tuple) and len(value) == 2:
        if all(isinstance(v, BoundsPolicy) for v in value):
            return value
        # Tuple of 2 numbers = fill values
        if all(isinstance(v, (int, float)) for v in value):
            return (
                BoundsPolicy(fill_value=float(value[0])),
                BoundsPolicy(fill_value=float(value[1])),
            )
        # Tuple of 2 convertible values (e.g., strings, dicts)
        return (
            BoundsPolicy.convert(value[0]),
            BoundsPolicy.convert(value[1]),
        )

    # Dict with "lower" and/or "upper" keys
    if isinstance(value, Mapping):
        if "lower" in value or "upper" in value:
            return (
                BoundsPolicy.convert(value.get("lower", {})),
                BoundsPolicy.convert(value.get("upper", {})),
            )

    # Single value (string, number, dict, BoundsPolicy): symmetric
    policy = BoundsPolicy.convert(value)
    return (policy, policy)


@attrs.define
class ErrorHandlingPolicy:
    """
    Error handling policy for a single coordinate.

    Parameters
    ----------
    missing : ErrorHandlingAction
        Action when coordinate is missing from dataset.

    scalar : ErrorHandlingAction
        Action when coordinate dimension is scalar.

    bounds : tuple[BoundsPolicy, BoundsPolicy]
        Policies for (``lower_bound``, ``upper_bound``).
        Values are automatically converted by :meth:`.BoundsPolicy.convert`.
        A single value can be passed and will be interpreted as a symmetric case.
    """

    missing: ErrorHandlingAction = attrs.field(repr=lambda x: f"<{x.name}>")
    scalar: ErrorHandlingAction = attrs.field(repr=lambda x: f"<{x.name}>")
    bounds: tuple[BoundsPolicy, BoundsPolicy] = attrs.field(converter=_convert_bounds)

    @classmethod
    def convert(cls, value):
        """
        Convert a value to an :class:`.ErrorHandlingPolicy`.

        Parameters
        ----------
        value
            Value to convert. Can be:

            * an :class:`.ErrorHandlingPolicy` instance (returned as-is).
            * a dict with keys ``missing``, ``scalar``, and ``bounds``.

              The ``bounds`` value is passed to :func:`._convert_bounds` and can be:

              * a string/number/dict/:class:`.BoundsPolicy` (symmetric)
              * a tuple of 2 :class:`.BoundsPolicy` instances
              * a tuple of 2 numbers (fill values)
              * a dict with ``lower`` and/or ``upper`` keys

        Returns
        -------
        ErrorHandlingPolicy

        Raises
        ------
        ValueError
            If the value cannot be converted to an :class:`.ErrorHandlingPolicy`
            instance.
        """

        if isinstance(value, cls):
            return value

        if isinstance(value, Mapping):
            kwargs = {}
            for k, v in value.items():
                if k == "missing":
                    kwargs["missing"] = ErrorHandlingAction(v)
                elif k == "scalar":
                    kwargs["scalar"] = ErrorHandlingAction(v)
                elif k == "bounds":
                    kwargs["bounds"] = v
                else:
                    # Unknown key, pass through
                    kwargs[k] = v

            return cls(**kwargs)
        else:
            raise ValueError(
                f"cannot convert value to ErrorHandlingPolicy (got {value!r})"
            )


@attrs.define
class ErrorHandlingConfiguration:
    """
    Error handling configuration.

    Parameters
    ----------
    x : ErrorHandlingPolicy
        Error handling policy for species concentrations.

    p : ErrorHandlingPolicy
        Error handling policy for pressure.

    t : ErrorHandlingPolicy
        Error handling policy for temperature.
    """

    x: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)
    p: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)
    t: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)

    @classmethod
    def convert(cls, value):
        """
        Convert a value to an :class:`.ErrorHandlingConfiguration`.

        Parameters
        ----------
        value
            Value to convert. Dictionaries values are passed as keyword arguments
            to the constructor.

        Returns
        -------
        ErrorHandlingConfiguration
        """
        if isinstance(value, Mapping):
            return cls(**value)
        else:
            return value


def handle_error(error: InterpolationError, action: ErrorHandlingAction):
    """
    Apply an error handling policy.

    Parameters
    ----------
    error : .InterpolationError
        The error that is handled.

    action : ErrorHandlingAction
        If ``IGNORE``, do nothing; if ``WARN``, emit a warning; if ``RAISE``,
        raise the error.
    """
    if action is ErrorHandlingAction.IGNORE:
        return

    if action is ErrorHandlingAction.WARN:
        warnings.warn(str(error), UserWarning)
        return

    if action is ErrorHandlingAction.RAISE:
        raise error

    raise NotImplementedError


#: Global default error handling configuration
_DEFAULT_ERROR_HANDLING_CONFIG: ErrorHandlingConfiguration | None = None


def set_error_handling_config(value: Mapping | ErrorHandlingConfiguration) -> None:
    """
    Set the global default error handling configuration.

    Parameters
    ----------
    value : Mapping | ErrorHandlingConfiguration
        Error handling configuration.

    Raises
    ------
    ValueError
        If ``value`` cannot be converted to an :class:`.ErrorHandlingConfiguration`.
    """
    global _DEFAULT_ERROR_HANDLING_CONFIG
    value = ErrorHandlingConfiguration.convert(value)
    if not isinstance(value, ErrorHandlingConfiguration):
        raise ValueError("could not convert value to ErrorHandlingConfiguration")
    _DEFAULT_ERROR_HANDLING_CONFIG = value


def get_error_handling_config() -> ErrorHandlingConfiguration:
    """
    Retrieve the current global default error handling configuration.

    Returns
    -------
    ErrorHandlingConfiguration
    """
    global _DEFAULT_ERROR_HANDLING_CONFIG
    if _DEFAULT_ERROR_HANDLING_CONFIG is None:  # No config yet: assign a default
        set_error_handling_config(
            {
                # This default configuration ignores bound errors on pressure and temperature
                # variables because this usually occurs at high altitude, where the absorption
                # coefficient is very low and can be safely forced to 0.
                "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
                "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
                # Ignore missing molecule coordinates, raise on bound error.
                "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
            }
        )

    return _DEFAULT_ERROR_HANDLING_CONFIG
