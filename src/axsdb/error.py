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


@attrs.define
class ErrorHandlingPolicy:
    """
    Error handling policy.

    Parameters
    ----------
    missing : ErrorHandlingAction
        Action to perform when a variable is missing.

    scalar : ErrorHandlingAction
        Action to perform when a dimension is scalar.

    bounds : ErrorHandlingAction
        Action to perform when an off-bounds query is made.
    """

    missing: ErrorHandlingAction
    scalar: ErrorHandlingAction
    bounds: ErrorHandlingAction

    @classmethod
    def convert(cls, value):
        """
        Convert a value to an :class:`.ErrorHandlingPolicy`.

        Parameters
        ----------
        value
            Value to convert. Dictionaries values are tentatively converted to
            :class:`.ErrorHandlingAction`, then passed as keyword arguments to
            the constructor.

        Returns
        -------
        ErrorHandlingPolicy
        """
        if isinstance(value, Mapping):
            kwargs = {k: ErrorHandlingAction(v) for k, v in value.items()}
            return cls(**kwargs)
        else:
            return value


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


_ERROR_HANDLING_CONFIG: ErrorHandlingConfiguration | None = None


def set_error_handling_config(value: Mapping | ErrorHandlingConfiguration) -> None:
    """
    Set the error handling configuration.

    Parameters
    ----------
    value : Mapping | ErrorHandlingConfiguration
        Error handling configuration.

    Raises
    ------
    ValueError
        If ``value`` cannot be converted to an :class:`.ErrorHandlingConfiguration`.
    """
    global _ERROR_HANDLING_CONFIG
    value = ErrorHandlingConfiguration.convert(value)
    if not isinstance(value, ErrorHandlingConfiguration):
        raise ValueError("could not convert value to ErrorHandlingConfiguration")
    _ERROR_HANDLING_CONFIG = value


def get_error_handling_config() -> ErrorHandlingConfiguration:
    """
    Retrieve the current error handling configuration.

    Returns
    -------
    ErrorHandlingConfiguration
    """
    if _ERROR_HANDLING_CONFIG is None:  # No config yet: assign a default
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

    return _ERROR_HANDLING_CONFIG
