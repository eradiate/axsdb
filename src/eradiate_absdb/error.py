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
    IGNORE = "ignore"
    RAISE = "raise"
    WARN = "warn"


@attrs.define
class ErrorHandlingPolicy:
    missing: ErrorHandlingAction
    scalar: ErrorHandlingAction
    bounds: ErrorHandlingAction

    @classmethod
    def convert(cls, value):
        if isinstance(value, Mapping):
            kwargs = {k: ErrorHandlingAction(v) for k, v in value.items()}
            return cls(**kwargs)
        else:
            return value


@attrs.define
class ErrorHandlingConfiguration:
    x: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)
    p: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)
    t: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)

    @classmethod
    def convert(cls, value):
        if isinstance(value, Mapping):
            return cls(**value)
        else:
            return value


def handle_error(error: InterpolationError, action: ErrorHandlingAction):
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
    global _ERROR_HANDLING_CONFIG
    value = ErrorHandlingConfiguration.convert(value)
    if not isinstance(value, ErrorHandlingConfiguration):
        raise ValueError("could not convert value to ErrorHandlingConfiguration")
    _ERROR_HANDLING_CONFIG = value


def get_error_handling_config() -> ErrorHandlingConfiguration:
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
