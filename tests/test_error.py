"""
Tests for axsdb.error module.
"""

import warnings

import pytest

from axsdb.error import (
    ErrorHandlingAction,
    ErrorHandlingConfiguration,
    ErrorHandlingPolicy,
    InterpolationError,
    handle_error,
)


class TestHandleError:
    """Tests for the handle_error function."""

    def test_ignore_does_nothing(self):
        """Test that IGNORE action does not raise or warn."""
        error = InterpolationError("test error")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise and not warn
            result = handle_error(error, ErrorHandlingAction.IGNORE)

        assert result is None

    def test_warn_emits_user_warning(self):
        """Test that WARN action emits a UserWarning."""
        error = InterpolationError("test error message")

        with pytest.warns(UserWarning, match="test error message"):
            handle_error(error, ErrorHandlingAction.WARN)

    def test_raise_raises_error(self):
        """Test that RAISE action raises the InterpolationError."""
        error = InterpolationError("test error")

        with pytest.raises(InterpolationError, match="test error"):
            handle_error(error, ErrorHandlingAction.RAISE)

    def test_raise_preserves_error_identity(self):
        """Test that RAISE raises the exact same error object."""
        error = InterpolationError("specific error")

        with pytest.raises(InterpolationError) as exc_info:
            handle_error(error, ErrorHandlingAction.RAISE)

        assert exc_info.value is error

    def test_warn_message_matches_error(self):
        """Test that warning message matches the error string."""
        error = InterpolationError("detailed OOB info: dim='t', delta=-5.0")

        with pytest.warns(UserWarning) as record:
            handle_error(error, ErrorHandlingAction.WARN)

        assert len(record) == 1
        assert "detailed OOB info" in str(record[0].message)


class TestErrorHandlingAction:
    """Tests for ErrorHandlingAction enum."""

    def test_values(self):
        """Test that enum values match expected strings."""
        assert ErrorHandlingAction.IGNORE.value == "ignore"
        assert ErrorHandlingAction.RAISE.value == "raise"
        assert ErrorHandlingAction.WARN.value == "warn"

    def test_from_string(self):
        """Test construction from string values."""
        assert ErrorHandlingAction("ignore") is ErrorHandlingAction.IGNORE
        assert ErrorHandlingAction("raise") is ErrorHandlingAction.RAISE
        assert ErrorHandlingAction("warn") is ErrorHandlingAction.WARN

    def test_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ErrorHandlingAction("invalid")


class TestErrorHandlingPolicy:
    """Tests for ErrorHandlingPolicy dataclass."""

    def test_convert_from_dict(self):
        """Test conversion from a dictionary."""
        policy = ErrorHandlingPolicy.convert(
            {"missing": "raise", "scalar": "warn", "bounds": "ignore"}
        )

        assert policy.missing is ErrorHandlingAction.RAISE
        assert policy.scalar is ErrorHandlingAction.WARN
        assert policy.bounds is ErrorHandlingAction.IGNORE

    def test_convert_passthrough(self):
        """Test that an existing policy passes through unchanged."""
        original = ErrorHandlingPolicy(
            missing=ErrorHandlingAction.RAISE,
            scalar=ErrorHandlingAction.RAISE,
            bounds=ErrorHandlingAction.RAISE,
        )
        result = ErrorHandlingPolicy.convert(original)
        assert result is original

    def test_convert_invalid_action(self):
        """Test that invalid action string in dict raises ValueError."""
        with pytest.raises(ValueError):
            ErrorHandlingPolicy.convert(
                {"missing": "invalid", "scalar": "raise", "bounds": "raise"}
            )


class TestErrorHandlingConfiguration:
    """Tests for ErrorHandlingConfiguration dataclass."""

    def test_convert_from_dict(self):
        """Test conversion from a nested dictionary."""
        config = ErrorHandlingConfiguration.convert(
            {
                "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
                "t": {"missing": "raise", "scalar": "raise", "bounds": "warn"},
                "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
            }
        )

        assert isinstance(config, ErrorHandlingConfiguration)
        assert config.p.bounds is ErrorHandlingAction.IGNORE
        assert config.t.bounds is ErrorHandlingAction.WARN
        assert config.x.bounds is ErrorHandlingAction.RAISE

    def test_convert_passthrough(self):
        """Test that an existing configuration passes through unchanged."""
        original = ErrorHandlingConfiguration(
            x={"missing": "raise", "scalar": "raise", "bounds": "raise"},
            p={"missing": "raise", "scalar": "raise", "bounds": "raise"},
            t={"missing": "raise", "scalar": "raise", "bounds": "raise"},
        )
        result = ErrorHandlingConfiguration.convert(original)
        assert result is original
