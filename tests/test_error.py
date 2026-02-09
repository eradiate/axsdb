"""Tests for error handling configuration."""

from axsdb import (
    BoundsPolicy,
    ErrorHandlingAction,
    ErrorHandlingConfiguration,
    ErrorHandlingPolicy,
    get_error_handling_config,
)
from axsdb.error import BoundsMode
from axsdb.error import _convert_bounds


class TestBoundsPolicy:
    """Test BoundsPolicy class."""

    def test_convert(self):
        # Convert action strings to BoundsPolicy
        for action_str in ["ignore", "warn", "raise"]:
            assert BoundsPolicy.convert(action_str) == BoundsPolicy(
                action=ErrorHandlingAction(action_str),
                mode=BoundsMode.FILL,
                fill_value=0.0,
            ), f"{action_str = }"

        # Convert mode strings to BoundsPolicy
        for mode_str in ["fill", "clamp"]:
            assert BoundsPolicy.convert(mode_str) == BoundsPolicy(
                action=ErrorHandlingAction.IGNORE,
                mode=BoundsMode(mode_str),
                fill_value=0.0,
            ), f"{mode_str = }"

        # Convert numeric value to fill_value
        assert BoundsPolicy.convert(0.5) == BoundsPolicy(
            action=ErrorHandlingAction.IGNORE,
            mode=BoundsMode.FILL,
            fill_value=0.5,
        )

        # Convert None to default policy
        assert BoundsPolicy.convert(None) == BoundsPolicy(
            action=ErrorHandlingAction.IGNORE,
            mode=BoundsMode.FILL,
            fill_value=0.0,
        )

        # Convert dict
        assert BoundsPolicy.convert(
            {"action": "warn", "mode": "clamp"}
        ) == BoundsPolicy(
            action=ErrorHandlingAction.WARN,
            mode=BoundsMode.CLAMP,
            fill_value=0.0,
        )


def test_convert_bounds():
    """Test the _convert_bounds helper function."""

    # Tuple of 2 numbers interpreted as fill values
    lower, upper = _convert_bounds((0.5, 1.5))
    assert lower.fill_value == 0.5
    assert upper.fill_value == 1.5

    # Single string creates symmetric policies
    lower, upper = _convert_bounds("warn")
    assert lower.action == ErrorHandlingAction.WARN
    assert upper.action == ErrorHandlingAction.WARN

    # Dict with lower/upper keys
    lower, upper = _convert_bounds(
        {"lower": {"action": "ignore", "fill_value": 0.5}, "upper": {"action": "raise"}}
    )
    assert lower.action == ErrorHandlingAction.IGNORE
    assert lower.fill_value == 0.5
    assert upper.action == ErrorHandlingAction.RAISE


class TestErrorHandlingPolicy:
    """Test ErrorHandlingPolicy."""

    def test_create_with_symmetric_bounds(self):
        """Create policy with symmetric bounds (single value)."""
        policy = ErrorHandlingPolicy(
            missing=ErrorHandlingAction.RAISE,
            scalar=ErrorHandlingAction.RAISE,
            bounds="ignore",
        )
        assert policy.bounds[0] == BoundsPolicy(
            action=ErrorHandlingAction.IGNORE, mode=BoundsMode.FILL
        )
        assert policy.bounds[1] == BoundsPolicy(
            action=ErrorHandlingAction.IGNORE, mode=BoundsMode.FILL
        )

    def test_create_with_asymmetric_bounds_tuple(self):
        """Create policy with explicit tuple of BoundsPolicy."""
        lower = BoundsPolicy(action=ErrorHandlingAction.IGNORE, fill_value=0.0)
        upper = BoundsPolicy(action=ErrorHandlingAction.RAISE, mode=BoundsMode.CLAMP)
        policy = ErrorHandlingPolicy(
            missing=ErrorHandlingAction.RAISE,
            scalar=ErrorHandlingAction.RAISE,
            bounds=(lower, upper),
        )
        assert policy.bounds[0] is lower
        assert policy.bounds[1] is upper

    def test_create_with_dict_bounds(self):
        """Create policy with dict bounds (lower/upper keys)."""
        policy = ErrorHandlingPolicy.convert(
            {
                "missing": "raise",
                "scalar": "raise",
                "bounds": {
                    "lower": {"action": "ignore", "fill_value": 0.0},
                    "upper": {"action": "raise", "mode": "clamp"},
                },
            }
        )
        assert policy.bounds[0] == BoundsPolicy(
            action=ErrorHandlingAction.IGNORE, fill_value=0.0
        )
        assert policy.bounds[1] == BoundsPolicy(
            action=ErrorHandlingAction.RAISE, mode=BoundsMode.CLAMP
        )


class TestErrorHandlingConfiguration:
    """Test ErrorHandlingConfiguration with partial configs."""

    def test_partial_single_dimension(self):
        """Override only one dimension."""
        config = ErrorHandlingConfiguration.convert(
            {"p": {"missing": "raise", "scalar": "raise", "bounds": "raise"}}
        )
        # Check overridden dimension
        assert config.p.missing == ErrorHandlingAction.RAISE
        assert config.p.scalar == ErrorHandlingAction.RAISE
        assert config.p.bounds[0].action == ErrorHandlingAction.RAISE
        assert config.p.bounds[1].action == ErrorHandlingAction.RAISE

    def test_partial_multiple_dimensions(self):
        """Override multiple dimensions."""
        config = ErrorHandlingConfiguration.convert(
            {
                "p": {"missing": "raise", "scalar": "raise", "bounds": "raise"},
                "t": {"bounds": "warn"},
            }
        )
        # Check overridden dimensions
        assert config.p.bounds[0].action == ErrorHandlingAction.RAISE
        assert config.t.bounds[0].action == ErrorHandlingAction.WARN

    def test_nested_partial_policy(self):
        """Override only bounds within a policy."""
        config = ErrorHandlingConfiguration.convert({"t": {"bounds": "raise"}})
        default = get_error_handling_config()
        # Check overridden field
        assert config.t.bounds[0].action == ErrorHandlingAction.RAISE
        assert config.t.bounds[1].action == ErrorHandlingAction.RAISE
        # Check that non-overridden fields match defaults
        assert config.t.missing == default.t.missing
        assert config.t.scalar == default.t.scalar

    def test_nested_partial_policy_multiple_fields(self):
        """Override bounds and scalar within a policy."""
        config = ErrorHandlingConfiguration.convert(
            {"t": {"bounds": "raise", "scalar": "warn"}}
        )
        default = get_error_handling_config()
        # Check overridden fields
        assert config.t.bounds[0].action == ErrorHandlingAction.RAISE
        assert config.t.scalar == ErrorHandlingAction.WARN
        # Check that non-overridden field matches default
        assert config.t.missing == default.t.missing

    def test_empty_dict_uses_all_defaults(self):
        """Empty dict should return a copy of the default config."""
        config = ErrorHandlingConfiguration.convert({})
        default = get_error_handling_config()
        assert config == default

    def test_complete_configuration_still_works(self):
        """Complete configurations should work as before (regression test)."""
        config = ErrorHandlingConfiguration.convert(
            {
                "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
                "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
                "t": {"missing": "raise", "scalar": "raise", "bounds": "warn"},
            }
        )
        # Verify all fields are set correctly
        assert config.x.missing == ErrorHandlingAction.IGNORE
        assert config.x.scalar == ErrorHandlingAction.IGNORE
        assert config.x.bounds[0].action == ErrorHandlingAction.RAISE

        assert config.p.missing == ErrorHandlingAction.RAISE
        assert config.p.scalar == ErrorHandlingAction.RAISE
        assert config.p.bounds[0].action == ErrorHandlingAction.IGNORE

        assert config.t.missing == ErrorHandlingAction.RAISE
        assert config.t.scalar == ErrorHandlingAction.RAISE
        assert config.t.bounds[0].action == ErrorHandlingAction.WARN

    def test_direct_instantiation_without_args(self):
        """Directly instantiating ErrorHandlingConfiguration() uses defaults."""
        config = ErrorHandlingConfiguration()
        default = get_error_handling_config()
        assert config == default
