"""Tests for error handling configuration."""

from axsdb import BoundsPolicy, ErrorHandlingAction, ErrorHandlingPolicy
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
