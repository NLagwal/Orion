"""Tests for Orion's deterministic guardrail layer."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.guardrails import apply_guardrails


def make_state(**overrides):
    """A safe baseline state; override only what a test needs."""
    state = {
        "acwr": 1.0,
        "rhr_delta": 0.0,
        "fatigue": 3,
        "injury_reported": 0,
        "readiness": 8,
        "soreness": 3,
    }
    state.update(overrides)
    return state


def test_injury_forces_escalation():
    result = apply_guardrails(
        make_state(injury_reported=1, acwr=1.0),
        "MAINTAIN_PLAN",
    )

    assert result.action == "ESCALATE_TO_HUMAN"
    assert result.overridden is True
    assert result.reason == "injury_signal"
    assert result.original_action == "MAINTAIN_PLAN"


def test_injury_keeps_existing_escalation():
    result = apply_guardrails(
        make_state(injury_reported=1),
        "ESCALATE_TO_HUMAN",
    )

    assert result.action == "ESCALATE_TO_HUMAN"
    assert result.overridden is False
    assert result.reason == "injury_signal"


def test_injury_beats_acwr_danger():
    result = apply_guardrails(
        make_state(injury_reported=1, acwr=2.2),
        "MAINTAIN_PLAN",
    )

    assert result.action == "ESCALATE_TO_HUMAN"
    assert result.reason == "injury_signal"


def test_acwr_danger_overrides_maintain():
    result = apply_guardrails(
        make_state(acwr=1.50),
        "MAINTAIN_PLAN",
    )

    assert result.action == "MODIFY_INTERVALS"
    assert result.overridden is True
    assert result.reason == "acwr_danger"


def test_acwr_danger_overrides_swap():
    result = apply_guardrails(
        make_state(acwr=1.60),
        "SWAP_TO_ZONE2",
    )

    assert result.action == "MODIFY_INTERVALS"
    assert result.overridden is True
    assert result.reason == "acwr_danger"


def test_acwr_danger_leaves_rest_alone():
    result = apply_guardrails(
        make_state(acwr=1.60),
        "REST_DAY",
    )

    assert result.action == "REST_DAY"
    assert result.overridden is False
    assert result.reason is None


def test_acwr_danger_leaves_modify_alone():
    result = apply_guardrails(
        make_state(acwr=1.60),
        "MODIFY_INTERVALS",
    )

    assert result.action == "MODIFY_INTERVALS"
    assert result.overridden is False
    assert result.reason is None


def test_acwr_extreme_forces_rest():
    result = apply_guardrails(
        make_state(acwr=2.00),
        "MODIFY_INTERVALS",
    )

    assert result.action == "REST_DAY"
    assert result.overridden is True
    assert result.reason == "acwr_extreme"


def test_acwr_extreme_keeps_rest_but_reports_rule():
    result = apply_guardrails(
        make_state(acwr=2.50),
        "REST_DAY",
    )

    assert result.action == "REST_DAY"
    assert result.overridden is False
    assert result.reason == "acwr_extreme"


def test_acwr_extreme_does_not_downgrade_escalation():
    result = apply_guardrails(
        make_state(acwr=2.50),
        "ESCALATE_TO_HUMAN",
    )

    assert result.action == "ESCALATE_TO_HUMAN"
    assert result.overridden is False
    assert result.reason is None


def test_rhr_and_fatigue_override_maintain():
    result = apply_guardrails(
        make_state(rhr_delta=2.0, fatigue=4, acwr=1.0),
        "MAINTAIN_PLAN",
    )

    assert result.action == "SWAP_TO_ZONE2"
    assert result.overridden is True
    assert result.reason == "rhr_fatigue"


def test_rhr_and_extreme_fatigue_override_swap():
    result = apply_guardrails(
        make_state(rhr_delta=2.0, fatigue=5, acwr=1.0),
        "SWAP_TO_ZONE2",
    )

    assert result.action == "REST_DAY"
    assert result.overridden is True
    assert result.reason == "rhr_fatigue_extreme"


def test_rhr_and_fatigue_do_not_override_modify():
    result = apply_guardrails(
        make_state(rhr_delta=2.0, fatigue=4, acwr=1.0),
        "MODIFY_INTERVALS",
    )

    assert result.action == "MODIFY_INTERVALS"
    assert result.overridden is False
    assert result.reason is None


def test_rhr_boundary_is_strictly_greater_than_1_5():
    result = apply_guardrails(
        make_state(rhr_delta=1.5, fatigue=5, acwr=1.0),
        "MAINTAIN_PLAN",
    )

    assert result.action == "MAINTAIN_PLAN"
    assert result.overridden is False


def test_missing_rhr_does_not_trigger_rhr_rule():
    result = apply_guardrails(
        make_state(rhr_delta=None, fatigue=5, acwr=1.0),
        "MAINTAIN_PLAN",
    )

    assert result.action == "MAINTAIN_PLAN"
    assert result.overridden is False


def test_missing_acwr_does_not_override_maintain():
    result = apply_guardrails(
        make_state(acwr=None),
        "MAINTAIN_PLAN",
    )

    assert result.action == "MAINTAIN_PLAN"
    assert result.overridden is False
    assert result.reason is None


def test_missing_acwr_still_allows_rhr_rule():
    result = apply_guardrails(
        make_state(acwr=None, rhr_delta=2.0, fatigue=4),
        "MAINTAIN_PLAN",
    )

    assert result.action == "SWAP_TO_ZONE2"
    assert result.overridden is True
    assert result.reason == "rhr_fatigue"


def test_missing_acwr_leaves_rest_alone():
    result = apply_guardrails(
        make_state(acwr=None),
        "REST_DAY",
    )

    assert result.action == "REST_DAY"
    assert result.overridden is False
    assert result.reason is None


def test_safe_state_is_unchanged():
    result = apply_guardrails(make_state(), "MAINTAIN_PLAN")

    assert result.action == "MAINTAIN_PLAN"
    assert result.overridden is False
    assert result.reason is None
    assert result.original_action == "MAINTAIN_PLAN"


def test_unknown_action_raises():
    with pytest.raises(ValueError, match="Unknown action"):
        apply_guardrails(make_state(), "NOT_A_REAL_ACTION")


@pytest.mark.parametrize(
    ("overrides", "router_action", "expected_action", "expected_reason"),
    [
        (
            {"injury_reported": 1, "acwr": 1.8, "fatigue": 4},
            "MAINTAIN_PLAN",
            "ESCALATE_TO_HUMAN",
            "injury_signal",
        ),
        (
            {"injury_reported": 1, "acwr": 2.5},
            "MAINTAIN_PLAN",
            "ESCALATE_TO_HUMAN",
            "injury_signal",
        ),
        (
            {"acwr": 1.8, "fatigue": 2},
            "MAINTAIN_PLAN",
            "MODIFY_INTERVALS",
            "acwr_danger",
        ),
        (
            {"acwr": 2.5, "fatigue": 2},
            "MAINTAIN_PLAN",
            "REST_DAY",
            "acwr_extreme",
        ),
        (
            {"acwr": 1.0, "rhr_delta": 2.5, "fatigue": 4},
            "MAINTAIN_PLAN",
            "SWAP_TO_ZONE2",
            "rhr_fatigue",
        ),
        (
            {"acwr": 1.0, "rhr_delta": 2.5, "fatigue": 5},
            "SWAP_TO_ZONE2",
            "REST_DAY",
            "rhr_fatigue_extreme",
        ),
        (
            {"acwr": None, "fatigue": 3},
            "MAINTAIN_PLAN",
            "MAINTAIN_PLAN",
            None,
        ),
        (
            {"acwr": 1.0, "rhr_delta": 0.0, "fatigue": 3},
            "MAINTAIN_PLAN",
            "MAINTAIN_PLAN",
            None,
        ),
    ],
)
def test_adversarial_safety_cases(
    overrides,
    router_action,
    expected_action,
    expected_reason,
):
    result = apply_guardrails(make_state(**overrides), router_action)

    assert result.action == expected_action
    assert result.reason == expected_reason
    assert result.overridden is (router_action != expected_action)
