from dataclasses import dataclass 

ACWR_DANGER = 1.50
ACWR_EXTREME = 2.00
RHR_DELTA_ALERT = 1.5
FATIGUE_HIGH = 4
FATIGUE_EXTREME = 5

ACTIONS = {
    "MAINTAIN_PLAN",
    "SWAP_TO_ZONE2",
    "REST_DAY",
    "MODIFY_INTERVALS",
    "ESCALATE_TO_HUMAN",
}


@dataclass
class GuardrailResult:
    action: str
    overridden: bool
    reason: str | None
    original_action: str


def apply_guardrails(state: dict, action: str) -> GuardrailResult:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    original = action

    # 1. Injury always wins.
    if state.get("injury_reported") == 1:
        return GuardrailResult(
            action="ESCALATE_TO_HUMAN",
            overridden=action != "ESCALATE_TO_HUMAN",
            reason="injury_signal",
            original_action=original,
        )

    # Never downgrade an escalation.
    if action == "ESCALATE_TO_HUMAN":
        return GuardrailResult(action, False, None, original)

    acwr = state.get("acwr")
    fatigue = state.get("fatigue")
    rhr_delta = state.get("rhr_delta")

    # 2. ACWR danger zone.
    if acwr is not None and acwr >= ACWR_DANGER:
        if acwr >= ACWR_EXTREME:
            return GuardrailResult(
                action="REST_DAY",
                overridden=action != "REST_DAY",
                reason="acwr_extreme",
                original_action=original,
            )
        if action in ("MAINTAIN_PLAN", "SWAP_TO_ZONE2"):
            return GuardrailResult(
                action="MODIFY_INTERVALS",
                overridden=True,
                reason="acwr_danger",
                original_action=original,
            )

    # 3. Elevated RHR plus high fatigue.
    if rhr_delta is not None and rhr_delta > RHR_DELTA_ALERT:
        if fatigue is not None and fatigue >= FATIGUE_HIGH:
            if action == "MAINTAIN_PLAN":
                return GuardrailResult(
                    action="SWAP_TO_ZONE2",
                    overridden=True,
                    reason="rhr_fatigue",
                    original_action=original,
                )
            if action == "SWAP_TO_ZONE2" and fatigue >= FATIGUE_EXTREME:
                return GuardrailResult(
                    action="REST_DAY",
                    overridden=True,
                    reason="rhr_fatigue_extreme",
                    original_action=original,
                )

    # 4. Missing ACWR is a data-quality flag, not an automatic action change.
    # The model was trained with missing values; overriding here broke 16/18
    # correct MAINTAIN_PLAN decisions on the held-out split.
    return GuardrailResult(action, False, None, original)
