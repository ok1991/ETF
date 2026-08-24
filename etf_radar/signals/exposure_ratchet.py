from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

EXPOSURE_RATCHET_DAILY_STEP = 0.30
EXPOSURE_RATCHET_RECOVERY_COOLDOWN_DAYS = 5

EXPOSURE_RATCHET_TRIGGER_STATES = {"RISK_OFF", "HARD_DEFENSIVE"}

def apply_exposure_ratchet(
    policy: Mapping[str, Any],
    recent_history: Iterable[Mapping[str, Any]] = (),
    daily_step: float = EXPOSURE_RATCHET_DAILY_STEP,
    recovery_cooldown_days: int = EXPOSURE_RATCHET_RECOVERY_COOLDOWN_DAYS,
) -> Dict[str, Any]:
    """Ratchet exposure upward gradually after RISK_OFF or HARD_DEFENSIVE regime.

    Exposure is only throttled when increasing relative to previous day; decreases
    are never delayed. Only engages if a RISK_OFF/HARD_DEFENSIVE regime appeared
    in the recent history window; otherwise the policy passes through untouched.
    """
    result: Dict[str, Any] = dict(policy)
    target_exposure = max(0.0, min(1.0, float(result.get("max_exposure_ratio", 0.0) or 0.0)))
    result["exposure_ratchet_applied"] = False
    result["exposure_ratchet_pre_ratchet_max_exposure_ratio"] = target_exposure
    result["exposure_ratchet_cooldown_remaining_days"] = 0

    history = list(recent_history or [])
    if not history:
        return result

    previous_exposure = max(0.0, min(1.0, float(history[-1].get("max_exposure_ratio", 0.0) or 0.0)))
    if target_exposure <= previous_exposure:
        return result

    window = history[-recovery_cooldown_days:]
    trigger_offset = None
    for offset, record in enumerate(reversed(window)):
        state = str(record.get("state") or record.get("regime_level") or "").upper()
        if state in EXPOSURE_RATCHET_TRIGGER_STATES:
            trigger_offset = offset
            break

    if trigger_offset is None:
        return result

    days_since_trigger = trigger_offset + 1
    cooldown_remaining = max(0, recovery_cooldown_days - days_since_trigger)
    capped_exposure = min(target_exposure, previous_exposure + daily_step)
    if capped_exposure >= target_exposure:
        result["exposure_ratchet_cooldown_remaining_days"] = cooldown_remaining
        return result

    result["max_exposure_ratio"] = round(capped_exposure, 6)
    if capped_exposure >= 0.999:
        result["entry_permission"] = "TRADEABLE"
    elif capped_exposure > 0.0:
        result["entry_permission"] = "MAINLINE_ONLY"
    else:
        result["entry_permission"] = "BLOCKED"
    result["exposure_ratchet_applied"] = True
    result["exposure_ratchet_cooldown_remaining_days"] = cooldown_remaining
    return result


__all__ = [
    "EXPOSURE_RATCHET_DAILY_STEP",
    "EXPOSURE_RATCHET_RECOVERY_COOLDOWN_DAYS",
    "EXPOSURE_RATCHET_TRIGGER_STATES",
    "apply_exposure_ratchet",
]
