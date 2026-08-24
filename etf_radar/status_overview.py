"""Aggregated production status overview across ETF pipeline artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SEVERITY_P0 = "P0_CRITICAL"
SEVERITY_P1 = "P1_HIGH"
SEVERITY_P2 = "P2_WARNING"
SEVERITY_P3 = "P3_NOTICE"
SEVERITY_P4 = "P4_HEALTHY"

_SEVERITY_ORDER = [SEVERITY_P0, SEVERITY_P1, SEVERITY_P2, SEVERITY_P3, SEVERITY_P4]

BLOCKED_CYCLE_STATUSES = {
    "CALIBRATION_FAILED_SAFE_FALLBACK",
    "COST_MODEL_RECALIBRATION_REQUIRED",
    "COST_MODEL_SHADOW_VALIDATION_FAILED_SAFE_CASH",
    "EXECUTION_FEEDBACK_EVIDENCE_BLOCKED_SAFE_CASH",
    "LIVE_PERFORMANCE_EVIDENCE_BLOCKED_SAFE_CASH",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _worst(severities: List[str]) -> str:
    for level in _SEVERITY_ORDER:
        if level in severities:
            return level
    return SEVERITY_P4


def _check_cycle_status(public_dir: Path) -> Dict[str, Any]:
    payload = _read_json(public_dir / "cycle_status_latest.json")
    if not isinstance(payload, dict):
        return {"severity": SEVERITY_P1, "detail": "cycle_status_latest.json unavailable"}
    status = str(payload.get("status", ""))
    severity = SEVERITY_P0 if status in BLOCKED_CYCLE_STATUSES else SEVERITY_P4
    return {"severity": severity, "status": status}


def _check_factor_health(public_dir: Path) -> Dict[str, Any]:
    payload = _read_json(public_dir / "factor_health_latest.json")
    if not isinstance(payload, dict):
        return {"severity": SEVERITY_P1, "detail": "factor_health_latest.json unavailable"}
    status = str(payload.get("status", ""))
    if status == "SUSPENDED":
        severity = SEVERITY_P1
    elif status in ("WATCH", "WARMUP"):
        severity = SEVERITY_P2
    else:
        severity = SEVERITY_P4
    return {
        "severity": severity,
        "status": status,
        "approved_for_live_use": bool(payload.get("approved_for_live_use", False)),
    }


def _check_live_performance(public_dir: Path) -> Dict[str, Any]:
    payload = _read_json(public_dir / "live_performance_audit_latest.json")
    if not isinstance(payload, dict):
        return {"severity": SEVERITY_P2, "detail": "live_performance_audit_latest.json unavailable"}
    recal = bool(payload.get("recalibration_required", False))
    status = str(payload.get("status", ""))
    if recal:
        severity = SEVERITY_P0
    elif status == "WARMUP":
        severity = SEVERITY_P2
    else:
        severity = SEVERITY_P4
    return {
        "severity": severity,
        "status": status,
        "recalibration_required": recal,
    }


def _check_execution_feedback(public_dir: Path) -> Dict[str, Any]:
    payload = _read_json(public_dir / "execution_feedback_audit_latest.json")
    if not isinstance(payload, dict):
        return {"severity": SEVERITY_P2, "detail": "execution_feedback_audit_latest.json unavailable"}
    overdue = int(payload.get("overdue_execution_count", 0) or 0)

    overdue += int(payload.get("overdue_confirmation_count", 0) or 0)
    degraded = bool(payload.get("sustained_cost_degradation", False))
    if degraded or overdue > 0:
        severity = SEVERITY_P1
    elif payload.get("errors"):
        severity = SEVERITY_P2
    else:
        severity = SEVERITY_P4
    return {
        "severity": severity,
        "status": str(payload.get("status", "")),
        "overdue_count": overdue,
        "sustained_cost_degradation": degraded,
    }


def _check_exposure_ratchet(state_dir: Path) -> Dict[str, Any]:
    payload = _read_json(state_dir / "exposure_ratchet_history.json")
    if not isinstance(payload, list) or not payload:
        return {"severity": SEVERITY_P4, "exposure_ratchet_applied": False}
    latest = payload[-1]
    applied = bool(isinstance(latest, dict) and latest.get("exposure_ratchet_applied"))
    severity = SEVERITY_P3 if applied else SEVERITY_P4
    return {
        "severity": severity,
        "exposure_ratchet_applied": applied,
        "max_exposure_ratio": (latest.get("max_exposure_ratio") if isinstance(latest, dict) else None),
        "cooldown_note": "ratchet limited exposure recovery pace" if applied else None,
    }


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_status_overview(
    public_dir: Path,
    state_dir: Path,
    *,
    generated_at: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    checks = {
        "cycle_status": _check_cycle_status(public_dir),
        "factor_health": _check_factor_health(public_dir),
        "live_performance": _check_live_performance(public_dir),
        "execution_feedback": _check_execution_feedback(public_dir),
        "exposure_ratchet": _check_exposure_ratchet(state_dir),
    }
    overall = _worst([c["severity"] for c in checks.values()])
    overview = {
        "schema_version": 1,
        "policy_version": "production-status-overview-v1",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "overall_severity": overall,
        "checks": checks,
    }
    if output_path is not None:
        _atomic_json_write(output_path, overview)
    return overview


__all__ = [
    "build_status_overview",
    "SEVERITY_P0",
    "SEVERITY_P1",
    "SEVERITY_P2",
    "SEVERITY_P3",
    "SEVERITY_P4",
]

