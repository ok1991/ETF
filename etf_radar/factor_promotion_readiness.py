"""Auditable readiness for adaptive-factor promotion, separate from rotation authority."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .factor_evolution import factor_registry_identity


POLICY_VERSION = "adaptive-factor-promotion-readiness-v2"


def _read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "ic_mean",
            "ic_ir",
            "recent_ic_mean",
            "recent_ic_ir",
            "turnover",
            "ic_observations",
            "multiple_testing_q_value",
            "status",
            "reasons",
        )
        if key in value
    }


def build_factor_promotion_readiness(
    registry_path: Path,
    factor_health_path: Path,
    output_path: Path,
    *,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    registry = _read_object(registry_path)
    health = _read_object(factor_health_path)
    expected_identity = factor_registry_identity(registry)
    identity_errors = [
        f"{field.upper()}_MISMATCH"
        for field, expected in expected_identity.items()
        if str(health.get(field, "")) != str(expected)
    ]
    identity_match = not identity_errors
    approval_reasons = [str(value) for value in registry.get("approval_reasons", [])]
    seasoned = registry.get("policy_seasoned") is True
    selection_passed = "FACTOR_SELECTION_GATE_FAILED" not in approval_reasons
    independent_passed = (
        "INDEPENDENT_ENSEMBLE_HOLDOUT_GATE_FAILED" not in approval_reasons
    )
    effective_count_passed = "EFFECTIVE_FACTOR_COUNT_BELOW_2" not in approval_reasons
    approved = bool(
        registry.get("approved") is True
        and health.get("approved_for_live_use") is True
        and identity_match
    )
    if not identity_match:
        status = "LIVE_HEALTH_REGISTRY_MISMATCH"
    elif approved:
        status = "APPROVED_FOR_LIVE_OVERLAY"
    elif not seasoned and not selection_passed:
        status = "WAITING_FOR_NEW_LABELLED_DATES_AND_STRONGER_CANDIDATES"
    elif not seasoned:
        status = "WAITING_FOR_POLICY_SEASONING"
    elif not selection_passed:
        status = "STRONGER_CANDIDATES_REQUIRED"
    elif not independent_passed:
        status = "INDEPENDENT_HOLDOUT_REJECTED"
    elif not effective_count_passed:
        status = "INSUFFICIENT_EFFECTIVE_FACTOR_COUNT"
    else:
        status = "LIVE_HEALTH_NOT_APPROVED"

    diagnostics = []
    for item in registry.get("candidate_diagnostics", []):
        if not isinstance(item, Mapping):
            continue
        diagnostics.append(
            {
                "name": str(item.get("name", "")),
                "candidate_origin": str(item.get("candidate_origin", "")),
                "expression_text": str(item.get("expression_text", "")),
                "selection_score": item.get("selection_score"),
                "accepted": item.get("accepted") is True,
                "rejection_reasons": [
                    str(value) for value in item.get("rejection_reasons", [])
                ],
                "train_metrics": _metric_summary(item.get("train_metrics") or {}),
                "selection_metrics": _metric_summary(
                    item.get("selection_metrics") or {}
                ),
            }
        )
    diagnostics.sort(
        key=lambda item: float(item.get("selection_score") or float("-inf")),
        reverse=True,
    )
    candidate_gate_summary = registry.get("candidate_gate_summary") or {}
    accepted_candidate_count = int(
        candidate_gate_summary.get(
            "accepted_count", sum(item["accepted"] for item in diagnostics)
        )
        or 0
    )
    rejection_counts = dict(
        candidate_gate_summary.get("rejection_counts")
        or registry.get("rejection_counts")
        or {}
    )
    unseen = int(registry.get("policy_unseen_date_count", 0) or 0)
    minimum = int(registry.get("policy_seasoning_min_dates", 0) or 0)
    result = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "generated_at": (generated_at or datetime.now()).astimezone().isoformat(
            timespec="seconds"
        ),
        "status": status,
        "promotion_allowed": approved,
        "rotation_authority_independent": True,
        "registry": {
            "path": str(registry_path),
            "sha256": _sha256(registry_path),
            "generated_at": registry.get("generated_at"),
            "trained_until": registry.get("trained_until"),
            "approved": registry.get("approved") is True,
            "approval_reasons": approval_reasons,
            "evolution_policy_version": registry.get("evolution_policy_version"),
        },
        "live_health": {
            "path": str(factor_health_path),
            "sha256": _sha256(factor_health_path),
            "status": health.get("status"),
            "approved_for_live_use": health.get("approved_for_live_use") is True,
            "reasons": [str(value) for value in health.get("reasons", [])],
            **{
                field: health.get(field)
                for field in expected_identity
            },
        },
        "registry_health_identity": {
            "match": identity_match,
            "errors": identity_errors,
            "expected": expected_identity,
        },
        "gates": {
            "registry_health_identity_match": identity_match,
            "policy_seasoned": seasoned,
            "factor_selection_passed": selection_passed,
            "independent_ensemble_holdout_passed": independent_passed,
            "effective_factor_count_passed": effective_count_passed,
            "live_health_approved": health.get("approved_for_live_use") is True,
        },
        "policy_seasoning": {
            "required": registry.get("policy_seasoning_required") is True,
            "anchor": registry.get("policy_seasoning_anchor"),
            "unseen_labelled_date_count": unseen,
            "minimum_unseen_labelled_dates": minimum,
            "remaining_unseen_labelled_dates": max(minimum - unseen, 0),
            "candidate_specification_fingerprint": registry.get(
                "candidate_specification_fingerprint"
            ),
            "candidate_specification_changed": registry.get(
                "policy_candidate_specification_changed"
            )
            is True,
            "previous_candidate_specification_fingerprint_valid": registry.get(
                "previous_candidate_specification_fingerprint_valid", True
            )
            is True,
        },
        "candidate_summary": {
            "candidate_count": int(registry.get("candidate_count", len(diagnostics)) or 0),
            "accepted_candidate_count": accepted_candidate_count,
            "research_challengers": list(registry.get("research_challengers", [])),
            "llm_research_challengers": list(
                registry.get("llm_research_challengers", [])
            ),
            "llm_proposals_submitted": int(
                registry.get("llm_proposals_submitted", 0) or 0
            ),
            "llm_proposals_considered": int(
                registry.get("llm_proposals_considered", 0) or 0
            ),
            "llm_proposals_skipped_rejected_cooldown": list(
                registry.get("llm_proposals_skipped_rejected_cooldown", []) or []
            ),
            "llm_candidate_trial_history_count": len(
                registry.get("llm_candidate_trial_history", []) or []
            ),
            "candidate_origins": dict(registry.get("candidate_origins") or {}),
            "rejection_counts": rejection_counts,
            "gate_summary": dict(candidate_gate_summary),
            "origin_gate_summary": dict(
                registry.get("candidate_origin_gate_summary") or {}
            ),
            "diagnostic_coverage": dict(
                registry.get("candidate_diagnostic_coverage") or {}
            ),
            "top_candidate_diagnostics": diagnostics[:10],
        },
        "next_action": (
            "REGENERATE_LIVE_FACTOR_HEALTH_FOR_CURRENT_REGISTRY"
            if not identity_match
            else "KEEP_CURRENT_ROTATION_AND_REEVALUATE_AFTER_NEW_LABELLED_DATES"
            if not approved
            else "MONITOR_LIVE_FACTOR_HEALTH"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return result


__all__ = ["POLICY_VERSION", "build_factor_promotion_readiness"]
