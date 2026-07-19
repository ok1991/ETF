"""Joint ETF producer and Swing executor health gate."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from jsonschema import Draft202012Validator

from .execution_feedback_audit import feedback_evidence_errors
from .live_performance_audit import live_performance_errors
from .rotation_contract import validate_rotation_contract


BLOCKED_CYCLE_STATUSES = {
    "CALIBRATION_FAILED_SAFE_FALLBACK",
    "COST_MODEL_RECALIBRATION_REQUIRED",
    "COST_MODEL_SHADOW_VALIDATION_FAILED_SAFE_CASH",
    "EXECUTION_FEEDBACK_EVIDENCE_BLOCKED_SAFE_CASH",
    "LIVE_PERFORMANCE_EVIDENCE_BLOCKED_SAFE_CASH",
}
IDENTITY_FIELDS = (
    "model_version",
    "execution_date",
    "execution_policy_version",
    "acceptance_policy_version",
    "strategy_specification_fingerprint",
    "target_weights",
    "cash_weight",
)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feedback_hash_valid(value: Mapping[str, Any]) -> bool:
    expected = str(value.get("feedback_id", ""))
    if len(expected) != 64:
        return False
    payload = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "generated_at",
            "feedback_id",
            "state_reconciliation_applied",
            "state_reconciliation",
        }
    }
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return actual == expected


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _schema_errors(payload: Mapping[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    return [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(dict(payload)),
            key=lambda error: list(error.path),
        )
    ]


def build_joint_health(
    etf_root: Path,
    swing_root: Path,
    *,
    now: Optional[datetime] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    current = (now or datetime.now().astimezone()).astimezone()
    today = current.strftime("%Y-%m-%d")
    blocking: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}

    calibration = etf_root / "artifacts" / "calibration"
    public = etf_root / "public"
    state = etf_root / ".runtime" / "state"
    swing_public = swing_root / "public"
    swing_state = swing_root / "runtime" / "state"
    swing_cache = swing_root / "runtime" / "cache" / "etf_rotation_latest.json"

    try:
        manifest = _read_json(calibration / "calibration_bundle.json")
        bundle_id = str(manifest.get("artifact_bundle_id", ""))
        file_errors = []
        for name, metadata in (manifest.get("files") or {}).items():
            path = calibration / str(name)
            expected = str((metadata or {}).get("sha256", ""))
            if not path.is_file() or not expected or _sha256(path) != expected:
                file_errors.append(str(name))
                continue
            artifact = _read_json(path)
            artifact_bundle = str(artifact.get("artifact_bundle_id", ""))
            if artifact_bundle and artifact_bundle != bundle_id:
                file_errors.append(str(name) + ":BUNDLE_ID_MISMATCH")
        checks["calibration_bundle"] = {
            "bundle_id": bundle_id,
            "valid": bool(bundle_id and not file_errors),
            "file_errors": file_errors,
            "valid_purged_fold_count": manifest.get("valid_purged_fold_count"),
        }
        if not bundle_id or file_errors:
            blocking.append("CALIBRATION_BUNDLE_INVALID")
    except Exception as error:
        checks["calibration_bundle"] = {"valid": False, "error": str(error)[:1000]}
        blocking.append("CALIBRATION_BUNDLE_UNAVAILABLE")
        manifest = {}

    rotation_path = public / "etf_rotation_latest.json"
    try:
        rotation = _read_json(rotation_path)
        rotation_errors = validate_rotation_contract(rotation) + _schema_errors(
            rotation,
            etf_root / "contracts" / "etf_rotation_v2.schema.json",
        )
        checks["rotation"] = {
            "valid": not rotation_errors,
            "model_version": rotation.get("model_version"),
            "execution_date": rotation.get("execution_date"),
            "errors": rotation_errors[:20],
        }
        if rotation_errors:
            blocking.append("ROTATION_CONTRACT_INVALID")
        authority_model = _read_json(calibration / "rotation_model.json")
        authority_match = bool(
            str(authority_model.get("version", ""))
            == str(rotation.get("model_version", ""))
            and str(authority_model.get("artifact_bundle_id", ""))
            == str(manifest.get("artifact_bundle_id", ""))
            and authority_model.get("approved") is True
        )
        checks["rotation"]["calibration_authority_match"] = authority_match
        if not authority_match:
            blocking.append("PUBLIC_ROTATION_AUTHORITY_MISMATCH")
    except Exception as error:
        rotation = {}
        checks["rotation"] = {"valid": False, "error": str(error)[:1000]}
        blocking.append("ROTATION_UNAVAILABLE")

    try:
        data_manifest = _read_json(public / "data_manifest_latest.json")
        approved = data_manifest.get("approved") is True
        checks["market_data"] = {
            "approved": approved,
            "expected_latest_data_date": data_manifest.get("expected_latest_data_date"),
            "current_count": data_manifest.get("current_count"),
            "required_count": data_manifest.get("required_count"),
        }
        if not approved:
            blocking.append("MARKET_DATA_NOT_APPROVED")
    except Exception as error:
        checks["market_data"] = {"approved": False, "error": str(error)[:1000]}
        blocking.append("MARKET_DATA_MANIFEST_UNAVAILABLE")

    try:
        promotion_path = public / "factor_promotion_readiness_latest.json"
        promotion = _read_json(promotion_path)
        promotion_registry = promotion.get("registry") or {}
        promotion_health = promotion.get("live_health") or {}
        promotion_identity = promotion.get("registry_health_identity") or {}
        promotion_errors = []
        if promotion.get("rotation_authority_independent") is not True:
            promotion_errors.append("ROTATION_AUTHORITY_NOT_INDEPENDENT")
        if str(promotion_registry.get("sha256", "")) != _sha256(
            calibration / "adaptive_factor_registry.json"
        ):
            promotion_errors.append("FACTOR_REGISTRY_SHA256_MISMATCH")
        if str(promotion_health.get("sha256", "")) != _sha256(
            public / "factor_health_latest.json"
        ):
            promotion_errors.append("FACTOR_HEALTH_SHA256_MISMATCH")
        if promotion_identity.get("match") is False:
            promotion_errors.append("FACTOR_HEALTH_REGISTRY_IDENTITY_MISMATCH")
        checks["factor_promotion_readiness"] = {
            "valid": not promotion_errors,
            "status": promotion.get("status"),
            "promotion_allowed": promotion.get("promotion_allowed") is True,
            "remaining_unseen_labelled_dates": (
                (promotion.get("policy_seasoning") or {}).get(
                    "remaining_unseen_labelled_dates"
                )
            ),
            "accepted_candidate_count": (
                (promotion.get("candidate_summary") or {}).get(
                    "accepted_candidate_count"
                )
            ),
            "registry_health_identity_match": promotion_identity.get("match"),
            "errors": promotion_errors,
        }
        if promotion_errors:
            warnings.append("FACTOR_PROMOTION_READINESS_STALE_OR_INVALID")
        elif promotion.get("promotion_allowed") is not True:
            warnings.append("ADAPTIVE_FACTOR_PROMOTION_NOT_READY")
    except Exception as error:
        checks["factor_promotion_readiness"] = {
            "valid": False,
            "error": str(error)[:1000],
        }
        warnings.append("FACTOR_PROMOTION_READINESS_UNAVAILABLE")

    execution_date = str(rotation.get("execution_date", ""))[:10]
    pretrade_path = (
        etf_root
        / ".runtime"
        / "audits"
        / f"pretrade_shadow_{execution_date.replace('-', '')}.json"
    )
    try:
        pretrade = _read_json(pretrade_path)
        pretrade_rotation = pretrade.get("rotation") or {}
        pretrade_manifest = pretrade.get("market_data_manifest") or {}
        pretrade_errors = []
        if pretrade.get("status") != "READY_FOR_EXECUTION_DATE_QUOTE_REVALIDATION":
            pretrade_errors.append("STATUS_NOT_READY")
        if pretrade.get("shadow_only") is not True:
            pretrade_errors.append("SHADOW_ONLY_NOT_TRUE")
        if pretrade.get("order_submission_allowed") is not False:
            pretrade_errors.append("ORDER_SUBMISSION_NOT_DISABLED")
        if pretrade.get("state_persisted") is not False:
            pretrade_errors.append("STATE_PERSISTED")
        if str(pretrade_rotation.get("model_version", "")) != str(
            rotation.get("model_version", "")
        ):
            pretrade_errors.append("MODEL_VERSION_MISMATCH")
        if str(pretrade_rotation.get("execution_date", ""))[:10] != execution_date:
            pretrade_errors.append("EXECUTION_DATE_MISMATCH")
        if str(pretrade_rotation.get("strategy_specification_fingerprint", "")) != str(
            rotation.get("strategy_specification_fingerprint", "")
        ):
            pretrade_errors.append("STRATEGY_FINGERPRINT_MISMATCH")
        if str(pretrade_rotation.get("sha256", "")) != _sha256(rotation_path):
            pretrade_errors.append("ROTATION_SHA256_MISMATCH")
        manifest_path = public / "data_manifest_latest.json"
        if str(pretrade_manifest.get("sha256", "")) != _sha256(manifest_path):
            pretrade_errors.append("MARKET_MANIFEST_SHA256_MISMATCH")
        if pretrade.get("errors") not in ([], None):
            pretrade_errors.append("PRETRADE_REPORTED_ERRORS")
        checks["pretrade_shadow"] = {
            "valid": not pretrade_errors,
            "status": pretrade.get("status"),
            "path": str(pretrade_path),
            "order_count": len(pretrade.get("reference_orders") or []),
            "estimated_execution_cost": (
                (pretrade.get("portfolio_result") or {}).get("estimated_execution_cost")
            ),
            "errors": pretrade_errors,
        }
        if pretrade_errors:
            blocking.append("PRETRADE_SHADOW_INVALID")
    except Exception as error:
        checks["pretrade_shadow"] = {"valid": False, "error": str(error)[:1000]}
        blocking.append("PRETRADE_SHADOW_UNAVAILABLE")

    try:
        cycle = _read_json(public / "cycle_status_latest.json")
        cycle_status = str(cycle.get("status", ""))
        checks["cycle"] = {"status": cycle_status}
        if cycle_status in BLOCKED_CYCLE_STATUSES:
            blocking.append("ETF_CYCLE_BLOCKED:" + cycle_status)
    except Exception as error:
        checks["cycle"] = {"status": "UNKNOWN", "error": str(error)[:1000]}
        blocking.append("ETF_CYCLE_STATUS_UNAVAILABLE")

    try:
        distribution = _read_json(public / "distribution_audit_latest.json")
        local_allowed = distribution.get("same_host_execution_allowed") is True
        remote_allowed = distribution.get("remote_only_execution_allowed") is True
        checks["distribution"] = {
            "status": distribution.get("status"),
            "same_host_execution_allowed": local_allowed,
            "remote_only_execution_allowed": remote_allowed,
        }
        if not local_allowed:
            blocking.append("LOCAL_DISTRIBUTION_AUTHORITY_BLOCKED")
        if not remote_allowed:
            warnings.append("REMOTE_ONLY_DISTRIBUTION_BLOCKED")
    except Exception as error:
        local_allowed = False
        remote_allowed = False
        checks["distribution"] = {"status": "UNKNOWN", "error": str(error)[:1000]}
        blocking.append("DISTRIBUTION_AUDIT_UNAVAILABLE")

    release_root = (etf_root / ".runtime" / "distribution-release").resolve()
    try:
        release = _read_json(release_root / "distribution_release_latest.json")
        payload_path = Path(str(release.get("payload_path", ""))).resolve()
        release_errors = []
        if payload_path != release_root and release_root not in payload_path.parents:
            release_errors.append("PAYLOAD_PATH_OUTSIDE_RELEASE_ROOT")
        if not payload_path.is_file():
            release_errors.append("PAYLOAD_FILE_UNAVAILABLE")
        elif str(release.get("payload_file_sha256", "")) != _sha256(payload_path):
            release_errors.append("PAYLOAD_FILE_SHA256_MISMATCH")
        if str(release.get("payload_canonical_sha256", "")) != str(
            distribution.get("local_payload_sha256", "")
        ):
            release_errors.append("PAYLOAD_CANONICAL_SHA256_MISMATCH")
        if str(release.get("model_version", "")) != str(
            rotation.get("model_version", "")
        ):
            release_errors.append("MODEL_VERSION_MISMATCH")
        if str(release.get("execution_date", ""))[:10] != execution_date:
            release_errors.append("EXECUTION_DATE_MISMATCH")
        if release.get("status") not in {
            "READY_FOR_EXTERNAL_PUBLISH",
            "ALREADY_DISTRIBUTED",
        }:
            release_errors.append("RELEASE_STATUS_INVALID")
        checks["distribution_release"] = {
            "valid": not release_errors,
            "status": release.get("status"),
            "release_id": release.get("release_id"),
            "payload_path": str(payload_path),
            "errors": release_errors,
        }
        if not remote_allowed:
            warnings.append(
                "REMOTE_RELEASE_READY_FOR_PUBLISH"
                if not release_errors
                else "REMOTE_RELEASE_NOT_READY"
            )
    except Exception as error:
        checks["distribution_release"] = {
            "valid": False,
            "error": str(error)[:1000],
        }
        if not remote_allowed:
            warnings.append("REMOTE_RELEASE_NOT_READY")

    try:
        cached = _read_json(swing_cache)
        mismatches = [
            field for field in IDENTITY_FIELDS
            if cached.get(field) != rotation.get(field)
        ]
        checks["swing_cache"] = {
            "valid": not mismatches,
            "model_version": cached.get("model_version"),
            "mismatched_fields": mismatches,
        }
        if mismatches:
            blocking.append("SWING_ROTATION_CACHE_MISMATCH")
    except Exception as error:
        checks["swing_cache"] = {"valid": False, "error": str(error)[:1000]}
        blocking.append("SWING_ROTATION_CACHE_UNAVAILABLE")

    lock_path = swing_state / "swing_execution.lock"
    if lock_path.exists():
        lock_age = max(0.0, current.timestamp() - lock_path.stat().st_mtime)
        checks["swing_lock"] = {"present": True, "age_seconds": round(lock_age, 3)}
        if lock_age <= 120 * 60:
            blocking.append("SWING_EXECUTION_ALREADY_RUNNING")
        else:
            warnings.append("SWING_STALE_LOCK_PRESENT")
    else:
        checks["swing_lock"] = {"present": False}

    scheduler_ready = False
    scheduler_path = etf_root / ".runtime" / "audits" / "windows_scheduler_latest.json"
    try:
        scheduler = _read_json(scheduler_path)
        verified_at = datetime.fromisoformat(str(scheduler.get("generated_at", "")))
        if verified_at.tzinfo is None:
            verified_at = verified_at.astimezone()
        age_hours = (current - verified_at.astimezone(current.tzinfo)).total_seconds() / 3600
        expected_count = int(scheduler.get("expected_task_count", 0) or 0)
        installed_count = int(scheduler.get("installed_task_count", 0) or 0)
        enabled_count = int(scheduler.get("enabled_task_count", 0) or 0)
        scheduler_ready = bool(
            scheduler.get("policy_version")
            == "windows-closed-loop-scheduler-audit-v1"
            and scheduler.get("status") == "READY"
            and scheduler.get("automation_execution_ready") is True
            and expected_count == 3
            and installed_count == expected_count
            and enabled_count == expected_count
            and -0.1 <= age_hours <= 24
        )
        checks["automation_scheduler"] = {
            "status": scheduler.get("status"),
            "automation_execution_ready": scheduler_ready,
            "expected_task_count": expected_count,
            "installed_task_count": installed_count,
            "enabled_task_count": enabled_count,
            "audit_age_hours": round(age_hours, 3),
            "path": str(scheduler_path.resolve()),
        }
        if not scheduler_ready:
            warnings.append("AUTOMATION_SCHEDULER_NOT_READY")
    except Exception as error:
        checks["automation_scheduler"] = {
            "status": "NOT_AUDITED",
            "automation_execution_ready": False,
            "path": str(scheduler_path.resolve()),
            "error": str(error)[:1000],
        }
        warnings.append("AUTOMATION_SCHEDULER_NOT_AUDITED")

    try:
        feedback_audit = _read_json(public / "execution_feedback_audit_latest.json")
        feedback_authority = feedback_audit.get("rotation_authority_allowed") is True
        checks["execution_feedback_audit"] = {
            "status": feedback_audit.get("status"),
            "rotation_authority_allowed": feedback_authority,
        }
        if not feedback_authority:
            blocking.append("EXECUTION_FEEDBACK_AUTHORITY_BLOCKED")
    except Exception as error:
        checks["execution_feedback_audit"] = {"error": str(error)[:1000]}
        blocking.append("EXECUTION_FEEDBACK_AUDIT_UNAVAILABLE")

    try:
        performance_audit = _read_json(public / "live_performance_audit_latest.json")
        performance_authority = performance_audit.get("rotation_authority_allowed") is True
        checks["live_performance_audit"] = {
            "status": performance_audit.get("status"),
            "rotation_authority_allowed": performance_authority,
        }
        if not performance_authority:
            blocking.append("LIVE_PERFORMANCE_AUTHORITY_BLOCKED")
    except Exception as error:
        checks["live_performance_audit"] = {"error": str(error)[:1000]}
        blocking.append("LIVE_PERFORMANCE_AUDIT_UNAVAILABLE")

    model_version = str(rotation.get("model_version", ""))
    direct_evidence = {"phase": "UNKNOWN"}
    if execution_date:
        if today < execution_date:
            direct_evidence = {"phase": "AWAITING_EXECUTION_DATE"}
        else:
            feedback_path = swing_public / "execution_feedback_latest.json"
            feedback_history_path = swing_public / "execution_feedback_history.json"
            performance_path = swing_public / "live_performance_latest.json"
            feedback_ok = False
            performance_ok = False
            feedback_source = ""
            feedback_errors: list[str] = []

            def feedback_valid(value: Mapping[str, Any]) -> tuple[bool, list[str]]:
                errors = _schema_errors(
                    value,
                    swing_root / "contracts" / "execution_feedback_v1.schema.json",
                )
                errors.extend(feedback_evidence_errors(value))
                if not _feedback_hash_valid(value):
                    errors.append("FEEDBACK_FINGERPRINT_MISMATCH")
                orders = value.get("orders")
                order_rows = orders if isinstance(orders, list) else []
                evidence_level = str(value.get("evidence_level", ""))
                broker_confirmed = value.get("broker_confirmed") is True
                if order_rows:
                    if evidence_level != "BROKER_CONFIRMED":
                        errors.append("ORDER_FEEDBACK_NOT_BROKER_CONFIRMED")
                    if not broker_confirmed:
                        errors.append("BROKER_CONFIRMATION_FALSE")
                    digest = str(value.get("broker_evidence_file_sha256", ""))
                    if len(digest) != 64 or any(
                        character not in "0123456789abcdef"
                        for character in digest.lower()
                    ):
                        errors.append("BROKER_EVIDENCE_FINGERPRINT_INVALID")
                    if not dict(value.get("broker_evidence") or {}):
                        errors.append("BROKER_EVIDENCE_EMPTY")
                    if str(value.get("broker_fill_completion_status", "")) not in {
                        "COMPLETE",
                        "PARTIAL",
                        "UNFILLED",
                    }:
                        errors.append("BROKER_FILL_COMPLETION_INVALID")
                else:
                    if evidence_level != "NO_ORDERS":
                        errors.append("ZERO_ORDER_EVIDENCE_LEVEL_INVALID")
                    if broker_confirmed:
                        errors.append("ZERO_ORDER_BROKER_CONFIRMATION_INVALID")
                    reason_codes = value.get("decision_reason_codes")
                    reason_set = (
                        {str(item) for item in reason_codes}
                        if isinstance(reason_codes, list)
                        else set()
                    )
                    rebalance_required = value.get("rebalance_required") is True
                    valid_no_order_reason = bool(
                        (
                            reason_set == {"PLAN_ALREADY_APPLIED"}
                            and not rebalance_required
                        )
                        or (
                            reason_set == {"PORTFOLIO_ALREADY_AT_TARGET"}
                            and rebalance_required
                        )
                    )
                    if not valid_no_order_reason:
                        errors.append("ZERO_ORDER_DECISION_REASON_INVALID")
                valid = bool(
                    not errors
                    and str(value.get("model_version", "")) == model_version
                    and str(value.get("execution_date", ""))[:10] == execution_date
                    and str(value.get("run_date", ""))[:10] == execution_date
                    and value.get("quote_tradeable") is True
                    and value.get("state_write_allowed") is True
                )
                return valid, errors

            feedback_events: list[tuple[str, Dict[str, Any]]] = []
            try:
                history = _read_json(feedback_history_path)
                events = history.get("events")
                if not isinstance(events, list):
                    raise ValueError("execution feedback history events is not an array")
                feedback_events.extend(
                    ("history", item) for item in reversed(events) if isinstance(item, dict)
                )
            except Exception as error:
                feedback_errors.append("HISTORY_UNAVAILABLE:" + str(error)[:500])
            try:
                feedback_events.append(("latest", _read_json(feedback_path)))
            except Exception as error:
                feedback_errors.append("LATEST_UNAVAILABLE:" + str(error)[:500])
            for source_name, event in feedback_events:
                valid, event_errors = feedback_valid(event)
                if valid:
                    feedback_ok = True
                    feedback_source = source_name
                    feedback_errors = []
                    break
                feedback_errors.extend(event_errors[:5])
            if not feedback_ok:
                feedback_errors.append("NO_MATCHING_VALID_EXECUTION_EVENT")
            try:
                performance = _read_json(performance_path)
                performance_errors = _schema_errors(
                    performance,
                    swing_root / "contracts" / "live_performance_v1.schema.json",
                )
                performance_errors.extend(live_performance_errors(performance))
                if str(performance.get("portfolio_state_evidence", "")) not in {
                    "BROKER_RECONCILED",
                    "NO_EXECUTION_REQUIRED",
                }:
                    performance_errors.append(
                        "LIVE_PERFORMANCE_STATE_EVIDENCE_NOT_FINAL"
                    )
                performance_ok = bool(
                    not performance_errors
                    and str(performance.get("model_version", "")) == model_version
                    and str(performance.get("data_date", ""))[:10] == execution_date
                )
            except Exception:
                performance_errors = ["UNAVAILABLE"]
            direct_evidence = {
                "phase": "EXECUTION_DATE" if today == execution_date else "POST_EXECUTION_DATE",
                "feedback_valid": feedback_ok,
                "feedback_source": feedback_source,
                "performance_valid": performance_ok,
                "feedback_errors": feedback_errors[:10],
                "performance_errors": performance_errors[:10],
            }
            if today == execution_date:
                if not feedback_ok:
                    warnings.append("EXECUTION_FEEDBACK_NOT_YET_VALID")
                if not performance_ok:
                    warnings.append("LIVE_PERFORMANCE_NOT_YET_VALID")
            else:
                if not feedback_ok:
                    blocking.append("POST_EXECUTION_FEEDBACK_MISSING_OR_INVALID")
                if not performance_ok:
                    blocking.append("POST_EXECUTION_PERFORMANCE_MISSING_OR_INVALID")
    checks["direct_execution_evidence"] = direct_evidence

    blocking = list(dict.fromkeys(blocking))
    warnings = list(dict.fromkeys(warnings))
    status = "BLOCKED" if blocking else ("READY_LOCAL_ONLY" if not remote_allowed else "READY")
    result = {
        "schema_version": 1,
        "policy_version": "etf-swing-joint-health-v1",
        "generated_at": current.isoformat(timespec="seconds"),
        "status": status,
        "same_host_execution_allowed": not blocking and local_allowed,
        "remote_only_execution_allowed": not blocking and remote_allowed,
        "automation_execution_ready": scheduler_ready,
        "model_version": model_version,
        "execution_date": execution_date,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "checks": checks,
    }
    if output_path is not None:
        _atomic_json(result, output_path)
    return result


__all__ = ["BLOCKED_CYCLE_STATUSES", "IDENTITY_FIELDS", "build_joint_health"]
