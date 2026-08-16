"""Transactional daily refresh and calibration-bundle promotion controller."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .config import PATHS, configure_runtime_paths
from .factor_evolution import FACTOR_EVOLUTION_POLICY_VERSION
from .llm_provider_health import run_provider_health_check
from .model_governance import validate_artifact_time, validate_bundle_member
from .rotation import (
    ROTATION_ACCEPTANCE_POLICY_VERSION,
    ROTATION_EXECUTION_POLICY_VERSION,
)
from .signals.contract import fingerprint_joint_price_directory


CALIBRATION_TRIGGER_GENERATED_DAYS = 14
CALIBRATION_TRIGGER_TRAINING_LAG_DAYS = 53
CALIBRATION_FINGERPRINT_POLICY = "qfq-raw-joint-v2"
FACTOR_HEALTH_RECALIBRATION_COOLDOWN_DAYS = 7
MODEL_SELECTION_POLICY_VERSION = "candidate-first-incumbent-comparison-v1"
FACTOR_HEALTH_STRUCTURAL_REASONS = {
    "UNSUPPORTED_LIVE_MONITOR_FEATURES",
    "ACTIVE_FACTOR_COUNT_BELOW_2",
    "EFFECTIVE_FACTOR_COUNT_BELOW_2",
}
FACTOR_HEALTH_STATISTICAL_REASONS = {
    "LIVE_ENSEMBLE_NEGATIVE_IC",
    "LIVE_ENSEMBLE_EXCESSIVE_TURNOVER",
}
CALIBRATION_FILES = (
    "v4_calibration.json",
    "v4_acceptance_report.json",
    "adaptive_factor_registry.json",
    "llm_factor_proposals.json",
    "rotation_model.json",
)
CORE_MODEL_FILES = (
    "v4_calibration.json",
    "adaptive_factor_registry.json",
    "rotation_model.json",
)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_metric(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rotation_performance(rotation: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Return the common performance evidence used to choose a production model."""
    metrics = rotation.get("portfolio_metrics") or {}
    if not isinstance(metrics, Mapping):
        return None
    result = {
        "excess_return": _finite_metric(metrics.get("excess_return")),
        "information_ratio": _finite_metric(metrics.get("information_ratio")),
        "max_drawdown": _finite_metric(metrics.get("max_drawdown")),
    }
    if any(value is None for value in result.values()):
        return None
    return {key: float(value) for key, value in result.items()}


def compare_rotation_candidates(
    candidate: Mapping[str, Any],
    incumbent: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select the new model unless the incumbent is demonstrably better.

    The comparison is deliberately narrow: it only uses post-cost excess return,
    information ratio, and drawdown from the same rotation backtest contract.
    Older approval, seasoning, and freeze metadata are evidence only, not a veto.
    """
    candidate_metrics = _rotation_performance(candidate)
    incumbent_metrics = _rotation_performance(incumbent or {})
    result: Dict[str, Any] = {
        "policy_version": MODEL_SELECTION_POLICY_VERSION,
        "candidate_model_version": str(candidate.get("version", "")),
        "incumbent_model_version": str((incumbent or {}).get("version", "")),
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
    }
    if candidate_metrics is None:
        return {
            **result,
            "selected": False,
            "reason": "CANDIDATE_PERFORMANCE_EVIDENCE_INVALID",
        }
    if incumbent_metrics is None:
        return {
            **result,
            "selected": True,
            "reason": "CANDIDATE_FIRST_NO_COMPARABLE_INCUMBENT",
        }
    incumbent_fingerprint = str((incumbent or {}).get("data_fingerprint", ""))
    if incumbent_fingerprint:
        current_fingerprint = fingerprint_joint_price_directory(
            str(PATHS.data),
            str((incumbent or {}).get("trained_until", "")),
            policy=CALIBRATION_FINGERPRINT_POLICY,
        )
        if current_fingerprint and current_fingerprint != incumbent_fingerprint:
            return {
                **result,
                "selected": True,
                "reason": "FORCED_PROMOTION_INCUMBENT_FINGERPRINT_STALE",
                "forced": True,
            }

    epsilon = 1e-12
    comparisons = (
        ("excess_return", 1.0),
        ("information_ratio", 1.0),
        ("max_drawdown", -1.0),
    )
    for metric, direction in comparisons:
        difference = direction * (
            candidate_metrics[metric] - incumbent_metrics[metric]
        )
        if difference > epsilon:
            return {
                **result,
                "selected": True,
                "reason": f"CANDIDATE_BETTER_{metric.upper()}",
            }
        if difference < -epsilon:
            return {
                **result,
                "selected": False,
                "reason": f"INCUMBENT_BETTER_{metric.upper()}",
            }
    return {
        **result,
        "selected": True,
        "reason": "CANDIDATE_FIRST_PERFORMANCE_TIE",
    }


def _read_rotation_model(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _apply_selected_candidate(
    staging_dir: Path,
    selection: Mapping[str, Any],
) -> None:
    """Make a comparison winner directly usable by the production loaders."""
    if selection.get("selected") is not True:
        raise ValueError("cannot mark a rejected candidate as production-ready")
    public_selection = dict(selection)
    public_selection["selected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rotation_path = staging_dir / "rotation_model.json"
    rotation = _read_json(rotation_path)
    rotation["approved"] = True
    rotation["production_selection"] = public_selection
    _atomic_json(rotation, rotation_path)

    v4_path = staging_dir / "v4_calibration.json"
    v4 = _read_json(v4_path)
    thresholds = dict(v4.get("thresholds") or {})
    thresholds["approved"] = True
    v4["thresholds"] = thresholds
    v4["production_selection"] = public_selection
    _atomic_json(v4, v4_path)

    report_path = staging_dir / "v4_acceptance_report.json"
    report = _read_json(report_path)
    report["strategy_approved"] = True
    report["production_selection"] = public_selection
    _atomic_json(report, report_path)


@contextmanager
def cycle_lock(path: Path, stale_after_hours: int = 6):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age = datetime.now().timestamp() - path.stat().st_mtime
        if age > max(1, int(stale_after_hours)) * 3600:
            path.unlink(missing_ok=True)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another ETF production cycle holds {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            )
        yield
    finally:
        path.unlink(missing_ok=True)


def calibration_due(
    calibration_dir: Path,
    data_dir: Path,
    *,
    now: Any = None,
    generated_trigger_days: int = CALIBRATION_TRIGGER_GENERATED_DAYS,
    training_trigger_days: int = CALIBRATION_TRIGGER_TRAINING_LAG_DAYS,
) -> List[str]:
    """Return concrete reasons why a full, non-cached walk-forward rerun is due."""
    reasons: List[str] = []
    fingerprints: Dict[str, str] = {}
    for name in CORE_MODEL_FILES:
        path = calibration_dir / name
        if not path.exists():
            reasons.append(f"{name}:MISSING")
            continue
        try:
            value = _read_json(path)
            bundle_status = validate_bundle_member(path, value)
            if bundle_status != "APPROVED":
                reasons.append(f"{name}:{bundle_status}")
            time_status = validate_artifact_time(
                value,
                now=now,
                generated_max_age_days=generated_trigger_days,
                trained_max_lag_days=training_trigger_days,
            )
            if not time_status.approved:
                reasons.append(f"{name}:{time_status.reason}")
            cutoff = str(value.get("trained_until", ""))[:10]
            expected = str(value.get("data_fingerprint", ""))
            if not cutoff or not expected:
                reasons.append(f"{name}:FINGERPRINT_AUTHORITY_MISSING")
                continue
            if cutoff not in fingerprints:
                fingerprints[cutoff] = fingerprint_joint_price_directory(
                    str(data_dir),
                    cutoff,
                    policy=CALIBRATION_FINGERPRINT_POLICY,
                )
            if not fingerprints[cutoff] or fingerprints[cutoff] != expected:
                reasons.append(f"{name}:DATA_FINGERPRINT_MISMATCH")
            if name == "adaptive_factor_registry.json" and str(
                value.get("evolution_policy_version", "")
            ) != FACTOR_EVOLUTION_POLICY_VERSION:
                reasons.append(f"{name}:FACTOR_EVOLUTION_POLICY_MISMATCH")
            if name == "rotation_model.json":
                if str(value.get("factor_evolution_policy_version", "")) != (
                    FACTOR_EVOLUTION_POLICY_VERSION
                ):
                    reasons.append(f"{name}:FACTOR_EVOLUTION_POLICY_MISMATCH")
                if str(value.get("execution_policy_version", "")) != (
                    ROTATION_EXECUTION_POLICY_VERSION
                ):
                    reasons.append(f"{name}:EXECUTION_POLICY_MISMATCH")
                if str(value.get("acceptance_policy_version", "")) != (
                    ROTATION_ACCEPTANCE_POLICY_VERSION
                ):
                    reasons.append(f"{name}:ACCEPTANCE_POLICY_MISMATCH")
        except Exception as error:
            reasons.append(f"{name}:INVALID:{str(error)[:160]}")
    return list(dict.fromkeys(reasons))


def factor_health_recalibration_due(
    calibration_dir: Path,
    factor_health_path: Path,
    *,
    now: Any = None,
    cooldown_days: int = FACTOR_HEALTH_RECALIBRATION_COOLDOWN_DAYS,
) -> List[str]:
    """Trigger GP/ML evolution only for mature failures of a promoted registry."""
    registry_path = calibration_dir / "adaptive_factor_registry.json"
    if not registry_path.exists() or not factor_health_path.exists():
        return []
    try:
        registry = _read_json(registry_path)
        health = _read_json(factor_health_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    if registry.get("approved") is not True:
        return []
    if health.get("status") != "SUSPENDED":
        return []
    evidence_mature = health.get("evidence_mature") is True
    hard_reasons = [
        str(reason)
        for reason in list(health.get("reasons") or [])
        if str(reason) in FACTOR_HEALTH_STRUCTURAL_REASONS
        or (
            evidence_mature
            and (
                str(reason) in FACTOR_HEALTH_STATISTICAL_REASONS
                or str(reason).startswith("LIVE_FACTOR_NEGATIVE_IC:")
            )
        )
    ]
    if not hard_reasons:
        return []
    generated_at = pd.to_datetime(registry.get("generated_at"), errors="coerce")
    reference = pd.Timestamp(now or datetime.now())
    if pd.isna(generated_at):
        return []
    if getattr(generated_at, "tzinfo", None) is not None:
        generated_at = generated_at.tz_localize(None)
    if getattr(reference, "tzinfo", None) is not None:
        reference = reference.tz_localize(None)
    age_days = (reference - pd.Timestamp(generated_at)).total_seconds() / 86400.0
    if age_days < max(0, int(cooldown_days)):
        return []
    return [f"FACTOR_LIVE_HEALTH_HARD_FAILURE:{reason}" for reason in hard_reasons]


def validate_staged_bundle(
    staging_dir: Path,
    data_dir: Path,
    *,
    now: Any = None,
) -> Dict[str, Any]:
    """Validate a complete calibration set before any production file is replaced."""
    missing = [name for name in CALIBRATION_FILES if not (staging_dir / name).exists()]
    if missing:
        raise ValueError("staged calibration bundle missing: " + ", ".join(missing))
    artifacts = {name: _read_json(staging_dir / name) for name in CALIBRATION_FILES}
    v4 = artifacts["v4_calibration.json"]
    report = artifacts["v4_acceptance_report.json"]
    registry = artifacts["adaptive_factor_registry.json"]
    llm = artifacts["llm_factor_proposals.json"]
    rotation = artifacts["rotation_model.json"]

    schemas = {
        "v4_calibration.json": 4,
        "v4_acceptance_report.json": 4,
        "adaptive_factor_registry.json": 2,
        "rotation_model.json": 1,
    }
    for name, expected_schema in schemas.items():
        if int(artifacts[name].get("schema_version", 0)) != expected_schema:
            raise ValueError(f"{name} schema mismatch")
    if not str(llm.get("status", "")) or not str(llm.get("generated_at", "")):
        raise ValueError("LLM proposal audit is incomplete")
    if str(llm.get("status", "")) in {
        "OK",
        "CACHED",
        "CACHED_OFFLINE",
        "CACHED_PROVIDER_FAILURE",
    } and not all(
        str(llm.get(field, "")).strip()
        for field in ("provider", "model", "model_identity", "endpoint_fingerprint")
    ):
        raise ValueError("successful LLM proposal audit has no provider identity")

    bundle_ids = {
        str(value.get("artifact_bundle_id", ""))
        for value in (v4, report, registry, rotation)
    }
    if len(bundle_ids) != 1 or not next(iter(bundle_ids), ""):
        raise ValueError("calibration artifacts do not share one bundle id")
    bundle_id = next(iter(bundle_ids))

    fingerprints = {
        str(value.get("data_fingerprint", ""))
        for value in (v4, report, registry, rotation)
    }
    if len(fingerprints) != 1 or not next(iter(fingerprints), ""):
        raise ValueError("calibration artifacts do not share one data fingerprint")
    fingerprint = next(iter(fingerprints))

    cutoffs = {
        str(value.get("trained_until", ""))[:10]
        for value in (v4, report, registry, rotation)
    }
    if len(cutoffs) != 1 or not next(iter(cutoffs), ""):
        raise ValueError("calibration artifacts do not share one training cutoff")
    trained_until = next(iter(cutoffs))

    for name, value in (
        ("v4_calibration.json", v4),
        ("adaptive_factor_registry.json", registry),
        ("rotation_model.json", rotation),
    ):
        status = validate_artifact_time(value, now=now)
        if not status.approved:
            raise ValueError(f"{name} time authority failed: {status.reason}")

    current_fingerprint = fingerprint_joint_price_directory(
        str(data_dir),
        trained_until,
        policy=CALIBRATION_FINGERPRINT_POLICY,
    )
    if not current_fingerprint or current_fingerprint != fingerprint:
        raise ValueError("staged bundle does not match current QFQ+RAW history")
    if str(report.get("calibration_version", "")) != str(v4.get("version", "")):
        raise ValueError("acceptance report and V4 model versions differ")
    if report.get("walk_forward_method") != (
        "expanding_calendar_windows_with_20d_purge_and_5d_embargo"
    ):
        raise ValueError("walk-forward method is not the approved purged protocol")

    folds = list(report.get("folds", []))
    ok_folds = [item for item in folds if item.get("status") == "OK"]
    if len(ok_folds) < 6:
        raise ValueError("fewer than six valid purged walk-forward folds")
    for fold in ok_folds:
        if fold.get("factor_purge_method") != (
            "28_calendar_day_approx_20_trading_day_purge"
        ):
            raise ValueError(f"{fold.get('name', 'UNKNOWN')} factor purge is invalid")
        train_end = pd.to_datetime(fold.get("train_end"), errors="coerce")
        validate_start = pd.to_datetime(fold.get("validate_start"), errors="coerce")
        if pd.isna(train_end) or pd.isna(validate_start):
            raise ValueError(f"{fold.get('name', 'UNKNOWN')} fold dates are invalid")
        if (pd.Timestamp(validate_start) - pd.Timestamp(train_end)).days < 28:
            raise ValueError(f"{fold.get('name', 'UNKNOWN')} purge gap is too short")

    if str(rotation.get("acceptance_policy_version", "")) != (
        ROTATION_ACCEPTANCE_POLICY_VERSION
    ):
        raise ValueError("rotation acceptance policy is stale")
    rotation_metrics = _rotation_performance(rotation)
    if rotation_metrics is None:
        raise ValueError("rotation model has no comparable performance evidence")

    return {
        "schema_version": 1,
        "artifact_bundle_id": bundle_id,
        "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_at": str(report.get("generated_at", "")),
        "trained_until": trained_until,
        "data_fingerprint": fingerprint,
        "calibration_version": str(v4.get("version", "")),
        "rotation_model_version": str(rotation.get("version", "")),
        "rotation_performance": rotation_metrics,
        "factor_registry_approved": bool(registry.get("approved", False)),
        "rotation_approved": bool(rotation.get("approved", False)),
        "llm_status": str(llm.get("status", "")),
        "valid_purged_fold_count": len(ok_folds),
        "files": {
            name: {"sha256": _sha256(staging_dir / name)}
            for name in CALIBRATION_FILES
        },
    }


def assert_production_promotion_allowed(
    manifest: Mapping[str, Any],
    calibration_dir: Path,
) -> None:
    """Reject only a candidate that the comparable-performance selector rejected.

    Historical approval flags and frozen production pins do not override a newer
    candidate.  `calibration_dir` remains part of the signature for callers that
    perform the comparison before promotion.
    """
    del calibration_dir
    selection = manifest.get("production_selection")
    if isinstance(selection, Mapping) and selection.get("selected") is not True:
        raise ValueError(
            "production promotion refused: incumbent has better comparable performance"
        )


def promote_staged_bundle(
    staging_dir: Path,
    calibration_dir: Path,
    manifest: Mapping[str, Any],
    *,
    bypass_protection: bool = False,
) -> None:
    """Promote all artifacts with rollback; publish the authority manifest last."""
    if not bypass_protection:
        assert_production_promotion_allowed(manifest, calibration_dir)
    bundle_id = str(manifest.get("artifact_bundle_id", ""))
    if not bundle_id:
        raise ValueError("bundle manifest has no id")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = calibration_dir / ".bundle_backups" / bundle_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, bool] = {}
    for name in CALIBRATION_FILES:
        target = calibration_dir / name
        existing[name] = target.exists()
        if target.exists():
            shutil.copy2(target, backup_dir / name)

    promoted: List[str] = []
    try:
        for name in CALIBRATION_FILES:
            source = staging_dir / name
            temporary = calibration_dir / f".{name}.{bundle_id}.tmp"
            shutil.copy2(source, temporary)
            os.replace(temporary, calibration_dir / name)
            promoted.append(name)
        _atomic_json(manifest, calibration_dir / "calibration_bundle.json")
    except Exception:
        for name in reversed(promoted):
            target = calibration_dir / name
            backup = backup_dir / name
            if existing.get(name) and backup.exists():
                temporary = calibration_dir / f".{name}.{bundle_id}.rollback"
                shutil.copy2(backup, temporary)
                os.replace(temporary, target)
            elif not existing.get(name):
                target.unlink(missing_ok=True)
        raise


def _seed_staging(staging_dir: Path, calibration_dir: Path) -> None:
    registry_source = calibration_dir / "adaptive_factor_registry.json"
    if registry_source.exists():
        shutil.copy2(registry_source, staging_dir / registry_source.name)
    configured_llm_cache = os.environ.get("LLM_FACTOR_CACHE_SOURCE", "").strip()
    llm_source = (
        Path(configured_llm_cache)
        if configured_llm_cache
        else calibration_dir / "llm_factor_proposals.json"
    )
    if configured_llm_cache and not llm_source.is_file():
        raise ValueError("configured LLM factor cache source is missing")
    if llm_source.exists():
        shutil.copy2(llm_source, staging_dir / "llm_factor_proposals.json")


def refresh_llm_staging_cache(
    staging_dir: Path,
    runtime_dir: Path,
) -> Dict[str, Any]:
    """Refresh Gemini research candidates without weakening cached fallback safety."""
    if os.environ.get("LLM_FACTOR_CACHE_SOURCE", "").strip():
        return {"status": "EXPLICIT_CACHE_PINNED", "refreshed": False}
    if os.environ.get("LLM_FACTOR_PROPOSALS_ENABLED", "auto").strip().lower() == "false":
        return {"status": "LLM_PROPOSALS_DISABLED", "refreshed": False}
    if os.environ.get("LLM_CYCLE_PROVIDER_REFRESH", "true").strip().lower() == "false":
        return {"status": "CYCLE_PROVIDER_REFRESH_DISABLED", "refreshed": False}
    health_path = runtime_dir / "audits" / "llm_provider_health_latest.json"
    proposal_path = (
        runtime_dir
        / "llm-shadow"
        / "provider-health"
        / "llm_factor_proposals.json"
    )
    try:
        proposal_count = int(os.environ.get("LLM_FACTOR_PROPOSAL_COUNT", "6"))
    except ValueError:
        proposal_count = 6
    try:
        health = run_provider_health_check(
            artifact_path=health_path,
            proposal_path=proposal_path,
            proposal_count=proposal_count,
        )
    except Exception as error:
        return {
            "status": "PROVIDER_HEALTH_ERROR_CACHE_PRESERVED",
            "refreshed": False,
            "health_artifact": str(health_path),
            "error": str(error)[:1000],
        }
    result = {
        "status": "PROVIDER_UNHEALTHY_CACHE_PRESERVED",
        "refreshed": False,
        "health_artifact": str(health_path),
        "provider": health.get("provider"),
        "model": health.get("model"),
        "proposal_count": int(health.get("proposal_count", 0) or 0),
        "error_code": str(health.get("error_code", "")),
    }
    expected_sha = str(health.get("cache_sha256", ""))
    if (
        health.get("status") == "OK"
        and health.get("refresh_allowed") is True
        and proposal_path.is_file()
        and len(expected_sha) == 64
        and _sha256(proposal_path) == expected_sha
    ):
        target = staging_dir / "llm_factor_proposals.json"
        shutil.copy2(proposal_path, target)
        result.update(
            {
                "status": "REFRESHED_FROM_HEALTHY_GEMINI",
                "refreshed": True,
                "proposal_artifact": str(proposal_path),
                "cache_sha256": expected_sha,
            }
        )
    return result


def _run_calibration(
    staging_dir: Path,
    sample_step: int,
    workers: int,
) -> Dict[str, Any]:
    llm_refresh = refresh_llm_staging_cache(staging_dir, PATHS.runtime)
    command = [
        sys.executable,
        str(PATHS.root / "calibrate_v4.py"),
        "--data-dir",
        str(PATHS.data),
        "--sample-step",
        str(max(1, int(sample_step))),
        "--workers",
        str(max(1, int(workers))),
        "--rows-cache",
        str(PATHS.state / "v4_calibration_rows.json"),
        "--reuse-rows-cache",
        "--calibration-out",
        str(staging_dir / "v4_calibration.json"),
        "--report-out",
        str(staging_dir / "v4_acceptance_report.json"),
        "--factor-registry-out",
        str(staging_dir / "adaptive_factor_registry.json"),
        "--rotation-model-out",
        str(staging_dir / "rotation_model.json"),
    ]
    subprocess.run(command, cwd=PATHS.root, check=True)
    return llm_refresh


def _run_cost_shadow_validation(
    recommendation_path: Path,
    output_dir: Path,
    data_dir: Path,
    sample_step: int,
    workers: int,
) -> None:
    command = [
        sys.executable,
        str(PATHS.root / "calibrate_v4.py"),
        "--data-dir",
        str(data_dir),
        "--sample-step",
        str(max(1, int(sample_step))),
        "--workers",
        str(max(1, int(workers))),
        "--cost-model-candidate",
        str(recommendation_path),
        "--shadow-output-dir",
        str(output_dir),
        "--reuse-rows-cache",
    ]
    subprocess.run(command, cwd=PATHS.root, check=True)


def ensure_cost_shadow_validation(
    recommendation_path: Path,
    runtime_dir: Path,
    data_dir: Path,
    *,
    sample_step: int = 5,
    workers: int = 6,
) -> Dict[str, Any]:
    if not recommendation_path.exists():
        return {"status": "CANDIDATE_MISSING", "attempted": False}
    try:
        candidate = _read_json(recommendation_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "CANDIDATE_INVALID",
            "attempted": False,
            "error": str(error)[:500],
        }
    if candidate.get("status") != "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION":
        return {
            "status": "CANDIDATE_NOT_READY",
            "attempted": False,
            "candidate_status": str(candidate.get("status", "")),
        }
    fingerprint = str(candidate.get("candidate_fingerprint", "")).lower()
    if not (
        len(fingerprint) == 64
        and all(char in "0123456789abcdef" for char in fingerprint)
    ):
        return {"status": "CANDIDATE_INVALID", "attempted": False}
    output_dir = runtime_dir / "cost-shadow" / fingerprint[:16]
    manifest_path = output_dir / "shadow_cost_validation_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}
        if (
            manifest.get("status") == "SHADOW_VALIDATION_COMPLETE"
            and manifest.get("shadow_only") is True
            and manifest.get("promotion_allowed") is False
            and str(manifest.get("candidate_fingerprint", "")) == fingerprint
        ):
            return {
                "status": "REUSED",
                "attempted": False,
                "candidate_fingerprint": fingerprint,
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "rotation_strategy_approved_under_candidate_costs": bool(
                    manifest.get("rotation_strategy_approved_under_candidate_costs", False)
                ),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_cost_shadow_validation(
        recommendation_path,
        output_dir,
        data_dir,
        sample_step,
        workers,
    )
    manifest = _read_json(manifest_path)
    if not (
        manifest.get("status") == "SHADOW_VALIDATION_COMPLETE"
        and manifest.get("shadow_only") is True
        and manifest.get("promotion_allowed") is False
        and str(manifest.get("candidate_fingerprint", "")) == fingerprint
    ):
        raise RuntimeError("cost shadow validation manifest failed integrity checks")
    return {
        "status": "COMPLETED",
        "attempted": True,
        "candidate_fingerprint": fingerprint,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "rotation_strategy_approved_under_candidate_costs": bool(
            manifest.get("rotation_strategy_approved_under_candidate_costs", False)
        ),
    }


def _production_run() -> None:
    from .pipeline import run

    run()


def _refresh_joint_health() -> None:
    """Regenerate joint_health_latest.json after the cycle so the dashboard
    does not display a stale distribution or execution-link status."""
    try:
        from .joint_health import build_joint_health

        swing_root = PATHS.root.parent / "Swing-trading"
        build_joint_health(
            PATHS.root,
            swing_root,
            output_path=PATHS.public / "joint_health_latest.json",
        )
    except Exception:
        pass


def _write_cycle_status(value: Mapping[str, Any]) -> None:
    _refresh_joint_health()
    payload = dict(value)
    distribution_path = PATHS.public / "distribution_audit_latest.json"
    if distribution_path.is_file():
        try:
            distribution = _read_json(distribution_path)
            payload.update(
                {
                    "distribution_audit": str(distribution_path),
                    "remote_distribution_status": str(
                        distribution.get("status", "UNKNOWN")
                    ),
                    "same_host_execution_allowed": bool(
                        distribution.get("same_host_execution_allowed", False)
                    ),
                    "remote_only_execution_allowed": bool(
                        distribution.get("remote_only_execution_allowed", False)
                    ),
                }
            )
        except Exception as error:
            payload.update(
                {
                    "distribution_audit": str(distribution_path),
                    "remote_distribution_status": "AUDIT_ARTIFACT_INVALID",
                    "same_host_execution_allowed": False,
                    "remote_only_execution_allowed": False,
                    "distribution_audit_error": str(error)[:1000],
                }
            )
    if isinstance(value, dict):
        value.update(payload)
    _atomic_json(payload, PATHS.state / "cycle_status_latest.json")
    _atomic_json(payload, PATHS.public / "cycle_status_latest.json")


def run_cycle(
    *,
    force_calibration: bool = False,
    sample_step: int = 5,
    workers: int = 6,
) -> Dict[str, Any]:
    configure_runtime_paths()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with cycle_lock(PATHS.state / "production_cycle.lock"):
        _production_run()
        execution_audit_path = PATHS.public / "execution_feedback_audit_latest.json"
        if execution_audit_path.exists():
            execution_audit = _read_json(execution_audit_path)
            if execution_audit.get("status") == "COST_MODEL_RECALIBRATION_REQUIRED":
                recommendation_path = (
                    PATHS.public / "execution_cost_recalibration_latest.json"
                )
                try:
                    shadow_validation = ensure_cost_shadow_validation(
                        recommendation_path,
                        PATHS.runtime,
                        PATHS.data,
                        sample_step=sample_step,
                        workers=workers,
                    )
                except Exception as error:
                    status = {
                        "schema_version": 1,
                        "status": "COST_MODEL_SHADOW_VALIDATION_FAILED_SAFE_CASH",
                        "started_at": started_at,
                        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "calibration_attempted": False,
                        "shadow_calibration_attempted": True,
                        "reasons": ["BROKER_CONFIRMED_COST_DEGRADATION"],
                        "rotation_authority_allowed": False,
                        "execution_feedback_audit": str(execution_audit_path),
                        "cost_recalibration_candidate": str(recommendation_path),
                        "error": str(error)[:2000],
                    }
                    _write_cycle_status(status)
                    return status
                status = {
                    "schema_version": 1,
                    "status": "COST_MODEL_RECALIBRATION_REQUIRED",
                    "started_at": started_at,
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calibration_attempted": False,
                    "shadow_calibration_attempted": bool(
                        shadow_validation.get("attempted", False)
                    ),
                    "reasons": ["BROKER_CONFIRMED_COST_DEGRADATION"],
                    "rotation_authority_allowed": False,
                    "execution_feedback_audit": str(execution_audit_path),
                    "cost_recalibration_candidate": str(recommendation_path),
                    "shadow_validation": shadow_validation,
                }
                _write_cycle_status(status)
                return status
            if execution_audit.get("rotation_authority_allowed") is False:
                status = {
                    "schema_version": 1,
                    "status": "EXECUTION_FEEDBACK_EVIDENCE_BLOCKED_SAFE_CASH",
                    "started_at": started_at,
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calibration_attempted": False,
                    "reasons": [str(execution_audit.get("status", "UNKNOWN"))],
                    "rotation_authority_allowed": False,
                    "execution_feedback_audit": str(execution_audit_path),
                }
                _write_cycle_status(status)
                return status
        reasons = factor_health_recalibration_due(
            PATHS.calibration,
            PATHS.public / "factor_health_latest.json",
        )
        live_performance_audit_path = (
            PATHS.public / "live_performance_audit_latest.json"
        )
        if live_performance_audit_path.exists():
            live_performance_audit = _read_json(live_performance_audit_path)
            if (
                live_performance_audit.get("rotation_authority_allowed") is False
                and live_performance_audit.get("recalibration_required") is not True
            ):
                status = {
                    "schema_version": 1,
                    "status": "LIVE_PERFORMANCE_EVIDENCE_BLOCKED_SAFE_CASH",
                    "started_at": started_at,
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calibration_attempted": False,
                    "reasons": [
                        str(live_performance_audit.get("status", "UNKNOWN"))
                    ],
                    "rotation_authority_allowed": False,
                    "live_performance_audit": str(live_performance_audit_path),
                }
                _write_cycle_status(status)
                return status
            if live_performance_audit.get("recalibration_required") is True:
                reasons.append(
                    "LIVE_PERFORMANCE_RECALIBRATION_REQUIRED:"
                    + str(live_performance_audit.get("status", "UNKNOWN"))
                )
        reasons.extend(calibration_due(PATHS.calibration, PATHS.data))
        if force_calibration:
            reasons.insert(0, "FORCED_CALIBRATION")
        if not reasons:
            status = {
                "schema_version": 1,
                "status": "UP_TO_DATE",
                "started_at": started_at,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_attempted": False,
                "reasons": [],
            }
            _write_cycle_status(status)
            return status

        staging_dir = Path(
            tempfile.mkdtemp(prefix="calibration-", dir=str(PATHS.runtime))
        )
        _seed_staging(staging_dir, PATHS.calibration)
        llm_refresh: Dict[str, Any] = {}
        try:
            llm_refresh = _run_calibration(staging_dir, sample_step, workers)
            candidate_manifest = validate_staged_bundle(staging_dir, PATHS.data)
            candidate_rotation = _read_json(staging_dir / "rotation_model.json")
            incumbent_rotation = _read_rotation_model(
                PATHS.calibration / "rotation_model.json"
            )
            selection = compare_rotation_candidates(
                candidate_rotation,
                incumbent_rotation,
            )
            if selection.get("selected") is not True:
                status = {
                    "schema_version": 1,
                    "status": "CALIBRATION_RETAINED_INCUMBENT",
                    "started_at": started_at,
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calibration_attempted": True,
                    "reasons": reasons,
                    "staging_dir": str(staging_dir),
                    "artifact_bundle_id": candidate_manifest["artifact_bundle_id"],
                    "rotation_approved": candidate_manifest["rotation_approved"],
                    "factor_registry_approved": candidate_manifest[
                        "factor_registry_approved"
                    ],
                    "llm_status": candidate_manifest["llm_status"],
                    "llm_research_refresh": llm_refresh,
                    "production_bundle_preserved": True,
                    "model_selection": selection,
                }
                _write_cycle_status(status)
                return status
            _apply_selected_candidate(staging_dir, selection)
            manifest = validate_staged_bundle(staging_dir, PATHS.data)
            manifest["production_selection"] = selection
            manifest["rotation_approved"] = True
            try:
                assert_production_promotion_allowed(manifest, PATHS.calibration)
            except ValueError as gate_error:
                status = {
                    "schema_version": 1,
                    "status": "CALIBRATION_STAGED_NOT_PROMOTED",
                    "started_at": started_at,
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calibration_attempted": True,
                    "reasons": reasons,
                    "error": str(gate_error)[:2000],
                    "staging_dir": str(staging_dir),
                    "artifact_bundle_id": manifest["artifact_bundle_id"],
                    "rotation_approved": manifest["rotation_approved"],
                    "factor_registry_approved": manifest["factor_registry_approved"],
                    "llm_status": manifest["llm_status"],
                    "llm_research_refresh": llm_refresh,
                    "production_bundle_preserved": True,
                    "model_selection": selection,
                }
                _write_cycle_status(status)
                return status
            promote_staged_bundle(staging_dir, PATHS.calibration, manifest)
            previous_force = os.environ.get("FORCE_DOWNLOAD")
            os.environ["FORCE_DOWNLOAD"] = "false"
            try:
                _production_run()
            finally:
                if previous_force is None:
                    os.environ.pop("FORCE_DOWNLOAD", None)
                else:
                    os.environ["FORCE_DOWNLOAD"] = previous_force
            status = {
                "schema_version": 1,
                "status": "CALIBRATION_PROMOTED",
                "started_at": started_at,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_attempted": True,
                "reasons": reasons,
                "artifact_bundle_id": manifest["artifact_bundle_id"],
                "rotation_approved": manifest["rotation_approved"],
                "factor_registry_approved": manifest["factor_registry_approved"],
                "llm_status": manifest["llm_status"],
                "llm_research_refresh": llm_refresh,
                "model_selection": selection,
            }
            _write_cycle_status(status)
            shutil.rmtree(staging_dir, ignore_errors=True)
            return status
        except Exception as error:
            status = {
                "schema_version": 1,
                "status": "CALIBRATION_FAILED_SAFE_FALLBACK",
                "started_at": started_at,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_attempted": True,
                "reasons": reasons,
                "error": str(error)[:2000],
                "staging_dir": str(staging_dir),
                "llm_research_refresh": llm_refresh,
            }
            _write_cycle_status(status)
            return status


def assert_last_cycle_healthy() -> None:
    configure_runtime_paths()
    status = _read_json(PATHS.state / "cycle_status_latest.json")
    if status.get("status") == "CALIBRATION_FAILED_SAFE_FALLBACK":
        raise RuntimeError("calibration failed; safe outputs were retained and published")
    if status.get("status") == "COST_MODEL_RECALIBRATION_REQUIRED":
        raise RuntimeError(
            "broker-confirmed costs invalidated the approved cost authority; "
            "rotation remains cash-only until the cost policy is recalibrated"
        )
    if status.get("status") == "COST_MODEL_SHADOW_VALIDATION_FAILED_SAFE_CASH":
        raise RuntimeError(
            "cost authority is revoked and its isolated shadow validation failed; "
            "rotation remains cash-only"
        )
    if status.get("status") == "EXECUTION_FEEDBACK_EVIDENCE_BLOCKED_SAFE_CASH":
        raise RuntimeError(
            "execution feedback evidence is rejected or broker confirmation is overdue; "
            "rotation remains cash-only"
        )
    if status.get("status") == "LIVE_PERFORMANCE_EVIDENCE_BLOCKED_SAFE_CASH":
        raise RuntimeError(
            "live performance evidence is invalid or a risk breach is cooling down; "
            "rotation remains cash-only"
        )


__all__ = [
    "CALIBRATION_FILES",
    "assert_last_cycle_healthy",
    "assert_production_promotion_allowed",
    "calibration_due",
    "factor_health_recalibration_due",
    "ensure_cost_shadow_validation",
    "promote_staged_bundle",
    "run_cycle",
    "validate_staged_bundle",
]
