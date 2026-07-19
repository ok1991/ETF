"""ETF V4 production pipeline."""

from __future__ import annotations

import json
import os
import shutil

from jsonschema import Draft202012Validator

from . import _core
from .config import PATHS, configure_runtime_paths
from .distribution_audit import audit_rotation_distribution
from .distribution_release import prepare_distribution_release
from .factor_promotion_readiness import build_factor_promotion_readiness
from .reporting import HTMLReporter
from .rotation_contract import validate_rotation_contract


def _publish_contract_assets() -> None:
    schema_source = PATHS.root / "contracts" / "etf_signal_v4.schema.json"
    schema_target = PATHS.public / "schema" / "etf-signal-v4.json"
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_source, schema_target)


def verify_public_signal() -> None:
    signal_path = PATHS.public / "etf_signals_latest.json"
    if not signal_path.exists():
        raise RuntimeError("V4 signal output was not generated")
    payload = json.loads(signal_path.read_text(encoding="utf-8"))
    schema_path = PATHS.root / "contracts" / "etf_signal_v4.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    errors = _core.validate_signal_contract(payload)
    errors.extend(error.message for error in schema_errors)
    if payload.get("schema_version") != _core.V4_SCHEMA_VERSION:
        errors.insert(0, "payload.schema_version must be 4")
    if errors:
        raise RuntimeError("V4 signal contract invalid: " + "; ".join(errors[:20]))


def verify_data_manifest() -> None:
    path = PATHS.public / "data_manifest_latest.json"
    if not path.exists():
        raise RuntimeError("market data manifest was not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "generated_at",
        "expected_latest_data_date",
        "approved",
        "required_count",
        "current_count",
        "records",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError("market data manifest missing fields: " + ", ".join(missing))
    if os.environ.get("REQUIRE_FRESH_MARKET_DATA", "false").lower() == "true" and not payload.get("approved"):
        raise RuntimeError(
            "fresh market data required but manifest is blocked: "
            + str(payload.get("reason", "UNKNOWN"))
        )


def verify_rotation_target() -> None:
    path = PATHS.public / "etf_rotation_latest.json"
    if not path.exists():
        raise RuntimeError("rotation V2 target was not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_path = PATHS.root / "contracts" / "etf_rotation_v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    errors = validate_rotation_contract(payload)
    errors.extend(error.message for error in schema_errors)
    if errors:
        raise RuntimeError("rotation V2 contract invalid: " + "; ".join(errors[:20]))


def verify_factor_health() -> None:
    path = PATHS.public / "factor_health_latest.json"
    if not path.exists():
        raise RuntimeError("live factor health artifact was not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "generated_at", "status", "approved_for_live_use", "reasons"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError("live factor health artifact missing fields: " + ", ".join(missing))
    if payload.get("status") not in {"ACTIVE", "WATCH", "WARMUP", "SUSPENDED"}:
        raise RuntimeError("live factor health status is invalid")


def verify_factor_promotion_readiness() -> None:
    path = PATHS.public / "factor_promotion_readiness_latest.json"
    if not path.exists():
        raise RuntimeError("factor promotion readiness artifact was not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "generated_at",
        "status",
        "promotion_allowed",
        "gates",
        "policy_seasoning",
        "candidate_summary",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(
            "factor promotion readiness artifact missing fields: " + ", ".join(missing)
        )


def run() -> None:
    configure_runtime_paths()
    _core.HTMLReporter = HTMLReporter
    _core.main()
    verify_data_manifest()
    verify_public_signal()
    verify_rotation_target()
    verify_factor_health()
    build_factor_promotion_readiness(
        PATHS.calibration / "adaptive_factor_registry.json",
        PATHS.public / "factor_health_latest.json",
        PATHS.public / "factor_promotion_readiness_latest.json",
    )
    verify_factor_promotion_readiness()
    distribution = audit_rotation_distribution(
        PATHS.public / "etf_rotation_latest.json",
        PATHS.public / "distribution_audit_latest.json",
        PATHS.root / "contracts" / "etf_rotation_v2.schema.json",
    )
    prepare_distribution_release(
        PATHS.public / "etf_rotation_latest.json",
        PATHS.root / "contracts" / "etf_rotation_v2.schema.json",
        distribution,
        PATHS.runtime / "distribution-release",
    )
    _publish_contract_assets()


__all__ = [
    "run",
    "verify_data_manifest",
    "verify_factor_health",
    "verify_factor_promotion_readiness",
    "verify_public_signal",
    "verify_rotation_target",
]
