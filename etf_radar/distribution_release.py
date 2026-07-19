"""Prepare an exact, tamper-evident rotation payload for an external publisher."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

from jsonschema import Draft202012Validator

from .rotation_contract import validate_rotation_contract


POLICY_VERSION = "rotation-distribution-release-v1"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_distribution_release(
    rotation_path: Path,
    schema_path: Path,
    distribution_audit: Mapping[str, Any],
    release_root: Path,
) -> Dict[str, Any]:
    payload = json.loads(rotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rotation release source is not a JSON object")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    contract_errors = validate_rotation_contract(payload) + [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(payload),
            key=lambda error: list(error.path),
        )
    ]
    if contract_errors:
        raise ValueError("rotation release source is invalid: " + "; ".join(contract_errors[:20]))
    payload_sha = _canonical_sha256(payload)
    release_id = (
        f"{str(payload.get('execution_date', '')).replace('-', '')}-"
        f"{payload_sha[:16]}"
    )
    release_dir = release_root / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    payload_target = release_dir / "etf_rotation_latest.json"
    temporary_payload = payload_target.with_suffix(payload_target.suffix + ".tmp")
    temporary_payload.write_bytes(rotation_path.read_bytes())
    os.replace(temporary_payload, payload_target)
    if _canonical_sha256(json.loads(payload_target.read_text(encoding="utf-8"))) != payload_sha:
        raise RuntimeError("distribution release payload readback verification failed")
    remote_ready = distribution_audit.get("remote_only_execution_allowed") is True
    manifest = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "ALREADY_DISTRIBUTED" if remote_ready else "READY_FOR_EXTERNAL_PUBLISH",
        "release_id": release_id,
        "model_version": payload.get("model_version"),
        "execution_date": payload.get("execution_date"),
        "strategy_specification_fingerprint": payload.get(
            "strategy_specification_fingerprint"
        ),
        "distribution_url": distribution_audit.get("distribution_url"),
        "payload_path": str(payload_target.resolve()),
        "payload_filename": "etf_rotation_latest.json",
        "payload_file_sha256": _file_sha256(payload_target),
        "payload_canonical_sha256": payload_sha,
        "expected_content_type": "application/json",
        "remote_audit_status_before_release": distribution_audit.get("status"),
        "remote_payload_sha256_before_release": distribution_audit.get(
            "remote_payload_sha256", ""
        ),
        "post_publish_requirements": {
            "remote_contract_valid": True,
            "remote_payload_canonical_sha256": payload_sha,
            "remote_model_version": payload.get("model_version"),
            "remote_execution_date": payload.get("execution_date"),
        },
        "external_publish_performed": False,
    }
    _atomic_json(manifest, release_dir / "release_manifest.json")
    _atomic_json(manifest, release_root / "distribution_release_latest.json")
    return manifest


__all__ = ["POLICY_VERSION", "prepare_distribution_release"]
