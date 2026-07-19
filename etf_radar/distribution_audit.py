"""Audit whether the publicly distributed rotation matches local authority."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from jsonschema import Draft202012Validator

from .rotation_contract import validate_rotation_contract


POLICY_VERSION = "rotation-distribution-integrity-v1"
DEFAULT_DISTRIBUTION_URL = "https://etf.imlam.com/etf_rotation_latest.json"
IDENTITY_FIELDS = (
    "model_version",
    "execution_date",
    "execution_policy_version",
    "acceptance_policy_version",
    "strategy_specification_fingerprint",
    "target_weights",
    "cash_weight",
    "max_exposure_ratio",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _normalise_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("distribution URL must be absolute HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("distribution URL must not contain credentials, query, or fragment")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "")
    )


def _contract_errors(payload: Mapping[str, Any], schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(payload)),
        key=lambda error: list(error.path),
    )
    return validate_rotation_contract(dict(payload)) + [
        error.message for error in schema_errors
    ]


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_rotation_distribution(
    local_path: Path,
    output_path: Path,
    schema_path: Path,
    *,
    url: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Write a non-authorising distribution audit and always fail closed remotely."""
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    configured_url = url or os.environ.get(
        "ROTATION_DISTRIBUTION_URL", DEFAULT_DISTRIBUTION_URL
    )
    timeout = max(
        3,
        int(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get("ROTATION_DISTRIBUTION_TIMEOUT_SECONDS", "10")
        ),
    )
    try:
        normalised_url = _normalise_url(configured_url)
    except Exception as error:
        result = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "generated_at": generated_at,
            "status": "DISTRIBUTION_URL_INVALID",
            "distribution_url": "",
            "local_authority_valid": False,
            "same_host_execution_allowed": False,
            "remote_only_execution_allowed": False,
            "identity_match": False,
            "errors": [str(error)[:1000]],
        }
        _atomic_json(result, output_path)
        return result

    try:
        local_payload = json.loads(local_path.read_text(encoding="utf-8"))
        if not isinstance(local_payload, dict):
            raise ValueError("local rotation is not a JSON object")
        local_errors = _contract_errors(local_payload, schema_path)
    except Exception as error:
        local_payload = {}
        local_errors = [str(error)]
    if local_errors:
        result = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "generated_at": generated_at,
            "status": "LOCAL_AUTHORITY_INVALID",
            "distribution_url": normalised_url,
            "local_authority_valid": False,
            "same_host_execution_allowed": False,
            "remote_only_execution_allowed": False,
            "identity_match": False,
            "local_contract_errors": local_errors[:20],
            "errors": [],
        }
        _atomic_json(result, output_path)
        return result

    base = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "generated_at": generated_at,
        "distribution_url": normalised_url,
        "local_authority_valid": True,
        "same_host_execution_allowed": True,
        "local_model_version": str(local_payload.get("model_version", "")),
        "local_execution_date": str(local_payload.get("execution_date", "")),
        "local_strategy_specification_fingerprint": str(
            local_payload.get("strategy_specification_fingerprint", "")
        ),
        "local_payload_sha256": _canonical_sha256(local_payload),
    }
    try:
        request = urllib.request.Request(
            normalised_url,
            headers={"Accept": "application/json", "User-Agent": "etf-distribution-audit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        remote_payload = json.loads(raw.decode("utf-8"))
        if not isinstance(remote_payload, dict):
            raise ValueError("remote rotation is not a JSON object")
    except Exception as error:
        result = {
            **base,
            "status": "REMOTE_UNAVAILABLE",
            "remote_contract_valid": False,
            "remote_only_execution_allowed": False,
            "identity_match": False,
            "errors": [str(error)[:1000]],
        }
        _atomic_json(result, output_path)
        return result

    remote_errors = _contract_errors(remote_payload, schema_path)
    remote_summary = {
        "remote_model_version": str(remote_payload.get("model_version", "")),
        "remote_execution_date": str(remote_payload.get("execution_date", "")),
        "remote_strategy_specification_fingerprint": str(
            remote_payload.get("strategy_specification_fingerprint", "")
        ),
        "remote_payload_sha256": _canonical_sha256(remote_payload),
    }
    if remote_errors:
        result = {
            **base,
            **remote_summary,
            "status": "REMOTE_CONTRACT_INVALID",
            "remote_contract_valid": False,
            "remote_only_execution_allowed": False,
            "identity_match": False,
            "remote_contract_errors": remote_errors[:20],
            "errors": [],
        }
        _atomic_json(result, output_path)
        return result

    mismatches = [
        field for field in IDENTITY_FIELDS
        if remote_payload.get(field) != local_payload.get(field)
    ]
    matched = not mismatches
    result = {
        **base,
        **remote_summary,
        "status": "MATCH" if matched else "REMOTE_IDENTITY_MISMATCH",
        "remote_contract_valid": True,
        "remote_only_execution_allowed": matched,
        "identity_match": matched,
        "mismatched_fields": mismatches,
        "errors": [],
    }
    _atomic_json(result, output_path)
    return result


__all__ = [
    "DEFAULT_DISTRIBUTION_URL",
    "IDENTITY_FIELDS",
    "POLICY_VERSION",
    "audit_rotation_distribution",
]
