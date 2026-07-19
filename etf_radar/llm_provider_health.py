from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from .factor_evolution import PRIMITIVE_FEATURES
from .llm_factor_proposals import (
    BUILTIN_CHAT_API_KEY,
    BUILTIN_CHAT_ENDPOINT,
    BUILTIN_CHAT_MODEL,
    load_or_generate_llm_proposals,
)


POLICY_VERSION = "scheduled-llm-provider-health-v2"
MAX_HEALTH_PROPOSAL_COUNT = 8


def _atomic_json(value: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous: Dict[str, Optional[str]] = {
        name: os.environ.get(name) for name in values
    }
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_provider_health_check(
    *,
    artifact_path: Path,
    proposal_path: Path,
    proposal_count: int = 1,
) -> Dict[str, Any]:
    configured_endpoint = os.environ.get(
        "LLM_LOCAL_ENDPOINT", BUILTIN_CHAT_ENDPOINT
    ).strip()
    configured_model = os.environ.get(
        "LLM_LOCAL_MODEL", BUILTIN_CHAT_MODEL
    ).strip()
    configured_key = os.environ.get("LLM_LOCAL_API_KEY", BUILTIN_CHAT_API_KEY)
    environment = {
        "LLM_FACTOR_PROPOSALS_ENABLED": "true",
        "LLM_FACTOR_PROPOSALS_REFRESH": "true",
        "LLM_FACTOR_PROPOSAL_COUNT": str(
            max(1, min(MAX_HEALTH_PROPOSAL_COUNT, int(proposal_count)))
        ),
    }
    try:
        with _temporary_environment(environment):
            result = load_or_generate_llm_proposals(
                PRIMITIVE_FEATURES,
                {},
                proposal_path,
            )
        proposals = list(result.get("proposals") or [])
        fallback_used = bool(result.get("fallback_used"))
        provider_ok = (
            result.get("status") == "OK"
            and bool(proposals)
            and not fallback_used
            and str(result.get("model", "")) == configured_model
        )
        document = (
            proposal_path.read_text(encoding="utf-8")
            if proposal_path.is_file()
            else ""
        )
        credential_persisted = bool(configured_key and configured_key in document)
        status = "OK" if provider_ok and not credential_persisted else "FAILED"
        error_code = ""
        if result.get("status") != "OK" or not proposals:
            error_code = "PROVIDER_RETURNED_NO_VALID_PROPOSALS"
        elif not provider_ok:
            error_code = "ACTIVE_PROVIDER_MODEL_IDENTITY_MISMATCH"
        elif credential_persisted:
            error_code = "CREDENTIAL_PERSISTED_IN_PROPOSAL_ARTIFACT"
        health: Dict[str, Any] = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": status,
            "provider": result.get("provider"),
            "configured_endpoint": configured_endpoint,
            "configured_model": configured_model,
            "endpoint_fingerprint": result.get("endpoint_fingerprint"),
            "model": result.get("model"),
            "fallback_used": fallback_used,
            "health_mode": "PRIMARY",
            "primary_provider_healthy": provider_ok,
            "refresh_allowed": status == "OK",
            "provider_attempts": result.get("provider_attempts") or [],
            "proposal_count": len(proposals),
            "rejected_count": len(result.get("rejected") or []),
            "credential_value_persisted": credential_persisted,
            "error_code": error_code,
        }
        if proposal_path.is_file():
            health["cache_sha256"] = hashlib.sha256(
                proposal_path.read_bytes()
            ).hexdigest()
    except Exception as error:
        health = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "FAILED",
            "provider": "OPENAI_CHAT_COMPATIBLE",
            "configured_endpoint": configured_endpoint,
            "configured_model": configured_model,
            "model": configured_model,
            "fallback_used": False,
            "provider_attempts": [],
            "proposal_count": 0,
            "rejected_count": 0,
            "credential_value_persisted": False,
            "error_code": "PROVIDER_REQUEST_FAILED",
            "error": str(error)[:1000],
        }
    _atomic_json(health, artifact_path)
    return health


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an isolated LLM provider health check."
    )
    parser.add_argument(
        "--artifact",
        default=".runtime/audits/llm_provider_health_latest.json",
    )
    parser.add_argument(
        "--proposal-path",
        default=".runtime/llm-shadow/provider-health/llm_factor_proposals.json",
    )
    parser.add_argument("--proposal-count", type=int, default=1)
    arguments = parser.parse_args(argv)
    health = run_provider_health_check(
        artifact_path=Path(arguments.artifact),
        proposal_path=Path(arguments.proposal_path),
        proposal_count=arguments.proposal_count,
    )
    print(
        json.dumps(
            {
                "status": health["status"],
                "provider": health.get("provider"),
                "model": health.get("model"),
                "proposal_count": health.get("proposal_count", 0),
                "fallback_used": health.get("fallback_used", False),
                "error_code": health.get("error_code", ""),
                "health_artifact": str(arguments.artifact),
            },
            ensure_ascii=False,
        )
    )
    return 0 if health["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_HEALTH_PROPOSAL_COUNT",
    "POLICY_VERSION",
    "main",
    "run_provider_health_check",
]
