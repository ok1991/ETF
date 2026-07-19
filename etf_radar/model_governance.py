"""Fail-closed temporal governance for production model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


MODEL_GENERATED_MAX_AGE_DAYS = 21
MODEL_TRAINED_MAX_LAG_DAYS = 60
MODEL_FUTURE_TOLERANCE_MINUTES = 10


@dataclass(frozen=True)
class ArtifactTimeStatus:
    approved: bool
    reason: str
    generated_age_days: Optional[int] = None
    training_lag_days: Optional[int] = None


def _normalise_now(now: Any = None) -> pd.Timestamp:
    value = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value


def _normalise_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    result = pd.Timestamp(parsed)
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result


def validate_artifact_time(
    artifact: Mapping[str, Any],
    *,
    now: Any = None,
    generated_max_age_days: int = MODEL_GENERATED_MAX_AGE_DAYS,
    trained_max_lag_days: int = MODEL_TRAINED_MAX_LAG_DAYS,
    future_tolerance_minutes: int = MODEL_FUTURE_TOLERANCE_MINUTES,
) -> ArtifactTimeStatus:
    """Validate generation freshness separately from forward-label training lag."""
    current = _normalise_now(now)
    generated_at = _normalise_timestamp(artifact.get("generated_at"))
    if pd.isna(generated_at):
        return ArtifactTimeStatus(False, "GENERATED_AT_MISSING_OR_INVALID")
    if generated_at > current + pd.Timedelta(minutes=max(0, int(future_tolerance_minutes))):
        return ArtifactTimeStatus(False, "GENERATED_AT_IN_FUTURE")

    trained_until = _normalise_timestamp(artifact.get("trained_until"))
    if pd.isna(trained_until):
        return ArtifactTimeStatus(False, "TRAINED_UNTIL_MISSING_OR_INVALID")
    if trained_until.normalize() > current.normalize() + pd.Timedelta(days=1):
        return ArtifactTimeStatus(False, "TRAINED_UNTIL_IN_FUTURE")

    generated_age = max(0, int((current.normalize() - generated_at.normalize()).days))
    training_lag = max(0, int((current.normalize() - trained_until.normalize()).days))
    if generated_age > max(1, int(generated_max_age_days)):
        return ArtifactTimeStatus(
            False,
            "GENERATED_AT_STALE",
            generated_age_days=generated_age,
            training_lag_days=training_lag,
        )
    if training_lag > max(1, int(trained_max_lag_days)):
        return ArtifactTimeStatus(
            False,
            "TRAINED_UNTIL_STALE",
            generated_age_days=generated_age,
            training_lag_days=training_lag,
        )
    return ArtifactTimeStatus(
        True,
        "APPROVED",
        generated_age_days=generated_age,
        training_lag_days=training_lag,
    )


def validate_bundle_member(path: Any, artifact: Mapping[str, Any]) -> str:
    """Bind a live artifact to the last transactionally promoted bundle manifest."""
    artifact_path = Path(path)
    manifest_path = artifact_path.parent / "calibration_bundle.json"
    if not manifest_path.is_file():
        return "BUNDLE_MANIFEST_MISSING"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_id = str(artifact.get("artifact_bundle_id", ""))
        if not bundle_id or bundle_id != str(manifest.get("artifact_bundle_id", "")):
            return "BUNDLE_ID_MISMATCH"
        expected = str(
            ((manifest.get("files") or {}).get(artifact_path.name) or {}).get(
                "sha256", ""
            )
        ).lower()
        if not expected:
            return "BUNDLE_MEMBER_HASH_MISSING"
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        return "APPROVED" if actual == expected else "BUNDLE_MEMBER_HASH_MISMATCH"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "BUNDLE_MANIFEST_INVALID"


__all__ = [
    "ArtifactTimeStatus",
    "MODEL_FUTURE_TOLERANCE_MINUTES",
    "MODEL_GENERATED_MAX_AGE_DAYS",
    "MODEL_TRAINED_MAX_LAG_DAYS",
    "validate_artifact_time",
    "validate_bundle_member",
]
