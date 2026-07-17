"""ETF V4 production pipeline."""

from __future__ import annotations

import json
import shutil

from jsonschema import Draft202012Validator

from . import _core
from .config import PATHS, configure_runtime_paths
from .reporting import HTMLReporter


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


def run() -> None:
    configure_runtime_paths()
    _core.HTMLReporter = HTMLReporter
    _core.main()
    verify_public_signal()
    _publish_contract_assets()


__all__ = ["run", "verify_public_signal"]
