"""Application configuration facade."""

import os
from pathlib import Path

from ._core import Config, ETFScoringConfig
from .paths import PATHS


REMOTE_SWING_EXECUTION_FEEDBACK_SOURCE = (
    "https://raw.githubusercontent.com/ok1991/Swing-trading/main/"
    "execution_feedback_history.json"
)
REMOTE_SWING_LIVE_PERFORMANCE_SOURCE = (
    "https://raw.githubusercontent.com/ok1991/Swing-trading/main/"
    "live_performance_latest.json"
)


def _resolve_swing_source(
    environment_name: str,
    local_path: Path,
    remote_default: str,
) -> str:
    explicit = os.environ.get(environment_name, "").strip()
    if explicit:
        return explicit
    return str(local_path) if local_path.is_file() else remote_default


def configure_runtime_paths() -> None:
    PATHS.ensure()
    Config.HISTORY_FILE = str(PATHS.state / "etf_history_state.json")
    Config.MARKET_ENV_HISTORY_FILE = str(PATHS.state / "market_env_history.json")
    Config.MARKET_ENV_LATEST_FILE = str(PATHS.public / "market_env_latest.json")
    Config.ETF_SIGNALS_LATEST_FILE = str(PATHS.public / "etf_signals_latest.json")
    Config.LOG_FILE = str(PATHS.logs / "etf_radar.log")
    Config.V4_CALIBRATION_FILE = str(PATHS.calibration / "v4_calibration.json")
    Config.FACTOR_REGISTRY_FILE = str(PATHS.calibration / "adaptive_factor_registry.json")
    Config.FACTOR_HEALTH_FILE = str(PATHS.public / "factor_health_latest.json")
    Config.ROTATION_MODEL_FILE = str(PATHS.calibration / "rotation_model.json")
    Config.ROTATION_STATE_FILE = str(PATHS.state / "rotation_state.json")
    Config.ROTATION_LATEST_FILE = str(PATHS.public / "etf_rotation_latest.json")
    Config.DATA_MANIFEST_FILE = str(PATHS.public / "data_manifest_latest.json")
    Config.EXECUTION_FEEDBACK_LEDGER_FILE = str(
        PATHS.state / "execution_feedback_ledger.json"
    )
    Config.EXECUTION_FEEDBACK_AUDIT_FILE = str(
        PATHS.public / "execution_feedback_audit_latest.json"
    )
    Config.EXECUTION_COST_RECALIBRATION_FILE = str(
        PATHS.public / "execution_cost_recalibration_latest.json"
    )
    Config.LIVE_PERFORMANCE_AUDIT_FILE = str(
        PATHS.public / "live_performance_audit_latest.json"
    )
    sibling_public = PATHS.root.parent / "Swing-trading" / "public"
    Config.EXECUTION_FEEDBACK_SOURCE = _resolve_swing_source(
        "SWING_EXECUTION_FEEDBACK_SOURCE",
        sibling_public / "execution_feedback_history.json",
        REMOTE_SWING_EXECUTION_FEEDBACK_SOURCE,
    )
    Config.LIVE_PERFORMANCE_SOURCE = _resolve_swing_source(
        "SWING_LIVE_PERFORMANCE_SOURCE",
        sibling_public / "live_performance_latest.json",
        REMOTE_SWING_LIVE_PERFORMANCE_SOURCE,
    )
    Config.DATA_DIR = str(PATHS.data)


__all__ = ["Config", "ETFScoringConfig", "PATHS", "configure_runtime_paths"]
