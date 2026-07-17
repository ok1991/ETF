"""Application configuration facade."""

from ._core import Config, ETFScoringConfig
from .paths import PATHS


def configure_runtime_paths() -> None:
    PATHS.ensure()
    Config.HISTORY_FILE = str(PATHS.state / "etf_history_state.json")
    Config.MARKET_ENV_HISTORY_FILE = str(PATHS.state / "market_env_history.json")
    Config.MARKET_ENV_LATEST_FILE = str(PATHS.public / "market_env_latest.json")
    Config.ETF_SIGNALS_LATEST_FILE = str(PATHS.public / "etf_signals_latest.json")
    Config.LOG_FILE = str(PATHS.logs / "etf_radar.log")
    Config.V4_CALIBRATION_FILE = str(PATHS.calibration / "v4_calibration.json")
    Config.DATA_DIR = str(PATHS.data)


__all__ = ["Config", "ETFScoringConfig", "PATHS", "configure_runtime_paths"]
