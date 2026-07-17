#!/usr/bin/env python3
"""ETF V4 calibration entrypoint."""

from etf_radar.calibration.pipeline import main_cli
from etf_radar.config import configure_runtime_paths


if __name__ == "__main__":
    configure_runtime_paths()
    main_cli()
