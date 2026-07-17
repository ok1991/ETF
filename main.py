#!/usr/bin/env python3
"""ETF V4 production entrypoint."""

from etf_radar._core import validate_signal_contract

__all__ = ["run", "validate_signal_contract"]


def run() -> None:
    from etf_radar.pipeline import run as pipeline_run

    pipeline_run()


if __name__ == "__main__":
    run()
