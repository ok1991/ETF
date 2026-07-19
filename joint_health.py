#!/usr/bin/env python3
"""CLI for the combined ETF producer and Swing executor health gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from etf_radar.joint_health import build_joint_health
from etf_radar.paths import PATHS


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-remote-distribution", action="store_true")
    parser.add_argument("--output", default=str(PATHS.public / "joint_health_latest.json"))
    args = parser.parse_args()
    result = build_joint_health(
        PATHS.root,
        PATHS.root.parent / "Swing-trading",
        output_path=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)
    if args.require_remote_distribution and not result["remote_only_execution_allowed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main_cli()
