#!/usr/bin/env python3
"""Disaster-recovery self-check CLI combining status overview and joint health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from etf_radar.joint_health import build_joint_health
from etf_radar.paths import PATHS
from etf_radar.status_overview import build_status_overview, SEVERITY_P0, SEVERITY_P1


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Disaster-recovery self-check")
    parser.add_argument("--swing-root", default=str(PATHS.root.parent / "Swing-trading"))
    args = parser.parse_args()

    overview = build_status_overview(
        PATHS.public,
        PATHS.state,
        output_path=PATHS.public / "disaster_recovery_selfcheck_latest.json",
    )

    joint = build_joint_health(
        PATHS.root,
        Path(args.swing_root),
    )

    combined = {
        "status_overview": overview,
        "joint_health": joint,
    }
    print(json.dumps(combined, ensure_ascii=False, indent=2))

    severity = overview.get("overall_severity")
    if severity in (SEVERITY_P0, SEVERITY_P1):
        raise SystemExit(2)
    if joint.get("status") == "BLOCKED":
        raise SystemExit(3)


if __name__ == "__main__":
    main_cli()
