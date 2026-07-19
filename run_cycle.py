#!/usr/bin/env python3
"""ETF production-cycle entrypoint."""

from __future__ import annotations

import argparse
import json

from etf_radar.cycle import assert_last_cycle_healthy, run_cycle


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--check-last-status", action="store_true")
    args = parser.parse_args()
    if args.check_last_status:
        assert_last_cycle_healthy()
        return
    status = run_cycle(
        force_calibration=args.force_calibration,
        sample_step=args.sample_step,
        workers=args.workers,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
