"""Generate point-in-time ETF v3 calibration and acceptance artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import main
from calibration_pipeline import build_forward_label, select_entry_thresholds
from v3_signals import CalibrationTable, calibration_features, fingerprint_price_frames


def _atomic_json(value: Dict[str, Any], path: str) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_prices(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    paths = sorted([
        path for path in glob.glob(os.path.join(data_dir, "*.csv"))
        if "_raw_" not in os.path.basename(path)
    ])
    prices: Dict[str, pd.DataFrame] = {}
    for path in paths:
        code = os.path.basename(path).split("_", 1)[0]
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if len(frame) >= 275:
            prices[code] = frame
    if main.Config.DEFAULT_INDEX_CODE not in prices:
        raise ValueError(f"benchmark {main.Config.DEFAULT_INDEX_CODE} is missing")
    return prices, paths


def _regime(benchmark: pd.DataFrame) -> str:
    if len(benchmark) < 60:
        return "NEUTRAL"
    close = benchmark["close"].astype(float)
    price = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    if price > ma20 > ma60:
        return "BULL"
    if price < ma20 < ma60:
        return "BEAR"
    return "NEUTRAL"


def _max_drawdown(dated_returns: pd.DataFrame) -> float:
    if dated_returns.empty:
        return 0.0
    daily = dated_returns.groupby("date")["return_10d"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return abs(float(drawdown.min()))


def apply_drawdown_gate(
    thresholds: Dict[str, Any],
    baseline_drawdown: float,
    selected_drawdown: float | None,
) -> Dict[str, Any]:
    result = dict(thresholds)
    result["baseline_max_drawdown"] = round(baseline_drawdown, 4)
    if not result.get("approved") or selected_drawdown is None:
        result["selected_max_drawdown"] = None
        result["drawdown_reduction"] = None
        result["approved"] = False
        return result
    reduction = (
        (baseline_drawdown - selected_drawdown) / baseline_drawdown
        if baseline_drawdown > 0 else 0.0
    )
    result["selected_max_drawdown"] = round(selected_drawdown, 4)
    result["drawdown_reduction"] = round(reduction, 4)
    result["approved"] = bool(reduction >= 0.15)
    return result


def _cross_validated_predictions(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"])
    folds = [
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2024", "2023-01-01", "2024-12-31"),
        ("2025-current", "2025-01-01", "2099-12-31"),
    ]
    predicted: List[Dict[str, Any]] = []
    fold_reports: List[Dict[str, Any]] = []
    for name, start, end in folds:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        train = data[data["date"] < start_ts]
        validate = data[(data["date"] >= start_ts) & (data["date"] <= end_ts)]
        if len(train) < 100 or validate.empty:
            fold_reports.append({"name": name, "train": len(train), "validate": len(validate), "status": "INSUFFICIENT"})
            continue
        table = CalibrationTable.fit(train.to_dict("records"), version=f"fold-{name}")
        fold_rows = []
        for row in validate.to_dict("records"):
            calibration = table.lookup(
                row["entry_bin"], row["setup"], row["regime"], row["stop_bin"], row["rps_bin"]
            )
            enriched = dict(row)
            enriched.update(calibration)
            enriched["entry_score"] = round(np.clip((float(row["daily_score"]) + 1.0) / 4.0, 0, 1) * 100, 1)
            enriched["confirmed_setup"] = row["setup"] != "NONE"
            predicted.append(enriched)
            fold_rows.append(enriched)
        fold_frame = pd.DataFrame(fold_rows)
        fold_reports.append(
            {
                "name": name,
                "train": len(train),
                "validate": len(validate),
                "status": "OK",
                "baseline_early_stop_rate": round(float(fold_frame[fold_frame["legacy_candidate"]]["early_stop"].mean()), 4),
                "baseline_excess_return_10d": round(float(fold_frame[fold_frame["legacy_candidate"]]["excess_return_10d"].mean()), 6),
            }
        )
    return predicted, fold_reports


def generate_rows(data_dir: str, sample_step: int = 5) -> Tuple[List[Dict[str, Any]], str]:
    prices, paths = _load_prices(data_dir)
    benchmark = prices[main.Config.DEFAULT_INDEX_CODE]
    calendar = pd.DatetimeIndex(benchmark["date"])
    dates = benchmark["date"].iloc[252:-21:sample_step]
    rows: List[Dict[str, Any]] = []
    main.Logger.info = lambda *args, **kwargs: None
    main.Logger.warning = lambda *args, **kwargs: None
    main.Logger.error = lambda *args, **kwargs: None

    for date_index, signal_date in enumerate(dates, start=1):
        benchmark_slice = benchmark[benchmark["date"] <= signal_date].copy()
        analyzers = []
        profiles: Dict[str, Dict[str, float]] = {}
        for code, frame in prices.items():
            current = frame[frame["date"] <= signal_date].copy()
            if len(current) < 252 or len(frame[frame["date"] > signal_date]) < 21:
                continue
            analyzer = main.ETFAnalyzer(code, code, market_safe=True, atr_multiplier=2.0)
            analyzer.set_price_frames(current, current, trading_calendar=calendar)
            profile = main.calc_risk_adjusted_alpha(current, benchmark_slice, window=120)
            if profile["vol_adj_alpha"] == -999.0:
                continue
            analyzers.append(analyzer)
            profiles[code] = profile

        ranked = sorted(analyzers, key=lambda item: profiles[item.code]["vol_adj_alpha"])
        divisor = max(1, len(ranked) - 1)
        regime = _regime(benchmark_slice)
        for rank, analyzer in enumerate(ranked):
            analyzer.rps = rank / divisor * 100.0
            profile = profiles[analyzer.code]
            analyzer.alpha_score = profile["alpha_score"]
            analyzer.vol_adj_alpha = profile["vol_adj_alpha"]
            analyzer.beta_to_benchmark = profile["beta_to_benchmark"]
            result = analyzer.analyze()
            if not result:
                continue
            full_frame = prices[analyzer.code]
            try:
                label = build_forward_label(
                    full_frame,
                    benchmark,
                    signal_date=signal_date,
                    stop_price=float(result["stop_loss"]),
                    slippage_bps=5,
                )
            except ValueError:
                continue
            features = calibration_features(result, regime=regime)
            legacy_candidate = (
                float(result["raw_total_score"]) >= 7.0
                and float(result["weekly_score"]) >= 1.0
                and float(result["daily_score"]) >= -1.0
                and float(result["rps"]) >= 45.0
                and 1.5 <= float(result["stop_dist"]) <= 12.0
            )
            rows.append(
                {
                    "date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                    "code": analyzer.code,
                    "daily_score": float(result["daily_score"]),
                    "legacy_candidate": legacy_candidate,
                    **features,
                    **label,
                }
            )
        if date_index % 25 == 0:
            print(f"calibration progress: {date_index}/{len(dates)} dates, {len(rows)} rows")
    trained_until = max(row["date"] for row in rows)
    return rows, fingerprint_price_frames(prices, trained_until)


def build_artifacts(
    rows: List[Dict[str, Any]],
    fingerprint: str,
    calibration_path: str,
    report_path: str,
) -> Dict[str, Any]:
    predictions, fold_reports = _cross_validated_predictions(rows)
    thresholds = select_entry_thresholds(predictions) if predictions else {"approved": False}
    predicted_frame = pd.DataFrame(predictions)
    baseline = predicted_frame[predicted_frame["legacy_candidate"]] if not predicted_frame.empty else pd.DataFrame()
    selected = pd.DataFrame()
    if thresholds.get("approved") and not baseline.empty:
        selected = baseline[
            (baseline["entry_score"] >= thresholds["entry_score_min"])
            & (baseline["early_stop_probability_3d"] <= thresholds["early_stop_probability_max"])
            & (baseline["confirmed_setup"])
        ]
    baseline_drawdown = _max_drawdown(baseline) if not baseline.empty else 0.0
    selected_drawdown = _max_drawdown(selected) if not selected.empty else None
    thresholds = apply_drawdown_gate(thresholds, baseline_drawdown, selected_drawdown)

    trained_until = max(row["date"] for row in rows)
    version = f"v3-{datetime.now().strftime('%Y%m%d')}-{fingerprint[:8]}"
    table = CalibrationTable.fit(
        rows,
        prior_strength=20,
        version=version,
        trained_until=trained_until,
        data_fingerprint=fingerprint,
    )
    table.thresholds = thresholds
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_version": version,
        "survivorship_bias_warning": True,
        "row_count": len(rows),
        "folds": fold_reports,
        "thresholds": thresholds,
    }
    _atomic_json(table.to_dict(), calibration_path)
    _atomic_json(report, report_path)
    return report


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=main.Config.DATA_DIR)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--calibration-out", default=main.Config.V3_CALIBRATION_FILE)
    parser.add_argument("--report-out", default="v3_acceptance_report.json")
    args = parser.parse_args()
    rows, fingerprint = generate_rows(args.data_dir, sample_step=max(1, args.sample_step))
    report = build_artifacts(rows, fingerprint, args.calibration_out, args.report_out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
