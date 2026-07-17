"""Generate purged walk-forward V4 calibration and portfolio acceptance artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .. import _core as main
from ..signals.labeling import build_forward_label
from ..signals.contract import (
    V4CalibrationModel,
    fit_v4_calibration,
    fingerprint_price_frames,
    v4_calibration_features,
)
from ..signals.factors import (
    final_priority,
    market_policy,
    normalised_atr_percentile,
    rank_relative_strength,
    weekly_trend_factor,
)


def _atomic_json(value: Mapping[str, Any], path: str) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_price_pairs(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    qfq: Dict[str, pd.DataFrame] = {}
    raw: Dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.basename(path)
        code = name.split("_", 1)[0]
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if "_raw_" in name:
            raw[code] = frame
        else:
            qfq[code] = frame
    usable = set(qfq).intersection(raw)
    qfq = {code: qfq[code] for code in usable if len(qfq[code]) >= 275}
    raw = {code: raw[code] for code in qfq}
    if main.Config.DEFAULT_INDEX_CODE not in qfq:
        raise ValueError("benchmark raw/qfq pair is missing")
    return qfq, raw


def _adjustment_crossed(
    raw: pd.DataFrame,
    qfq: pd.DataFrame,
    signal_date: Any,
    horizon_days: int = 20,
    threshold: float = 0.005,
) -> bool:
    left = raw[["date", "close"]].rename(columns={"close": "raw"})
    right = qfq[["date", "close"]].rename(columns={"close": "qfq"})
    aligned = left.merge(right, on="date", how="inner").sort_values("date")
    aligned = aligned[aligned["date"] >= pd.Timestamp(signal_date)].head(horizon_days + 2)
    if aligned.empty:
        return True
    factor = aligned["qfq"].astype(float) / aligned["raw"].astype(float).replace(0, np.nan)
    return bool((factor.pct_change().abs() > threshold).any())


def generate_rows(data_dir: str, sample_step: int = 5) -> Tuple[List[Dict[str, Any]], str, Dict[str, pd.DataFrame]]:
    qfq, raw = _load_price_pairs(data_dir)
    benchmark_qfq = qfq[main.Config.DEFAULT_INDEX_CODE]
    benchmark_raw = raw[main.Config.DEFAULT_INDEX_CODE]
    calendar = pd.DatetimeIndex(benchmark_qfq["date"])
    dates = benchmark_qfq["date"].iloc[252:-21:max(1, sample_step)]
    rows: List[Dict[str, Any]] = []
    main.Logger.info = lambda *args, **kwargs: None
    main.Logger.warning = lambda *args, **kwargs: None
    main.Logger.error = lambda *args, **kwargs: None

    for date_index, signal_date in enumerate(dates, start=1):
        current_frames: Dict[str, pd.DataFrame] = {}
        analyzers: Dict[str, main.ETFAnalyzer] = {}
        raw_slices: Dict[str, pd.DataFrame] = {}
        for code, frame in qfq.items():
            current = frame[frame["date"] <= signal_date].copy()
            raw_current = raw[code][raw[code]["date"] <= signal_date].copy()
            if len(current) < 252 or len(frame[frame["date"] > signal_date]) < 21:
                continue
            analyzer = main.ETFAnalyzer(code, code, market_safe=True, atr_multiplier=2.0)
            try:
                analyzer.set_price_frames(current, raw_current, trading_calendar=calendar)
                result = analyzer.analyze()
            except Exception:
                continue
            if not result:
                continue
            analyzer._v4_result = result  # calibration-only transient state
            analyzers[code] = analyzer
            current_frames[code] = current
            raw_slices[code] = raw_current

        benchmark_slice = benchmark_qfq[benchmark_qfq["date"] <= signal_date].copy()
        relative_strength = rank_relative_strength(
            {
                code: frame
                for code, frame in current_frames.items()
                if code != main.Config.DEFAULT_INDEX_CODE
            },
            benchmark_slice,
        )
        provisional: List[Dict[str, Any]] = []
        for code, analyzer in analyzers.items():
            if code == main.Config.DEFAULT_INDEX_CODE:
                continue
            result = dict(analyzer._v4_result)
            result["relative_strength"] = dict(relative_strength.get(code, {}))
            result["v4_priority"] = final_priority(result.get("v4_factors", {}), result["relative_strength"])
            provisional.append(result)

        benchmark_analyzer = analyzers.get(main.Config.DEFAULT_INDEX_CODE)
        benchmark_weekly = (
            float(weekly_trend_factor(benchmark_analyzer.df_weekly).get("score", 0.0))
            if benchmark_analyzer is not None else 0.0
        )
        policy = market_policy(
            provisional,
            benchmark_weekly_score=benchmark_weekly,
            benchmark_natr_percentile=normalised_atr_percentile(benchmark_slice),
        )

        for result in provisional:
            code = str(result["code"])
            result["v4_market"] = dict(policy)
            risk = result.get("v4_factors", {}).get("risk", {})
            if _adjustment_crossed(raw[code], qfq[code], signal_date):
                continue
            try:
                label = build_forward_label(
                    raw[code],
                    benchmark_raw,
                    signal_date=signal_date,
                    stop_price=float(risk.get("stop_loss", 0.0)),
                    slippage_bps=5,
                    fee_rate=0.00015,
                )
            except ValueError:
                continue
            factors = v4_calibration_features(result)
            monthly_score = float((result.get("v4_factors", {}).get("monthly", {}) or {}).get("score", 0.0))
            weekly_score = float((result.get("v4_factors", {}).get("weekly", {}) or {}).get("score", 0.0))
            setup_score = float((result.get("v4_factors", {}).get("setup", {}) or {}).get("score", 0.0))
            baseline_candidate = (
                weekly_score >= 0.25
                and monthly_score >= -0.15
                and str(result.get("v4_factors", {}).get("setup", {}).get("setup", "NONE"))
                in {"BREAKOUT", "PULLBACK"}
                and setup_score >= 55.0
                and bool(risk.get("executable", False))
                and float(result.get("v4_priority", 0.0)) >= 55.0
            )
            rows.append(
                {
                    "date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                    "code": code,
                    "priority": float(result.get("v4_priority", 0.0)),
                    "setup": str(result.get("v4_factors", {}).get("setup", {}).get("setup", "NONE")),
                    "setup_score_raw": float(result.get("v4_factors", {}).get("setup", {}).get("score", 0.0)),
                    "risk_executable": bool(risk.get("executable", False)),
                    "stop_loss": float(risk.get("stop_loss", 0.0)),
                    "baseline_candidate": baseline_candidate,
                    **factors,
                    **label,
                }
            )
        if date_index % 25 == 0:
            print(f"v4 calibration progress: {date_index}/{len(dates)} dates, {len(rows)} rows", flush=True)
    trained_until = max(row["date"] for row in rows)
    return rows, fingerprint_price_frames(qfq, trained_until), raw


def _predict(model: V4CalibrationModel, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {**dict(row), **model.predict(row)}


def walk_forward_predictions(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"])
    unique_dates = sorted(data["date"].unique())
    folds = [
        ("2021-2022", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
        ("2023-2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
        ("2025-current", pd.Timestamp("2025-01-01"), pd.Timestamp("2099-12-31")),
    ]
    predictions: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    for name, start, end in folds:
        prior_dates = [pd.Timestamp(value) for value in unique_dates if pd.Timestamp(value) < start]
        purge_cutoff = prior_dates[-21] if len(prior_dates) > 20 else pd.Timestamp.min
        train = data[data["date"] <= purge_cutoff]
        validate = data[(data["date"] >= start) & (data["date"] <= end)]
        if len(train) < 200 or validate.empty:
            reports.append({"name": name, "train": len(train), "validate": len(validate), "status": "INSUFFICIENT"})
            continue
        model = fit_v4_calibration(train.to_dict("records"), regularisation=1.0, version=f"fold-{name}")
        fold_rows = [_predict(model, row) for row in validate.to_dict("records")]
        predictions.extend(fold_rows)
        reports.append({"name": name, "train": len(train), "validate": len(validate), "status": "OK"})
    return predictions, reports


def select_thresholds(predictions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    data = pd.DataFrame(list(predictions))
    baseline = data[data["baseline_candidate"].astype(bool)].copy()
    if baseline.empty:
        return {"approved": False, "reason": "NO_BASELINE"}
    baseline_early = float(baseline["early_stop"].mean())
    baseline_excess = float(baseline["excess_return_10d"].mean())
    choices: List[Dict[str, Any]] = []
    for priority_min in (60.0, 65.0, 70.0, 75.0):
        for setup_min in (55.0, 60.0, 65.0):
            for early_max in (0.18, 0.20, 0.2143, 0.24, 0.26):
                selected = data[
                    (data["priority"] >= priority_min)
                    & (data["setup_score_raw"] >= setup_min)
                    & (data["risk_executable"].astype(bool))
                    & (data["early_stop_probability_3d"] <= early_max)
                    & (data["expected_excess_return_10d"] > 0)
                ]
                retention = len(selected) / len(baseline)
                if not 0.30 <= retention <= 0.70 or selected.empty:
                    continue
                early = float(selected["early_stop"].mean())
                excess = float(selected["excess_return_10d"].mean())
                choices.append(
                    {
                        "priority_min": priority_min,
                        "setup_score_min": setup_min,
                        "early_stop_probability_max": early_max,
                        "retention_rate": retention,
                        "baseline_early_stop_rate": baseline_early,
                        "selected_early_stop_rate": early,
                        "early_stop_reduction": (baseline_early - early) / baseline_early if baseline_early > 0 else 0.0,
                        "baseline_excess_return_10d": baseline_excess,
                        "selected_excess_return_10d": excess,
                        "selected_count": int(len(selected)),
                    }
                )
    approved = [
        choice for choice in choices
        if choice["early_stop_reduction"] >= 0.25
        and choice["selected_excess_return_10d"] > 0
        and choice["selected_excess_return_10d"] >= choice["baseline_excess_return_10d"]
    ]
    if not approved:
        return {
            "approved": False,
            "reason": "NO_THRESHOLD_PASSED",
            "baseline_early_stop_rate": round(baseline_early, 6),
            "baseline_excess_return_10d": round(baseline_excess, 8),
        }
    best = min(approved, key=lambda item: (item["selected_early_stop_rate"], -item["selected_excess_return_10d"]))
    return {key: round(value, 8) if isinstance(value, float) else value for key, value in best.items()}


def _selected(frame: pd.DataFrame, thresholds: Mapping[str, Any]) -> pd.DataFrame:
    if not thresholds.get("approved"):
        return frame.iloc[0:0].copy()
    return frame[
        (frame["priority"] >= float(thresholds["priority_min"]))
        & (frame["setup_score_raw"] >= float(thresholds["setup_score_min"]))
        & (frame["risk_executable"].astype(bool))
        & (frame["early_stop_probability_3d"] <= float(thresholds["early_stop_probability_max"]))
        & (frame["expected_excess_return_10d"] > 0)
    ].copy()


def simulate_portfolio(
    rows: pd.DataFrame,
    raw_frames: Mapping[str, pd.DataFrame],
    max_positions: int = 3,
) -> Dict[str, Any]:
    if rows.empty:
        return {"max_drawdown": None, "return": None, "trade_count": 0}
    frames = {
        code: frame.assign(date=pd.to_datetime(frame["date"])).set_index("date").sort_index()
        for code, frame in raw_frames.items()
    }
    candidates: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for row in rows.to_dict("records"):
        candidates.setdefault(pd.Timestamp(row["entry_date"]), []).append(row)
    dates = sorted({date for frame in frames.values() for date in frame.index})
    cash = 1.0
    positions: Dict[str, Dict[str, Any]] = {}
    equity_curve: List[float] = []
    trade_count = 0
    fee, slippage = 0.00015, 0.0005
    for date in dates:
        for code in list(positions):
            if code not in frames or date not in frames[code].index:
                continue
            bar = frames[code].loc[date]
            position = positions[code]
            position["bars"] += 1
            exit_price = None
            if float(bar["low"]) <= position["stop"]:
                exit_price = position["stop"] * (1.0 - slippage) * (1.0 - fee)
            elif position["bars"] >= 10:
                exit_price = float(bar["close"]) * (1.0 - slippage) * (1.0 - fee)
            if exit_price is not None:
                cash += position["shares"] * exit_price
                del positions[code]
        marked = cash + sum(
            position["shares"] * float(frames[code].loc[date]["close"])
            for code, position in positions.items()
            if code in frames and date in frames[code].index
        )
        for row in sorted(candidates.get(date, []), key=lambda item: float(item.get("priority", 0.0)), reverse=True):
            code = str(row["code"])
            if code in positions or len(positions) >= max_positions or code not in frames or date not in frames[code].index:
                continue
            entry = float(frames[code].loc[date]["open"]) * (1.0 + slippage) * (1.0 + fee)
            stop = float(row["stop_loss"])
            risk_per_share = entry - stop
            if entry <= 0 or risk_per_share <= 0:
                continue
            shares = min(marked * 0.01 / risk_per_share, marked * 0.25 / entry, cash / entry)
            if shares <= 0:
                continue
            cash -= shares * entry
            positions[code] = {"shares": shares, "stop": stop, "bars": 0}
            trade_count += 1
        equity = cash + sum(
            position["shares"] * float(frames[code].loc[date]["close"])
            for code, position in positions.items()
            if code in frames and date in frames[code].index
        )
        equity_curve.append(equity)
    curve = pd.Series(equity_curve, dtype=float)
    drawdown = curve / curve.cummax() - 1.0
    return {
        "max_drawdown": round(abs(float(drawdown.min())), 6),
        "return": round(float(curve.iloc[-1] - 1.0), 6),
        "trade_count": trade_count,
    }


def build_artifacts(
    rows: List[Dict[str, Any]],
    fingerprint: str,
    raw_frames: Mapping[str, pd.DataFrame],
    calibration_path: str,
    report_path: str,
) -> Dict[str, Any]:
    predictions, folds = walk_forward_predictions(rows)
    thresholds = select_thresholds(predictions) if predictions else {"approved": False, "reason": "NO_PREDICTIONS"}
    predicted = pd.DataFrame(predictions)
    baseline = predicted[predicted["baseline_candidate"].astype(bool)].copy() if not predicted.empty else predicted
    selected = _selected(predicted, thresholds) if not predicted.empty else predicted
    baseline_portfolio = simulate_portfolio(baseline, raw_frames)
    selected_portfolio = simulate_portfolio(selected, raw_frames)
    fold_checks = []
    if not selected.empty:
        selected["date"] = pd.to_datetime(selected["date"])
        for fold in folds:
            name = fold["name"]
            if name == "2021-2022":
                part = selected[(selected["date"] >= "2021-01-01") & (selected["date"] <= "2022-12-31")]
            elif name == "2023-2024":
                part = selected[(selected["date"] >= "2023-01-01") & (selected["date"] <= "2024-12-31")]
            else:
                part = selected[selected["date"] >= "2025-01-01"]
            fold_checks.append(
                {
                    "name": name,
                    "selected_count": int(len(part)),
                    "excess_return_10d": round(float(part["excess_return_10d"].mean()), 8) if not part.empty else None,
                }
            )
    baseline_dd = baseline_portfolio.get("max_drawdown")
    selected_dd = selected_portfolio.get("max_drawdown")
    drawdown_reduction = (
        (baseline_dd - selected_dd) / baseline_dd
        if baseline_dd and selected_dd is not None else 0.0
    )
    approved = bool(
        thresholds.get("approved")
        and len(selected) >= 200
        and all(item["selected_count"] >= 40 for item in fold_checks)
        and all(item["excess_return_10d"] is not None and item["excess_return_10d"] >= -0.002 for item in fold_checks)
        and drawdown_reduction >= 0.20
    )
    thresholds["approved"] = approved
    thresholds["baseline_max_drawdown"] = baseline_dd
    thresholds["selected_max_drawdown"] = selected_dd
    thresholds["drawdown_reduction"] = round(drawdown_reduction, 6)
    trained_until = max(row["date"] for row in rows)
    version = f"v4-{datetime.now().strftime('%Y%m%d')}-{fingerprint[:8]}"
    model = fit_v4_calibration(
        rows,
        regularisation=1.0,
        version=version,
        trained_until=trained_until,
        data_fingerprint=fingerprint,
        thresholds=thresholds,
    )
    report = {
        "schema_version": 4,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_version": version,
        "survivorship_bias_warning": True,
        "fixed_universe_only": True,
        "row_count": len(rows),
        "folds": folds,
        "fold_checks": fold_checks,
        "thresholds": thresholds,
        "baseline_portfolio": baseline_portfolio,
        "selected_portfolio": selected_portfolio,
    }
    _atomic_json(model.to_dict(), calibration_path)
    _atomic_json(report, report_path)
    return report


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=main.Config.DATA_DIR)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--calibration-out", default=main.Config.V4_CALIBRATION_FILE)
    parser.add_argument(
        "--report-out",
        default=os.path.join(os.path.dirname(main.Config.V4_CALIBRATION_FILE), "v4_acceptance_report.json"),
    )
    args = parser.parse_args()
    rows, fingerprint, raw_frames = generate_rows(args.data_dir, sample_step=max(1, args.sample_step))
    report = build_artifacts(rows, fingerprint, raw_frames, args.calibration_out, args.report_out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
