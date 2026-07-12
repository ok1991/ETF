"""Point-in-time labels and deterministic v3 entry-threshold selection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import pandas as pd


def _prepared(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"market frame missing columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def build_forward_label(
    etf_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    signal_date: Any,
    stop_price: float,
    slippage_bps: float = 5.0,
) -> Dict[str, Any]:
    """Label a close-generated signal using a T+1 executable entry."""
    etf = _prepared(etf_df)
    benchmark = _prepared(benchmark_df)
    signal_ts = pd.Timestamp(signal_date)
    future = etf[etf["date"] > signal_ts].reset_index(drop=True)
    if len(future) < 11:
        raise ValueError("at least 11 post-signal trading days are required")

    entry_row = future.iloc[0]
    entry_price = float(entry_row["open"]) * (1.0 + float(slippage_bps) / 10000.0)
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("entry and stop prices must be positive")

    first_three = future.iloc[:3]
    early_stop = int(bool((first_three["low"].astype(float) <= float(stop_price)).any()))
    horizons: Dict[int, float] = {}
    for horizon in (5, 10, 20):
        if len(future) > horizon:
            horizons[horizon] = float(future.iloc[horizon]["close"]) / entry_price - 1.0

    target_date = future.iloc[10]["date"]
    benchmark_future = benchmark[benchmark["date"] >= entry_row["date"]].reset_index(drop=True)
    benchmark_entry = benchmark_future[benchmark_future["date"] == entry_row["date"]]
    benchmark_target = benchmark_future[benchmark_future["date"] == target_date]
    if benchmark_entry.empty or benchmark_target.empty:
        raise ValueError("benchmark is missing aligned entry or horizon dates")
    benchmark_return = (
        float(benchmark_target.iloc[0]["close"]) / float(benchmark_entry.iloc[0]["open"]) - 1.0
    )

    adverse_window = future.iloc[: min(20, len(future))]
    max_adverse_excursion = float(adverse_window["low"].min()) / entry_price - 1.0
    return {
        "signal_date": signal_ts.strftime("%Y-%m-%d"),
        "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
        "entry_price": round(entry_price, 8),
        "early_stop": early_stop,
        "return_5d": round(horizons.get(5, 0.0), 8),
        "return_10d": round(horizons[10], 8),
        "return_20d": round(horizons.get(20, 0.0), 8),
        "benchmark_return_10d": round(benchmark_return, 8),
        "excess_return_10d": round(horizons[10] - benchmark_return, 8),
        "win_10d": int(horizons[10] > 0),
        "max_adverse_excursion": round(max_adverse_excursion, 8),
    }


def _mean(rows: pd.DataFrame, column: str) -> float:
    return float(rows[column].astype(float).mean()) if not rows.empty else 0.0


def select_entry_thresholds(
    rows: Iterable[Mapping[str, Any]],
    retention_min: float = 0.50,
    retention_max: float = 0.70,
    minimum_stop_reduction: float = 0.25,
) -> Dict[str, Any]:
    """Choose conservative thresholds subject to the approved acceptance gates."""
    data = pd.DataFrame(list(rows))
    required = {
        "legacy_candidate",
        "entry_score",
        "early_stop_probability_3d",
        "early_stop",
        "excess_return_10d",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"threshold rows missing columns: {sorted(missing)}")
    baseline = data[data["legacy_candidate"].astype(bool)].copy()
    if baseline.empty:
        raise ValueError("no legacy candidates available")

    baseline_stop = _mean(baseline, "early_stop")
    baseline_excess = _mean(baseline, "excess_return_10d")
    confirmed_setup = (
        baseline["confirmed_setup"].astype(bool)
        if "confirmed_setup" in baseline.columns
        else pd.Series(True, index=baseline.index)
    )
    score_thresholds = sorted(set([40.0, 50.0, 60.0, 70.0, 80.0] + baseline["entry_score"].astype(float).tolist()))
    risk_thresholds = sorted(set([0.20, 0.25, 0.30, 0.35, 0.40] + baseline["early_stop_probability_3d"].astype(float).tolist()))
    choices = []

    for score_threshold in score_thresholds:
        for risk_threshold in risk_thresholds:
            selected = baseline[
                (baseline["entry_score"].astype(float) >= score_threshold)
                & (baseline["early_stop_probability_3d"].astype(float) <= risk_threshold)
                & confirmed_setup
            ]
            retention = len(selected) / len(baseline)
            if not retention_min <= retention <= retention_max:
                continue
            selected_stop = _mean(selected, "early_stop")
            stop_reduction = (
                (baseline_stop - selected_stop) / baseline_stop
                if baseline_stop > 0
                else 0.0
            )
            selected_excess = _mean(selected, "excess_return_10d")
            approved = (
                stop_reduction >= minimum_stop_reduction
                and selected_excess >= baseline_excess
            )
            choices.append(
                {
                    "approved": approved,
                    "entry_score_min": round(float(score_threshold), 4),
                    "early_stop_probability_max": round(float(risk_threshold), 4),
                    "retention_rate": round(retention, 4),
                    "baseline_early_stop_rate": round(baseline_stop, 4),
                    "selected_early_stop_rate": round(selected_stop, 4),
                    "early_stop_reduction": round(stop_reduction, 4),
                    "baseline_excess_return_10d": round(baseline_excess, 6),
                    "selected_excess_return_10d": round(selected_excess, 6),
                    "selected_count": int(len(selected)),
                }
            )

    approved_choices = [choice for choice in choices if choice["approved"]]
    if approved_choices:
        return min(
            approved_choices,
            key=lambda choice: (
                choice["selected_early_stop_rate"],
                -choice["selected_excess_return_10d"],
                -choice["retention_rate"],
            ),
        )
    return {
        "approved": False,
        "entry_score_min": None,
        "early_stop_probability_max": None,
        "retention_rate": 0.0,
        "baseline_early_stop_rate": round(baseline_stop, 4),
        "selected_early_stop_rate": None,
        "early_stop_reduction": 0.0,
        "baseline_excess_return_10d": round(baseline_excess, 6),
        "selected_excess_return_10d": None,
        "selected_count": 0,
    }
