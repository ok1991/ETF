"""Point-in-time forward labels for V4 calibration."""

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
    fee_rate: float = 0.00015,
) -> Dict[str, Any]:
    """Label a close-generated signal using a T+1 executable entry."""
    etf = _prepared(etf_df)
    benchmark = _prepared(benchmark_df)
    signal_ts = pd.Timestamp(signal_date)
    future = etf[etf["date"] > signal_ts].reset_index(drop=True)
    if len(future) < 11:
        raise ValueError("at least 11 post-signal trading days are required")

    entry_row = future.iloc[0]
    entry_price = (
        float(entry_row["open"])
        * (1.0 + float(slippage_bps) / 10000.0)
        * (1.0 + float(fee_rate))
    )
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("entry and stop prices must be positive")

    first_three = future.iloc[:3]
    early_stop = int(bool((first_three["low"].astype(float) <= float(stop_price)).any()))
    horizons: Dict[int, float] = {}
    for horizon in (5, 10, 20):
        if len(future) > horizon:
            exit_value = (
                float(future.iloc[horizon]["close"])
                * (1.0 - float(slippage_bps) / 10000.0)
                * (1.0 - float(fee_rate))
            )
            horizons[horizon] = exit_value / entry_price - 1.0

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
        "target_date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
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


