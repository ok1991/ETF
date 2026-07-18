"""Point-in-time forward labels for V4 calibration."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from ..trading import TradingCostModel


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
    cost_model: Optional[TradingCostModel] = None,
    assumed_notional: float = 100_000.0,
) -> Dict[str, Any]:
    """Label a close-generated signal using a T+1 executable entry and full costs."""
    etf = _prepared(etf_df)
    benchmark = _prepared(benchmark_df)
    signal_ts = pd.Timestamp(signal_date)
    future = etf[etf["date"] > signal_ts].reset_index(drop=True)
    if len(future) < 11:
        raise ValueError("at least 11 post-signal trading days are required")

    entry_row = future.iloc[0]
    model = cost_model or TradingCostModel(
        commission_rate=float(fee_rate),
        minimum_commission=0.0,
        exchange_handling_rate=0.0,
        bid_ask_half_spread_bps=0.0,
        base_slippage_bps=float(slippage_bps),
        impact_bps_at_full_adv=0.0,
    )
    open_price = float(entry_row["open"])
    shares = max(float(assumed_notional) / max(open_price, 1e-9), 1.0)
    if "amount" in future.columns:
        average_amount = float(future["amount"].astype(float).head(20).mean())
    else:
        volume = future.get("volume", pd.Series(0.0, index=future.index)).astype(float)
        average_amount = float((future["close"].astype(float) * volume).head(20).mean())
    entry_cost = model.estimate("BUY", open_price, shares, average_daily_amount=average_amount)
    entry_price = float(entry_cost["effective_price"])
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("entry and stop prices must be positive")

    first_three = future.iloc[:3]
    early_stop = int(bool((first_three["low"].astype(float) <= float(stop_price)).any()))
    horizons: Dict[int, float] = {}
    for horizon in (5, 10, 20):
        if len(future) > horizon:
            exit_cost = model.estimate(
                "SELL",
                float(future.iloc[horizon]["close"]),
                shares,
                average_daily_amount=average_amount,
            )
            exit_value = float(exit_cost["effective_price"])
            horizons[horizon] = exit_value / entry_price - 1.0

    benchmark_future = benchmark[benchmark["date"] >= entry_row["date"]].reset_index(drop=True)
    benchmark_entry = benchmark_future[benchmark_future["date"] == entry_row["date"]]
    if benchmark_entry.empty:
        raise ValueError("benchmark is missing aligned entry date")
    benchmark_returns: Dict[int, float] = {}
    for horizon in (5, 10, 20):
        if len(future) <= horizon:
            continue
        target_date = future.iloc[horizon]["date"]
        benchmark_target = benchmark_future[benchmark_future["date"] == target_date]
        if benchmark_target.empty:
            raise ValueError("benchmark is missing aligned horizon dates")
        benchmark_returns[horizon] = (
            float(benchmark_target.iloc[0]["close"]) / float(benchmark_entry.iloc[0]["open"]) - 1.0
        )
    target_date = future.iloc[10]["date"]

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
        "benchmark_return_5d": round(benchmark_returns.get(5, 0.0), 8),
        "benchmark_return_10d": round(benchmark_returns[10], 8),
        "benchmark_return_20d": round(benchmark_returns.get(20, 0.0), 8),
        "excess_return_5d": round(horizons.get(5, 0.0) - benchmark_returns.get(5, 0.0), 8),
        "excess_return_10d": round(horizons[10] - benchmark_returns[10], 8),
        "excess_return_20d": round(horizons.get(20, 0.0) - benchmark_returns.get(20, 0.0), 8),
        "win_10d": int(horizons[10] > 0),
        "max_adverse_excursion": round(max_adverse_excursion, 8),
        "estimated_round_trip_cost": round(
            (float(entry_cost["total_cost"]) + float(model.estimate(
                "SELL", float(future.iloc[10]["close"]), shares, average_daily_amount=average_amount
            )["total_cost"])) / max(float(assumed_notional), 1.0),
            8,
        ),
    }

