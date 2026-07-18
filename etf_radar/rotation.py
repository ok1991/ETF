"""Staggered industry-ETF rotation portfolio and live target planning."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .trading import DEFAULT_ETF_COST_MODEL, TradingCostModel
from .universe import industry_group


ROTATION_WEIGHTS: Dict[str, float] = {
    "relative_strength": 0.50,
    "trend_efficiency_20": 0.20,
    "volume_confirmation": 0.15,
    "priority": 0.15,
}

ROTATION_ECONOMIC_LOGIC = {
    "relative_strength": "行业相对沪深300的强势具有中期延续性，是轮动主驱动。",
    "trend_efficiency_20": "趋势效率过滤高换手、低方向性的伪强势。",
    "volume_confirmation": "量能确认区分真实资金流入与无量价格漂移。",
    "priority": "原V4综合优先级补充形态、风险与多周期趋势质量。",
}


def score_rotation_candidates(
    frame: pd.DataFrame,
    weights: Mapping[str, float] = ROTATION_WEIGHTS,
) -> pd.DataFrame:
    """Create point-in-time cross-sectional ranks and an economic rotation score."""
    output = frame.copy()
    date_column = "entry_date" if "entry_date" in output.columns else "date"
    output[date_column] = pd.to_datetime(output[date_column])
    output["rotation_score"] = 0.0
    for name, weight in weights.items():
        values = pd.to_numeric(output.get(name, 0.0), errors="coerce").fillna(0.0)
        ranks = values.groupby(output[date_column]).rank(pct=True, method="average")
        output[f"rotation_rank__{name}"] = ranks
        output["rotation_score"] += float(weight) * ranks
    return output


def select_rotation_targets(
    frame: pd.DataFrame,
    top_n: int = 3,
    weekly_trend_min: float = -0.25,
) -> List[Dict[str, Any]]:
    """Select one ETF per broad industry, ordered by rotation score."""
    eligible = frame[pd.to_numeric(frame["weekly_trend"], errors="coerce") >= weekly_trend_min]
    selected: List[Dict[str, Any]] = []
    used_groups = set()
    for row in eligible.sort_values("rotation_score", ascending=False).to_dict("records"):
        group = str(row.get("industry_group") or industry_group(str(row.get("code", "")), str(row.get("name", ""))))
        if group in used_groups:
            continue
        row["industry_group"] = group
        selected.append(row)
        used_groups.add(group)
        if len(selected) >= max(1, int(top_n)):
            break
    return selected


def _prepared_frames(raw_frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {
        str(code): frame.assign(date=pd.to_datetime(frame["date"])).set_index("date").sort_index()
        for code, frame in raw_frames.items()
    }


def _price(frame: pd.DataFrame, date: pd.Timestamp, column: str = "open") -> Optional[float]:
    if date in frame.index:
        value = frame.loc[date, column]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return float(value)
    history = frame.loc[:date]
    return float(history[column].iloc[-1]) if not history.empty else None


def _average_amount(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    history = frame.loc[:date].tail(20)
    if history.empty:
        return 0.0
    if "amount" in history.columns:
        return float(history["amount"].astype(float).mean())
    volume = history.get("volume", pd.Series(0.0, index=history.index)).astype(float)
    return float((history["close"].astype(float) * volume).mean())


def _sleeve_equity(
    sleeve: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    date: pd.Timestamp,
) -> float:
    return float(sleeve["cash"]) + sum(
        float(shares) * float(_price(frames[code], date, "open") or 0.0)
        for code, shares in sleeve["positions"].items()
        if code in frames
    )


def _rebalance_sleeve(
    sleeve: Dict[str, Any],
    targets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    date: pd.Timestamp,
    cost_model: TradingCostModel,
) -> Tuple[float, float]:
    equity = _sleeve_equity(sleeve, frames, date)
    target_value = equity / len(targets) if targets else 0.0
    desired: Dict[str, int] = {}
    for code in targets:
        price = _price(frames[code], date, "open") if code in frames else None
        desired[code] = cost_model.round_lot(target_value / price) if price and price > 0 else 0

    total_cost = 0.0
    traded_value = 0.0
    positions = dict(sleeve["positions"])
    for code in set(positions) | set(desired):
        current = int(positions.get(code, 0))
        target = int(desired.get(code, 0))
        if target >= current or current <= 0 or code not in frames:
            continue
        shares = current - target
        price = float(_price(frames[code], date, "open") or 0.0)
        execution = cost_model.estimate("SELL", price, shares, _average_amount(frames[code], date))
        sleeve["cash"] += float(execution["cash_delta"])
        total_cost += float(execution["total_cost"])
        traded_value += float(execution["gross"])
        if target > 0:
            positions[code] = target
        else:
            positions.pop(code, None)

    for code in targets:
        current = int(positions.get(code, 0))
        target = int(desired.get(code, 0))
        if target <= current or code not in frames:
            continue
        price = float(_price(frames[code], date, "open") or 0.0)
        shares = target - current
        execution = cost_model.estimate("BUY", price, shares, _average_amount(frames[code], date))
        while shares > 0 and -float(execution["cash_delta"]) > float(sleeve["cash"]):
            shares -= cost_model.lot_size
            execution = cost_model.estimate("BUY", price, shares, _average_amount(frames[code], date))
        if shares <= 0:
            continue
        sleeve["cash"] += float(execution["cash_delta"])
        positions[code] = current + shares
        total_cost += float(execution["total_cost"])
        traded_value += float(execution["gross"])
    sleeve["positions"] = positions
    return total_cost, traded_value


def simulate_staggered_rotation(
    rows: pd.DataFrame,
    raw_frames: Mapping[str, pd.DataFrame],
    top_n: int = 3,
    weekly_trend_min: float = -0.25,
    sleeve_count: int = 2,
    initial_capital: float = 1_000_000.0,
    benchmark_code: str = "510300",
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
) -> Dict[str, Any]:
    """Backtest weekly staggered sleeves, each holding its targets for ten days."""
    if rows.empty:
        return {"return": None, "benchmark_return": None, "information_ratio": None}
    data = score_rotation_candidates(rows)
    data["entry_date"] = pd.to_datetime(data["entry_date"])
    frames = _prepared_frames(raw_frames)
    dates = sorted(pd.Timestamp(value) for value in data["entry_date"].unique())
    if len(dates) < 3 or benchmark_code not in frames:
        return {"return": None, "benchmark_return": None, "information_ratio": None}
    grouped = {pd.Timestamp(date): part for date, part in data.groupby("entry_date")}
    sleeves = [
        {"cash": float(initial_capital) / max(1, sleeve_count), "positions": {}}
        for _ in range(max(1, sleeve_count))
    ]
    equity_values = [float(initial_capital)]
    equity_dates = [dates[0]]
    strategy_returns: List[float] = []
    benchmark_returns: List[float] = []
    period_records: List[Dict[str, Any]] = []
    total_cost = 0.0
    traded_value = 0.0

    initial_targets = [str(row["code"]) for row in select_rotation_targets(grouped[dates[0]], top_n, weekly_trend_min)]
    for sleeve in sleeves:
        cost, traded = _rebalance_sleeve(sleeve, initial_targets, frames, dates[0], cost_model)
        total_cost += cost
        traded_value += traded

    for index, (date, next_date) in enumerate(zip(dates, dates[1:])):
        if index > 0:
            sleeve_index = index % len(sleeves)
            targets = [
                str(row["code"])
                for row in select_rotation_targets(grouped[date], top_n, weekly_trend_min)
            ]
            cost, traded = _rebalance_sleeve(
                sleeves[sleeve_index], targets, frames, date, cost_model
            )
            total_cost += cost
            traded_value += traded
        current_equity = sum(_sleeve_equity(sleeve, frames, date) for sleeve in sleeves)
        next_equity = sum(_sleeve_equity(sleeve, frames, next_date) for sleeve in sleeves)
        strategy_return = next_equity / max(current_equity, 1e-9) - 1.0
        benchmark_open = _price(frames[benchmark_code], date, "open")
        benchmark_next = _price(frames[benchmark_code], next_date, "open")
        benchmark_return = (
            benchmark_next / benchmark_open - 1.0
            if benchmark_open and benchmark_next else 0.0
        )
        strategy_returns.append(float(strategy_return))
        benchmark_returns.append(float(benchmark_return))
        equity_values.append(float(next_equity))
        equity_dates.append(next_date)
        period_records.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "strategy_return": float(strategy_return),
                "benchmark_return": float(benchmark_return),
                "active_return": float(strategy_return - benchmark_return),
            }
        )

    strategy = pd.Series(strategy_returns, index=pd.DatetimeIndex(equity_dates[1:]), dtype=float)
    benchmark = pd.Series(benchmark_returns, index=strategy.index, dtype=float)
    active = strategy - benchmark
    curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), dtype=float)
    benchmark_curve = (1.0 + benchmark).cumprod()
    strategy_total = float(curve.iloc[-1] / initial_capital - 1.0)
    benchmark_total = float(benchmark_curve.iloc[-1] - 1.0)
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1.0 / 52.0)
    drawdown = curve / curve.cummax() - 1.0
    benchmark_drawdown = benchmark_curve / benchmark_curve.cummax() - 1.0
    information_ratio = float(active.mean() / max(float(active.std()), 1e-8) * math.sqrt(52.0))
    sharpe = float(strategy.mean() / max(float(strategy.std()), 1e-8) * math.sqrt(52.0))

    year_checks: List[Dict[str, Any]] = []
    for year, part in pd.DataFrame({"strategy": strategy, "benchmark": benchmark}).groupby(strategy.index.year):
        strategy_year = float((1.0 + part["strategy"]).prod() - 1.0)
        benchmark_year = float((1.0 + part["benchmark"]).prod() - 1.0)
        year_checks.append(
            {
                "year": int(year),
                "strategy_return": round(strategy_year, 6),
                "benchmark_return": round(benchmark_year, 6),
                "excess_return": round(strategy_year - benchmark_year, 6),
            }
        )
    positive_years = sum(item["excess_return"] > 0 for item in year_checks)
    return {
        "return": round(strategy_total, 6),
        "cagr": round((1.0 + strategy_total) ** (1.0 / years) - 1.0, 6),
        "benchmark_return": round(benchmark_total, 6),
        "benchmark_cagr": round((1.0 + benchmark_total) ** (1.0 / years) - 1.0, 6),
        "excess_return": round(strategy_total - benchmark_total, 6),
        "information_ratio": round(information_ratio, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(abs(float(drawdown.min())), 6),
        "benchmark_max_drawdown": round(abs(float(benchmark_drawdown.min())), 6),
        "turnover": round(traded_value / max(float(curve.mean()), 1.0), 6),
        "total_cost": round(total_cost, 2),
        "cost_ratio": round(total_cost / max(initial_capital, 1.0), 8),
        "top_n": int(top_n),
        "sleeve_count": int(sleeve_count),
        "holding_period_trading_days": 10,
        "weekly_trend_min": float(weekly_trend_min),
        "factor_weights": dict(ROTATION_WEIGHTS),
        "factor_economic_logic": dict(ROTATION_ECONOMIC_LOGIC),
        "industry_constraint": "one_etf_per_broad_industry_per_sleeve",
        "year_checks": year_checks,
        "positive_year_ratio": round(positive_years / len(year_checks), 6) if year_checks else 0.0,
        "period_records": period_records,
        "cost_model": cost_model.to_dict(),
    }


def update_live_rotation_state(
    candidates: pd.DataFrame,
    previous_state: Optional[Mapping[str, Any]],
    data_date: Any,
    top_n: int = 3,
    weekly_trend_min: float = -0.25,
) -> Dict[str, Any]:
    """Update one of two live sleeves once per ISO week and return target weights."""
    date = pd.Timestamp(data_date)
    scored = score_rotation_candidates(candidates)
    targets = [str(row["code"]) for row in select_rotation_targets(scored, top_n, weekly_trend_min)]
    state = dict(previous_state or {})
    sleeves = [list(value) for value in state.get("sleeves", [])]
    if len(sleeves) != 2:
        sleeves = [list(targets), list(targets)]
    week_key = f"{date.isocalendar().year}-W{date.isocalendar().week:02d}"
    if state.get("last_rebalance_week") != week_key:
        sleeves[int(date.isocalendar().week) % 2] = list(targets)
    weights: Dict[str, float] = {}
    for sleeve in sleeves:
        if not sleeve:
            continue
        for code in sleeve:
            weights[code] = weights.get(code, 0.0) + 0.5 / len(sleeve)
    ranked = scored.sort_values("rotation_score", ascending=False)
    return {
        "schema_version": 1,
        "data_date": date.strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_rebalance_week": week_key,
        "sleeves": sleeves,
        "target_weights": {code: round(weight, 6) for code, weight in sorted(weights.items())},
        "new_sleeve_targets": targets,
        "top_candidates": [
            {
                "code": str(row.get("code", "")),
                "name": str(row.get("name", "")),
                "industry_group": str(row.get("industry_group", "other")),
                "rotation_score": round(float(row.get("rotation_score", 0.0)), 6),
            }
            for row in ranked.head(10).to_dict("records")
        ],
        "factor_weights": dict(ROTATION_WEIGHTS),
        "factor_economic_logic": dict(ROTATION_ECONOMIC_LOGIC),
        "holding_period_trading_days": 10,
        "rebalance_frequency_trading_days": 5,
    }


def load_rotation_state(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return dict(value) if int(value.get("schema_version", 0)) == 1 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def load_rotation_model(path: str, max_age_days: int = 180) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if int(value.get("schema_version", 0)) != 1 or not bool(value.get("approved", False)):
            return None
        trained_until = pd.to_datetime(value.get("trained_until"), errors="coerce")
        if pd.isna(trained_until):
            return None
        age = (pd.Timestamp.now().normalize() - pd.Timestamp(trained_until).normalize()).days
        return dict(value) if age <= max(1, int(max_age_days)) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_rotation_state(value: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


__all__ = [
    "ROTATION_ECONOMIC_LOGIC",
    "ROTATION_WEIGHTS",
    "load_rotation_state",
    "load_rotation_model",
    "save_rotation_state",
    "score_rotation_candidates",
    "select_rotation_targets",
    "simulate_staggered_rotation",
    "update_live_rotation_state",
]
