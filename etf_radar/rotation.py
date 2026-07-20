"""Staggered industry-ETF rotation portfolio and live target planning."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model_governance import (
    MODEL_GENERATED_MAX_AGE_DAYS,
    MODEL_TRAINED_MAX_LAG_DAYS,
    validate_artifact_time,
    validate_bundle_member,
)
from .trading import DEFAULT_ETF_COST_MODEL, TradingCostModel
from .universe import industry_group


ROTATION_WEIGHTS: Dict[str, float] = {
    "relative_strength": 0.50,
    "trend_efficiency_20": 0.20,
    "volume_confirmation": 0.15,
    "priority": 0.15,
}

ROTATION_SCHEMA_VERSION = 2
ROTATION_EXECUTION_POLICY_VERSION = "single-exposure-authority-v4"
ROTATION_ACCEPTANCE_POLICY_VERSION = "rolling-excess-stability-v1"
ROTATION_CAPACITY_REFERENCE_CAPITAL = 10_000.0
ROTATION_PUBLICATION_IDENTITY_POLICY_VERSION = "stable-economic-payload-v1"
ROTATION_EXPOSURE_AUTHORITY = "v4_market_policy"
RISK_CONTROL_EXPOSURE_AUTHORITY = "risk_control_fail_closed"

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
    incumbent_codes: Sequence[str] = (),
    rank_buffer: int = 0,
    minimum_average_daily_amount: float = 0.0,
) -> List[Dict[str, Any]]:
    """Select one ETF per industry and retain incumbents inside a rank buffer."""
    eligible = frame[pd.to_numeric(frame["weekly_trend"], errors="coerce") >= weekly_trend_min]
    if float(minimum_average_daily_amount) > 0.0:
        liquidity = pd.to_numeric(
            eligible.get(
                "average_daily_amount_20",
                pd.Series(0.0, index=eligible.index, dtype=float),
            ),
            errors="coerce",
        ).fillna(0.0)
        eligible = eligible[liquidity >= float(minimum_average_daily_amount)]
    ranked: List[Dict[str, Any]] = []
    used_groups = set()
    for row in eligible.sort_values("rotation_score", ascending=False).to_dict("records"):
        group = str(row.get("industry_group") or industry_group(str(row.get("code", "")), str(row.get("name", ""))))
        if group in used_groups:
            continue
        row["industry_group"] = group
        ranked.append(row)
        used_groups.add(group)
        if len(ranked) >= max(1, int(top_n)) + max(0, int(rank_buffer)):
            break
    selected: List[Dict[str, Any]] = []
    selected_groups = set()
    buffered = {str(row.get("code", "")): row for row in ranked}
    for code in incumbent_codes:
        row = buffered.get(str(code))
        if row is None or row["industry_group"] in selected_groups:
            continue
        selected.append(row)
        selected_groups.add(row["industry_group"])
        if len(selected) >= max(1, int(top_n)):
            return selected
    for row in ranked:
        if row["industry_group"] in selected_groups:
            continue
        selected.append(row)
        selected_groups.add(row["industry_group"])
        if len(selected) >= max(1, int(top_n)):
            break
    return selected


def _capacity_aware_rotation_targets(
    frame: pd.DataFrame,
    portfolio_value: float,
    cost_model: TradingCostModel,
    top_n: int = 3,
    weekly_trend_min: float = -0.25,
    incumbent_codes: Sequence[str] = (),
    rank_buffer: int = 0,
) -> List[Dict[str, Any]]:
    """Select only targets whose point-in-time ADV can carry the full allocation."""
    divisor = max(1, int(top_n))
    selected: List[Dict[str, Any]] = []
    for _ in range(max(1, int(top_n)) + 1):
        minimum_adv = (
            max(float(portfolio_value), 0.0)
            / divisor
            / max(float(cost_model.max_participation_rate), 1e-12)
        )
        refined = select_rotation_targets(
            frame,
            top_n=top_n,
            weekly_trend_min=weekly_trend_min,
            incumbent_codes=incumbent_codes,
            rank_buffer=rank_buffer,
            minimum_average_daily_amount=minimum_adv,
        )
        refined_codes = [str(row.get("code", "")) for row in refined]
        selected_codes = [str(row.get("code", "")) for row in selected]
        selected = refined
        next_divisor = max(1, len(selected))
        if refined_codes == selected_codes or next_divisor == divisor:
            break
        divisor = next_divisor
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


def _exposure_ratio(
    frame: pd.DataFrame,
) -> float:
    """Read the single point-in-time exposure authority without inference."""
    if "max_exposure_ratio" not in frame.columns:
        raise ValueError("authoritative max_exposure_ratio is required")
    values = pd.to_numeric(frame["max_exposure_ratio"], errors="coerce")
    if values.empty or values.isna().any():
        raise ValueError("authoritative max_exposure_ratio is not numeric")
    if float(values.max()) - float(values.min()) > 1e-9:
        raise ValueError("authoritative max_exposure_ratio is inconsistent within the date")
    return float(np.clip(values.iloc[0], 0.0, 1.0))


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
    exposure_ratio: float = 1.0,
    used_buy_shares: Optional[Dict[str, int]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    equity = _sleeve_equity(sleeve, frames, date)
    exposure_ratio = float(np.clip(exposure_ratio, 0.0, 1.0))
    target_value = equity * exposure_ratio / len(targets) if targets else 0.0
    desired: Dict[str, int] = {}
    for code in targets:
        price = _price(frames[code], date, "open") if code in frames else None
        desired[code] = cost_model.round_lot(target_value / price) if price and price > 0 else 0

    total_cost = 0.0
    traded_value = 0.0
    capacity_audit = {
        "buy_order_count": 0.0,
        "capacity_truncation_count": 0.0,
        "requested_buy_value": 0.0,
        "capacity_executable_buy_value": 0.0,
        "executed_buy_value": 0.0,
        "capacity_truncated_buy_value": 0.0,
        "cash_limited_buy_value": 0.0,
        "unfilled_buy_value": 0.0,
        "max_requested_participation_rate": 0.0,
        "max_executed_participation_rate": 0.0,
    }
    daily_usage = used_buy_shares if used_buy_shares is not None else {}
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
        requested_shares = target - current
        shares = requested_shares
        average_amount = _average_amount(frames[code], date)
        capacity_shares = cost_model.capacity_lot(price, average_amount)
        prior_used_shares = max(int(daily_usage.get(code, 0)), 0)
        available_capacity_shares = max(capacity_shares - prior_used_shares, 0)
        shares = min(shares, available_capacity_shares)
        requested_value = float(requested_shares) * price
        capacity_value = float(shares) * price
        requested_participation = (
            requested_value / float(average_amount)
            if average_amount and float(average_amount) > 0.0 else 1.0
        )
        capacity_audit["buy_order_count"] += 1.0
        capacity_audit["requested_buy_value"] += requested_value
        capacity_audit["capacity_executable_buy_value"] += capacity_value
        capacity_audit["capacity_truncated_buy_value"] += max(
            requested_value - capacity_value, 0.0
        )
        capacity_audit["max_requested_participation_rate"] = max(
            capacity_audit["max_requested_participation_rate"],
            requested_participation,
        )
        if shares < requested_shares:
            capacity_audit["capacity_truncation_count"] += 1.0
        execution = cost_model.estimate("BUY", price, shares, average_amount)
        while shares > 0 and -float(execution["cash_delta"]) > float(sleeve["cash"]):
            shares -= cost_model.lot_size
            execution = cost_model.estimate("BUY", price, shares, average_amount)
        executed_value = float(shares) * price if shares > 0 else 0.0
        capacity_audit["executed_buy_value"] += executed_value
        capacity_audit["cash_limited_buy_value"] += max(
            capacity_value - executed_value, 0.0
        )
        capacity_audit["unfilled_buy_value"] += max(
            requested_value - executed_value, 0.0
        )
        if average_amount and float(average_amount) > 0.0:
            daily_usage[code] = prior_used_shares + max(int(shares), 0)
            capacity_audit["max_executed_participation_rate"] = max(
                capacity_audit["max_executed_participation_rate"],
                daily_usage[code] * price / float(average_amount),
            )
        if shares <= 0:
            continue
        sleeve["cash"] += float(execution["cash_delta"])
        positions[code] = current + shares
        total_cost += float(execution["total_cost"])
        traded_value += float(execution["gross"])
    sleeve["positions"] = positions
    return total_cost, traded_value, capacity_audit


def _rolling_rotation_stability(
    strategy: pd.Series,
    benchmark: pd.Series,
    rolling_periods: int = 52,
) -> Dict[str, Any]:
    """Measure whether excess performance persists beyond aggregate backtest totals."""
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
    ).dropna()
    window = max(4, int(rolling_periods))
    if aligned.empty:
        return {
            "rolling_12m_observations": 0,
            "rolling_12m_positive_excess_ratio": 0.0,
            "rolling_12m_median_excess_return": 0.0,
            "rolling_12m_worst_excess_return": 0.0,
            "max_relative_drawdown": 0.0,
            "longest_relative_underwater_periods": 0,
        }
    rolling_strategy = (1.0 + aligned["strategy"]).rolling(window).apply(
        np.prod,
        raw=True,
    ) - 1.0
    rolling_benchmark = (1.0 + aligned["benchmark"]).rolling(window).apply(
        np.prod,
        raw=True,
    ) - 1.0
    rolling_excess = (rolling_strategy - rolling_benchmark).dropna()
    relative_curve = (1.0 + aligned["strategy"]).cumprod() / (
        1.0 + aligned["benchmark"]
    ).cumprod().replace(0.0, np.nan)
    relative_drawdown = relative_curve / relative_curve.cummax() - 1.0
    underwater = relative_drawdown < -1e-12
    longest_underwater = 0
    current_underwater = 0
    for value in underwater.fillna(False):
        current_underwater = current_underwater + 1 if bool(value) else 0
        longest_underwater = max(longest_underwater, current_underwater)
    return {
        "rolling_12m_observations": int(len(rolling_excess)),
        "rolling_12m_positive_excess_ratio": round(
            float((rolling_excess > 0.0).mean()) if len(rolling_excess) else 0.0,
            6,
        ),
        "rolling_12m_median_excess_return": round(
            float(rolling_excess.median()) if len(rolling_excess) else 0.0,
            6,
        ),
        "rolling_12m_worst_excess_return": round(
            float(rolling_excess.min()) if len(rolling_excess) else 0.0,
            6,
        ),
        "max_relative_drawdown": round(
            abs(float(relative_drawdown.min())) if len(relative_drawdown) else 0.0,
            6,
        ),
        "longest_relative_underwater_periods": int(longest_underwater),
    }


def simulate_staggered_rotation(
    rows: pd.DataFrame,
    raw_frames: Mapping[str, pd.DataFrame],
    top_n: int = 3,
    weekly_trend_min: float = -0.25,
    sleeve_count: int = 2,
    initial_capital: float = ROTATION_CAPACITY_REFERENCE_CAPITAL,
    benchmark_code: str = "510300",
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
    rank_buffer: int = 0,
) -> Dict[str, Any]:
    """Backtest weekly staggered sleeves, each holding its targets for ten days."""
    if rows.empty:
        return {"return": None, "benchmark_return": None, "information_ratio": None}
    frames = _prepared_frames(raw_frames)
    data = score_rotation_candidates(rows)
    data["entry_date"] = pd.to_datetime(data["entry_date"])
    data["average_daily_amount_20"] = [
        _average_amount(frames[str(code)], pd.Timestamp(date))
        if str(code) in frames else 0.0
        for code, date in zip(data["code"], data["entry_date"])
    ]
    dates = sorted(pd.Timestamp(value) for value in data["entry_date"].unique())
    if len(dates) < 3 or benchmark_code not in frames:
        return {"return": None, "benchmark_return": None, "information_ratio": None}
    grouped = {pd.Timestamp(date): part for date, part in data.groupby("entry_date")}
    sleeves = [
        {"cash": float(initial_capital) / max(1, sleeve_count), "positions": {}, "targets": []}
        for _ in range(max(1, sleeve_count))
    ]
    equity_values = [float(initial_capital)]
    equity_dates = [dates[0]]
    strategy_returns: List[float] = []
    benchmark_returns: List[float] = []
    period_records: List[Dict[str, Any]] = []
    total_cost = 0.0
    traded_value = 0.0
    exposure_values: List[float] = []
    rebalance_count = 0
    skipped_unchanged_rebalances = 0
    capacity_audit = {
        "buy_order_count": 0.0,
        "capacity_truncation_count": 0.0,
        "requested_buy_value": 0.0,
        "capacity_executable_buy_value": 0.0,
        "executed_buy_value": 0.0,
        "capacity_truncated_buy_value": 0.0,
        "cash_limited_buy_value": 0.0,
        "unfilled_buy_value": 0.0,
        "max_requested_participation_rate": 0.0,
        "max_executed_participation_rate": 0.0,
    }

    def merge_capacity_audit(update: Mapping[str, float]) -> None:
        for key, value in update.items():
            if key.startswith("max_"):
                capacity_audit[key] = max(capacity_audit[key], float(value))
            else:
                capacity_audit[key] += float(value)

    # Candidate capacity is screened against the full reference portfolio even
    # when the authoritative market policy currently requires cash.
    capacity_exposure = 1.0
    initial_targets = [
        str(row["code"])
        for row in _capacity_aware_rotation_targets(
            grouped[dates[0]],
            float(initial_capital) * float(np.clip(capacity_exposure, 0.0, 1.0)),
            cost_model,
            top_n=top_n,
            weekly_trend_min=weekly_trend_min,
            rank_buffer=rank_buffer,
        )
    ]
    current_exposure = _exposure_ratio(grouped[dates[0]])
    daily_capacity_usage: Dict[str, int] = {}
    for sleeve in sleeves:
        sleeve["targets"] = list(initial_targets)
        cost, traded, audit = _rebalance_sleeve(
            sleeve,
            initial_targets,
            frames,
            dates[0],
            cost_model,
            current_exposure,
            daily_capacity_usage,
        )
        total_cost += cost
        traded_value += traded
        merge_capacity_audit(audit)
        rebalance_count += 1
    exposure_values.append(current_exposure)

    for index, (date, next_date) in enumerate(zip(dates, dates[1:])):
        if index > 0:
            sleeve_index = index % len(sleeves)
            next_exposure = _exposure_ratio(grouped[date])
            portfolio_equity = sum(
                _sleeve_equity(sleeve, frames, date) for sleeve in sleeves
            )
            targets = [
                str(row["code"])
                for row in _capacity_aware_rotation_targets(
                    grouped[date],
                    portfolio_equity
                    * float(np.clip(capacity_exposure, 0.0, 1.0)),
                    cost_model,
                    top_n=top_n,
                    weekly_trend_min=weekly_trend_min,
                    incumbent_codes=sleeves[sleeve_index].get("targets", []),
                    rank_buffer=rank_buffer,
                )
            ]
            targets_changed = list(targets) != list(sleeves[sleeve_index].get("targets", []))
            sleeves[sleeve_index]["targets"] = list(targets)
            if next_exposure != current_exposure:
                rebalance_indexes: Sequence[int] = range(len(sleeves))
            elif targets_changed:
                rebalance_indexes = (sleeve_index,)
            else:
                rebalance_indexes = ()
                skipped_unchanged_rebalances += 1
            daily_capacity_usage = {}
            for rebalance_index in rebalance_indexes:
                sleeve = sleeves[rebalance_index]
                cost, traded, audit = _rebalance_sleeve(
                    sleeve,
                    sleeve.get("targets", []),
                    frames,
                    date,
                    cost_model,
                    next_exposure,
                    daily_capacity_usage,
                )
                total_cost += cost
                traded_value += traded
                merge_capacity_audit(audit)
                rebalance_count += 1
            current_exposure = next_exposure
        exposure_values.append(current_exposure)
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
                "max_exposure_ratio": float(current_exposure),
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
    rolling_stability = _rolling_rotation_stability(strategy, benchmark)
    requested_buy_value = capacity_audit["requested_buy_value"]
    executed_buy_value = capacity_audit["executed_buy_value"]
    capacity_executable_buy_value = capacity_audit["capacity_executable_buy_value"]
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
        "buy_order_count": int(capacity_audit["buy_order_count"]),
        "capacity_truncation_count": int(capacity_audit["capacity_truncation_count"]),
        "requested_buy_value": round(requested_buy_value, 2),
        "capacity_executable_buy_value": round(capacity_executable_buy_value, 2),
        "executed_buy_value": round(executed_buy_value, 2),
        "capacity_truncated_buy_value": round(
            capacity_audit["capacity_truncated_buy_value"], 2
        ),
        "cash_limited_buy_value": round(capacity_audit["cash_limited_buy_value"], 2),
        "unfilled_buy_value": round(capacity_audit["unfilled_buy_value"], 2),
        "buy_fill_ratio": round(
            executed_buy_value / requested_buy_value if requested_buy_value > 0.0 else 1.0,
            8,
        ),
        "capacity_fill_ratio": round(
            capacity_executable_buy_value / requested_buy_value
            if requested_buy_value > 0.0 else 1.0,
            8,
        ),
        "max_requested_participation_rate": round(
            capacity_audit["max_requested_participation_rate"], 8
        ),
        "max_executed_participation_rate": round(
            capacity_audit["max_executed_participation_rate"], 8
        ),
        "top_n": int(top_n),
        "sleeve_count": int(sleeve_count),
        "holding_period_trading_days": 10,
        "weekly_trend_min": float(weekly_trend_min),
        "factor_weights": dict(ROTATION_WEIGHTS),
        "factor_economic_logic": dict(ROTATION_ECONOMIC_LOGIC),
        "industry_constraint": "one_etf_per_broad_industry_per_sleeve",
        "year_checks": year_checks,
        "positive_year_ratio": round(positive_years / len(year_checks), 6) if year_checks else 0.0,
        **rolling_stability,
        "average_exposure_ratio": round(float(np.mean(exposure_values)), 6) if exposure_values else 1.0,
        "risk_overlay": "authoritative_v4_market_policy_max_exposure",
        "exposure_authority": ROTATION_EXPOSURE_AUTHORITY,
        "rebalance_count": int(rebalance_count),
        "skipped_unchanged_rebalances": int(skipped_unchanged_rebalances),
        "rank_buffer": int(rank_buffer),
        "period_records": period_records,
        "cost_model": cost_model.to_dict(),
        "capacity_reference_capital": round(float(initial_capital), 2),
        "capacity_selection_policy": "point_in_time_adv_supports_full_reference_target",
    }


def update_live_rotation_state(
    candidates: pd.DataFrame,
    previous_state: Optional[Mapping[str, Any]],
    data_date: Any,
    top_n: int = 3,
    weekly_trend_min: float = -0.25,
    market_policy: Optional[Mapping[str, Any]] = None,
    rank_buffer: int = 0,
    execution_date: Any = None,
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
    capacity_reference_capital: float = ROTATION_CAPACITY_REFERENCE_CAPITAL,
    model_authority: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Update one of two live sleeves once per ISO week and return target weights."""
    date = pd.Timestamp(data_date)
    execution = pd.to_datetime(execution_date, errors="coerce")
    scored = score_rotation_candidates(candidates)
    policy = dict(market_policy or {})
    if "max_exposure_ratio" not in policy:
        raise ValueError("market_policy.max_exposure_ratio is the required exposure authority")
    policy.setdefault("state", "UNKNOWN")
    exposure_ratio = float(np.clip(float(policy["max_exposure_ratio"]), 0.0, 1.0))
    policy.setdefault("entry_permission", "TRADEABLE" if exposure_ratio > 0 else "BLOCKED")
    if str(policy["entry_permission"]) == "BLOCKED" and exposure_ratio > 0.0:
        raise ValueError("BLOCKED market policy cannot authorize positive exposure")
    policy["max_exposure_ratio"] = exposure_ratio
    capacity_exposure = 1.0
    capacity_portfolio_value = max(float(capacity_reference_capital), 0.0) * float(
        np.clip(capacity_exposure, 0.0, 1.0)
    )
    state = dict(previous_state or {})
    authority = {
        key: str((model_authority or {}).get(key, ""))
        for key in (
            "model_version",
            "execution_policy_version",
            "acceptance_policy_version",
            "strategy_specification_fingerprint",
        )
        if str((model_authority or {}).get(key, ""))
    }
    authority_mismatches = (
        [
            key
            for key, expected in authority.items()
            if str(state.get(key, "")) != expected
        ]
        if state else []
    )
    state_reset_reason = ""
    if authority_mismatches:
        state = {}
        state_reset_reason = "MODEL_AUTHORITY_CHANGED"
    sleeves = [list(value) for value in state.get("sleeves", [])]
    week_key = f"{date.isocalendar().year}-W{date.isocalendar().week:02d}"
    sleeve_index = int(date.isocalendar().week) % 2
    if len(sleeves) != 2:
        targets = [
            str(row["code"])
            for row in _capacity_aware_rotation_targets(
                scored,
                capacity_portfolio_value,
                cost_model,
                top_n=top_n,
                weekly_trend_min=weekly_trend_min,
                rank_buffer=rank_buffer,
            )
        ]
        sleeves = [list(targets), list(targets)]
    elif state.get("last_rebalance_week") != week_key:
        targets = [
            str(row["code"])
            for row in _capacity_aware_rotation_targets(
                scored,
                capacity_portfolio_value,
                cost_model,
                top_n=top_n,
                weekly_trend_min=weekly_trend_min,
                incumbent_codes=sleeves[sleeve_index],
                rank_buffer=rank_buffer,
            )
        ]
        sleeves[sleeve_index] = list(targets)
    else:
        targets = list(sleeves[sleeve_index])
    weights: Dict[str, float] = {}
    for sleeve in sleeves:
        if not sleeve:
            continue
        for code in sleeve:
            weights[code] = weights.get(code, 0.0) + exposure_ratio * 0.5 / len(sleeve)
    ranked = scored.sort_values("rotation_score", ascending=False)
    result = {
        "schema_version": ROTATION_SCHEMA_VERSION,
        "data_date": date.strftime("%Y-%m-%d"),
        "execution_date": (
            pd.Timestamp(execution).strftime("%Y-%m-%d") if not pd.isna(execution) else ""
        ),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_rebalance_week": week_key,
        "sleeves": sleeves,
        "target_weights": {
            code: round(weight, 6)
            for code, weight in sorted(weights.items())
            if weight > 0.0
        },
        "execution_liquidity": {
            str(row.get("code", "")): {
                "average_daily_amount_20": round(
                    max(float(row.get("average_daily_amount_20", 0.0) or 0.0), 0.0),
                    2,
                ),
                "max_new_risk_amount": round(
                    max(float(row.get("average_daily_amount_20", 0.0) or 0.0), 0.0)
                    * max(float(cost_model.max_participation_rate), 0.0),
                    2,
                ),
                "max_participation_rate": float(cost_model.max_participation_rate),
                "as_of_date": date.strftime("%Y-%m-%d"),
            }
            for row in ranked.to_dict("records")
            if str(row.get("code", ""))
            and float(row.get("average_daily_amount_20", 0.0) or 0.0) > 0.0
        },
        "max_exposure_ratio": round(exposure_ratio, 6),
        "cash_weight": round(1.0 - exposure_ratio, 6),
        "capacity_reference_capital": round(float(capacity_reference_capital), 2),
        "market_policy": policy,
        "exposure_authority": ROTATION_EXPOSURE_AUTHORITY,
        "rank_buffer": int(rank_buffer),
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
    if authority:
        result.update(authority)
    if state_reset_reason:
        result["state_reset_reason"] = state_reset_reason
        result["state_reset_fields"] = authority_mismatches
    return result


def build_cash_rotation_target(
    data_date: Any,
    market_policy: Optional[Mapping[str, Any]] = None,
    reason: str = "ROTATION_MODEL_NOT_APPROVED_OR_UNAVAILABLE",
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
    execution_date: Any = None,
) -> Dict[str, Any]:
    """Publish an actionable cash target when alpha authority is withdrawn."""
    date = pd.Timestamp(data_date)
    execution = pd.to_datetime(execution_date, errors="coerce")
    policy = dict(market_policy or {})
    policy.update(
        {
            "state": "RISK_OFF",
            "entry_permission": "BLOCKED",
            "max_exposure_ratio": 0.0,
            "risk_reason": str(reason),
        }
    )
    week_key = f"{date.isocalendar().year}-W{date.isocalendar().week:02d}"
    return {
        "schema_version": ROTATION_SCHEMA_VERSION,
        "data_date": date.strftime("%Y-%m-%d"),
        "execution_date": (
            pd.Timestamp(execution).strftime("%Y-%m-%d") if not pd.isna(execution) else ""
        ),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_rebalance_week": week_key,
        "sleeves": [[], []],
        "target_weights": {},
        "execution_liquidity": {},
        "max_exposure_ratio": 0.0,
        "cash_weight": 1.0,
        "capacity_reference_capital": round(
            float(ROTATION_CAPACITY_REFERENCE_CAPITAL), 2
        ),
        "market_policy": policy,
        "exposure_authority": RISK_CONTROL_EXPOSURE_AUTHORITY,
        "approved": True,
        "execution_policy_version": ROTATION_EXECUTION_POLICY_VERSION,
        "acceptance_policy_version": ROTATION_ACCEPTANCE_POLICY_VERSION,
        "alpha_model_approved": False,
        "risk_control_only": True,
        "reason": str(reason),
        "model_version": "risk-control-cash-v4",
        "walk_forward_metrics": {
            "information_ratio": 0.0,
            "capacity_truncation_count": 0,
            "requested_buy_value": 0.0,
            "executed_buy_value": 0.0,
            "capacity_truncated_buy_value": 0.0,
            "unfilled_buy_value": 0.0,
            "buy_fill_ratio": 1.0,
            "capacity_fill_ratio": 1.0,
            "cost_model": cost_model.to_dict(),
        },
    }


def load_rotation_state(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return dict(value) if int(value.get("schema_version", 0)) in {1, ROTATION_SCHEMA_VERSION} else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def stabilize_rotation_publication(
    candidate: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Preserve publication time only when every non-time field is unchanged."""
    current = dict(candidate)
    prior = dict(previous or {})
    model_version = str(current.get("model_version", "")).strip()
    strategy_fingerprint = str(
        current.get("strategy_specification_fingerprint", "")
    ).strip()
    same_authority = bool(
        model_version
        and strategy_fingerprint
        and model_version == str(prior.get("model_version", "")).strip()
        and strategy_fingerprint
        == str(prior.get("strategy_specification_fingerprint", "")).strip()
    )
    reset_audit_keys = ("state_reset_reason", "state_reset_fields")
    missing_reset_keys = [
        key
        for key in reset_audit_keys
        if same_authority and key in prior and key not in current
    ]
    if missing_reset_keys:
        prior_keys = list(prior)
        last_reset_index = max(prior_keys.index(key) for key in missing_reset_keys)
        anchor = next(
            (key for key in prior_keys[last_reset_index + 1 :] if key in current),
            None,
        )
        restored: Dict[str, Any] = {}
        for key, value in current.items():
            if key == anchor:
                for reset_key in missing_reset_keys:
                    reset_value = prior[reset_key]
                    restored[reset_key] = (
                        list(reset_value or [])
                        if reset_key == "state_reset_fields"
                        else reset_value
                    )
            restored[key] = value
        if anchor is None:
            for reset_key in missing_reset_keys:
                reset_value = prior[reset_key]
                restored[reset_key] = (
                    list(reset_value or [])
                    if reset_key == "state_reset_fields"
                    else reset_value
                )
        current = restored
    current_generated_at = str(current.get("generated_at", ""))[:19]
    prior_generated_at = str(prior.get("generated_at", ""))[:19]
    try:
        current_time = datetime.strptime(current_generated_at, "%Y-%m-%d %H:%M:%S")
        prior_time = datetime.strptime(prior_generated_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return current
    if prior_time > current_time:
        return current
    current_identity = {
        key: value for key, value in current.items() if key != "generated_at"
    }
    prior_identity = {
        key: value for key, value in prior.items() if key != "generated_at"
    }
    if current_identity == prior_identity:
        current["generated_at"] = prior_generated_at
    return current


def load_rotation_model_with_status(
    path: str,
    max_age_days: int = MODEL_TRAINED_MAX_LAG_DAYS,
    generated_max_age_days: int = MODEL_GENERATED_MAX_AGE_DAYS,
    now: Any = None,
    require_bundle_integrity: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if int(value.get("schema_version", 0)) != 1:
            return None, "ROTATION_MODEL_SCHEMA_INVALID"
        if not bool(value.get("approved", False)):
            return None, "ROTATION_MODEL_NOT_APPROVED"
        if require_bundle_integrity:
            bundle_status = validate_bundle_member(path, value)
            if bundle_status != "APPROVED":
                return None, f"ROTATION_MODEL_{bundle_status}"
        time_status = validate_artifact_time(
            value,
            now=now,
            generated_max_age_days=generated_max_age_days,
            trained_max_lag_days=max_age_days,
        )
        if not time_status.approved:
            return None, f"ROTATION_MODEL_{time_status.reason}"
        if str(value.get("execution_policy_version", "")) != (
            ROTATION_EXECUTION_POLICY_VERSION
        ):
            return None, "ROTATION_MODEL_EXECUTION_POLICY_MISMATCH"
        if "risk_budget_profile" in value:
            return None, "ROTATION_MODEL_LEGACY_EXPOSURE_AUTHORITY"
        return dict(value), "APPROVED"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, "ROTATION_MODEL_UNAVAILABLE"


def load_rotation_model(
    path: str,
    max_age_days: int = MODEL_TRAINED_MAX_LAG_DAYS,
    generated_max_age_days: int = MODEL_GENERATED_MAX_AGE_DAYS,
    now: Any = None,
    require_bundle_integrity: bool = False,
) -> Optional[Dict[str, Any]]:
    model, _ = load_rotation_model_with_status(
        path,
        max_age_days=max_age_days,
        generated_max_age_days=generated_max_age_days,
        now=now,
        require_bundle_integrity=require_bundle_integrity,
    )
    return model


def save_rotation_state(value: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


__all__ = [
    "ROTATION_ECONOMIC_LOGIC",
    "ROTATION_CAPACITY_REFERENCE_CAPITAL",
    "ROTATION_ACCEPTANCE_POLICY_VERSION",
    "ROTATION_EXPOSURE_AUTHORITY",
    "RISK_CONTROL_EXPOSURE_AUTHORITY",
    "ROTATION_SCHEMA_VERSION",
    "ROTATION_PUBLICATION_IDENTITY_POLICY_VERSION",
    "build_cash_rotation_target",
    "ROTATION_WEIGHTS",
    "load_rotation_state",
    "load_rotation_model",
    "load_rotation_model_with_status",
    "save_rotation_state",
    "score_rotation_candidates",
    "select_rotation_targets",
    "simulate_staggered_rotation",
    "stabilize_rotation_publication",
    "update_live_rotation_state",
]
