"""Post-training live health gate for promoted adaptive factors."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .factor_evolution import (
    PRIMITIVE_FEATURES,
    apply_factor_registry,
    evaluate_expression,
    expression_features,
    factor_metrics,
)
from .signals.factors import rank_relative_strength
from .trading import DEFAULT_ETF_COST_MODEL, TradingCostModel
from .universe import industry_group


PRICE_MONITOR_FEATURES = {
    "relative_strength",
    "momentum_20",
    "momentum_60",
    "reversal_5",
    "trend_efficiency_20",
    "volatility_20",
    "downside_volatility_60",
    "volume_confirmation",
    "liquidity_log",
}


def _price_features(
    code: str,
    frame: pd.DataFrame,
    relative_strength: float,
) -> Dict[str, Any]:
    data = frame.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce").astype(float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    def log_return(days: int) -> float:
        if len(close) <= days or close.iloc[-days - 1] <= 0 or close.iloc[-1] <= 0:
            return 0.0
        return float(math.log(close.iloc[-1] / close.iloc[-days - 1]))

    momentum_20 = log_return(20)
    momentum_60 = log_return(60)
    tail20 = returns.tail(20)
    tail60 = returns.tail(60)
    downside = tail60[tail60 < 0]
    movement = float(close.diff().abs().tail(20).sum()) if len(close) >= 21 else 0.0
    efficiency = (
        float((close.iloc[-1] - close.iloc[-21]) / movement)
        if len(close) >= 21 and movement > 0 else 0.0
    )
    volume = pd.to_numeric(
        data.get("volume", pd.Series(0.0, index=data.index)), errors="coerce"
    ).fillna(0.0)
    volume5 = float(volume.tail(5).mean()) if len(volume) >= 5 else 0.0
    volume20 = float(volume.tail(20).mean()) if len(volume) >= 20 else volume5
    amount = float((close * volume).tail(20).mean()) if not close.empty else 0.0
    values = {
        "date": pd.Timestamp(data["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "code": str(code),
        "industry_group": industry_group(str(code)),
        "relative_strength": float(relative_strength) / 100.0,
        "momentum_20": momentum_20,
        "momentum_60": momentum_60,
        "reversal_5": -log_return(5),
        "trend_efficiency_20": float(np.clip(efficiency, -1.0, 1.0)),
        "volatility_20": float(tail20.std() * math.sqrt(252.0)) if len(tail20) >= 10 else 0.0,
        "downside_volatility_60": (
            float(downside.std() * math.sqrt(252.0)) if len(downside) >= 5 else 0.0
        ),
        "volume_confirmation": float(
            np.clip(math.log(max(volume5, 1.0) / max(volume20, 1.0)), -3.0, 3.0)
        ),
        "liquidity_log": math.log1p(max(amount, 0.0)),
    }
    for name in PRIMITIVE_FEATURES:
        values.setdefault(name, 0.0)
    return values


def _cost_adjusted_excess_return(
    raw: pd.DataFrame,
    benchmark: pd.DataFrame,
    signal_date: Any,
    horizon: int,
    cost_model: TradingCostModel,
    notional: float = 100_000.0,
) -> float | None:
    signal = pd.Timestamp(signal_date)
    future = raw[pd.to_datetime(raw["date"]) > signal].sort_values("date").reset_index(drop=True)
    if len(future) <= horizon:
        return None
    entry = future.iloc[0]
    exit_row = future.iloc[horizon]
    benchmark_future = benchmark[
        pd.to_datetime(benchmark["date"]) >= pd.Timestamp(entry["date"])
    ].sort_values("date")
    benchmark_entry = benchmark_future[
        pd.to_datetime(benchmark_future["date"]) == pd.Timestamp(entry["date"])
    ]
    benchmark_exit = benchmark_future[
        pd.to_datetime(benchmark_future["date"]) == pd.Timestamp(exit_row["date"])
    ]
    if benchmark_entry.empty or benchmark_exit.empty:
        return None
    open_price = float(entry["open"])
    if open_price <= 0:
        return None
    shares = max(notional / open_price, 1.0)
    volume = pd.to_numeric(
        future.get("volume", pd.Series(0.0, index=future.index)), errors="coerce"
    ).fillna(0.0)
    average_amount = float(
        (pd.to_numeric(future["close"], errors="coerce") * volume).head(horizon + 1).mean()
    )
    buy = cost_model.estimate("BUY", open_price, shares, average_daily_amount=average_amount)
    sell = cost_model.estimate(
        "SELL", float(exit_row["close"]), shares, average_daily_amount=average_amount
    )
    if buy["effective_price"] <= 0:
        return None
    asset_return = sell["effective_price"] / buy["effective_price"] - 1.0
    benchmark_return = (
        float(benchmark_exit.iloc[0]["close"]) / float(benchmark_entry.iloc[0]["open"]) - 1.0
    )
    return float(asset_return - benchmark_return)


def assess_live_factor_health(
    registry: Mapping[str, Any],
    factor_results: Mapping[str, Mapping[str, Any]],
    ensemble_metrics: Mapping[str, Any],
    signal_date_count: int,
    evaluated_through: str,
    unsupported_features: Sequence[str] = (),
    minimum_observations: int = 8,
) -> Dict[str, Any]:
    factors = [item for item in registry.get("factors", []) if item.get("status") == "ACTIVE"]
    coefficients = [max(float(value), 0.0) for value in (registry.get("ensemble") or {}).get("coefficients", [])]
    total = sum(coefficients)
    normalised = [value / total if total > 0 else 0.0 for value in coefficients]
    weights = {
        str(item.get("name")): round(normalised[index], 8)
        for index, item in enumerate(factors)
        if index < len(normalised)
    }
    effective = [name for name, weight in weights.items() if weight >= 0.05]
    reasons = []
    if not registry.get("approved"):
        reasons.append("REGISTRY_NOT_APPROVED")
    if unsupported_features:
        reasons.append("UNSUPPORTED_LIVE_MONITOR_FEATURES")
    if len(factors) < 2:
        reasons.append("ACTIVE_FACTOR_COUNT_BELOW_2")
    if len(effective) < 2:
        reasons.append("EFFECTIVE_FACTOR_COUNT_BELOW_2")
    observations = int(ensemble_metrics.get("ic_observations", 0) or 0)
    enough_history = signal_date_count >= minimum_observations and observations >= minimum_observations
    if enough_history:
        if (
            float(ensemble_metrics.get("ic_mean", 0.0) or 0.0) <= -0.02
            and float(ensemble_metrics.get("ic_ir", 0.0) or 0.0) <= -0.25
        ):
            reasons.append("LIVE_ENSEMBLE_NEGATIVE_IC")
        if float(ensemble_metrics.get("turnover", 0.0) or 0.0) > 0.90:
            reasons.append("LIVE_ENSEMBLE_EXCESSIVE_TURNOVER")
        for name in effective:
            metrics = factor_results.get(name, {}) or {}
            if (
                float(metrics.get("ic_mean", 0.0) or 0.0) <= -0.03
                and float(metrics.get("ic_ir", 0.0) or 0.0) <= -0.25
            ):
                reasons.append(f"LIVE_FACTOR_NEGATIVE_IC:{name}")
    hard_failure = bool(reasons)
    if hard_failure:
        status = "SUSPENDED"
    elif not enough_history:
        status = "WARMUP"
    elif (
        float(ensemble_metrics.get("ic_mean", 0.0) or 0.0) >= 0.01
        and float(ensemble_metrics.get("ic_ir", 0.0) or 0.0) >= 0.15
    ):
        status = "ACTIVE"
    else:
        status = "WATCH"
    factor_decisions = {}
    for item in factors:
        name = str(item.get("name"))
        metrics = factor_results.get(name, {}) or {}
        if not enough_history:
            decision = "WARMUP"
        elif any(reason == f"LIVE_FACTOR_NEGATIVE_IC:{name}" for reason in reasons):
            decision = "SUSPENDED"
        elif (
            float(metrics.get("ic_mean", 0.0) or 0.0) >= 0.01
            and float(metrics.get("ic_ir", 0.0) or 0.0) >= 0.15
        ):
            decision = "ACTIVE"
        else:
            decision = "WATCH"
        factor_decisions[name] = decision
    return {
        "schema_version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "registry_trained_until": str(registry.get("trained_until", "")),
        "evaluated_through": str(evaluated_through),
        "signal_date_count": int(signal_date_count),
        "minimum_observations": int(minimum_observations),
        "evidence_mature": bool(enough_history),
        "status": status,
        "approved_for_live_use": bool(not hard_failure),
        "reasons": list(dict.fromkeys(reasons)),
        "active_factor_count": len(factors),
        "effective_factor_count": len(effective),
        "effective_factor_weights": weights,
        "effective_factors": effective,
        "unsupported_features": sorted(set(str(value) for value in unsupported_features)),
        "factor_metrics": {str(key): dict(value) for key, value in factor_results.items()},
        "factor_decisions": factor_decisions,
        "ensemble_metrics": dict(ensemble_metrics),
        "method": "post_training_non_overlapping_10d_cross_sectional_ic_with_t_plus_1_real_cost_labels",
    }


def build_live_factor_health(
    qfq_frames: Mapping[str, pd.DataFrame],
    raw_frames: Mapping[str, pd.DataFrame],
    registry: Mapping[str, Any],
    benchmark_code: str,
    horizon: int = 10,
    maximum_signal_dates: int = 26,
    minimum_observations: int = 8,
    signal_step: int | None = None,
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
) -> Dict[str, Any]:
    factors = [item for item in registry.get("factors", []) if item.get("status") == "ACTIVE"]
    used_features = sorted(
        {
            feature
            for item in factors
            for feature in expression_features(item.get("expression") or {})
        }
    )
    unsupported = sorted(set(used_features) - PRICE_MONITOR_FEATURES)
    benchmark_qfq = qfq_frames.get(str(benchmark_code))
    benchmark_raw = raw_frames.get(str(benchmark_code))
    if benchmark_qfq is None or benchmark_raw is None or unsupported:
        return assess_live_factor_health(
            registry, {}, {}, 0, "", unsupported or ["BENCHMARK_FRAME_MISSING"], minimum_observations
        )
    benchmark_dates = pd.DatetimeIndex(pd.to_datetime(benchmark_qfq["date"])).sort_values()
    trained_until = pd.to_datetime(registry.get("trained_until"), errors="coerce")
    eligible = []
    for index, value in enumerate(benchmark_dates):
        if pd.isna(trained_until) or pd.Timestamp(value) <= pd.Timestamp(trained_until):
            continue
        if len(benchmark_dates) - index - 1 >= horizon + 1:
            eligible.append(pd.Timestamp(value))
    step = max(1, int(signal_step or horizon))
    non_overlapping_dates = eligible[::step]
    signal_dates = non_overlapping_dates[-max(1, int(maximum_signal_dates)):]
    rows = []
    asset_codes = sorted(set(qfq_frames).intersection(raw_frames) - {str(benchmark_code)})
    for signal_date in signal_dates:
        current = {
            code: frame[pd.to_datetime(frame["date"]) <= signal_date].copy()
            for code, frame in qfq_frames.items()
            if code in asset_codes
        }
        benchmark_slice = benchmark_qfq[pd.to_datetime(benchmark_qfq["date"]) <= signal_date].copy()
        relative = rank_relative_strength(current, benchmark_slice)
        for code in asset_codes:
            frame = current.get(code)
            if frame is None or len(frame) < 121:
                continue
            target = _cost_adjusted_excess_return(
                raw_frames[code], benchmark_raw, signal_date, horizon, cost_model
            )
            if target is None:
                continue
            row = _price_features(code, frame, float((relative.get(code) or {}).get("score", 0.0)))
            row["excess_return_10d"] = target
            rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return assess_live_factor_health(
            registry, {}, {}, len(signal_dates), "", unsupported, minimum_observations
        )
    factor_results = {
        str(item.get("name")): factor_metrics(
            panel,
            evaluate_expression(item.get("expression") or {}, panel),
        )
        for item in factors
    }
    scored = apply_factor_registry(panel, registry)
    ensemble_metrics = factor_metrics(panel, scored["adaptive_raw"])
    return assess_live_factor_health(
        registry,
        factor_results,
        ensemble_metrics,
        len(signal_dates),
        max(pd.to_datetime(panel["date"])).strftime("%Y-%m-%d"),
        unsupported,
        minimum_observations,
    )


__all__ = [
    "PRICE_MONITOR_FEATURES",
    "assess_live_factor_health",
    "build_live_factor_health",
]
