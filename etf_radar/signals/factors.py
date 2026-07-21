"""Scale-free V4 trend, setup, relative-strength, and market factors."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _prepared(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return (
        out.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def wilder_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR, retaining the original index."""
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    previous_close = frame["close"].astype(float).shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def normalised_atr_percentile(frame: pd.DataFrame, lookback: int = 120) -> float:
    if frame.empty or len(frame) < 30:
        return 50.0
    atr = wilder_atr(frame)
    close = frame["close"].astype(float).replace(0, np.nan)
    natr = (atr / close).dropna().tail(lookback)
    if len(natr) < 20:
        return 50.0
    current = float(natr.iloc[-1])
    return float((natr < current).sum() / len(natr) * 100.0)


def monthly_trend_factor(monthly: pd.DataFrame) -> Dict[str, Any]:
    frame = _prepared(monthly)
    if len(frame) < 13:
        return {
            "score": 0.0,
            "history_ok": False,
            "distance": 0.0,
            "slope": 0.0,
        }
    close = frame["close"].astype(float)
    ema10 = close.ewm(span=10, adjust=False).mean()
    current = float(close.iloc[-1])
    current_ema = float(ema10.iloc[-1])
    previous_ema = float(ema10.iloc[-4])
    if min(current, current_ema, previous_ema) <= 0:
        return {"score": 0.0, "history_ok": False, "distance": 0.0, "slope": 0.0}
    distance = math.tanh(math.log(current / current_ema) / 0.08)
    slope = math.tanh(math.log(current_ema / previous_ema) / 0.06)
    score = _clip(0.55 * distance + 0.45 * slope, -1.0, 1.0)
    return {
        "score": round(score, 6),
        "history_ok": True,
        "distance": round(distance, 6),
        "slope": round(slope, 6),
    }


def weekly_trend_factor(weekly: pd.DataFrame) -> Dict[str, Any]:
    frame = _prepared(weekly)
    if len(frame) < 21:
        return {
            "score": 0.0,
            "history_ok": False,
            "distance": 0.0,
            "slope": 0.0,
            "efficiency": 0.0,
        }
    close = frame["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    atr14 = wilder_atr(frame)
    current = float(close.iloc[-1])
    atr = _number(atr14.iloc[-1])
    current_ema = _number(ema20.iloc[-1])
    previous_ema = _number(ema20.iloc[-5])
    if min(current, atr, current_ema, previous_ema) <= 0:
        return {
            "score": 0.0,
            "history_ok": False,
            "distance": 0.0,
            "slope": 0.0,
            "efficiency": 0.0,
        }
    distance = math.tanh(((current - current_ema) / atr) / 2.0)
    natr = max(atr / current, 0.005)
    slope = math.tanh(math.log(current_ema / previous_ema) / (4.0 * natr))
    changes = close.diff().abs().tail(13).sum()
    direction = current - float(close.iloc[-14])
    efficiency = 0.0 if changes <= 0 else _clip(direction / changes, -1.0, 1.0)
    score = _clip(0.40 * distance + 0.35 * slope + 0.25 * efficiency, -1.0, 1.0)
    return {
        "score": round(score, 6),
        "history_ok": True,
        "distance": round(distance, 6),
        "slope": round(slope, 6),
        "efficiency": round(efficiency, 6),
    }


def daily_setup_factor(daily: pd.DataFrame, weekly_score: float) -> Dict[str, Any]:
    frame = _prepared(daily)
    if len(frame) < 25:
        return {"setup": "NONE", "score": 0.0, "breakout_score": 0.0, "pullback_score": 0.0}
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame.get("volume", pd.Series(0.0, index=frame.index)).astype(float)
    atr14 = wilder_atr(frame)
    atr5 = wilder_atr(frame, 5)
    atr20 = wilder_atr(frame, 20)
    ema20 = close.ewm(span=20, adjust=False).mean()

    current = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    current_high = float(high.iloc[-1])
    current_low = float(low.iloc[-1])
    current_open = float(open_.iloc[-1])
    atr = _number(atr14.iloc[-1])
    if current <= 0 or atr <= 0:
        return {"setup": "NONE", "score": 0.0, "breakout_score": 0.0, "pullback_score": 0.0}

    prior_high20 = float(high.iloc[-21:-1].max())
    price_range = max(current_high - current_low, current * 1e-6)
    clv = _clip((current - current_low) / price_range, 0.0, 1.0)
    vma20 = float(volume.iloc[-21:-1].mean())
    volume_ratio = float(volume.iloc[-1] / vma20) if vma20 > 0 else 1.0
    breakout_atr = (current - prior_high20) / atr

    breakout_score = 0.0
    breakout_eligible = (
        weekly_score >= 0.25
        and current >= prior_high20
        and 0.0 <= breakout_atr <= 1.0
        and clv >= 0.65
        and volume_ratio >= 1.20
    )
    if breakout_eligible:
        position_quality = _clip(1.0 - abs(breakout_atr - 0.25) / 0.75, 0.0, 1.0)
        volume_quality = _clip(math.log(max(volume_ratio, 1.0)) / math.log(2.0), 0.0, 1.0)
        close_quality = _clip((clv - 0.5) / 0.5, 0.0, 1.0)
        contraction = _clip(1.0 - _number(atr5.iloc[-1]) / max(_number(atr20.iloc[-1]), 1e-9), 0.0, 1.0)
        breakout_score = 100.0 * (
            0.30 * position_quality
            + 0.25 * volume_quality
            + 0.25 * close_quality
            + 0.20 * contraction
        )

    ema = _number(ema20.iloc[-1])
    ema_distance = (current - ema) / atr if ema > 0 else 99.0
    pullback_score = 0.0
    pullback_eligible = (
        weekly_score >= 0.25
        and -0.75 <= ema_distance <= 1.25
        and current > previous_close
        and current >= current_open
        and clv >= 0.55
        and volume_ratio <= 1.20
    )
    if pullback_eligible:
        proximity = _clip(1.0 - abs(ema_distance) / 1.5, 0.0, 1.0)
        close_quality = _clip((clv - 0.5) / 0.5, 0.0, 1.0)
        dry_up = _clip((1.20 - volume_ratio) / 0.70, 0.0, 1.0)
        repair = _clip((current - previous_close) / atr, 0.0, 1.0)
        pullback_score = 100.0 * (
            0.35 * proximity
            + 0.25 * close_quality
            + 0.20 * dry_up
            + 0.20 * repair
        )

    if breakout_score >= pullback_score and breakout_score > 0:
        setup, score = "BREAKOUT", breakout_score
    elif pullback_score > 0:
        setup, score = "PULLBACK", pullback_score
    else:
        setup, score = "NONE", 0.0
    return {
        "setup": setup,
        "score": round(_clip(score, 0.0, 100.0), 2),
        "breakout_score": round(_clip(breakout_score, 0.0, 100.0), 2),
        "pullback_score": round(_clip(pullback_score, 0.0, 100.0), 2),
        "clv": round(clv, 4),
        "volume_ratio": round(volume_ratio, 4),
        "breakout_atr": round(breakout_atr, 4),
        "ema20_atr": round(ema_distance, 4),
    }


def structural_risk(
    daily: pd.DataFrame,
    setup: str,
    executable_price: float,
    adjustment_factor: float,
) -> Dict[str, Any]:
    frame = _prepared(daily)
    if len(frame) < 21 or executable_price <= 0 or adjustment_factor <= 0:
        return {
            "stop_loss": 0.0,
            "stop_dist_pct": 0.0,
            "atr_multiple": 0.0,
            "quality": 0.0,
            "executable": False,
        }
    close_qfq = float(frame["close"].iloc[-1])
    atr = _number(wilder_atr(frame).iloc[-1])
    if close_qfq <= 0 or atr <= 0:
        return {
            "stop_loss": 0.0,
            "stop_dist_pct": 0.0,
            "atr_multiple": 0.0,
            "quality": 0.0,
            "executable": False,
        }
    swing_low = float(frame["low"].astype(float).iloc[-11:-1].min())
    if setup == "BREAKOUT":
        prior_high20 = float(frame["high"].astype(float).iloc[-21:-1].max())
        raw_stop_qfq = max(prior_high20 - atr, swing_low - 0.25 * atr)
    else:
        raw_stop_qfq = swing_low - 0.25 * atr
    stop_qfq = min(raw_stop_qfq, close_qfq - 1.20 * atr)
    stop_raw = stop_qfq / adjustment_factor
    stop_dist = (executable_price - stop_raw) / executable_price * 100.0
    atr_raw = atr / adjustment_factor
    atr_multiple = (executable_price - stop_raw) / atr_raw if atr_raw > 0 else 0.0
    percent_quality = _clip(1.0 - abs(stop_dist - 5.5) / 4.5, 0.0, 1.0)
    atr_quality = _clip(1.0 - abs(atr_multiple - 2.2) / 1.8, 0.0, 1.0)
    quality = 100.0 * (0.55 * atr_quality + 0.45 * percent_quality)
    executable = bool(stop_raw > 0 and stop_raw < executable_price and 2.0 <= stop_dist <= 10.0)
    if not executable:
        quality = 0.0
    return {
        "stop_loss": round(stop_raw, 8),
        "stop_dist_pct": round(stop_dist, 4),
        "atr_multiple": round(atr_multiple, 4),
        "quality": round(quality, 2),
        "executable": executable,
    }


def compute_asset_factors(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    executable_price: float,
    adjustment_factor: float,
) -> Dict[str, Any]:
    monthly_factor = monthly_trend_factor(monthly)
    weekly_factor = weekly_trend_factor(weekly)
    setup = daily_setup_factor(daily, _number(weekly_factor.get("score")))
    risk = structural_risk(
        daily,
        str(setup.get("setup", "NONE")),
        executable_price=executable_price,
        adjustment_factor=adjustment_factor,
    )
    return {
        "monthly": monthly_factor,
        "weekly": weekly_factor,
        "setup": setup,
        "risk": risk,
    }


def _aligned_returns(asset: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    left = _prepared(asset)[["date", "close"]].rename(columns={"close": "asset"})
    right = _prepared(benchmark)[["date", "close"]].rename(columns={"close": "benchmark"})
    aligned = left.merge(right, on="date", how="inner").sort_values("date")
    aligned["asset_return"] = aligned["asset"].astype(float).pct_change()
    aligned["benchmark_return"] = aligned["benchmark"].astype(float).pct_change()
    return aligned.dropna().reset_index(drop=True)


def relative_strength_raw(asset: pd.DataFrame, benchmark: pd.DataFrame) -> Dict[str, Any]:
    aligned = _aligned_returns(asset, benchmark)
    if len(aligned) < 21:
        return {"raw": {}, "beta": 1.0, "has_120": False}
    tail = aligned.tail(120)
    benchmark_variance = float(np.var(tail["benchmark_return"].to_numpy(dtype=float)))
    beta = 1.0
    if len(tail) >= 20 and benchmark_variance > 1e-10:
        beta = float(
            np.cov(
                tail["asset_return"].to_numpy(dtype=float),
                tail["benchmark_return"].to_numpy(dtype=float),
            )[0, 1]
            / benchmark_variance
        )
    beta = _clip(beta, -0.5, 2.5)
    residual = aligned["asset_return"] - beta * aligned["benchmark_return"]
    raw: Dict[int, float] = {}
    for horizon in (20, 60, 120):
        if len(aligned) <= horizon:
            continue
        asset_return = math.log(float(aligned["asset"].iloc[-1]) / float(aligned["asset"].iloc[-horizon - 1]))
        benchmark_return = math.log(
            float(aligned["benchmark"].iloc[-1]) / float(aligned["benchmark"].iloc[-horizon - 1])
        )
        alpha = asset_return - beta * benchmark_return
        tracking_error = float(residual.tail(max(60, horizon)).std()) * math.sqrt(horizon)
        raw[horizon] = alpha / max(tracking_error, 0.01)
    return {"raw": raw, "beta": round(beta, 4), "has_120": 120 in raw}


def rank_relative_strength(
    frames: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    profiles = {code: relative_strength_raw(frame, benchmark) for code, frame in frames.items()}
    percentile_by_horizon: Dict[int, Dict[str, float]] = {}
    for horizon in (20, 60, 120):
        values = {
            code: _number(profile.get("raw", {}).get(horizon), np.nan)
            for code, profile in profiles.items()
            if horizon in profile.get("raw", {})
        }
        if not values:
            percentile_by_horizon[horizon] = {}
            continue
        series = pd.Series(values, dtype=float)
        low, high = series.quantile(0.05), series.quantile(0.95)
        winsorised = series.clip(lower=low, upper=high)
        percentile_by_horizon[horizon] = (winsorised.rank(pct=True) * 100.0).to_dict()

    weights = {20: 0.50, 60: 0.30, 120: 0.20}
    output: Dict[str, Dict[str, Any]] = {}
    for code, profile in profiles.items():
        available = {
            horizon: percentile_by_horizon[horizon][code]
            for horizon in weights
            if code in percentile_by_horizon[horizon]
        }
        weight_sum = sum(weights[horizon] for horizon in available)
        score = (
            sum(available[horizon] * weights[horizon] for horizon in available) / weight_sum
            if weight_sum > 0 else 0.0
        )
        output[code] = {
            "score": round(_clip(score, 0.0, 100.0), 2),
            "percentiles": {str(k): round(v, 2) for k, v in available.items()},
            "beta": profile.get("beta", 1.0),
            "has_120": bool(profile.get("has_120", False)),
            "scope": "fixed_pool",
        }
    return output


def final_priority(factors: Mapping[str, Any], relative_strength: Mapping[str, Any]) -> float:
    monthly = (_number(factors.get("monthly", {}).get("score")) + 1.0) * 50.0
    weekly = (_number(factors.get("weekly", {}).get("score")) + 1.0) * 50.0
    setup = _number(factors.get("setup", {}).get("score"))
    risk = _number(factors.get("risk", {}).get("quality"))
    rps = _number(relative_strength.get("score"))
    return round(_clip(0.30 * setup + 0.30 * rps + 0.25 * weekly + 0.10 * risk + 0.05 * monthly, 0.0, 100.0), 2)



def point_in_time_benchmark_momentum(frame: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Close-to-close benchmark thrust ending on the latest bar (signal-date safe)."""
    empty = {"ret_1": 0.0, "ret_3": 0.0, "ret_5": 0.0, "ret_10": 0.0}
    if frame is None:
        return dict(empty)
    prepared = _prepared(frame)
    if prepared.empty or "close" not in prepared.columns:
        return dict(empty)
    close = prepared["close"].astype(float)
    output = dict(empty)
    for horizon in (1, 3, 5, 10):
        if len(close) <= horizon:
            continue
        previous = float(close.iloc[-horizon - 1])
        current = float(close.iloc[-1])
        if previous <= 0 or current <= 0 or math.isnan(previous) or math.isnan(current):
            continue
        output[f"ret_{horizon}"] = current / previous - 1.0
    return output


def classify_policy_pulse(
    momentum: Mapping[str, float],
    benchmark_natr_percentile: float = 50.0,
    raw_score: float = 0.0,
) -> Dict[str, Any]:
    """Classify broad-market policy pulses without using any future bars.

    Levels:
    * FULL / MILD: confirmed close-to-close thrust while stressed or under-invested
    * EARLY: modest thrust while still under-invested (captures first policy weeks
      such as late-2024 before the model score has turned)
    Ordinary bull weeks without underinvestment stay on the normal ladder.
    All inputs are signal-date close-to-close; no future bars.
    """
    ret_1 = _number(momentum.get("ret_1"))
    ret_3 = _number(momentum.get("ret_3"))
    ret_5 = _number(momentum.get("ret_5"))
    ret_10 = _number(momentum.get("ret_10"))
    stress = _number(benchmark_natr_percentile)
    score = _number(raw_score)
    strong = ret_5 >= 0.06 or ret_3 >= 0.04 or (ret_1 >= 0.03 and ret_3 >= 0.025)
    mild = ret_5 >= 0.035 or ret_3 >= 0.025 or (ret_1 >= 0.02 and ret_5 >= 0.02)
    early = (
        ret_3 >= 0.012
        or ret_5 >= 0.015
        or (ret_1 >= 0.008 and ret_3 >= 0.008)
    )
    stress_confirmed = stress >= 90.0
    underinvested = score < 0.10
    deep_cash = score < -0.45
    # FULL: confirmed thrust. Do not invent a bullish score; keep a high floor only
    # when the raw score is already non-negative (true breakout weeks like 2024-10).
    if strong and (stress_confirmed or (underinvested and not deep_cash)):
        level = "FULL"
        exposure_floor = 1.0 if score >= 0.0 else 0.85
        score_floor = 0.0
    elif mild and stress_confirmed and not deep_cash:
        # Mild only under crisis stress; deep cash stays on the normal ladder.
        level = "MILD"
        exposure_floor = 0.70
        score_floor = -0.10
    elif (
        early
        and not mild
        and score < 0.0
        and score >= -0.55
        and ret_10 > -0.05
    ):
        # First-week policy coverage (e.g. 2024-09-24).
        # Deep-cash EARLY needs a clean 3-day thrust (not a single 5d bounce).
        # Low-stress EARLY needs a positive day print so quiet bear bounces stay off.
        if deep_cash and ret_3 < 0.012:
            level = "NONE"
            exposure_floor = 0.0
            score_floor = -1.0
        elif (not deep_cash) and stress < 55.0 and ret_1 < 0.008:
            level = "NONE"
            exposure_floor = 0.0
            score_floor = -1.0
        else:
            level = "EARLY"
            exposure_floor = 0.85 if score >= -0.30 else 0.70
            score_floor = 0.0
    else:
        level = "NONE"
        exposure_floor = 0.0
        score_floor = -1.0
    return {
        "level": level,
        "exposure_floor": float(exposure_floor),
        "score_floor": float(score_floor),
        "ret_1": round(ret_1, 6),
        "ret_3": round(ret_3, 6),
        "ret_5": round(ret_5, 6),
        "ret_10": round(ret_10, 6),
    }


def decide_market_exposure(
    score: float,
    benchmark_natr_percentile: float,
    pulse_level: str = "NONE",
) -> Tuple[str, float, str]:
    """Map market score and stress into permission, exposure, and regime state.

    Ladder design goals:
    * keep full cash only for genuine crisis / extreme stress
    * avoid turning mildly negative rebounds into multi-month cash traps
    * restore risk gradually so bull catch-up does not require a perfect 0.25 score
    * allow confirmed point-in-time policy pulses to keep tradable coverage even when
      stress percentiles are extreme, because stress alone is lagging in thrust weeks
    """
    score = _clip(score, -1.0, 1.0)
    stress = _number(benchmark_natr_percentile)
    pulse = str(pulse_level or "NONE").upper()

    if pulse == "FULL":
        # Confirmed thrust: do not cash out solely because NATR is at extremes.
        # Keep a cautious band for near-zero / negative scores so score floors do not
        # auto-promote to 100% risk (2018 false FULL weeks).
        if score < -0.50:
            return "MAINLINE_ONLY", 0.70, "PULSE_HARD"
        if score < 0.05:
            return "MAINLINE_ONLY", 0.90, "PULSE_CAUTIOUS"
        return "TRADEABLE", 1.0, "PULSE_FULL"
    if pulse in {"MILD", "EARLY"}:
        if score < -0.50:
            return "MAINLINE_ONLY", 0.50, "PULSE_HARD"
        if score < 0.0:
            return "MAINLINE_ONLY", 0.85 if pulse == "EARLY" else 0.70, "PULSE_EARLY" if pulse == "EARLY" else "PULSE_CAUTIOUS"
        if score < 0.20:
            return "MAINLINE_ONLY", 0.90, "PULSE_EARLY" if pulse == "EARLY" else "PULSE_CAUTIOUS"
        return "TRADEABLE", 1.0, "PULSE_FULL"

    # Crisis cash when the score itself is deeply negative.
    # Extreme stress without a pulse still caps risk: a modestly positive score in a
    # 98th-percentile NATR week (e.g. 2018-02-09) is not a free full-risk day.
    if score < -0.50:
        return "BLOCKED", 0.0, "RISK_OFF"
    if stress >= 98.0 and score < 0.0:
        return "BLOCKED", 0.0, "RISK_OFF"
    if stress >= 98.0:
        if score >= 0.35:
            return "MAINLINE_ONLY", 0.70, "DEFENSIVE"
        if score >= 0.0:
            return "MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"
        return "MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"
    if stress >= 96.0:
        if score >= 0.25:
            return "MAINLINE_ONLY", 0.90, "CAUTIOUS"
        if score >= 0.0:
            return "MAINLINE_ONLY", 0.70, "DEFENSIVE"
        if score < -0.25:
            return "MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"
        return "MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"
    if score < -0.25:
        return "MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"
    if score < 0.0:
        return "MAINLINE_ONLY", 0.70, "DEFENSIVE"
    if score < 0.20:
        return "MAINLINE_ONLY", 0.90, "CAUTIOUS"
    return "TRADEABLE", 1.0, "NORMAL"


def market_policy(
    results: Iterable[Mapping[str, Any]],
    benchmark_weekly_score: float,
    benchmark_natr_percentile: float,
    benchmark_frame: Optional[pd.DataFrame] = None,
    benchmark_momentum: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    weekly_scores = [
        _number(row.get("v4_factors", {}).get("weekly", {}).get("score"))
        for row in results
        if row.get("v4_factors")
    ]
    total = len(weekly_scores)
    bull = sum(score >= 0.25 for score in weekly_scores)
    bear = sum(score <= -0.25 for score in weekly_scores)
    breadth_balance = (bull - bear) / total if total else 0.0
    volatility_penalty = -_clip((benchmark_natr_percentile - 50.0) / 50.0, 0.0, 1.0)
    raw_score = _clip(
        0.55 * benchmark_weekly_score + 0.35 * breadth_balance + 0.10 * volatility_penalty,
        -1.0,
        1.0,
    )
    if benchmark_momentum is None:
        momentum = point_in_time_benchmark_momentum(benchmark_frame)
    else:
        momentum = {
            "ret_1": _number(benchmark_momentum.get("ret_1")),
            "ret_3": _number(benchmark_momentum.get("ret_3")),
            "ret_5": _number(benchmark_momentum.get("ret_5")),
            "ret_10": _number(benchmark_momentum.get("ret_10")),
        }
    pulse = classify_policy_pulse(
        momentum,
        benchmark_natr_percentile=benchmark_natr_percentile,
        raw_score=raw_score,
    )
    score = max(raw_score, float(pulse["score_floor"])) if pulse["level"] != "NONE" else raw_score
    score = _clip(score, -1.0, 1.0)
    permission, max_exposure, state = decide_market_exposure(
        score,
        benchmark_natr_percentile,
        pulse_level=str(pulse["level"]),
    )
    if pulse["level"] != "NONE":
        max_exposure = max(float(max_exposure), float(pulse["exposure_floor"]))
        max_exposure = float(_clip(max_exposure, 0.0, 1.0))
        if max_exposure >= 0.999:
            permission = "TRADEABLE"
        elif permission == "BLOCKED" and max_exposure > 0.0:
            permission = "MAINLINE_ONLY"
    return {
        "state": state,
        "score": round(score, 4),
        "raw_score": round(raw_score, 4),
        "entry_permission": permission,
        "max_exposure_ratio": max_exposure,
        "benchmark_weekly_score": round(benchmark_weekly_score, 4),
        "benchmark_natr_percentile": round(benchmark_natr_percentile, 2),
        "breadth_balance": round(breadth_balance, 4),
        "bull_count": bull,
        "bear_count": bear,
        "total_count": total,
        "policy_pulse": str(pulse["level"]),
        "policy_pulse_ret_1": pulse["ret_1"],
        "policy_pulse_ret_3": pulse["ret_3"],
        "policy_pulse_ret_5": pulse["ret_5"],
        "policy_pulse_ret_10": pulse["ret_10"],
        "policy_pulse_exposure_floor": float(pulse["exposure_floor"]),
    }
