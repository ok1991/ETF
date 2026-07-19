"""Adaptive factor monitoring, symbolic genetic programming, and ML ensembling.

The module is deliberately self-contained and depends only on NumPy/Pandas so the
production job does not need a heavyweight AutoML runtime.  Every promoted factor
is stored with its expression, economic rationale, in/out-of-sample diagnostics,
and replacement history in ``adaptive_factor_registry.json``.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model_governance import (
    MODEL_GENERATED_MAX_AGE_DAYS,
    MODEL_TRAINED_MAX_LAG_DAYS,
    validate_artifact_time,
    validate_bundle_member,
)
from .universe import industry_group


PRIMITIVE_FEATURES: Tuple[str, ...] = (
    "monthly_trend",
    "weekly_trend",
    "setup_score",
    "relative_strength",
    "risk_quality",
    "market_score",
    "momentum_20",
    "momentum_60",
    "reversal_5",
    "trend_efficiency_20",
    "volatility_20",
    "downside_volatility_60",
    "volume_confirmation",
    "liquidity_log",
)
FACTOR_EVOLUTION_POLICY_VERSION = "complementary-stability-fdr-seasoning-v7"
POLICY_SEASONING_MIN_DATES = 13
DISCOVERY_FDR_MAX = 0.10


class PurgedHoldoutInsufficientError(ValueError):
    """Raised when strict purged selection and approval holdouts cannot be formed."""

FEATURE_LOGIC: Dict[str, str] = {
    "monthly_trend": "月线趋势刻画中长期资金方向",
    "weekly_trend": "周线趋势刻画行业景气的中期持续性",
    "setup_score": "形态质量衡量突破或回踩是否可执行",
    "relative_strength": "相对沪深300强度反映超额需求",
    "risk_quality": "止损结构质量约束单位风险回报",
    "market_score": "市场体制控制系统性风险暴露",
    "momentum_20": "20日动量捕捉短中期资金延续",
    "momentum_60": "60日动量代理行业景气趋势",
    "reversal_5": "5日反转捕捉趋势中的短期过度交易",
    "trend_efficiency_20": "趋势效率过滤高噪声的虚假上涨",
    "volatility_20": "20日波动率衡量短期风险拥挤",
    "downside_volatility_60": "下行波动率聚焦投资者真正承担的风险",
    "volume_confirmation": "量能确认用于区分真实资金流与无量波动",
    "liquidity_log": "流动性降低冲击成本和不可成交风险",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def build_primitive_row(
    code: str,
    name: str,
    daily: pd.DataFrame,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build point-in-time primitive features from an existing analyzer result."""
    frame = daily.copy().sort_values("date").reset_index(drop=True)
    close = frame.get("close", pd.Series(dtype=float)).astype(float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    def log_return(days: int) -> float:
        if len(close) <= days or float(close.iloc[-days - 1]) <= 0 or float(close.iloc[-1]) <= 0:
            return 0.0
        return float(math.log(float(close.iloc[-1]) / float(close.iloc[-days - 1])))

    momentum_20 = log_return(20)
    momentum_60 = log_return(60)
    reversal_5 = -log_return(5)
    tail20 = returns.tail(20)
    tail60 = returns.tail(60)
    volatility_20 = float(tail20.std() * math.sqrt(252.0)) if len(tail20) >= 10 else 0.0
    downside = tail60[tail60 < 0]
    downside_volatility_60 = float(downside.std() * math.sqrt(252.0)) if len(downside) >= 5 else 0.0
    if len(close) >= 21:
        movement = float(close.diff().abs().tail(20).sum())
        trend_efficiency = float((close.iloc[-1] - close.iloc[-21]) / movement) if movement > 0 else 0.0
    else:
        trend_efficiency = 0.0
    volume = frame.get("volume", pd.Series(0.0, index=frame.index)).astype(float)
    volume5 = float(volume.tail(5).mean()) if len(volume) >= 5 else 0.0
    volume20 = float(volume.tail(20).mean()) if len(volume) >= 20 else volume5
    volume_confirmation = math.log(max(volume5, 1.0) / max(volume20, 1.0))
    if "amount" in frame.columns:
        amount = frame["amount"].astype(float).tail(20).mean()
    else:
        amount = (close * volume).tail(20).mean() if not close.empty else 0.0

    factors = result.get("v4_factors", {}) or {}
    setup = factors.get("setup", {}) or {}
    relative = result.get("relative_strength", {}) or {}
    market = result.get("v4_market", {}) or {}
    return {
        "code": str(code),
        "name": str(name),
        "industry_group": industry_group(str(code), str(name)),
        "monthly_trend": _number((factors.get("monthly") or {}).get("score")),
        "weekly_trend": _number((factors.get("weekly") or {}).get("score")),
        "setup_score": _number(setup.get("score")) / 100.0,
        "relative_strength": _number(relative.get("score")) / 100.0,
        "risk_quality": _number((factors.get("risk") or {}).get("quality")) / 100.0,
        "market_score": _number(market.get("score")),
        "market_state": str(market.get("state", "UNKNOWN")),
        "entry_permission": str(market.get("entry_permission", "BLOCKED")),
        "max_exposure_ratio": float(np.clip(_number(market.get("max_exposure_ratio")), 0.0, 1.0)),
        "momentum_20": momentum_20,
        "momentum_60": momentum_60,
        "reversal_5": reversal_5,
        "trend_efficiency_20": float(np.clip(trend_efficiency, -1.0, 1.0)),
        "volatility_20": max(0.0, volatility_20),
        "downside_volatility_60": max(0.0, downside_volatility_60),
        "volume_confirmation": float(np.clip(volume_confirmation, -3.0, 3.0)),
        "liquidity_log": math.log1p(max(_number(amount), 0.0)),
        "average_daily_amount_20": max(_number(amount), 0.0),
    }


def _feature(name: str) -> Dict[str, Any]:
    return {"feature": name}


def _op(name: str, *args: Mapping[str, Any]) -> Dict[str, Any]:
    return {"op": name, "args": [dict(arg) for arg in args]}


def seeded_factor_specs() -> List[Dict[str, Any]]:
    """Economically motivated seeds supplied to every GP generation."""
    return [
        {
            "name": "quality_relative_momentum",
            "expression": _op("mul", _feature("relative_strength"), _feature("trend_efficiency_20")),
            "economic_logic": "相对强度只有在价格路径高效率时更可信，可过滤震荡型伪强势。",
            "generation": 0,
        },
        {
            "name": "low_risk_industry_momentum",
            "expression": _op(
                "div",
                _op("add", _feature("momentum_20"), _feature("momentum_60")),
                _op("add", _feature("downside_volatility_60"), _feature("volatility_20")),
            ),
            "economic_logic": "以短中期动量除以下行与总波动，偏好风险调整后更稳健的行业趋势。",
            "generation": 0,
        },
        {
            "name": "volume_confirmed_trend",
            "expression": _op(
                "mul",
                _op("add", _feature("weekly_trend"), _feature("momentum_20")),
                _op("add", _feature("volume_confirmation"), _feature("liquidity_log")),
            ),
            "economic_logic": "趋势与量能、流动性共振时更可能由真实资金推动，并具有更低冲击成本。",
            "generation": 0,
        },
        {
            "name": "pullback_with_structural_quality",
            "expression": _op(
                "add",
                _op("mul", _feature("reversal_5"), _feature("weekly_trend")),
                _op("mul", _feature("setup_score"), _feature("risk_quality")),
            ),
            "economic_logic": "在周线趋势中买入短期回撤，并要求入场形态与止损结构同时有效。",
            "generation": 0,
        },
    ]


def primitive_factor_specs() -> List[Dict[str, Any]]:
    """Keep transparent primitives as challengers to complex GP expressions."""
    return [
        {
            "name": f"primitive_{name}",
            "expression": _feature(name),
            "economic_logic": FEATURE_LOGIC[name],
            "generation": 0,
            "candidate_origin": "primitive_challenger",
        }
        for name in PRIMITIVE_FEATURES
    ]


def expression_complexity(expression: Mapping[str, Any]) -> int:
    if "feature" in expression:
        return 1
    return 1 + sum(expression_complexity(arg) for arg in expression.get("args", []))


def expression_text(expression: Mapping[str, Any]) -> str:
    if "feature" in expression:
        return str(expression["feature"])
    args = [expression_text(arg) for arg in expression.get("args", [])]
    op = str(expression.get("op", ""))
    if len(args) == 1:
        return f"{op}({args[0]})"
    return f"{op}({', '.join(args)})"


def _expression_key(expression: Mapping[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(dict(expression), sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _expression_family_key(expression: Mapping[str, Any]) -> str:
    """Treat a factor and its top-level sign reversal as one cooldown family."""
    canonical: Mapping[str, Any] = expression
    while (
        str(canonical.get("op", "")) == "neg"
        and len(canonical.get("args", [])) == 1
        and isinstance(canonical.get("args", [None])[0], Mapping)
    ):
        canonical = canonical["args"][0]
    return _expression_key(canonical)


def _expression_features(expression: Mapping[str, Any]) -> List[str]:
    if "feature" in expression:
        return [str(expression["feature"])]
    values: List[str] = []
    for arg in expression.get("args", []):
        values.extend(_expression_features(arg))
    return list(dict.fromkeys(values))


def expression_features(expression: Mapping[str, Any]) -> List[str]:
    """Return the unique primitive features referenced by an expression."""
    return _expression_features(expression)


def economic_logic(expression: Mapping[str, Any]) -> str:
    features = _expression_features(expression)
    descriptions = [FEATURE_LOGIC.get(name, name) for name in features[:3]]
    joined = "；".join(descriptions)
    return f"遗传编程组合：{joined}。表达式经行业中性化后，仅保留行业组内相对机会。"


def _standardised_feature(frame: pd.DataFrame, name: str) -> np.ndarray:
    cached_name = f"__z__{name}"
    if cached_name in frame.columns:
        return frame[cached_name].to_numpy(dtype=float)
    values = pd.to_numeric(frame.get(name, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median() if not values.dropna().empty else 0.0)
    if "date" not in frame.columns:
        std = float(values.std())
        return ((values - float(values.mean())) / max(std, 1e-8)).to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"])
    means = values.groupby(dates).transform("mean")
    scales = values.groupby(dates).transform(lambda part: part.std(ddof=0)).clip(lower=1e-8)
    return ((values - means) / scales).fillna(0.0).to_numpy(dtype=float)


def _prepare_feature_cache(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for name in PRIMITIVE_FEATURES:
        cached_name = f"__z__{name}"
        if cached_name not in output.columns:
            output[cached_name] = _standardised_feature(output, name)
    return output


def evaluate_expression(expression: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if "feature" in expression:
        return _standardised_feature(frame, str(expression["feature"]))
    op = str(expression.get("op", ""))
    args = [evaluate_expression(arg, frame) for arg in expression.get("args", [])]
    if not args:
        return np.zeros(len(frame), dtype=float)
    if op == "neg":
        value = -args[0]
    elif op == "abs":
        value = np.abs(args[0])
    elif op == "signed_sqrt":
        value = np.sign(args[0]) * np.sqrt(np.abs(args[0]))
    elif op == "add":
        value = args[0] + args[1]
    elif op == "sub":
        value = args[0] - args[1]
    elif op == "mul":
        value = args[0] * args[1]
    elif op == "div":
        denominator = np.where(np.abs(args[1]) < 0.25, np.sign(args[1]) * 0.25 + (args[1] == 0) * 0.25, args[1])
        value = args[0] / denominator
    elif op == "min":
        value = np.minimum(args[0], args[1])
    elif op == "max":
        value = np.maximum(args[0], args[1])
    else:
        value = args[0]
    return np.nan_to_num(np.clip(value, -20.0, 20.0), nan=0.0, posinf=20.0, neginf=-20.0)


def industry_neutralise(frame: pd.DataFrame, values: Sequence[float]) -> pd.Series:
    """Demean inside broad industries and z-score each signal date."""
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(frame.get("date", pd.Series("current", index=frame.index))),
            "industry_group": frame.get("industry_group", pd.Series("other", index=frame.index)).astype(str),
            "value": pd.Series(values, index=frame.index, dtype=float).replace([np.inf, -np.inf], np.nan),
        },
        index=frame.index,
    )
    work["value"] = work["value"].fillna(0.0)
    date_group = work.groupby("date")["value"]
    global_mean = date_group.transform("mean")
    grouped = work.groupby(["date", "industry_group"])["value"]
    counts = grouped.transform("count")
    group_mean = grouped.transform("mean")
    centre = group_mean.where(counts >= 2, global_mean)
    residual = work["value"] - centre
    scales = residual.groupby(work["date"]).transform(lambda part: part.std(ddof=0)).clip(lower=1e-8)
    return (residual / scales).fillna(0.0)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def _factor_turnover(frame: pd.DataFrame, values: pd.Series, top_fraction: float = 0.25) -> float:
    work = frame[["date", "code"]].copy()
    work["factor"] = values
    work["date"] = pd.to_datetime(work["date"])
    counts = work.groupby("date")["code"].transform("count")
    top_counts = np.maximum(1, np.ceil(counts.astype(float) * top_fraction)).astype(int)
    ranks = work.groupby("date")["factor"].rank(method="first", ascending=False)
    work["weight"] = np.where(ranks <= top_counts, 1.0 / top_counts, 0.0)
    weights = work.pivot_table(index="date", columns="code", values="weight", aggfunc="last", fill_value=0.0)
    if len(weights) < 2:
        return 0.0
    return float(weights.diff().abs().sum(axis=1).iloc[1:].mean() * 0.5)


def _cross_sectional_ic(frame: pd.DataFrame, factor_column: str, target_column: str) -> pd.Series:
    work = frame[["date", factor_column, target_column]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work[factor_column] = pd.to_numeric(work[factor_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return pd.Series(dtype=float)
    work["_x"] = work.groupby("date")[factor_column].rank(method="average")
    work["_y"] = work.groupby("date")[target_column].rank(method="average")
    grouped = work.groupby("date")
    n = grouped["_x"].count().astype(float)
    sum_x = grouped["_x"].sum()
    sum_y = grouped["_y"].sum()
    sum_xy = (work["_x"] * work["_y"]).groupby(work["date"]).sum()
    sum_x2 = (work["_x"] ** 2).groupby(work["date"]).sum()
    sum_y2 = (work["_y"] ** 2).groupby(work["date"]).sum()
    numerator = sum_xy - sum_x * sum_y / n
    denominator = np.sqrt((sum_x2 - sum_x**2 / n) * (sum_y2 - sum_y**2 / n))
    result = numerator / denominator.replace(0.0, np.nan)
    return result[(n >= 3) & result.notna()]


@lru_cache(maxsize=4096)
def _positive_sign_tail_probability(observations: int, positives: int) -> float:
    count = max(int(observations), 0)
    wins = min(max(int(positives), 0), count)
    if count == 0:
        return 1.0
    return sum(math.comb(count, value) for value in range(wins, count + 1)) / float(
        2**count
    )


def factor_metrics(frame: pd.DataFrame, values: Sequence[float], recent_dates: int = 26) -> Dict[str, Any]:
    """Calculate IC, IC-IR, horizon decay, and cross-sectional turnover."""
    data = frame.copy()
    data["_factor"] = industry_neutralise(data, values)
    date_values = sorted(pd.to_datetime(data["date"]).unique())
    recent_set = set(date_values[-max(1, recent_dates):])
    ic_values = _cross_sectional_ic(data, "_factor", "excess_return_10d")
    ic_series = [float(value) for value in ic_values.to_numpy(dtype=float)]
    recent_ic = [
        float(value) for date, value in ic_values.items()
        if pd.Timestamp(date) in recent_set
    ]

    def summarise(values_: Sequence[float]) -> Tuple[float, float]:
        if not values_:
            return 0.0, 0.0
        mean = float(np.mean(values_))
        std = float(np.std(values_, ddof=1)) if len(values_) > 1 else 0.0
        ir = mean / max(std, 1e-8) * math.sqrt(52.0)
        return mean, ir

    ic_mean, ic_ir = summarise(ic_series)
    recent_mean, recent_ir = summarise(recent_ic)
    nonzero_ic = [value for value in ic_series if abs(value) > 1e-12]
    positive_ic_count = sum(value > 0.0 for value in nonzero_ic)
    sign_tail = _positive_sign_tail_probability(
        len(nonzero_ic), positive_ic_count
    )
    ic_std = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else 0.0
    ic_t_stat = (
        ic_mean / max(ic_std, 1e-8) * math.sqrt(len(ic_series))
        if ic_series
        else 0.0
    )
    normal_two_sided_p = math.erfc(abs(ic_t_stat) / math.sqrt(2.0))
    ic_p_value = (
        max(sign_tail, normal_two_sided_p)
        if ic_mean > 0.0
        else 1.0
    )
    decay: Dict[str, float] = {}
    for horizon in (5, 10, 20):
        target = f"excess_return_{horizon}d"
        if target not in data.columns:
            continue
        horizon_ics = _cross_sectional_ic(data, "_factor", target)
        decay[str(horizon)] = float(horizon_ics.mean()) if not horizon_ics.empty else 0.0
    short_ic = max(abs(decay.get("5", 0.0)), abs(decay.get("10", ic_mean)), 0.01)
    decay_ratio = abs(decay.get("20", decay.get("10", ic_mean))) / short_ic
    turnover = _factor_turnover(data, data["_factor"])
    sign_flip = bool(ic_mean * recent_mean < 0 and abs(ic_mean) >= 0.01 and abs(recent_mean) >= 0.01)
    reasons: List[str] = []
    if len(ic_series) < 8:
        reasons.append("INSUFFICIENT_IC_HISTORY")
    if recent_mean < -0.01:
        reasons.append("NEGATIVE_RECENT_IC")
    elif recent_mean < 0.01:
        reasons.append("RECENT_IC_WEAK")
    if recent_ir < 0.15:
        reasons.append("RECENT_IR_WEAK")
    if decay_ratio < 0.20:
        reasons.append("FAST_DECAY")
    if turnover > 0.85:
        reasons.append("EXCESSIVE_TURNOVER")
    if sign_flip:
        reasons.append("IC_SIGN_FLIP")
    hard_failure = sign_flip or "NEGATIVE_RECENT_IC" in reasons or "EXCESSIVE_TURNOVER" in reasons
    status = "ACTIVE" if not reasons else ("WATCH" if len(reasons) <= 2 and not hard_failure else "RETIRED")
    return {
        "ic_mean": round(ic_mean, 8),
        "ic_ir": round(ic_ir, 6),
        "ic_t_stat": round(ic_t_stat, 6),
        "ic_positive_ratio": round(
            positive_ic_count / max(len(nonzero_ic), 1), 6
        ),
        "ic_p_value": round(min(max(ic_p_value, 0.0), 1.0), 10),
        "recent_ic_mean": round(recent_mean, 8),
        "recent_ic_ir": round(recent_ir, 6),
        "ic_decay": {key: round(value, 8) for key, value in decay.items()},
        "decay_ratio_20_to_5": round(decay_ratio, 6),
        "turnover": round(turnover, 6),
        "ic_observations": len(ic_series),
        "status": status,
        "reasons": reasons,
    }


def _benjamini_hochberg_adjust(p_values: Sequence[float]) -> List[float]:
    """Return monotone Benjamini-Hochberg q-values for one candidate family."""
    count = len(p_values)
    if count == 0:
        return []
    ordered = sorted(
        range(count),
        key=lambda index: min(max(float(p_values[index]), 0.0), 1.0),
    )
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = ordered[reverse_rank]
        rank = reverse_rank + 1
        p_value = min(max(float(p_values[index]), 0.0), 1.0)
        running = min(running, p_value * count / rank)
        adjusted[index] = min(max(running, 0.0), 1.0)
    return adjusted


UNARY_OPS = ("neg", "abs", "signed_sqrt")
BINARY_OPS = ("add", "sub", "mul", "div", "min", "max")


def _random_expression(rng: random.Random, depth: int = 0, max_depth: int = 3) -> Dict[str, Any]:
    if depth >= max_depth or (depth > 0 and rng.random() < 0.30):
        return _feature(rng.choice(PRIMITIVE_FEATURES))
    if rng.random() < 0.20:
        return _op(rng.choice(UNARY_OPS), _random_expression(rng, depth + 1, max_depth))
    return _op(
        rng.choice(BINARY_OPS),
        _random_expression(rng, depth + 1, max_depth),
        _random_expression(rng, depth + 1, max_depth),
    )


def _mutate(expression: Mapping[str, Any], rng: random.Random) -> Dict[str, Any]:
    value = copy.deepcopy(dict(expression))
    if rng.random() < 0.45:
        return _op(rng.choice(BINARY_OPS), value, _feature(rng.choice(PRIMITIVE_FEATURES)))
    if rng.random() < 0.70:
        return _op(rng.choice(UNARY_OPS), value)
    return _random_expression(rng, max_depth=3)


def _fitness(metrics: Mapping[str, Any], complexity: int) -> float:
    return float(
        metrics.get("ic_mean", 0.0)
        + 0.015 * metrics.get("ic_ir", 0.0)
        - 0.020 * metrics.get("turnover", 0.0)
        - 0.0008 * complexity
    )


def _oriented(expression: Mapping[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    values = evaluate_expression(expression, frame)
    metrics = factor_metrics(frame, values)
    if float(metrics.get("ic_mean", 0.0)) < 0:
        return _op("neg", expression)
    return dict(expression)


def genetic_candidates(
    train: pd.DataFrame,
    seed: int = 42,
    population_size: int = 32,
    generations: int = 5,
) -> List[Dict[str, Any]]:
    """Evolve symbolic candidates using IC/IR/turnover-aware fitness."""
    rng = random.Random(seed)
    population = [dict(item["expression"]) for item in seeded_factor_specs()]
    population.extend(_random_expression(rng) for _ in range(max(0, population_size - len(population))))
    scored: Dict[str, Tuple[float, Dict[str, Any], Dict[str, Any]]] = {}
    for generation in range(max(1, generations)):
        ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for expression in population:
            expression = _oriented(expression, train)
            key = json.dumps(expression, sort_keys=True, ensure_ascii=True)
            if key not in scored:
                metrics = factor_metrics(train, evaluate_expression(expression, train))
                scored[key] = (_fitness(metrics, expression_complexity(expression)), expression, metrics)
            ranked.append(scored[key])
        ranked.sort(key=lambda item: item[0], reverse=True)
        elites = [copy.deepcopy(item[1]) for item in ranked[: max(4, population_size // 4)]]
        population = list(elites)
        while len(population) < population_size:
            left = copy.deepcopy(rng.choice(elites))
            right = copy.deepcopy(rng.choice(elites))
            child = _op(rng.choice(BINARY_OPS), left, right) if rng.random() < 0.55 else _mutate(left, rng)
            if expression_complexity(child) <= 15:
                population.append(child)
        for expression in elites:
            key = json.dumps(expression, sort_keys=True, ensure_ascii=True)
            score, _, metrics = scored[key]
            scored[key] = (score, expression, {**metrics, "generation": generation})
    ranked_all = sorted(scored.values(), key=lambda item: item[0], reverse=True)
    return [
        {
            "name": (
                f"gp_{index + 1}_"
                f"{hashlib.sha1(expression_text(expression).encode('utf-8')).hexdigest()[:8]}"
            ),
            "expression": expression,
            "economic_logic": economic_logic(expression),
            "generation": int(metrics.get("generation", generations - 1)),
            "train_metrics": metrics,
            "train_fitness": round(score, 8),
        }
        for index, (score, expression, metrics) in enumerate(ranked_all[: max(12, population_size // 2)])
    ]


def _ridge_ensemble(
    frame: pd.DataFrame,
    factor_values: pd.DataFrame,
    regularisation: float = 2.0,
    half_life_dates: float = 52.0,
) -> Dict[str, Any]:
    """Fit a recency-weighted non-negative ridge ensemble.

    Candidate expressions are oriented to positive IC before fitting.  Constraining
    their coefficients to be non-negative prevents an unstable regression from
    silently reversing an economically approved factor in the final ensemble.
    """
    matrix = factor_values.to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardised = np.nan_to_num((matrix - mean) / scale)
    design = np.column_stack([np.ones(len(standardised)), standardised])
    target = pd.to_numeric(frame["excess_return_10d"], errors="coerce").fillna(0.0).clip(-0.15, 0.15).to_numpy()
    dates = pd.to_datetime(frame["date"])
    unique_dates = pd.Index(sorted(dates.unique()))
    date_rank = {pd.Timestamp(value): index for index, value in enumerate(unique_dates)}
    ages = np.asarray([len(unique_dates) - 1 - date_rank[pd.Timestamp(value)] for value in dates], dtype=float)
    weights = np.power(0.5, ages / max(float(half_life_dates), 1.0))
    sqrt_weights = np.sqrt(weights / max(float(np.mean(weights)), 1e-8))
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = target * sqrt_weights
    penalty = np.eye(design.shape[1]) * float(regularisation)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    coefficients[1:] = np.maximum(coefficients[1:], 0.0)
    if not bool(np.any(coefficients[1:] > 1e-10)):
        coefficients[1:] = 1.0 / max(len(coefficients) - 1, 1)
    coefficients[0] = float(np.average(target - standardised @ coefficients[1:], weights=weights))
    return {
        "intercept": float(coefficients[0]),
        "coefficients": [float(value) for value in coefficients[1:]],
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "training_half_life_dates": float(half_life_dates),
        "non_negative_coefficients": True,
    }


def _factor_matrix(frame: pd.DataFrame, specs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    values: Dict[str, pd.Series] = {}
    for spec in specs:
        raw = evaluate_expression(spec["expression"], frame)
        values[str(spec["name"])] = industry_neutralise(frame, raw)
    return pd.DataFrame(values, index=frame.index)


def _normalised_coefficient_weights(ensemble: Mapping[str, Any]) -> List[float]:
    coefficients = [max(float(value), 0.0) for value in ensemble.get("coefficients", [])]
    total = sum(coefficients)
    return [value / total if total > 0 else 0.0 for value in coefficients]


def _selection_subperiod_stability(
    frame: pd.DataFrame,
    raw: Sequence[float],
    blocks: int = 3,
) -> Dict[str, Any]:
    """Require a combination to survive multiple contiguous selection regimes."""
    work = frame.copy()
    work["_combination_raw"] = np.asarray(raw, dtype=float)
    dates = np.asarray(sorted(pd.to_datetime(work["date"]).unique()))
    parts = [part for part in np.array_split(dates, max(2, int(blocks))) if len(part)]
    metrics = [
        factor_metrics(
            work[work["date"].isin(part)],
            work.loc[work["date"].isin(part), "_combination_raw"],
        )
        for part in parts
    ]
    ic_values = [float(item.get("ic_mean", 0.0)) for item in metrics]
    positive = sum(value >= 0.0 for value in ic_values)
    return {
        "block_count": len(metrics),
        "positive_block_count": positive,
        "positive_block_ratio": round(positive / max(len(metrics), 1), 6),
        "worst_block_ic": round(min(ic_values) if ic_values else 0.0, 8),
        "blocks": metrics,
    }


def _passes_selection_subperiod_stability(stability: Mapping[str, Any]) -> bool:
    block_count = max(int(stability.get("block_count", 0)), 1)
    positive_count = int(stability.get("positive_block_count", 0))
    return bool(
        positive_count * 3 >= block_count * 2
        and float(stability.get("worst_block_ic", 0.0)) >= -0.02
    )


def _select_complementary_factor_set(
    train: pd.DataFrame,
    selection: pd.DataFrame,
    evaluated: Sequence[Mapping[str, Any]],
    max_active: int,
    pool_limit: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select a jointly useful factor set without touching the approval holdout."""
    pool = [dict(item) for item in evaluated if item.get("accepted")][:max(2, int(pool_limit))]
    if len(pool) < 2:
        return [], {
            "candidate_pool_size": len(pool),
            "evaluated_combinations": 0,
            "status": "INSUFFICIENT",
            "discovery_control": {
                "method": "benjamini_hochberg",
                "family": "all_evaluated_factor_combinations",
                "family_size": 0,
                "maximum_fdr": DISCOVERY_FDR_MAX,
                "discovery_count": 0,
            },
        }
    train_matrix = _factor_matrix(train, pool)
    selection_matrix = _factor_matrix(selection, pool)
    names = [str(item.get("name")) for item in pool]
    evaluated_combinations = 0
    combination_rejection_counts: Dict[str, int] = {}

    def reject(reason: str) -> None:
        combination_rejection_counts[reason] = (
            combination_rejection_counts.get(reason, 0) + 1
        )

    ranked: List[
        Tuple[
            float,
            Tuple[int, ...],
            Dict[str, Any],
            List[float],
            Dict[str, Any],
            Dict[str, Any],
            int,
        ]
    ] = []
    combination_p_values: List[float] = []
    max_size = min(max(2, int(max_active)), 3, len(pool))
    for size in range(2, max_size + 1):
        for indices in itertools.combinations(range(len(pool)), size):
            columns = [names[index] for index in indices]
            correlations = selection_matrix[columns].corr().abs()
            pairwise_correlations = [
                float(correlations.iloc[left, right])
                for left in range(len(columns))
                for right in range(left + 1, len(columns))
            ]
            max_pairwise_correlation = max(pairwise_correlations, default=0.0)
            minimum_residual_variance = min(
                (1.0 - value**2 for value in pairwise_correlations),
                default=1.0,
            )
            if max_pairwise_correlation >= 0.95 or minimum_residual_variance < 0.10:
                reject("REDUNDANCY_OR_RESIDUAL_VARIANCE_GATE_FAILED")
                continue
            ensemble = _ridge_ensemble(train, train_matrix[columns])
            weights = _normalised_coefficient_weights(ensemble)
            if sum(weight >= 0.05 for weight in weights) < 2:
                reject("EFFECTIVE_WEIGHT_COUNT_BELOW_2")
                continue
            mean = np.asarray(ensemble.get("feature_mean", []), dtype=float)
            scale = np.asarray(ensemble.get("feature_scale", []), dtype=float)
            coefficients = np.asarray(ensemble.get("coefficients", []), dtype=float)
            matrix = selection_matrix[columns].to_numpy(dtype=float)
            if len(mean) != size or len(scale) != size or len(coefficients) != size:
                reject("ENSEMBLE_DIMENSION_MISMATCH")
                continue
            standardised = np.nan_to_num(
                (matrix - mean) / np.where(np.abs(scale) < 1e-8, 1.0, scale)
            )
            raw = float(ensemble.get("intercept", 0.0)) + standardised @ coefficients
            metrics = factor_metrics(selection, raw)
            trial_index = len(combination_p_values)
            combination_p_values.append(float(metrics.get("ic_p_value", 1.0)))
            evaluated_combinations += 1
            if not (
                metrics["ic_observations"] >= 3
                and metrics["ic_mean"] >= 0.01
                and metrics["ic_ir"] >= 0.10
                and metrics["recent_ic_mean"] >= 0.005
                and metrics["status"] == "ACTIVE"
                and metrics["turnover"] <= 0.90
            ):
                reject("SELECTION_METRIC_GATE_FAILED")
                continue
            single_metrics = []
            for index in indices:
                item_metrics = dict(pool[index].get("selection_metrics") or {})
                if not item_metrics:
                    item_metrics = factor_metrics(
                        selection,
                        selection_matrix[names[index]],
                    )
                single_metrics.append(item_metrics)
            best_single_ic = max(
                (float(item.get("ic_mean", 0.0)) for item in single_metrics),
                default=0.0,
            )
            incremental_ic_gain = float(metrics["ic_mean"]) - best_single_ic
            if incremental_ic_gain < 0.0005:
                reject("INCREMENTAL_IC_GAIN_BELOW_0_0005")
                continue
            stability = _selection_subperiod_stability(selection, raw)
            if not _passes_selection_subperiod_stability(stability):
                reject("SELECTION_SUBPERIOD_STABILITY_GATE_FAILED")
                continue
            score = float(
                metrics["ic_mean"]
                + 0.01 * metrics["ic_ir"]
                - 0.02 * metrics["turnover"]
                - 0.001 * (size - 2)
            )
            ranked.append(
                (
                    score,
                    indices,
                    metrics,
                    weights,
                    dict(ensemble),
                    {
                        "max_pairwise_correlation": round(max_pairwise_correlation, 8),
                        "minimum_residual_variance": round(minimum_residual_variance, 8),
                        "best_single_ic": round(best_single_ic, 8),
                        "incremental_ic_gain": round(incremental_ic_gain, 8),
                        "selection_subperiod_stability": stability,
                    },
                    trial_index,
                )
            )
    combination_q_values = _benjamini_hochberg_adjust(combination_p_values)
    fdr_ranked = []
    for entry in ranked:
        q_value = combination_q_values[entry[6]]
        if q_value <= DISCOVERY_FDR_MAX:
            fdr_ranked.append(entry)
        else:
            reject("SELECTION_COMBINATION_FDR_ABOVE_0_10")
    ranked = fdr_ranked
    if not ranked:
        return [], {
            "candidate_pool_size": len(pool),
            "evaluated_combinations": evaluated_combinations,
            "status": "NO_COMPLEMENTARY_SET_PASSED_SELECTION",
            "rejection_counts": combination_rejection_counts,
            "discovery_control": {
                "method": "benjamini_hochberg",
                "family": "all_evaluated_factor_combinations",
                "family_size": evaluated_combinations,
                "maximum_fdr": DISCOVERY_FDR_MAX,
                "discovery_count": sum(
                    value <= DISCOVERY_FDR_MAX for value in combination_q_values
                ),
            },
        }
    ranked.sort(key=lambda item: item[0], reverse=True)
    score, indices, metrics, weights, selected_ensemble, complementarity, trial_index = ranked[0]
    selected_combination_q = combination_q_values[trial_index]
    selected = [{**pool[index], "status": "ACTIVE"} for index in indices]
    return selected, {
        "candidate_pool_size": len(pool),
        "evaluated_combinations": evaluated_combinations,
        "status": "SELECTED",
        "selected_names": [str(item.get("name")) for item in selected],
        "selection_score": round(score, 8),
        "selection_metrics": metrics,
        "rejection_counts": combination_rejection_counts,
        "discovery_control": {
            "method": "benjamini_hochberg",
            "family": "all_evaluated_factor_combinations",
            "family_size": evaluated_combinations,
            "maximum_fdr": DISCOVERY_FDR_MAX,
            "discovery_count": sum(
                value <= DISCOVERY_FDR_MAX for value in combination_q_values
            ),
            "selected_q_value": round(selected_combination_q, 10),
        },
        "selected_ensemble": selected_ensemble,
        "ensemble_fit_scope": "train_only_frozen_before_selection",
        "training_fit_effective_weights": {
            str(selected[index].get("name")): round(weights[index], 8)
            for index in range(len(selected))
        },
        "complementarity": complementarity,
        "approval_holdout_used": False,
    }


def apply_factor_registry(frame: pd.DataFrame, registry: Mapping[str, Any]) -> pd.DataFrame:
    """Apply a promoted registry and return per-factor values plus adaptive score."""
    output = _prepare_feature_cache(frame)
    specs = [item for item in registry.get("factors", []) if item.get("status") == "ACTIVE"]
    if not registry.get("approved") or not specs:
        output["adaptive_score"] = 50.0
        output["adaptive_raw"] = 0.0
        return output.drop(columns=[name for name in output.columns if name.startswith("__z__")])
    matrix = _factor_matrix(output, specs)
    ensemble = registry.get("ensemble", {}) or {}
    mean = np.asarray(ensemble.get("feature_mean", [0.0] * len(specs)), dtype=float)
    scale = np.asarray(ensemble.get("feature_scale", [1.0] * len(specs)), dtype=float)
    coefficients = np.asarray(ensemble.get("coefficients", [1.0 / len(specs)] * len(specs)), dtype=float)
    if len(mean) != len(specs) or len(scale) != len(specs) or len(coefficients) != len(specs):
        output["adaptive_score"] = 50.0
        output["adaptive_raw"] = 0.0
        return output.drop(columns=[name for name in output.columns if name.startswith("__z__")])
    standardised = np.nan_to_num((matrix.to_numpy(dtype=float) - mean) / np.where(scale < 1e-8, 1.0, scale))
    raw = float(ensemble.get("intercept", 0.0)) + standardised @ coefficients
    output["adaptive_raw"] = industry_neutralise(output, raw)
    scores = pd.Series(50.0, index=output.index, dtype=float)
    scores = output["adaptive_raw"].groupby(pd.to_datetime(output["date"])).rank(
        pct=True,
        method="average",
    ) * 100.0
    output["adaptive_score"] = scores.clip(0.0, 100.0)
    for name in matrix.columns:
        output[f"factor__{name}"] = matrix[name]
    return output.drop(columns=[name for name in output.columns if name.startswith("__z__")])


def _factor_specification_fingerprint(factors: Sequence[Mapping[str, Any]]) -> str:
    expressions = sorted(
        {
            _expression_key(item.get("expression"))
            for item in factors
            if isinstance(item.get("expression"), Mapping)
        }
    )
    if not expressions:
        return ""
    return hashlib.sha256(
        json.dumps(expressions, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def blend_priority(base_priority: float, adaptive_score: float, adaptive_weight: float = 0.15) -> float:
    weight = float(np.clip(adaptive_weight, 0.0, 0.35))
    return round(float(np.clip((1.0 - weight) * base_priority + weight * adaptive_score, 0.0, 100.0)), 2)


def evolve_factor_registry(
    rows: Iterable[Mapping[str, Any]],
    previous_registry: Optional[Mapping[str, Any]] = None,
    llm_candidates: Sequence[Mapping[str, Any]] = (),
    seed: int = 42,
    max_active: int = 5,
    population_size: int = 32,
    generations: int = 5,
    require_policy_seasoning: bool = False,
) -> Dict[str, Any]:
    """Evolve on train, select on validation, and approve only on untouched holdout."""
    if previous_registry is not None:
        previous_registry = sanitize_factor_registry(previous_registry)
    data = pd.DataFrame(list(rows)).copy()
    if data.empty or "excess_return_10d" not in data.columns:
        raise ValueError("factor evolution requires labelled calibration rows")
    data["date"] = pd.to_datetime(data["date"])
    for name in PRIMITIVE_FEATURES:
        if name not in data.columns:
            data[name] = 0.0
    if "industry_group" not in data.columns:
        data["industry_group"] = [industry_group(code) for code in data["code"]]
    data = _prepare_feature_cache(data)
    unique_dates = sorted(data["date"].unique())
    if len(unique_dates) < 9:
        raise ValueError("factor evolution requires at least 9 distinct dates")
    train_index = min(max(int(len(unique_dates) * 0.70), 1), len(unique_dates) - 3)
    selection_index = min(max(int(len(unique_dates) * 0.85), train_index + 1), len(unique_dates) - 1)
    train_cutoff = pd.Timestamp(unique_dates[train_index - 1])
    selection_cutoff = pd.Timestamp(unique_dates[selection_index - 1])
    purge_gap = pd.Timedelta(days=28)
    train = data[data["date"] <= train_cutoff].copy()
    selection = data[
        (data["date"] > train_cutoff + purge_gap)
        & (data["date"] <= selection_cutoff)
    ].copy()
    approval = data[data["date"] > selection_cutoff + purge_gap].copy()
    if selection.empty or approval.empty:
        raise PurgedHoldoutInsufficientError(
            "factor evolution requires non-empty selection and approval holdouts after purge"
        )
    purge_method = "28_calendar_day_approx_20_trading_day_purge"
    development = pd.concat([train, selection], ignore_index=False).sort_values("date")

    prior_retirements = [
        dict(item) for item in (previous_registry or {}).get("retired_factors", [])
        if isinstance(item, Mapping)
    ]
    trained_until = pd.Timestamp(data["date"].max())
    previous_policy = str((previous_registry or {}).get("evolution_policy_version", ""))
    prior_anchor = pd.to_datetime(
        (previous_registry or {}).get("policy_seasoning_anchor"),
        errors="coerce",
    )
    if previous_policy == FACTOR_EVOLUTION_POLICY_VERSION and not pd.isna(prior_anchor):
        policy_seasoning_anchor = pd.Timestamp(prior_anchor)
    else:
        policy_seasoning_anchor = trained_until
    unseen_policy_dates = sum(
        pd.Timestamp(value) > policy_seasoning_anchor for value in unique_dates
    )
    policy_seasoned = bool(
        not require_policy_seasoning
        or (
            previous_policy == FACTOR_EVOLUTION_POLICY_VERSION
            and unseen_policy_dates >= POLICY_SEASONING_MIN_DATES
        )
    )
    cooldown_keys = set()
    for item in prior_retirements:
        cooldown_until = pd.to_datetime(item.get("cooldown_until"), errors="coerce")
        if (
            (item.get("expression_key") or item.get("expression_family_key") or item.get("expression"))
            and not pd.isna(cooldown_until)
            and pd.Timestamp(cooldown_until) >= trained_until
        ):
            expression = item.get("expression")
            cooldown_keys.add(
                str(item.get("expression_family_key"))
                if item.get("expression_family_key")
                else _expression_family_key(expression)
                if isinstance(expression, Mapping)
                else str(item["expression_key"])
            )

    approved_incumbents = [
        dict(item) for item in (previous_registry or {}).get("factors", [])
        if bool((previous_registry or {}).get("approved", False))
        and item.get("expression")
        and item.get("status") == "ACTIVE"
    ]
    incumbents = list(approved_incumbents)
    if not incumbents:
        incumbents = seeded_factor_specs()
    monitored: List[Dict[str, Any]] = []
    for item in incumbents:
        metrics = factor_metrics(development, evaluate_expression(item["expression"], development))
        monitored.append({**item, "monitor_metrics": metrics, "status": metrics["status"]})

    candidates = genetic_candidates(
        train,
        seed=seed,
        population_size=max(12, population_size),
        generations=max(1, generations),
    )
    candidates.extend(seeded_factor_specs())
    candidates.extend(primitive_factor_specs())
    candidates.extend(dict(item) for item in llm_candidates if item.get("expression"))
    candidates.extend(item for item in monitored if item.get("status") != "RETIRED")
    evaluated: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        expression = _oriented(item["expression"], train)
        expression_key = _expression_key(expression)
        expression_family_key = _expression_family_key(expression)
        if expression_family_key in cooldown_keys:
            continue
        key = json.dumps(expression, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        train_metrics = factor_metrics(train, evaluate_expression(expression, train))
        selection_metrics = factor_metrics(selection, evaluate_expression(expression, selection))
        rejection_reasons: List[str] = []
        if train_metrics["status"] == "RETIRED":
            rejection_reasons.append("TRAIN_STATUS_RETIRED")
        if selection_metrics["ic_observations"] < 3:
            rejection_reasons.append("SELECTION_IC_HISTORY_INSUFFICIENT")
        if selection_metrics["ic_mean"] < 0.01:
            rejection_reasons.append("SELECTION_IC_BELOW_0_01")
        if selection_metrics["ic_ir"] < 0.10:
            rejection_reasons.append("SELECTION_IR_BELOW_0_10")
        if selection_metrics["recent_ic_mean"] < 0.005:
            rejection_reasons.append("SELECTION_RECENT_IC_BELOW_0_005")
        if selection_metrics["status"] != "ACTIVE":
            rejection_reasons.append("SELECTION_STATUS_NOT_ACTIVE")
        if selection_metrics["turnover"] > 0.90:
            rejection_reasons.append("SELECTION_TURNOVER_ABOVE_0_90")
        if train_metrics["ic_mean"] * selection_metrics["ic_mean"] <= 0:
            rejection_reasons.append("TRAIN_SELECTION_SIGN_MISMATCH")
        accepted = not rejection_reasons
        stability = min(train_metrics["ic_mean"], selection_metrics["ic_mean"])
        score = stability + 0.01 * min(train_metrics["ic_ir"], selection_metrics["ic_ir"]) - 0.02 * selection_metrics["turnover"]
        evaluated.append(
            {
                "name": str(item.get("name") or f"gp_{len(evaluated) + 1}"),
                "expression": expression,
                "expression_key": expression_key,
                "expression_family_key": expression_family_key,
                "expression_text": expression_text(expression),
                "economic_logic": str(item.get("economic_logic") or economic_logic(expression)),
                "generation": int(item.get("generation", 0)),
                "candidate_origin": str(item.get("candidate_origin", "genetic_or_seeded")),
                "proposal_metadata": dict(item.get("proposal_metadata") or {}),
                "complexity": expression_complexity(expression),
                "train_metrics": train_metrics,
                "validation_metrics": selection_metrics,
                "selection_metrics": selection_metrics,
                "selection_score": round(score, 8),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
            }
        )
    candidate_q_values = _benjamini_hochberg_adjust(
        [
            float((item.get("selection_metrics") or {}).get("ic_p_value", 1.0))
            for item in evaluated
        ]
    )
    for item, q_value in zip(evaluated, candidate_q_values):
        rounded_q = round(float(q_value), 10)
        item["selection_metrics"]["multiple_testing_q_value"] = rounded_q
        item["validation_metrics"]["multiple_testing_q_value"] = rounded_q
        item["selection_discovery_control"] = {
            "method": "benjamini_hochberg",
            "family": "all_unique_candidate_expressions",
            "family_size": len(evaluated),
            "maximum_fdr": DISCOVERY_FDR_MAX,
            "q_value": rounded_q,
            "passed": bool(q_value <= DISCOVERY_FDR_MAX),
        }
        if item.get("accepted") and q_value > DISCOVERY_FDR_MAX:
            item["accepted"] = False
            item["rejection_reasons"].append("SELECTION_FDR_ABOVE_0_10")
    evaluated.sort(key=lambda item: item["selection_score"], reverse=True)

    selected, combination_search = _select_complementary_factor_set(
        train,
        selection,
        evaluated,
        max_active=max_active,
    )

    selection_passed = len(selected) >= 2
    if not selected:
        fallback = evaluated[: min(max(1, max_active), len(evaluated))]
        selected = [{**item, "status": "WATCH"} for item in fallback]
    active_for_model = [item for item in selected if item["status"] == "ACTIVE"]
    if not active_for_model:
        active_for_model = selected
    model_matrix = _factor_matrix(development, active_for_model)
    selected_ensemble = combination_search.get("selected_ensemble")
    if selection_passed and isinstance(selected_ensemble, Mapping):
        ensemble = dict(selected_ensemble)
        ensemble_fit_scope = "train_only_frozen_before_selection"
    else:
        ensemble = _ridge_ensemble(development, model_matrix)
        ensemble_fit_scope = "development_fallback_research_only"
    effective_factor_weights = _normalised_coefficient_weights(ensemble)
    effective_factor_count = sum(value >= 0.05 for value in effective_factor_weights)
    provisional = {
        "approved": True,
        "factors": [{**item, "status": "ACTIVE"} for item in active_for_model],
        "ensemble": ensemble,
        "ensemble_fit_scope": ensemble_fit_scope,
    }
    selection_scored = apply_factor_registry(selection, provisional)
    ensemble_selection_metrics = factor_metrics(selection, selection_scored["adaptive_raw"])
    approval_scored = apply_factor_registry(approval, provisional)
    ensemble_approval_metrics = factor_metrics(approval, approval_scored["adaptive_raw"])
    active_for_model = [
        {
            **item,
            "approval_metrics": factor_metrics(
                approval,
                evaluate_expression(item["expression"], approval),
            ),
        }
        for item in active_for_model
    ]
    candidate_specification_fingerprint = _factor_specification_fingerprint(
        active_for_model
    )
    previous_candidate_specification_fingerprint = str(
        (previous_registry or {}).get("candidate_specification_fingerprint", "")
    )
    if not previous_candidate_specification_fingerprint:
        previous_candidate_specification_fingerprint = _factor_specification_fingerprint(
            [
                item
                for item in (previous_registry or {}).get("factors", [])
                if isinstance(item, Mapping)
            ]
        )
    candidate_specification_changed = bool(
        require_policy_seasoning
        and previous_policy == FACTOR_EVOLUTION_POLICY_VERSION
        and previous_candidate_specification_fingerprint
        and candidate_specification_fingerprint
        and previous_candidate_specification_fingerprint
        != candidate_specification_fingerprint
    )
    if candidate_specification_changed:
        policy_seasoning_anchor = trained_until
        unseen_policy_dates = 0
        policy_seasoned = False
    approved = bool(
        policy_seasoned
        and
        selection_passed
        and len(active_for_model) >= 2
        and effective_factor_count >= 2
        and ensemble_approval_metrics["ic_observations"] >= 3
        and ensemble_approval_metrics["ic_mean"] >= 0.015
        and ensemble_approval_metrics["ic_ir"] >= 0.15
        and ensemble_approval_metrics["recent_ic_mean"] >= 0.005
        and ensemble_approval_metrics["status"] != "RETIRED"
        and ensemble_approval_metrics["turnover"] <= 0.85
    )
    incumbent_names = {str(item.get("name")) for item in approved_incumbents}
    selected_names = {str(item.get("name")) for item in active_for_model}
    current_retired = [
        {
            "name": str(item.get("name")),
            "expression": item.get("expression"),
            "expression_key": _expression_key(item["expression"]),
            "expression_family_key": _expression_family_key(item["expression"]),
            "expression_text": expression_text(item["expression"]),
            "retired_at": trained_until.strftime("%Y-%m-%d"),
            "cooldown_until": (trained_until + pd.Timedelta(days=183)).strftime("%Y-%m-%d"),
            "reasons": (
                (item.get("monitor_metrics") or {}).get("reasons")
                or ["OUTPERFORMED_BY_REPLACEMENT"]
            ),
        }
        for item in monitored
        if item.get("status") == "RETIRED"
        or (approved and str(item.get("name")) not in selected_names)
    ]
    replacement_candidates = [name for name in selected_names if name not in incumbent_names]
    replacements = replacement_candidates if approved else []
    retirement_by_key: Dict[str, Dict[str, Any]] = {}
    for item in prior_retirements + current_retired:
        expression = item.get("expression")
        family_key = item.get("expression_family_key")
        if not family_key and isinstance(expression, Mapping):
            family_key = _expression_family_key(expression)
            item["expression_family_key"] = family_key
        retirement_by_key[str(family_key or item.get("expression_key") or item.get("name"))] = item
    retirement_history = list(retirement_by_key.values())
    prior_replacement_events = [
        dict(item) for item in (previous_registry or {}).get("replacement_events", [])
        if isinstance(item, Mapping)
    ]
    if not prior_replacement_events and previous_registry:
        prior_active = [
            str(item.get("name")) for item in previous_registry.get("factors", [])
            if item.get("name")
        ]
        prior_retired = [
            str(item.get("name")) for item in previous_registry.get("retired_factors", [])
            if item.get("name")
        ]
        if prior_active and prior_retired:
            prior_replacement_events.append(
                {
                    "event_type": "registry_v2_migration_snapshot",
                    "new_factors": prior_active,
                    "retired_factors": prior_retired,
                    "approved_on_independent_holdout": bool(previous_registry.get("approved")),
                }
            )
    current_replacement_events = [
        {
            "new_factor": name,
            "replaces": current_retired[index % len(current_retired)]["name"] if current_retired else None,
            "approved_on_independent_holdout": bool(approved),
            "event_date": trained_until.strftime("%Y-%m-%d"),
        }
        for index, name in enumerate(sorted(replacements))
    ]
    replacement_events = prior_replacement_events + current_replacement_events
    approval_reasons: List[str] = []
    if not approved:
        if not policy_seasoned:
            approval_reasons.append("POLICY_SEASONING_INCOMPLETE")
        if candidate_specification_changed:
            approval_reasons.append("POLICY_CANDIDATE_SPEC_CHANGED_RESET_SEASONING")
        if not selection_passed:
            approval_reasons.append("FACTOR_SELECTION_GATE_FAILED")
        if effective_factor_count < 2:
            approval_reasons.append("EFFECTIVE_FACTOR_COUNT_BELOW_2")
        if not (
            ensemble_approval_metrics["ic_observations"] >= 3
            and ensemble_approval_metrics["ic_mean"] >= 0.015
            and ensemble_approval_metrics["ic_ir"] >= 0.15
            and ensemble_approval_metrics["recent_ic_mean"] >= 0.005
            and ensemble_approval_metrics["status"] != "RETIRED"
            and ensemble_approval_metrics["turnover"] <= 0.85
        ):
            approval_reasons.append("INDEPENDENT_ENSEMBLE_HOLDOUT_GATE_FAILED")
    rejection_counts: Dict[str, int] = {}
    for item in evaluated:
        for reason in item.get("rejection_reasons", []):
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
    return {
        "schema_version": 2,
        "evolution_policy_version": FACTOR_EVOLUTION_POLICY_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_until": data["date"].max().strftime("%Y-%m-%d"),
        "approved": approved,
        "approval_reasons": approval_reasons,
        "validation_method": "train_70_selection_15_approval_15_with_purged_boundaries",
        "purge_method": purge_method,
        "policy_seasoning_required": bool(require_policy_seasoning),
        "policy_seasoning_anchor": policy_seasoning_anchor.strftime("%Y-%m-%d"),
        "policy_unseen_date_count": int(unseen_policy_dates),
        "policy_seasoning_min_dates": POLICY_SEASONING_MIN_DATES,
        "policy_seasoned": policy_seasoned,
        "candidate_specification_fingerprint": candidate_specification_fingerprint,
        "previous_candidate_specification_fingerprint": (
            previous_candidate_specification_fingerprint
        ),
        "policy_candidate_specification_changed": candidate_specification_changed,
        "neutralisation": "broad_industry_demean_then_cross_sectional_zscore",
        "monitor_thresholds": {
            "recent_ic_min": 0.01,
            "recent_ir_min": 0.15,
            "decay_ratio_min": 0.20,
            "turnover_max": 0.85,
        },
        "factors": [
            {**item, "status": "ACTIVE" if approved else "RESEARCH"}
            for item in active_for_model
        ],
        "ensemble": ensemble,
        "ensemble_fit_scope": ensemble_fit_scope,
        "effective_factor_weights": {
            str(item.get("name")): round(effective_factor_weights[index], 8)
            for index, item in enumerate(active_for_model)
            if index < len(effective_factor_weights)
        },
        "effective_factor_count": int(effective_factor_count),
        "combination_search": combination_search,
        "ensemble_selection_metrics": ensemble_selection_metrics,
        "ensemble_approval_metrics": ensemble_approval_metrics,
        "ensemble_validation_metrics": ensemble_approval_metrics,
        "retired_factors": retirement_history,
        "new_replacements": replacements,
        "research_challengers": sorted(selected_names) if not approved else [],
        "replacement_events": replacement_events,
        "candidate_count": len(evaluated),
        "candidate_discovery_control": {
            "method": "benjamini_hochberg",
            "family": "all_unique_candidate_expressions",
            "family_size": len(evaluated),
            "maximum_fdr": DISCOVERY_FDR_MAX,
            "discovery_count": sum(
                float(
                    (item.get("selection_metrics") or {}).get(
                        "multiple_testing_q_value", 1.0
                    )
                )
                <= DISCOVERY_FDR_MAX
                for item in evaluated
            ),
            "accepted_after_all_gates": sum(
                bool(item.get("accepted")) for item in evaluated
            ),
        },
        "candidate_gate_summary": {
            "accepted_count": sum(bool(item.get("accepted")) for item in evaluated),
            "rejected_count": sum(not bool(item.get("accepted")) for item in evaluated),
            "rejection_counts": rejection_counts,
        },
        "candidate_diagnostics": [
            {
                "name": str(item.get("name", "")),
                "candidate_origin": str(item.get("candidate_origin", "")),
                "expression_text": str(item.get("expression_text", "")),
                "selection_score": float(item.get("selection_score", 0.0)),
                "accepted": bool(item.get("accepted", False)),
                "rejection_reasons": list(item.get("rejection_reasons", [])),
                "train_metrics": dict(item.get("train_metrics", {})),
                "selection_metrics": dict(item.get("selection_metrics", {})),
            }
            for item in evaluated[:20]
        ],
        "candidate_origins": {
            origin: sum(item.get("candidate_origin") == origin for item in evaluated)
            for origin in sorted({str(item.get("candidate_origin")) for item in evaluated})
        },
        "llm_proposals_considered": sum(
            item.get("candidate_origin") == "llm_structured_proposal" for item in evaluated
        ),
        "llm_proposals_selected": (
            [
                str(item.get("name"))
                for item in active_for_model
                if item.get("candidate_origin") == "llm_structured_proposal"
            ]
            if approved
            else []
        ),
        "llm_research_challengers": (
            [
                str(item.get("name"))
                for item in active_for_model
                if item.get("candidate_origin") == "llm_structured_proposal"
            ]
            if not approved
            else []
        ),
        "train_rows": len(train),
        "selection_rows": len(selection),
        "approval_rows": len(approval),
        "validation_rows": len(selection),
        "train_cutoff": train_cutoff.strftime("%Y-%m-%d"),
        "selection_cutoff": selection_cutoff.strftime("%Y-%m-%d"),
        "validation_cutoff": selection_cutoff.strftime("%Y-%m-%d"),
    }


def load_factor_registry_with_status(
    path: str,
    max_age_days: int = MODEL_TRAINED_MAX_LAG_DAYS,
    generated_max_age_days: int = MODEL_GENERATED_MAX_AGE_DAYS,
    now: Any = None,
    require_bundle_integrity: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if int(value.get("schema_version", 0)) not in {1, 2}:
            return None, "FACTOR_REGISTRY_SCHEMA_INVALID"
        if require_bundle_integrity:
            bundle_status = validate_bundle_member(path, value)
            if bundle_status != "APPROVED":
                return None, f"FACTOR_REGISTRY_{bundle_status}"
        time_status = validate_artifact_time(
            value,
            now=now,
            generated_max_age_days=generated_max_age_days,
            trained_max_lag_days=max_age_days,
        )
        if not time_status.approved:
            return None, f"FACTOR_REGISTRY_{time_status.reason}"
        return dict(value), "APPROVED"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, "FACTOR_REGISTRY_UNAVAILABLE"


def load_factor_registry(
    path: str,
    max_age_days: int = MODEL_TRAINED_MAX_LAG_DAYS,
    generated_max_age_days: int = MODEL_GENERATED_MAX_AGE_DAYS,
    now: Any = None,
    require_bundle_integrity: bool = False,
) -> Optional[Dict[str, Any]]:
    registry, _ = load_factor_registry_with_status(
        path,
        max_age_days=max_age_days,
        generated_max_age_days=generated_max_age_days,
        now=now,
        require_bundle_integrity=require_bundle_integrity,
    )
    return registry


def sanitize_factor_registry(registry: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove replacement claims that never passed the independent holdout."""
    value = copy.deepcopy(dict(registry))
    if bool(value.get("approved", False)):
        return value
    events = [
        dict(item)
        for item in value.get("replacement_events", [])
        if isinstance(item, Mapping)
    ]
    rejected_events = [
        item for item in events
        if item.get("approved_on_independent_holdout") is False
    ]
    rejected_replaced = {
        str(item.get("replaces"))
        for item in rejected_events
        if item.get("replaces")
    }
    approved_retired = {
        str(name)
        for item in events
        if item.get("approved_on_independent_holdout") is True
        for name in (
            list(item.get("retired_factors", []) or [])
            + ([item.get("replaces")] if item.get("replaces") else [])
        )
        if name
    }
    value["replacement_events"] = [
        item for item in events
        if item.get("approved_on_independent_holdout") is not False
    ]
    value["retired_factors"] = [
        item
        for item in value.get("retired_factors", [])
        if not (
            str(item.get("name")) in rejected_replaced
            and str(item.get("name")) not in approved_retired
            and list(item.get("reasons", [])) == ["OUTPERFORMED_BY_REPLACEMENT"]
        )
    ]
    existing_retired = {str(item.get("name")) for item in value["retired_factors"]}
    known_specs = {
        str(item.get("name")): item
        for item in seeded_factor_specs() + primitive_factor_specs()
        if item.get("name") and item.get("expression")
    }
    trained_until = pd.to_datetime(value.get("trained_until"), errors="coerce")
    for name in sorted(approved_retired - existing_retired):
        spec = known_specs.get(name)
        if spec is None or pd.isna(trained_until):
            continue
        expression = dict(spec["expression"])
        value["retired_factors"].append(
            {
                "name": name,
                "expression": expression,
                "expression_key": _expression_key(expression),
                "expression_family_key": _expression_family_key(expression),
                "expression_text": expression_text(expression),
                "retired_at": pd.Timestamp(trained_until).strftime("%Y-%m-%d"),
                "cooldown_until": (
                    pd.Timestamp(trained_until) + pd.Timedelta(days=183)
                ).strftime("%Y-%m-%d"),
                "reasons": ["OUTPERFORMED_BY_REPLACEMENT"],
                "restored_from_approved_event": True,
            }
        )
    candidates = list(
        dict.fromkeys(
            [str(name) for name in value.get("research_challengers", []) if name]
            + [str(name) for name in value.get("new_replacements", []) if name]
            + [str(item.get("new_factor")) for item in rejected_events if item.get("new_factor")]
        )
    )
    value["research_challengers"] = candidates
    value["new_replacements"] = []
    value["factors"] = [
        {**dict(item), "status": "RESEARCH"}
        for item in value.get("factors", [])
        if isinstance(item, Mapping)
    ]
    return value


def save_factor_registry(registry: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(registry), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


__all__ = [
    "PRIMITIVE_FEATURES",
    "apply_factor_registry",
    "blend_priority",
    "build_primitive_row",
    "evaluate_expression",
    "expression_features",
    "evolve_factor_registry",
    "PurgedHoldoutInsufficientError",
    "factor_metrics",
    "industry_neutralise",
    "load_factor_registry",
    "load_factor_registry_with_status",
    "primitive_factor_specs",
    "save_factor_registry",
    "sanitize_factor_registry",
    "seeded_factor_specs",
]
