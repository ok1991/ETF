"""Adaptive factor monitoring, symbolic genetic programming, and ML ensembling.

The module is deliberately self-contained and depends only on NumPy/Pandas so the
production job does not need a heavyweight AutoML runtime.  Every promoted factor
is stored with its expression, economic rationale, in/out-of-sample diagnostics,
and replacement history in ``adaptive_factor_registry.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
        "momentum_20": momentum_20,
        "momentum_60": momentum_60,
        "reversal_5": reversal_5,
        "trend_efficiency_20": float(np.clip(trend_efficiency, -1.0, 1.0)),
        "volatility_20": max(0.0, volatility_20),
        "downside_volatility_60": max(0.0, downside_volatility_60),
        "volume_confirmation": float(np.clip(volume_confirmation, -3.0, 3.0)),
        "liquidity_log": math.log1p(max(_number(amount), 0.0)),
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


def _expression_features(expression: Mapping[str, Any]) -> List[str]:
    if "feature" in expression:
        return [str(expression["feature"])]
    values: List[str] = []
    for arg in expression.get("args", []):
        values.extend(_expression_features(arg))
    return list(dict.fromkeys(values))


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
        "recent_ic_mean": round(recent_mean, 8),
        "recent_ic_ir": round(recent_ir, 6),
        "ic_decay": {key: round(value, 8) for key, value in decay.items()},
        "decay_ratio_20_to_5": round(decay_ratio, 6),
        "turnover": round(turnover, 6),
        "ic_observations": len(ic_series),
        "status": status,
        "reasons": reasons,
    }


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


def blend_priority(base_priority: float, adaptive_score: float, adaptive_weight: float = 0.15) -> float:
    weight = float(np.clip(adaptive_weight, 0.0, 0.35))
    return round(float(np.clip((1.0 - weight) * base_priority + weight * adaptive_score, 0.0, 100.0)), 2)


def evolve_factor_registry(
    rows: Iterable[Mapping[str, Any]],
    previous_registry: Optional[Mapping[str, Any]] = None,
    seed: int = 42,
    max_active: int = 5,
    population_size: int = 32,
    generations: int = 5,
) -> Dict[str, Any]:
    """Monitor incumbents, retire failures, evolve replacements, and fit ML weights."""
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
    split_index = min(max(int(len(unique_dates) * 0.80), 1), max(len(unique_dates) - 1, 1))
    cutoff = pd.Timestamp(unique_dates[split_index - 1])
    train = data[data["date"] <= cutoff].copy()
    validate = data[data["date"] > cutoff].copy()
    if validate.empty:
        validate = train.tail(max(1, len(train) // 5)).copy()

    incumbents = [
        dict(item) for item in (previous_registry or {}).get("factors", [])
        if item.get("expression") and item.get("status") == "ACTIVE"
    ]
    if not incumbents:
        incumbents = seeded_factor_specs()
    monitored: List[Dict[str, Any]] = []
    for item in incumbents:
        metrics = factor_metrics(data, evaluate_expression(item["expression"], data))
        monitored.append({**item, "monitor_metrics": metrics, "status": metrics["status"]})

    candidates = genetic_candidates(
        train,
        seed=seed,
        population_size=max(12, population_size),
        generations=max(1, generations),
    )
    candidates.extend(seeded_factor_specs())
    candidates.extend(item for item in monitored if item.get("status") != "RETIRED")
    evaluated: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        expression = _oriented(item["expression"], train)
        key = json.dumps(expression, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        train_metrics = factor_metrics(train, evaluate_expression(expression, train))
        validate_metrics = factor_metrics(validate, evaluate_expression(expression, validate))
        accepted = bool(
            validate_metrics["ic_observations"] >= 3
            and validate_metrics["ic_mean"] >= 0.01
            and validate_metrics["ic_ir"] >= 0.10
            and validate_metrics["recent_ic_mean"] >= 0.005
            and validate_metrics["status"] != "RETIRED"
            and validate_metrics["turnover"] <= 0.90
            and train_metrics["ic_mean"] * validate_metrics["ic_mean"] > 0
        )
        stability = min(train_metrics["ic_mean"], validate_metrics["ic_mean"])
        score = stability + 0.01 * min(train_metrics["ic_ir"], validate_metrics["ic_ir"]) - 0.02 * validate_metrics["turnover"]
        evaluated.append(
            {
                "name": str(item.get("name") or f"gp_{len(evaluated) + 1}"),
                "expression": expression,
                "expression_text": expression_text(expression),
                "economic_logic": str(item.get("economic_logic") or economic_logic(expression)),
                "generation": int(item.get("generation", 0)),
                "complexity": expression_complexity(expression),
                "train_metrics": train_metrics,
                "validation_metrics": validate_metrics,
                "selection_score": round(score, 8),
                "accepted": accepted,
            }
        )
    evaluated.sort(key=lambda item: item["selection_score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    validation_values: List[pd.Series] = []
    for item in evaluated:
        if not item["accepted"]:
            continue
        values = industry_neutralise(validate, evaluate_expression(item["expression"], validate))
        if any(abs(float(values.corr(existing))) >= 0.85 for existing in validation_values):
            continue
        selected.append({**item, "status": "ACTIVE"})
        validation_values.append(values)
        if len(selected) >= max(1, max_active):
            break

    if not selected:
        fallback = evaluated[: min(max(1, max_active), len(evaluated))]
        selected = [{**item, "status": "WATCH"} for item in fallback]
    active_for_model = [item for item in selected if item["status"] == "ACTIVE"]
    if not active_for_model:
        active_for_model = selected
    train_matrix = _factor_matrix(train, active_for_model)
    ensemble = _ridge_ensemble(train, train_matrix)
    provisional = {
        "approved": True,
        "factors": [{**item, "status": "ACTIVE"} for item in active_for_model],
        "ensemble": ensemble,
    }
    validation_scored = apply_factor_registry(validate, provisional)
    ensemble_metrics = factor_metrics(validate, validation_scored["adaptive_raw"])
    approved = bool(
        len(active_for_model) >= 2
        and ensemble_metrics["ic_observations"] >= 3
        and ensemble_metrics["ic_mean"] >= 0.015
        and ensemble_metrics["ic_ir"] >= 0.15
        and ensemble_metrics["turnover"] <= 0.85
    )
    incumbent_names = {str(item.get("name")) for item in incumbents}
    selected_names = {str(item.get("name")) for item in active_for_model}
    retired = [
        {
            "name": str(item.get("name")),
            "reasons": (
                (item.get("monitor_metrics") or {}).get("reasons")
                or ["OUTPERFORMED_BY_REPLACEMENT"]
            ),
        }
        for item in monitored
        if str(item.get("name")) not in selected_names or item.get("status") == "RETIRED"
    ]
    replacements = [name for name in selected_names if name not in incumbent_names]
    return {
        "schema_version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_until": data["date"].max().strftime("%Y-%m-%d"),
        "approved": approved,
        "approval_reasons": [] if approved else ["ENSEMBLE_OUT_OF_SAMPLE_GATE_FAILED"],
        "neutralisation": "broad_industry_demean_then_cross_sectional_zscore",
        "monitor_thresholds": {
            "recent_ic_min": 0.01,
            "recent_ir_min": 0.15,
            "decay_ratio_min": 0.20,
            "turnover_max": 0.85,
        },
        "factors": [{**item, "status": "ACTIVE"} for item in active_for_model],
        "ensemble": ensemble,
        "ensemble_validation_metrics": ensemble_metrics,
        "retired_factors": retired,
        "new_replacements": replacements,
        "candidate_count": len(evaluated),
        "train_rows": len(train),
        "validation_rows": len(validate),
        "validation_cutoff": cutoff.strftime("%Y-%m-%d"),
    }


def load_factor_registry(path: str, max_age_days: int = 180) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if int(value.get("schema_version", 0)) != 1:
            return None
        trained_until = pd.to_datetime(value.get("trained_until"), errors="coerce")
        if pd.isna(trained_until):
            return None
        if (pd.Timestamp.now().normalize() - pd.Timestamp(trained_until).normalize()).days > max(1, max_age_days):
            return None
        return dict(value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


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
    "evolve_factor_registry",
    "factor_metrics",
    "industry_neutralise",
    "load_factor_registry",
    "save_factor_registry",
    "seeded_factor_specs",
]
