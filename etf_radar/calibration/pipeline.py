"""Generate purged walk-forward V4 calibration and portfolio acceptance artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .. import _core as main
from ..signals.labeling import build_forward_label
from ..factor_evolution import (
    apply_factor_registry,
    blend_priority,
    build_primitive_row,
    evolve_factor_registry,
    load_factor_registry,
    save_factor_registry,
)
from ..trading import DEFAULT_ETF_COST_MODEL, TradingCostModel
from ..rotation import simulate_staggered_rotation
from ..universe import industry_group
from ..validation import WalkForwardConfig, expanding_walk_forward_splits, split_report
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


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def _atomic_json(value: Mapping[str, Any], path: str) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, default=_json_default)
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


def _analyse_snapshot(
    code: str,
    qfq_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    signal_date: Any,
    calendar: pd.DatetimeIndex,
) -> Optional[Tuple[str, main.ETFAnalyzer, pd.DataFrame]]:
    current = qfq_frame[qfq_frame["date"] <= signal_date].copy()
    if len(current) < 252 or len(qfq_frame[qfq_frame["date"] > signal_date]) < 21:
        return None
    raw_current = raw_frame[raw_frame["date"] <= signal_date].copy()
    analyzer = main.ETFAnalyzer(code, code, market_safe=True, atr_multiplier=2.0)
    try:
        analyzer.set_price_frames(current, raw_current, trading_calendar=calendar)
        result = analyzer.analyze()
    except Exception:
        return None
    if not result:
        return None
    analyzer._v4_result = result  # calibration-only transient state
    return code, analyzer, current


def save_rows_cache(
    rows: List[Dict[str, Any]],
    fingerprint: str,
    sample_step: int,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _atomic_json(
        {
            "schema_version": 1,
            "sample_step": int(sample_step),
            "trained_until": max(row["date"] for row in rows),
            "data_fingerprint": fingerprint,
            "row_count": len(rows),
            "rows": rows,
        },
        path,
    )


def load_rows_cache(
    data_dir: str,
    sample_step: int,
    path: str,
) -> Optional[Tuple[List[Dict[str, Any]], str, Dict[str, pd.DataFrame]]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if int(value.get("schema_version", 0)) != 1 or int(value.get("sample_step", 0)) != int(sample_step):
            return None
        rows = list(value.get("rows", []))
        if not rows or int(value.get("row_count", 0)) != len(rows):
            return None
        qfq, raw = _load_price_pairs(data_dir)
        trained_until = str(value.get("trained_until", ""))
        fingerprint = fingerprint_price_frames(qfq, trained_until)
        if not fingerprint or fingerprint != str(value.get("data_fingerprint", "")):
            return None
        return rows, fingerprint, raw
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


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


def generate_rows(
    data_dir: str,
    sample_step: int = 5,
    max_workers: int = 6,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, pd.DataFrame]]:
    qfq, raw = _load_price_pairs(data_dir)
    benchmark_qfq = qfq[main.Config.DEFAULT_INDEX_CODE]
    benchmark_raw = raw[main.Config.DEFAULT_INDEX_CODE]
    calendar = pd.DatetimeIndex(benchmark_qfq["date"])
    dates = benchmark_qfq["date"].iloc[252:-21:max(1, sample_step)]
    rows: List[Dict[str, Any]] = []
    main.Logger.info = lambda *args, **kwargs: None
    main.Logger.warning = lambda *args, **kwargs: None
    main.Logger.error = lambda *args, **kwargs: None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        for date_index, signal_date in enumerate(dates, start=1):
            current_frames: Dict[str, pd.DataFrame] = {}
            analyzers: Dict[str, main.ETFAnalyzer] = {}
            futures = [
                executor.submit(
                    _analyse_snapshot,
                    code,
                    frame,
                    raw[code],
                    signal_date,
                    calendar,
                )
                for code, frame in qfq.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                snapshot = future.result()
                if snapshot is None:
                    continue
                code, analyzer, current = snapshot
                analyzers[code] = analyzer
                current_frames[code] = current

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
                result["v4_priority"] = final_priority(
                    result.get("v4_factors", {}),
                    result["relative_strength"],
                )
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
                        cost_model=DEFAULT_ETF_COST_MODEL,
                    )
                except ValueError:
                    continue
                factors = v4_calibration_features(result)
                monthly_score = float(
                    (result.get("v4_factors", {}).get("monthly", {}) or {}).get("score", 0.0)
                )
                weekly_score = float(
                    (result.get("v4_factors", {}).get("weekly", {}) or {}).get("score", 0.0)
                )
                setup_score = float(
                    (result.get("v4_factors", {}).get("setup", {}) or {}).get("score", 0.0)
                )
                baseline_candidate = (
                    weekly_score >= 0.25
                    and monthly_score >= -0.15
                    and str(result.get("v4_factors", {}).get("setup", {}).get("setup", "NONE"))
                    in {"BREAKOUT", "PULLBACK"}
                    and setup_score >= 55.0
                    and bool(risk.get("executable", False))
                    and float(result.get("v4_priority", 0.0)) >= 55.0
                )
                primitive = build_primitive_row(
                    code,
                    str(result.get("name", code)),
                    analyzers[code].df_daily,
                    result,
                )
                rows.append(
                    {
                        "date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                        "code": code,
                        "priority": float(result.get("v4_priority", 0.0)),
                        "setup": str(
                            result.get("v4_factors", {}).get("setup", {}).get("setup", "NONE")
                        ),
                        "setup_score_raw": float(
                            result.get("v4_factors", {}).get("setup", {}).get("score", 0.0)
                        ),
                        "risk_executable": bool(risk.get("executable", False)),
                        "stop_loss": float(risk.get("stop_loss", 0.0)),
                        "baseline_candidate": baseline_candidate,
                        **primitive,
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


def walk_forward_predictions(
    rows: List[Dict[str, Any]],
    config: WalkForwardConfig = WalkForwardConfig(),
    evolution_population_size: int = 18,
    evolution_generations: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"])
    predictions: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    previous_registry: Optional[Dict[str, Any]] = None
    splits = expanding_walk_forward_splits(data["date"].unique(), config=config)
    for index, (train_end, validate_start, validate_end, _) in enumerate(splits, start=1):
        name = f"wf-{index:02d}-{validate_start.strftime('%Y%m')}"
        train = data[data["date"] <= train_end].copy()
        validate = data[(data["date"] >= validate_start) & (data["date"] < validate_end)].copy()
        if len(train) < config.min_train_rows or len(validate) < config.min_validate_rows:
            reports.append(split_report(name, train, validate, "INSUFFICIENT"))
            continue
        registry = evolve_factor_registry(
            train.to_dict("records"),
            previous_registry=previous_registry,
            seed=41 + index,
            max_active=4,
            population_size=max(8, evolution_population_size),
            generations=max(1, evolution_generations),
        )
        previous_registry = registry
        train_scored = apply_factor_registry(train, registry)
        validate_scored = apply_factor_registry(validate, registry)
        train_scored["priority_base"] = train_scored["priority"].astype(float)
        validate_scored["priority_base"] = validate_scored["priority"].astype(float)
        train_scored["priority"] = [
            blend_priority(base, adaptive)
            for base, adaptive in zip(train_scored["priority_base"], train_scored["adaptive_score"])
        ]
        validate_scored["priority"] = [
            blend_priority(base, adaptive)
            for base, adaptive in zip(validate_scored["priority_base"], validate_scored["adaptive_score"])
        ]
        model = fit_v4_calibration(
            train_scored.to_dict("records"),
            regularisation=1.0,
            version=f"fold-{name}",
        )
        fold_rows = [_predict(model, row) for row in validate_scored.to_dict("records")]
        predictions.extend(fold_rows)
        report = split_report(name, train, validate, "OK")
        report["factor_registry_approved"] = bool(registry.get("approved", False))
        report["active_factors"] = [
            {
                "name": item.get("name"),
                "economic_logic": item.get("economic_logic"),
                "validation_metrics": item.get("validation_metrics"),
            }
            for item in registry.get("factors", [])
        ]
        report["ensemble_validation_metrics"] = registry.get("ensemble_validation_metrics", {})
        reports.append(report)
    return predictions, reports


def select_thresholds(predictions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    data = pd.DataFrame(list(predictions))
    baseline = data[data["baseline_candidate"].astype(bool)].copy()
    if baseline.empty:
        return {"approved": False, "reason": "NO_BASELINE"}
    baseline_early = float(baseline["early_stop"].mean())
    baseline_excess = float(baseline["excess_return_10d"].mean())
    choices: List[Dict[str, Any]] = []
    for priority_min in (55.0, 60.0, 65.0, 70.0, 75.0):
        for setup_min in (50.0, 55.0, 60.0, 65.0):
            for early_max in (0.18, 0.20, 0.2143, 0.24, 0.26, 0.30, 0.35):
                for expected_excess_min in (-0.002, -0.001, 0.0):
                    selected = data[
                        (data["priority"] >= priority_min)
                        & (data["setup_score_raw"] >= setup_min)
                        & (data["risk_executable"].astype(bool))
                        & (data["early_stop_probability_3d"] <= early_max)
                        & (data["expected_excess_return_10d"] > expected_excess_min)
                    ]
                    retention = len(selected) / len(baseline)
                    if not 0.15 <= retention <= 0.70 or len(selected) < 15:
                        continue
                    early = float(selected["early_stop"].mean())
                    excess = float(selected["excess_return_10d"].mean())
                    choices.append(
                        {
                            "priority_min": priority_min,
                            "setup_score_min": setup_min,
                            "early_stop_probability_max": early_max,
                            "expected_excess_min": expected_excess_min,
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
        if choice["early_stop_reduction"] >= 0.20
        and choice["selected_excess_return_10d"] > 0
        and choice["selected_excess_return_10d"] >= choice["baseline_excess_return_10d"]
        and choice["selected_count"] >= 20
    ]
    if not approved:
        best_attempts = sorted(
            choices,
            key=lambda item: (
                item["early_stop_reduction"],
                item["selected_excess_return_10d"],
                item["selected_count"],
            ),
            reverse=True,
        )[:10]
        return {
            "approved": False,
            "reason": "NO_THRESHOLD_PASSED",
            "baseline_early_stop_rate": round(baseline_early, 6),
            "baseline_excess_return_10d": round(baseline_excess, 8),
            "candidate_count": len(choices),
            "best_attempts": [
                {key: round(value, 8) if isinstance(value, float) else value for key, value in item.items()}
                for item in best_attempts
            ],
        }
    best = min(approved, key=lambda item: (item["selected_early_stop_rate"], -item["selected_excess_return_10d"]))
    return {
        "approved": True,
        **{key: round(value, 8) if isinstance(value, float) else value for key, value in best.items()},
    }


def _selected(frame: pd.DataFrame, thresholds: Mapping[str, Any]) -> pd.DataFrame:
    if not thresholds.get("approved"):
        return frame.iloc[0:0].copy()
    return frame[
        (frame["priority"] >= float(thresholds["priority_min"]))
        & (frame["setup_score_raw"] >= float(thresholds["setup_score_min"]))
        & (frame["risk_executable"].astype(bool))
        & (frame["early_stop_probability_3d"] <= float(thresholds["early_stop_probability_max"]))
        & (frame["expected_excess_return_10d"] > float(thresholds.get("expected_excess_min", 0.0)))
    ].copy()


def simulate_portfolio(
    rows: pd.DataFrame,
    raw_frames: Mapping[str, pd.DataFrame],
    max_positions: int = 3,
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
    benchmark_code: str = "510300",
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    if rows.empty:
        return {
            "max_drawdown": None,
            "return": None,
            "benchmark_return": None,
            "excess_return": None,
            "information_ratio": None,
            "turnover": None,
            "total_cost": 0.0,
            "trade_count": 0,
        }
    frames = {
        code: frame.assign(date=pd.to_datetime(frame["date"])).set_index("date").sort_index()
        for code, frame in raw_frames.items()
    }
    candidates: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for row in rows.to_dict("records"):
        candidates.setdefault(pd.Timestamp(row["entry_date"]), []).append(row)
    first_entry = min(candidates)
    dates = sorted({date for frame in frames.values() for date in frame.index if date >= first_entry})
    cash = float(initial_capital)
    positions: Dict[str, Dict[str, Any]] = {}
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    trade_count = 0
    total_cost = 0.0
    traded_value = 0.0

    def average_amount(code: str, date: pd.Timestamp) -> float:
        history = frames[code].loc[:date].tail(20)
        if "amount" in history.columns:
            return float(history["amount"].astype(float).mean())
        volume = history.get("volume", pd.Series(0.0, index=history.index)).astype(float)
        return float((history["close"].astype(float) * volume).mean())

    def mark_price(code: str, date: pd.Timestamp) -> float:
        history = frames[code].loc[:date]
        return float(history["close"].iloc[-1]) if not history.empty else 0.0

    for date in dates:
        for code in list(positions):
            if code not in frames or date not in frames[code].index:
                continue
            bar = frames[code].loc[date]
            position = positions[code]
            position["bars"] += 1
            exit_reference = None
            if float(bar["low"]) <= position["stop"]:
                # A gap through the stop executes at the worse of open and stop.
                exit_reference = min(float(position["stop"]), float(bar["open"]))
            elif position["bars"] >= 10:
                exit_reference = float(bar["close"])
            if exit_reference is not None:
                execution = cost_model.estimate(
                    "SELL",
                    exit_reference,
                    position["shares"],
                    average_daily_amount=average_amount(code, date),
                )
                cash += float(execution["cash_delta"])
                total_cost += float(execution["total_cost"])
                traded_value += float(execution["gross"])
                del positions[code]
        marked = cash + sum(
            position["shares"] * mark_price(code, date)
            for code, position in positions.items()
            if code in frames
        )
        for row in sorted(candidates.get(date, []), key=lambda item: float(item.get("priority", 0.0)), reverse=True):
            code = str(row["code"])
            if code in positions or len(positions) >= max_positions or code not in frames or date not in frames[code].index:
                continue
            group = str(row.get("industry_group") or industry_group(code, str(row.get("name", ""))))
            group_positions = sum(position["industry_group"] == group for position in positions.values())
            if group_positions >= (2 if group == "other" else 1):
                continue
            entry_reference = float(frames[code].loc[date]["open"])
            stop = float(row["stop_loss"])
            risk_per_share = entry_reference - stop
            if entry_reference <= 0 or risk_per_share <= 0:
                continue
            group_value = sum(
                position["shares"] * mark_price(position_code, date)
                for position_code, position in positions.items()
                if position["industry_group"] == group
            )
            group_room = max(0.0, marked * 0.35 - group_value)
            shares = min(
                marked * 0.01 / risk_per_share,
                marked * 0.25 / entry_reference,
                group_room / entry_reference,
                cash / entry_reference,
            )
            shares = cost_model.round_lot(shares)
            if shares <= 0:
                continue
            execution = cost_model.estimate(
                "BUY",
                entry_reference,
                shares,
                average_daily_amount=average_amount(code, date),
            )
            while shares > 0 and -float(execution["cash_delta"]) > cash:
                shares -= cost_model.lot_size
                execution = cost_model.estimate(
                    "BUY",
                    entry_reference,
                    shares,
                    average_daily_amount=average_amount(code, date),
                )
            if shares <= 0:
                continue
            cash += float(execution["cash_delta"])
            total_cost += float(execution["total_cost"])
            traded_value += float(execution["gross"])
            positions[code] = {
                "shares": shares,
                "stop": stop,
                "bars": 0,
                "industry_group": group,
            }
            trade_count += 1
        equity = cash + sum(
            position["shares"] * mark_price(code, date)
            for code, position in positions.items()
            if code in frames
        )
        equity_curve.append((pd.Timestamp(date), equity))
    if positions and equity_curve:
        last_date = equity_curve[-1][0]
        for code, position in list(positions.items()):
            execution = cost_model.estimate(
                "SELL",
                mark_price(code, last_date),
                position["shares"],
                average_daily_amount=average_amount(code, last_date),
            )
            cash += float(execution["cash_delta"])
            total_cost += float(execution["total_cost"])
            traded_value += float(execution["gross"])
            del positions[code]
        equity_curve[-1] = (last_date, cash)
    curve = pd.Series(
        [value for _, value in equity_curve],
        index=pd.DatetimeIndex([date for date, _ in equity_curve]),
        dtype=float,
    )
    drawdown = curve / curve.cummax() - 1.0
    daily_returns = curve.pct_change().fillna(0.0)
    years = max(len(curve) / 252.0, 1.0 / 252.0)
    strategy_return = float(curve.iloc[-1] / float(initial_capital) - 1.0)
    cagr = float((curve.iloc[-1] / float(initial_capital)) ** (1.0 / years) - 1.0)
    sharpe = float(daily_returns.mean() / max(float(daily_returns.std()), 1e-8) * math.sqrt(252.0))
    benchmark_return = None
    benchmark_cagr = None
    information_ratio = None
    if benchmark_code in frames:
        benchmark_close = frames[benchmark_code]["close"].astype(float).reindex(curve.index).ffill().dropna()
        if len(benchmark_close) >= 2:
            benchmark_return = float(benchmark_close.iloc[-1] / benchmark_close.iloc[0] - 1.0)
            benchmark_cagr = float((1.0 + benchmark_return) ** (1.0 / years) - 1.0)
            aligned_strategy = daily_returns.reindex(benchmark_close.index).fillna(0.0)
            benchmark_daily = benchmark_close.pct_change().fillna(0.0)
            active = aligned_strategy - benchmark_daily
            information_ratio = float(active.mean() / max(float(active.std()), 1e-8) * math.sqrt(252.0))
    return {
        "max_drawdown": round(abs(float(drawdown.min())), 6),
        "return": round(strategy_return, 6),
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 6),
        "benchmark_return": round(benchmark_return, 6) if benchmark_return is not None else None,
        "benchmark_cagr": round(benchmark_cagr, 6) if benchmark_cagr is not None else None,
        "excess_return": round(strategy_return - benchmark_return, 6) if benchmark_return is not None else None,
        "information_ratio": round(information_ratio, 6) if information_ratio is not None else None,
        "turnover": round(traded_value / max(float(curve.mean()), 1.0), 6),
        "total_cost": round(total_cost, 2),
        "cost_ratio": round(total_cost / max(float(initial_capital), 1.0), 8),
        "cost_model": cost_model.to_dict(),
        "industry_constraint": "max_1_position_per_broad_industry_and_35pct_group_exposure",
        "trade_count": trade_count,
    }


def build_artifacts(
    rows: List[Dict[str, Any]],
    fingerprint: str,
    raw_frames: Mapping[str, pd.DataFrame],
    calibration_path: str,
    report_path: str,
    factor_registry_path: Optional[str] = None,
    predictions_path: Optional[str] = None,
    rotation_model_path: Optional[str] = None,
    precomputed_predictions: Optional[List[Dict[str, Any]]] = None,
    precomputed_folds: Optional[List[Dict[str, Any]]] = None,
    reuse_factor_registry: bool = False,
) -> Dict[str, Any]:
    factor_registry_path = factor_registry_path or os.path.join(
        os.path.dirname(calibration_path),
        "adaptive_factor_registry.json",
    )
    rotation_model_path = rotation_model_path or os.path.join(
        os.path.dirname(calibration_path),
        "rotation_model.json",
    )
    if precomputed_predictions is None:
        predictions, folds = walk_forward_predictions(rows)
    else:
        predictions = list(precomputed_predictions)
        folds = list(precomputed_folds or [])
    if predictions_path:
        os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
        _atomic_json(
            {
                "schema_version": 1,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "row_count": len(predictions),
                "rows": predictions,
            },
            predictions_path,
        )
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
            if fold.get("status") != "OK" or not fold.get("validate_start") or not fold.get("validate_end"):
                continue
            name = str(fold["name"])
            start = pd.Timestamp(fold["validate_start"])
            end = pd.Timestamp(fold["validate_end"])
            part = selected[(selected["date"] >= start) & (selected["date"] <= end)]
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
    previous_registry = load_factor_registry(factor_registry_path, max_age_days=3650)
    if reuse_factor_registry:
        if not previous_registry or not bool(previous_registry.get("approved", False)):
            raise ValueError("approved factor registry is required when reuse_factor_registry=True")
        factor_registry = previous_registry
    else:
        factor_registry = evolve_factor_registry(
            rows,
            previous_registry=previous_registry,
            seed=42,
            max_active=5,
            population_size=32,
            generations=5,
        )
        save_factor_registry(factor_registry, factor_registry_path)
    rotation_portfolio = simulate_staggered_rotation(predicted, raw_frames)
    rotation_year_checks = list(rotation_portfolio.get("year_checks", []))
    rotation_approved = bool(
        factor_registry.get("approved")
        and rotation_portfolio.get("excess_return") is not None
        and float(rotation_portfolio["excess_return"]) > 0.0
        and rotation_portfolio.get("information_ratio") is not None
        and float(rotation_portfolio["information_ratio"]) > 0.25
        and rotation_portfolio.get("max_drawdown") is not None
        and rotation_portfolio.get("benchmark_max_drawdown") is not None
        and float(rotation_portfolio["max_drawdown"])
        <= float(rotation_portfolio["benchmark_max_drawdown"])
        and len(rotation_year_checks) >= 5
        and float(rotation_portfolio.get("positive_year_ratio", 0.0)) >= 0.60
    )
    minimum_selected_rows = 30
    eligible_fold_checks = [item for item in fold_checks if item["selected_count"] >= 3]
    stable_folds = [
        item for item in eligible_fold_checks
        if item["selected_count"] >= 3
        and item["excess_return_10d"] is not None
        and item["excess_return_10d"] >= 0.0
    ]
    portfolio_excess = selected_portfolio.get("excess_return")
    portfolio_ir = selected_portfolio.get("information_ratio")
    approved = bool(
        thresholds.get("approved")
        and factor_registry.get("approved")
        and len(selected) >= minimum_selected_rows
        and len(eligible_fold_checks) >= 6
        and len(stable_folds) / len(eligible_fold_checks) >= 0.67
        and portfolio_excess is not None
        and float(portfolio_excess) > 0.0
        and portfolio_ir is not None
        and float(portfolio_ir) > 0.25
        and selected_dd is not None
        and float(selected_dd) <= max(float(baseline_dd or 0.0), 0.20)
    )
    thresholds["approved"] = approved
    thresholds["baseline_max_drawdown"] = baseline_dd
    thresholds["selected_max_drawdown"] = selected_dd
    thresholds["drawdown_reduction"] = round(drawdown_reduction, 6)
    thresholds["benchmark_excess_return"] = portfolio_excess
    thresholds["information_ratio"] = portfolio_ir
    thresholds["eligible_fold_count"] = len(eligible_fold_checks)
    thresholds["stable_fold_ratio"] = (
        round(len(stable_folds) / len(eligible_fold_checks), 6)
        if eligible_fold_checks else 0.0
    )
    thresholds["acceptance_gates"] = {
        "threshold_selection": bool(thresholds.get("priority_min") is not None),
        "factor_registry_oos": bool(factor_registry.get("approved", False)),
        "selected_rows_minimum": minimum_selected_rows,
        "selected_rows_gate": bool(len(selected) >= minimum_selected_rows),
        "eligible_folds_min_6": bool(len(eligible_fold_checks) >= 6),
        "stable_fold_ratio_min_0_67": bool(
            eligible_fold_checks and len(stable_folds) / len(eligible_fold_checks) >= 0.67
        ),
        "cost_after_benchmark_excess_positive": bool(portfolio_excess is not None and float(portfolio_excess) > 0.0),
        "information_ratio_above_0_25": bool(portfolio_ir is not None and float(portfolio_ir) > 0.25),
        "max_drawdown_within_limit": bool(
            selected_dd is not None and float(selected_dd) <= max(float(baseline_dd or 0.0), 0.20)
        ),
    }
    trained_until = max(row["date"] for row in rows)
    version = f"v4-{datetime.now().strftime('%Y%m%d')}-{fingerprint[:8]}"
    final_training = apply_factor_registry(pd.DataFrame(rows), factor_registry)
    final_training["priority"] = [
        blend_priority(base, adaptive)
        for base, adaptive in zip(final_training["priority"], final_training["adaptive_score"])
    ]
    model = fit_v4_calibration(
        final_training.to_dict("records"),
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
        "walk_forward_method": "expanding_calendar_windows_with_20d_purge_and_5d_embargo",
        "factor_registry_path": factor_registry_path,
        "factor_registry_approved": bool(factor_registry.get("approved", False)),
        "strategy_approved": rotation_approved,
        "factor_monitor": {
            "ensemble_validation_metrics": factor_registry.get("ensemble_validation_metrics", {}),
            "retired_factors": factor_registry.get("retired_factors", []),
            "new_replacements": factor_registry.get("new_replacements", []),
        },
        "row_count": len(rows),
        "prediction_diagnostics": {
            "prediction_count": int(len(predicted)),
            "baseline_candidate_count": int(len(baseline)),
            "early_stop_probability_quantiles": (
                {
                    str(quantile): round(float(predicted["early_stop_probability_3d"].quantile(quantile)), 8)
                    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
                }
                if not predicted.empty else {}
            ),
            "expected_excess_quantiles": (
                {
                    str(quantile): round(float(predicted["expected_excess_return_10d"].quantile(quantile)), 8)
                    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
                }
                if not predicted.empty else {}
            ),
        },
        "folds": folds,
        "fold_checks": fold_checks,
        "thresholds": thresholds,
        "baseline_portfolio": baseline_portfolio,
        "selected_portfolio": selected_portfolio,
        "rotation_portfolio": {
            key: value
            for key, value in rotation_portfolio.items()
            if key != "period_records"
        },
        "rotation_acceptance_gates": {
            "factor_registry_oos": bool(factor_registry.get("approved", False)),
            "benchmark_excess_positive": bool(
                rotation_portfolio.get("excess_return") is not None
                and float(rotation_portfolio["excess_return"]) > 0.0
            ),
            "information_ratio_above_0_25": bool(
                rotation_portfolio.get("information_ratio") is not None
                and float(rotation_portfolio["information_ratio"]) > 0.25
            ),
            "drawdown_not_worse_than_benchmark": bool(
                rotation_portfolio.get("max_drawdown") is not None
                and rotation_portfolio.get("benchmark_max_drawdown") is not None
                and float(rotation_portfolio["max_drawdown"])
                <= float(rotation_portfolio["benchmark_max_drawdown"])
            ),
            "positive_year_ratio_min_0_60": bool(
                len(rotation_year_checks) >= 5
                and float(rotation_portfolio.get("positive_year_ratio", 0.0)) >= 0.60
            ),
        },
    }
    _atomic_json(
        {
            "schema_version": 1,
            "version": f"rotation-{version}",
            "generated_at": report["generated_at"],
            "trained_until": trained_until,
            "data_fingerprint": fingerprint,
            "approved": rotation_approved,
            "approval_gates": report["rotation_acceptance_gates"],
            "portfolio_metrics": report["rotation_portfolio"],
            "top_n": int(rotation_portfolio.get("top_n", 3)),
            "sleeve_count": int(rotation_portfolio.get("sleeve_count", 2)),
            "holding_period_trading_days": int(
                rotation_portfolio.get("holding_period_trading_days", 10)
            ),
            "weekly_trend_min": float(rotation_portfolio.get("weekly_trend_min", -0.25)),
            "factor_weights": dict(rotation_portfolio.get("factor_weights", {})),
            "factor_economic_logic": dict(
                rotation_portfolio.get("factor_economic_logic", {})
            ),
            "industry_constraint": rotation_portfolio.get("industry_constraint"),
            "cost_model": rotation_portfolio.get("cost_model", {}),
        },
        rotation_model_path,
    )
    _atomic_json(model.to_dict(), calibration_path)
    _atomic_json(report, report_path)
    return report


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=main.Config.DATA_DIR)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--rows-cache",
        default=os.path.normpath(os.path.join(main.Config.DATA_DIR, "..", "state", "v4_calibration_rows.json")),
    )
    parser.add_argument("--reuse-rows-cache", action="store_true")
    parser.add_argument("--calibration-out", default=main.Config.V4_CALIBRATION_FILE)
    parser.add_argument(
        "--report-out",
        default=os.path.join(os.path.dirname(main.Config.V4_CALIBRATION_FILE), "v4_acceptance_report.json"),
    )
    parser.add_argument(
        "--factor-registry-out",
        default=main.Config.FACTOR_REGISTRY_FILE,
    )
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--predictions-cache", default="")
    parser.add_argument("--reuse-predictions-cache", action="store_true")
    parser.add_argument("--reuse-factor-registry", action="store_true")
    parser.add_argument(
        "--rotation-model-out",
        default=os.path.join(os.path.dirname(main.Config.V4_CALIBRATION_FILE), "rotation_model.json"),
    )
    args = parser.parse_args()
    sample_step = max(1, args.sample_step)
    cached = (
        load_rows_cache(args.data_dir, sample_step, args.rows_cache)
        if args.reuse_rows_cache else None
    )
    if cached is None:
        rows, fingerprint, raw_frames = generate_rows(
            args.data_dir,
            sample_step=sample_step,
            max_workers=max(1, args.workers),
        )
        save_rows_cache(rows, fingerprint, sample_step, args.rows_cache)
    else:
        rows, fingerprint, raw_frames = cached
        print(f"reused calibration row cache: {args.rows_cache} ({len(rows)} rows)", flush=True)
    cached_predictions = None
    cached_folds = None
    if args.reuse_predictions_cache:
        if not args.predictions_cache:
            raise ValueError("--predictions-cache is required with --reuse-predictions-cache")
        with open(args.predictions_cache, "r", encoding="utf-8") as handle:
            prediction_payload = json.load(handle)
        cached_predictions = list(prediction_payload.get("rows", []))
        if not cached_predictions or int(prediction_payload.get("row_count", 0)) != len(cached_predictions):
            raise ValueError("predictions cache is empty or inconsistent")
        if os.path.isfile(args.report_out):
            with open(args.report_out, "r", encoding="utf-8") as handle:
                cached_folds = list(json.load(handle).get("folds", []))
        print(
            f"reused walk-forward predictions: {args.predictions_cache} ({len(cached_predictions)} rows)",
            flush=True,
        )
    report = build_artifacts(
        rows,
        fingerprint,
        raw_frames,
        args.calibration_out,
        args.report_out,
        factor_registry_path=args.factor_registry_out,
        predictions_path=args.predictions_out or None,
        rotation_model_path=args.rotation_model_out,
        precomputed_predictions=cached_predictions,
        precomputed_folds=cached_folds,
        reuse_factor_registry=args.reuse_factor_registry,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
