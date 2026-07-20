"""Generate purged walk-forward V4 calibration and portfolio acceptance artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import _core as main
from ..signals.labeling import build_forward_label
from ..factor_evolution import (
    FACTOR_EVOLUTION_POLICY_VERSION,
    PRIMITIVE_FEATURES,
    PurgedHoldoutInsufficientError,
    apply_factor_registry,
    blend_priority,
    build_primitive_row,
    evolve_factor_registry,
    load_factor_registry,
    sanitize_factor_registry,
    save_factor_registry,
)
from ..llm_factor_proposals import load_or_generate_llm_proposals
from ..trading import DEFAULT_ETF_COST_MODEL, TradingCostModel
from ..rotation import (
    ROTATION_ACCEPTANCE_POLICY_VERSION,
    ROTATION_EXECUTION_POLICY_VERSION,
    simulate_staggered_rotation,
)
from ..universe import industry_group
from ..validation import WalkForwardConfig, expanding_walk_forward_splits, split_report
from ..signals.contract import (
    V4CalibrationModel,
    fit_v4_calibration,
    fingerprint_joint_price_frames,
    v4_calibration_features,
)
from ..signals.factors import (
    final_priority,
    market_policy,
    normalised_atr_percentile,
    rank_relative_strength,
    weekly_trend_factor,
)

CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION = "qfq-raw-joint-v2"


def cost_model_fingerprint(cost_model: TradingCostModel) -> str:
    return hashlib.sha256(
        json.dumps(
            cost_model.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def prediction_cache_compatible(
    payload: Mapping[str, Any],
    data_fingerprint: str,
    calibration_row_count: int,
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
) -> bool:
    return bool(
        int(payload.get("schema_version", 0) or 0) == 3
        and payload.get("data_fingerprint_policy")
        == CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION
        and str(payload.get("data_fingerprint", "")) == str(data_fingerprint)
        and int(payload.get("calibration_row_count", 0) or 0)
        == int(calibration_row_count)
        and dict(payload.get("cost_model") or {}) == cost_model.to_dict()
        and str(payload.get("cost_model_fingerprint", ""))
        == cost_model_fingerprint(cost_model)
    )


def load_cost_model_candidate(path: str) -> Tuple[TradingCostModel, Dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cost candidate must be a JSON object")
    if value.get("status") != "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION":
        raise ValueError("cost candidate is not ready for shadow validation")
    if (
        value.get("approved_for_live_use") is not False
        or value.get("auto_promotion_allowed") is not False
        or value.get("requires_full_purged_walk_forward") is not True
    ):
        raise ValueError("cost candidate governance flags are invalid")
    candidate_policy = str(value.get("candidate_execution_policy_version", ""))
    candidate_fingerprint = str(value.get("candidate_fingerprint", ""))
    if not candidate_policy or not (
        len(candidate_fingerprint) == 64
        and all(char in "0123456789abcdef" for char in candidate_fingerprint.lower())
    ):
        raise ValueError("cost candidate identity is incomplete")
    raw_model = value.get("recommended_cost_model")
    required = set(TradingCostModel.__dataclass_fields__)
    if not isinstance(raw_model, dict) or set(raw_model) != required:
        raise ValueError("recommended cost model fields do not match TradingCostModel")
    try:
        model = TradingCostModel(**raw_model)
    except (TypeError, ValueError) as error:
        raise ValueError("recommended cost model is invalid") from error
    numeric = model.to_dict()
    if any(float(numeric[field]) < 0.0 for field in required - {"lot_size"}):
        raise ValueError("recommended cost model contains negative values")
    if not 0.0 < float(model.max_participation_rate) <= 0.20:
        raise ValueError("recommended max participation rate is outside research bounds")
    if int(model.lot_size) <= 0:
        raise ValueError("recommended lot size must be positive")
    return model, value


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def rotation_model_identity(
    data_fingerprint: str,
    portfolio: Mapping[str, Any],
    generated_at: str,
    execution_policy_version: str = ROTATION_EXECUTION_POLICY_VERSION,
) -> Tuple[str, str]:
    """Version rotation models by both market data and executable strategy rules."""
    specification = {
        "execution_policy_version": execution_policy_version,
        "factor_evolution_policy_version": FACTOR_EVOLUTION_POLICY_VERSION,
        "acceptance_policy_version": ROTATION_ACCEPTANCE_POLICY_VERSION,
        "top_n": int(portfolio.get("top_n", 3)),
        "sleeve_count": int(portfolio.get("sleeve_count", 2)),
        "holding_period_trading_days": int(portfolio.get("holding_period_trading_days", 10)),
        "weekly_trend_min": float(portfolio.get("weekly_trend_min", -0.25)),
        "exposure_authority": str(portfolio.get("exposure_authority", "")),
        "rank_buffer": int(portfolio.get("rank_buffer", 0)),
        "factor_weights": dict(portfolio.get("factor_weights", {})),
        "industry_constraint": portfolio.get("industry_constraint"),
        "cost_model": dict(portfolio.get("cost_model", {})),
        "capacity_audit_version": "formal-capacity-metrics-v1",
        "capacity_reference_capital": float(
            portfolio.get("capacity_reference_capital", 10_000.0)
        ),
        "capacity_selection_policy": portfolio.get("capacity_selection_policy"),
    }
    encoded = json.dumps(
        specification,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    specification_fingerprint = hashlib.sha256(encoded).hexdigest()
    stamp = str(generated_at)[:10].replace("-", "")
    version = (
        f"rotation-v2-{stamp}-{str(data_fingerprint)[:8]}-"
        f"{specification_fingerprint[:8]}"
    )
    return version, specification_fingerprint


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
    eligible_signal_until: str,
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _atomic_json(
        {
            "schema_version": 4,
            "data_fingerprint_policy": CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION,
            "sample_step": int(sample_step),
            "trained_until": max(row["date"] for row in rows),
            "eligible_signal_until": str(eligible_signal_until)[:10],
            "cost_model": cost_model.to_dict(),
            "cost_model_fingerprint": cost_model_fingerprint(cost_model),
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
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
) -> Optional[Tuple[List[Dict[str, Any]], str, Dict[str, pd.DataFrame]]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if (
            int(value.get("schema_version", 0)) != 4
            or value.get("data_fingerprint_policy")
            != CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION
            or int(value.get("sample_step", 0)) != int(sample_step)
        ):
            return None
        if (
            dict(value.get("cost_model") or {}) != cost_model.to_dict()
            or str(value.get("cost_model_fingerprint", ""))
            != cost_model_fingerprint(cost_model)
        ):
            return None
        rows = list(value.get("rows", []))
        if not rows or int(value.get("row_count", 0)) != len(rows):
            return None
        qfq, raw = _load_price_pairs(data_dir)
        eligible_signal_until = _latest_eligible_signal_date(qfq, sample_step)
        if (
            not eligible_signal_until
            or str(value.get("eligible_signal_until", ""))[:10]
            != eligible_signal_until
        ):
            return None
        trained_until = str(value.get("trained_until", ""))
        fingerprint = calibration_data_fingerprint(qfq, raw, trained_until)
        if not fingerprint or fingerprint != str(value.get("data_fingerprint", "")):
            return None
        return rows, fingerprint, raw
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _latest_eligible_signal_date(
    qfq_frames: Mapping[str, pd.DataFrame],
    sample_step: int,
) -> str:
    """Return the latest sampled date with the full 21-day forward-label horizon."""
    benchmark = qfq_frames.get(main.Config.DEFAULT_INDEX_CODE)
    if benchmark is None or "date" not in benchmark or len(benchmark) < 274:
        return ""
    dates = pd.to_datetime(benchmark["date"], errors="coerce").dropna().sort_values()
    eligible = dates.iloc[252:-21:max(1, int(sample_step))]
    if eligible.empty:
        return ""
    return pd.Timestamp(eligible.iloc[-1]).strftime("%Y-%m-%d")


def calibration_data_fingerprint(
    qfq_frames: Mapping[str, pd.DataFrame],
    raw_frames: Mapping[str, pd.DataFrame],
    trained_until: Any,
) -> str:
    """Bind calibration evidence to both analysis and executable price bases."""
    return fingerprint_joint_price_frames(
        qfq_frames,
        raw_frames,
        trained_until,
        policy=CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION,
    )


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
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
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
                        cost_model=cost_model,
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
    return rows, calibration_data_fingerprint(qfq, raw, trained_until), raw


def _predict(model: V4CalibrationModel, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {**dict(row), **model.predict(row)}


def _apply_approved_adaptive_priority(
    frame: pd.DataFrame,
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Match live behavior: an unapproved registry must not alter base priority."""
    scored = apply_factor_registry(frame, registry)
    scored["priority_base"] = scored["priority"].astype(float)
    approved = bool(registry.get("approved", False))
    if approved:
        scored["priority"] = [
            blend_priority(base, adaptive)
            for base, adaptive in zip(scored["priority_base"], scored["adaptive_score"])
        ]
    else:
        scored["priority"] = scored["priority_base"]
    scored["adaptive_factor_applied"] = approved
    return scored


def _rotation_acceptance_gates(
    portfolio: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> Dict[str, bool]:
    year_checks = list(portfolio.get("year_checks", []))
    holdout_years = list(holdout.get("year_checks", []))
    return {
        "benchmark_excess_positive": bool(
            portfolio.get("excess_return") is not None
            and float(portfolio["excess_return"]) > 0.0
        ),
        "information_ratio_above_0_25": bool(
            portfolio.get("information_ratio") is not None
            and float(portfolio["information_ratio"]) > 0.25
        ),
        "drawdown_not_worse_than_benchmark": bool(
            portfolio.get("max_drawdown") is not None
            and portfolio.get("benchmark_max_drawdown") is not None
            and float(portfolio["max_drawdown"]) <= float(portfolio["benchmark_max_drawdown"])
        ),
        "absolute_max_drawdown_at_most_0_25": bool(
            portfolio.get("max_drawdown") is not None
            and float(portfolio["max_drawdown"]) <= 0.25
        ),
        "positive_year_ratio_min_0_60": bool(
            len(year_checks) >= 5
            and float(portfolio.get("positive_year_ratio", 0.0)) >= 0.60
        ),
        "rolling_12m_observations_min_104": bool(
            int(portfolio.get("rolling_12m_observations", 0)) >= 104
        ),
        "rolling_12m_positive_excess_ratio_min_0_60": bool(
            float(portfolio.get("rolling_12m_positive_excess_ratio", 0.0)) >= 0.60
        ),
        "rolling_12m_worst_excess_at_least_minus_0_25": bool(
            float(portfolio.get("rolling_12m_worst_excess_return", -1.0)) >= -0.25
        ),
        "max_relative_drawdown_at_most_0_30": bool(
            float(portfolio.get("max_relative_drawdown", 1.0)) <= 0.30
        ),
        "relative_underwater_periods_at_most_260": bool(
            int(portfolio.get("longest_relative_underwater_periods", 10**9)) <= 260
        ),
        "recent_holdout_excess_positive": bool(
            holdout.get("excess_return") is not None
            and float(holdout["excess_return"]) > 0.0
        ),
        "recent_holdout_information_ratio_above_0_25": bool(
            holdout.get("information_ratio") is not None
            and float(holdout["information_ratio"]) > 0.25
        ),
        "recent_holdout_max_drawdown_at_most_0_25": bool(
            holdout.get("max_drawdown") is not None
            and float(holdout["max_drawdown"]) <= 0.25
        ),
        "recent_holdout_positive_year_ratio_min_0_50": bool(
            len(holdout_years) >= 2
            and float(holdout.get("positive_year_ratio", 0.0)) >= 0.50
        ),
        "recent_holdout_rolling_12m_observations_min_26": bool(
            int(holdout.get("rolling_12m_observations", 0)) >= 26
        ),
        "recent_holdout_rolling_positive_excess_ratio_min_0_60": bool(
            float(holdout.get("rolling_12m_positive_excess_ratio", 0.0)) >= 0.60
        ),
        "recent_holdout_rolling_worst_excess_at_least_minus_0_20": bool(
            float(holdout.get("rolling_12m_worst_excess_return", -1.0)) >= -0.20
        ),
        "recent_holdout_max_relative_drawdown_at_most_0_25": bool(
            float(holdout.get("max_relative_drawdown", 1.0)) <= 0.25
        ),
        "recent_holdout_relative_underwater_periods_at_most_104": bool(
            int(holdout.get("longest_relative_underwater_periods", 10**9)) <= 104
        ),
        "capacity_fill_ratio_min_0_90": bool(
            float(portfolio.get("capacity_fill_ratio", 0.0)) >= 0.90
        ),
        "recent_holdout_capacity_fill_ratio_min_0_90": bool(
            float(holdout.get("capacity_fill_ratio", 0.0)) >= 0.90
        ),
    }


def walk_forward_predictions(
    rows: List[Dict[str, Any]],
    config: WalkForwardConfig = WalkForwardConfig(),
    evolution_population_size: int = 18,
    evolution_generations: int = 3,
    llm_candidates: Sequence[Mapping[str, Any]] = (),
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
        try:
            registry = evolve_factor_registry(
                train.to_dict("records"),
                previous_registry=previous_registry,
                llm_candidates=llm_candidates,
                seed=41 + index,
                max_active=4,
                population_size=max(8, evolution_population_size),
                generations=max(1, evolution_generations),
            )
        except PurgedHoldoutInsufficientError:
            reports.append(split_report(name, train, validate, "FACTOR_PURGE_INSUFFICIENT"))
            continue
        previous_registry = registry
        train_scored = _apply_approved_adaptive_priority(train, registry)
        validate_scored = _apply_approved_adaptive_priority(validate, registry)
        model = fit_v4_calibration(
            train_scored.to_dict("records"),
            regularisation=1.0,
            version=f"fold-{name}",
        )
        fold_rows = [_predict(model, row) for row in validate_scored.to_dict("records")]
        predictions.extend(fold_rows)
        report = split_report(name, train, validate, "OK")
        report["factor_registry_approved"] = bool(registry.get("approved", False))
        report["adaptive_factor_applied"] = bool(registry.get("approved", False))
        report["factor_purge_method"] = registry.get("purge_method")
        report["factor_train_cutoff"] = registry.get("train_cutoff")
        report["factor_selection_cutoff"] = registry.get("selection_cutoff")
        report["llm_proposals_considered"] = int(
            registry.get("llm_proposals_considered", 0)
        )
        report["llm_proposals_selected"] = list(
            registry.get("llm_proposals_selected", [])
        )
        report["llm_research_challengers"] = list(
            registry.get("llm_research_challengers", [])
        )
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


def _compact_rotation_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "return",
        "benchmark_return",
        "excess_return",
        "information_ratio",
        "max_drawdown",
        "benchmark_max_drawdown",
        "turnover",
        "total_cost",
        "cost_ratio",
        "buy_order_count",
        "capacity_truncation_count",
        "requested_buy_value",
        "capacity_executable_buy_value",
        "executed_buy_value",
        "capacity_truncated_buy_value",
        "cash_limited_buy_value",
        "unfilled_buy_value",
        "buy_fill_ratio",
        "capacity_fill_ratio",
        "max_requested_participation_rate",
        "max_executed_participation_rate",
        "rebalance_count",
        "skipped_unchanged_rebalances",
        "positive_year_ratio",
        "rolling_12m_observations",
        "rolling_12m_positive_excess_ratio",
        "rolling_12m_median_excess_return",
        "rolling_12m_worst_excess_return",
        "max_relative_drawdown",
        "longest_relative_underwater_periods",
        "year_checks",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def choose_rotation_rank_buffer(
    candidate_metrics: Mapping[int, Mapping[str, Any]],
) -> int:
    """Choose on development data only; prefer robust IR, then excess and turnover."""
    eligible = [
        int(buffer)
        for buffer, metrics in candidate_metrics.items()
        if metrics.get("excess_return") is not None
        and float(metrics["excess_return"]) > 0.0
        and metrics.get("information_ratio") is not None
        and float(metrics["information_ratio"]) > 0.0
        and metrics.get("max_drawdown") is not None
        and float(metrics["max_drawdown"]) <= 0.25
    ]
    pool = eligible or [int(buffer) for buffer in candidate_metrics]
    if not pool:
        return 0
    return max(
        pool,
        key=lambda buffer: (
            float(candidate_metrics[buffer].get("information_ratio") or -999.0),
            float(candidate_metrics[buffer].get("excess_return") or -999.0),
            -float(candidate_metrics[buffer].get("turnover") or 1e9),
            -int(buffer),
        ),
    )


def select_rotation_rank_buffer(
    rows: pd.DataFrame,
    raw_frames: Mapping[str, pd.DataFrame],
    candidates: Tuple[int, ...] = (0, 1, 2, 3),
    holdout_years: int = 3,
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
) -> Tuple[int, Dict[str, Any]]:
    """Select hysteresis on pre-holdout OOS data and audit it on untouched recent years."""
    if rows.empty:
        return 0, {"status": "NO_ROWS", "selected_rank_buffer": 0}
    data = rows.copy()
    date_column = "entry_date" if "entry_date" in data.columns else "date"
    data[date_column] = pd.to_datetime(data[date_column])
    last_year = int(data[date_column].max().year)
    holdout_start = pd.Timestamp(year=last_year - max(1, int(holdout_years)) + 1, month=1, day=1)
    development = data[data[date_column] < holdout_start].copy()
    holdout = data[data[date_column] >= holdout_start].copy()
    if development.empty or holdout.empty:
        return 0, {
            "status": "INSUFFICIENT_SPLIT",
            "selected_rank_buffer": 0,
            "holdout_start": holdout_start.strftime("%Y-%m-%d"),
        }
    development_metrics: Dict[int, Dict[str, Any]] = {}
    for buffer in candidates:
        development_metrics[int(buffer)] = _compact_rotation_metrics(
            simulate_staggered_rotation(
                development,
                raw_frames,
                rank_buffer=int(buffer),
                cost_model=cost_model,
            )
        )
    selected = choose_rotation_rank_buffer(development_metrics)
    holdout_metrics = _compact_rotation_metrics(
        simulate_staggered_rotation(
            holdout,
            raw_frames,
            rank_buffer=selected,
            cost_model=cost_model,
        )
    )
    return selected, {
        "status": "OK",
        "method": "development_oos_selection_then_recent_3y_holdout",
        "holdout_start": holdout_start.strftime("%Y-%m-%d"),
        "selected_rank_buffer": int(selected),
        "development_candidates": {
            str(buffer): metrics for buffer, metrics in sorted(development_metrics.items())
        },
        "holdout_metrics": holdout_metrics,
    }


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
    cost_model: TradingCostModel = DEFAULT_ETF_COST_MODEL,
    execution_policy_version: str = ROTATION_EXECUTION_POLICY_VERSION,
    shadow_cost_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    factor_registry_path = factor_registry_path or os.path.join(
        os.path.dirname(calibration_path),
        "adaptive_factor_registry.json",
    )
    rotation_model_path = rotation_model_path or os.path.join(
        os.path.dirname(calibration_path),
        "rotation_model.json",
    )
    previous_registry = load_factor_registry(
        factor_registry_path,
        max_age_days=3650,
        generated_max_age_days=3650,
    )
    llm_artifact_path = Path(os.path.dirname(factor_registry_path)) / "llm_factor_proposals.json"
    try:
        llm_proposal_audit = load_or_generate_llm_proposals(
            PRIMITIVE_FEATURES,
            {},
            llm_artifact_path,
        )
    except Exception as error:
        llm_proposal_audit = {
            "status": "ERROR",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "proposals": [],
            "rejected": [],
            "error": str(error)[:1000],
        }
        _atomic_json(llm_proposal_audit, str(llm_artifact_path))
    llm_candidates = list(llm_proposal_audit.get("proposals", []))
    if not llm_candidates:
        llm_walk_forward_status = "NOT_APPLICABLE_NO_VALID_PROPOSALS"
    elif reuse_factor_registry:
        llm_walk_forward_status = "DEFERRED_REUSE_FACTOR_REGISTRY"
    elif precomputed_predictions is None:
        llm_walk_forward_status = "APPLIED_TO_ALL_WALK_FORWARD_FOLDS"
    else:
        llm_walk_forward_status = "DEFERRED_REQUIRES_FULL_WALK_FORWARD_RERUN"
    eligible_llm_candidates = (
        llm_candidates if precomputed_predictions is None and not reuse_factor_registry else []
    )
    if precomputed_predictions is None:
        predictions, folds = walk_forward_predictions(
            rows,
            llm_candidates=eligible_llm_candidates,
        )
    else:
        predictions = list(precomputed_predictions)
        folds = list(precomputed_folds or [])
    if predictions_path:
        os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
        _atomic_json(
            {
                "schema_version": 3,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_fingerprint": fingerprint,
                "data_fingerprint_policy": CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION,
                "cost_model": cost_model.to_dict(),
                "cost_model_fingerprint": cost_model_fingerprint(cost_model),
                "calibration_row_count": len(rows),
                "row_count": len(predictions),
                "llm_walk_forward_status": llm_walk_forward_status,
                "llm_prompt_version": llm_proposal_audit.get("prompt_version"),
                "llm_expression_signatures": [
                    (item.get("proposal_metadata") or {}).get("expression_signature")
                    for item in eligible_llm_candidates
                ],
                "folds": folds,
                "rows": predictions,
            },
            predictions_path,
        )
    thresholds = select_thresholds(predictions) if predictions else {"approved": False, "reason": "NO_PREDICTIONS"}
    predicted = pd.DataFrame(predictions)
    baseline = predicted[predicted["baseline_candidate"].astype(bool)].copy() if not predicted.empty else predicted
    selected = _selected(predicted, thresholds) if not predicted.empty else predicted
    baseline_portfolio = simulate_portfolio(
        baseline, raw_frames, cost_model=cost_model
    )
    selected_portfolio = simulate_portfolio(
        selected, raw_frames, cost_model=cost_model
    )
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
    if reuse_factor_registry:
        if not previous_registry:
            raise ValueError("factor registry is required when reuse_factor_registry=True")
        factor_registry = sanitize_factor_registry(previous_registry)
    else:
        factor_registry = evolve_factor_registry(
            rows,
            previous_registry=previous_registry,
            llm_candidates=eligible_llm_candidates,
            seed=42,
            max_active=5,
            population_size=32,
            generations=5,
            require_policy_seasoning=True,
        )
    factor_registry["llm_proposal_audit"] = {
        "status": llm_proposal_audit.get("status"),
        "model": llm_proposal_audit.get("model"),
        "provider": llm_proposal_audit.get("provider"),
        "model_identity": llm_proposal_audit.get("model_identity"),
        "endpoint_fingerprint": llm_proposal_audit.get("endpoint_fingerprint"),
        "prompt_version": llm_proposal_audit.get("prompt_version"),
        "proposal_count": len(llm_candidates),
        "walk_forward_status": llm_walk_forward_status,
        "rejected_count": len(llm_proposal_audit.get("rejected", [])),
        "artifact_path": str(llm_artifact_path),
    }
    save_factor_registry(factor_registry, factor_registry_path)
    selected_rank_buffer, rotation_selection = select_rotation_rank_buffer(
        predicted,
        raw_frames,
        cost_model=cost_model,
    )
    rotation_portfolio = simulate_staggered_rotation(
        predicted,
        raw_frames,
        rank_buffer=selected_rank_buffer,
        cost_model=cost_model,
    )
    rotation_holdout = dict(rotation_selection.get("holdout_metrics", {}))
    rotation_gates = _rotation_acceptance_gates(rotation_portfolio, rotation_holdout)
    rotation_approved = bool(all(rotation_gates.values()))
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
    final_training = _apply_approved_adaptive_priority(
        pd.DataFrame(rows), factor_registry
    )
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
        "trained_until": trained_until,
        "data_fingerprint": fingerprint,
        "data_fingerprint_policy": CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION,
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
            "research_challengers": factor_registry.get("research_challengers", []),
        },
        "llm_factor_proposals": {
            "status": llm_proposal_audit.get("status"),
            "model": llm_proposal_audit.get("model"),
            "provider": llm_proposal_audit.get("provider"),
            "model_identity": llm_proposal_audit.get("model_identity"),
            "endpoint_fingerprint": llm_proposal_audit.get("endpoint_fingerprint"),
            "prompt_version": llm_proposal_audit.get("prompt_version"),
            "proposal_count": len(llm_candidates),
            "submitted_count": int(
                factor_registry.get("llm_proposals_submitted", len(llm_candidates))
                or 0
            ),
            "considered_count": int(
                factor_registry.get("llm_proposals_considered", 0) or 0
            ),
            "skipped_rejected_cooldown": list(
                factor_registry.get(
                    "llm_proposals_skipped_rejected_cooldown", []
                )
                or []
            ),
            "candidate_trial_history_count": len(
                factor_registry.get("llm_candidate_trial_history", []) or []
            ),
            "rejected_candidate_cooldown_days": int(
                factor_registry.get("llm_rejected_candidate_cooldown_days", 0)
                or 0
            ),
            "walk_forward_status": llm_walk_forward_status,
            "selected_factors": factor_registry.get("llm_proposals_selected", []),
            "research_challengers": factor_registry.get(
                "llm_research_challengers", []
            ),
            "rejected_count": len(llm_proposal_audit.get("rejected", [])),
            "artifact_path": str(llm_artifact_path),
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
        "rotation_model_selection": rotation_selection,
        "rotation_acceptance_gates": rotation_gates,
        "rotation_adaptive_context": {
            "adaptive_factor_registry_oos": bool(factor_registry.get("approved", False)),
            "adaptive_overlay_applied_in_walk_forward": bool(factor_registry.get("approved", False)),
            "independent_rotation_authority": True,
        },
        "execution_policy_version": execution_policy_version,
        "cost_model_fingerprint": cost_model_fingerprint(cost_model),
        "shadow_cost_validation": dict(shadow_cost_context or {}),
    }
    rotation_version, rotation_specification_fingerprint = rotation_model_identity(
        fingerprint,
        report["rotation_portfolio"],
        report["generated_at"],
        execution_policy_version=execution_policy_version,
    )
    artifact_bundle_id = hashlib.sha256(
        "|".join(
            [
                fingerprint,
                report["generated_at"],
                version,
                rotation_version,
            ]
        ).encode("utf-8")
    ).hexdigest()[:24]
    report["artifact_bundle_id"] = artifact_bundle_id
    factor_registry["artifact_bundle_id"] = artifact_bundle_id
    factor_registry["data_fingerprint"] = fingerprint
    factor_registry["data_fingerprint_policy"] = CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION
    factor_registry["trained_until"] = trained_until
    save_factor_registry(factor_registry, factor_registry_path)
    _atomic_json(
        {
            "schema_version": 1,
            "artifact_bundle_id": artifact_bundle_id,
            "version": rotation_version,
            "generated_at": report["generated_at"],
            "trained_until": trained_until,
            "data_fingerprint": fingerprint,
            "execution_policy_version": execution_policy_version,
            "acceptance_policy_version": ROTATION_ACCEPTANCE_POLICY_VERSION,
            "factor_evolution_policy_version": FACTOR_EVOLUTION_POLICY_VERSION,
            "strategy_specification_fingerprint": rotation_specification_fingerprint,
            "approved": rotation_approved,
            "approval_gates": report["rotation_acceptance_gates"],
            "portfolio_metrics": report["rotation_portfolio"],
            "top_n": int(rotation_portfolio.get("top_n", 3)),
            "sleeve_count": int(rotation_portfolio.get("sleeve_count", 2)),
            "holding_period_trading_days": int(
                rotation_portfolio.get("holding_period_trading_days", 10)
            ),
            "weekly_trend_min": float(rotation_portfolio.get("weekly_trend_min", -0.25)),
            "exposure_authority": str(rotation_portfolio.get("exposure_authority", "")),
            "rank_buffer": int(rotation_portfolio.get("rank_buffer", 0)),
            "selection_protocol": {
                "method": rotation_selection.get("method"),
                "holdout_start": rotation_selection.get("holdout_start"),
                "selected_rank_buffer": rotation_selection.get("selected_rank_buffer"),
                "holdout_metrics": rotation_selection.get("holdout_metrics", {}),
            },
            "factor_weights": dict(rotation_portfolio.get("factor_weights", {})),
            "factor_economic_logic": dict(
                rotation_portfolio.get("factor_economic_logic", {})
            ),
            "industry_constraint": rotation_portfolio.get("industry_constraint"),
            "cost_model": rotation_portfolio.get("cost_model", {}),
            "capacity_reference_capital": float(
                rotation_portfolio.get("capacity_reference_capital", 10_000.0)
            ),
            "capacity_selection_policy": rotation_portfolio.get(
                "capacity_selection_policy"
            ),
        },
        rotation_model_path,
    )
    model_payload = model.to_dict()
    model_payload["generated_at"] = report["generated_at"]
    model_payload["artifact_bundle_id"] = artifact_bundle_id
    _atomic_json(model_payload, calibration_path)
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
    parser.add_argument(
        "--cost-model-candidate",
        help="execution_cost_recalibration_latest.json candidate; shadow validation only",
    )
    parser.add_argument(
        "--shadow-output-dir",
        help="isolated output directory required with --cost-model-candidate",
    )
    args = parser.parse_args()
    if bool(args.cost_model_candidate) != bool(args.shadow_output_dir):
        parser.error(
            "--cost-model-candidate and --shadow-output-dir must be provided together"
        )
    active_cost_model = DEFAULT_ETF_COST_MODEL
    execution_policy_version = ROTATION_EXECUTION_POLICY_VERSION
    shadow_context: Dict[str, Any] = {}
    shadow_manifest_path: Optional[Path] = None
    if args.cost_model_candidate:
        active_cost_model, candidate = load_cost_model_candidate(
            args.cost_model_candidate
        )
        execution_policy_version = str(
            candidate["candidate_execution_policy_version"]
        )
        shadow_dir = Path(args.shadow_output_dir).resolve()
        protected_dirs = {
            Path(main.Config.V4_CALIBRATION_FILE).resolve().parent,
            Path(main.Config.FACTOR_REGISTRY_FILE).resolve().parent,
            Path(main.Config.ROTATION_MODEL_FILE).resolve().parent,
            Path(main.Config.ROTATION_LATEST_FILE).resolve().parent,
        }
        if any(
            shadow_dir == protected or protected in shadow_dir.parents
            for protected in protected_dirs
        ):
            raise ValueError("shadow cost validation cannot write inside production artifact directories")
        shadow_dir.mkdir(parents=True, exist_ok=True)
        args.rows_cache = str(shadow_dir / "v4_calibration_rows.json")
        args.calibration_out = str(shadow_dir / "v4_calibration.json")
        args.report_out = str(shadow_dir / "v4_acceptance_report.json")
        args.factor_registry_out = str(shadow_dir / "adaptive_factor_registry.json")
        args.predictions_out = str(shadow_dir / "walk_forward_predictions.json")
        args.predictions_cache = str(shadow_dir / "walk_forward_predictions.json")
        args.rotation_model_out = str(shadow_dir / "rotation_model.json")
        shadow_manifest_path = shadow_dir / "shadow_cost_validation_manifest.json"
        shadow_context = {
            "shadow_only": True,
            "promotion_allowed": False,
            "source_candidate_path": str(Path(args.cost_model_candidate).resolve()),
            "candidate_fingerprint": str(candidate.get("candidate_fingerprint", "")),
            "candidate_execution_policy_version": execution_policy_version,
            "current_cost_model": dict(candidate.get("current_cost_model") or {}),
            "recommended_cost_model": active_cost_model.to_dict(),
        }
    sample_step = max(1, args.sample_step)
    cached = (
        load_rows_cache(
            args.data_dir,
            sample_step,
            args.rows_cache,
            cost_model=active_cost_model,
        )
        if args.reuse_rows_cache else None
    )
    if cached is None:
        rows, fingerprint, raw_frames = generate_rows(
            args.data_dir,
            sample_step=sample_step,
            max_workers=max(1, args.workers),
            cost_model=active_cost_model,
        )
        qfq_frames, _ = _load_price_pairs(args.data_dir)
        eligible_signal_until = _latest_eligible_signal_date(qfq_frames, sample_step)
        if not eligible_signal_until:
            raise ValueError("market history has no fully labelled sampled calibration date")
        save_rows_cache(
            rows,
            fingerprint,
            sample_step,
            args.rows_cache,
            eligible_signal_until,
            active_cost_model,
        )
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
        if not prediction_cache_compatible(
            prediction_payload,
            fingerprint,
            len(rows),
            active_cost_model,
        ):
            raise ValueError("predictions cache market-data fingerprint is stale or incompatible")
        cached_predictions = list(prediction_payload.get("rows", []))
        if not cached_predictions or int(prediction_payload.get("row_count", 0)) != len(cached_predictions):
            raise ValueError("predictions cache is empty or inconsistent")
        cached_folds = list(prediction_payload.get("folds", []))
        if not cached_folds:
            raise ValueError("predictions cache does not contain its own fold evidence")
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
        cost_model=active_cost_model,
        execution_policy_version=execution_policy_version,
        shadow_cost_context=shadow_context,
    )
    if shadow_manifest_path is not None:
        _atomic_json(
            {
                "schema_version": 1,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "SHADOW_VALIDATION_COMPLETE",
                "shadow_only": True,
                "promotion_allowed": False,
                "candidate_fingerprint": shadow_context.get("candidate_fingerprint"),
                "candidate_execution_policy_version": execution_policy_version,
                "cost_model": active_cost_model.to_dict(),
                "cost_model_fingerprint": cost_model_fingerprint(active_cost_model),
                "rotation_strategy_approved_under_candidate_costs": bool(
                    report.get("strategy_approved", False)
                ),
                "rotation_acceptance_gates": dict(
                    report.get("rotation_acceptance_gates") or {}
                ),
                "output_directory": str(shadow_manifest_path.parent),
            },
            str(shadow_manifest_path),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
