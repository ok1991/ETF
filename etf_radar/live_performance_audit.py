"""Govern rotation authority using broker-reconciled live performance versus CSI 300."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib import error, parse, request


ALLOWED_REMOTE_HOSTS = {
    "raw.githubusercontent.com",
    "github.com",
    "ok1991.github.io",
}
AUDIT_POLICY_VERSION = "live-relative-performance-audit-v2"
MIN_LIVE_OBSERVATIONS = 20
LONG_LIVE_OBSERVATIONS = 60
MAX_RELATIVE_DRAWDOWN = -0.10
MAX_STRATEGY_DRAWDOWN = -0.15
MIN_ROLLING_20_RELATIVE_RETURN = -0.05
MIN_ROLLING_60_RELATIVE_RETURN = -0.08
MAX_EVIDENCE_AGE_DAYS = 7
MAX_MISSED_SESSION_GRACE_DAYS = 1
RECALIBRATION_COOLDOWN_DAYS = 7
FINAL_PORTFOLIO_STATE_EVIDENCE = {"BROKER_RECONCILED", "NO_EXECUTION_REQUIRED"}
ALLOWED_PORTFOLIO_STATE_EVIDENCE = FINAL_PORTFOLIO_STATE_EVIDENCE | {
    "MODEL_ESTIMATE_PENDING",
    "INITIAL_STATE",
    "LEGACY_UNVERIFIED",
}


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def fetch_live_performance(
    source: str, timeout: int = 5
) -> Tuple[Optional[Dict[str, Any]], str]:
    source = str(source or "").strip()
    if not source:
        return None, "NO_LIVE_PERFORMANCE_SOURCE"
    try:
        if source.startswith(("https://", "http://")):
            hostname = str(parse.urlparse(source).hostname or "").lower()
            if hostname not in ALLOWED_REMOTE_HOSTS:
                return None, "UNAPPROVED_LIVE_PERFORMANCE_SOURCE"
            req = request.Request(
                source,
                headers={"User-Agent": "etf-main-live-performance-audit/1.0"},
            )
            with request.urlopen(req, timeout=max(1, int(timeout))) as response:
                raw = response.read()
        else:
            raw = Path(source).read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            return None, "LIVE_PERFORMANCE_NOT_OBJECT"
        return value, "LOADED"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, error.URLError) as exc:
        return None, f"LIVE_PERFORMANCE_UNAVAILABLE:{str(exc)[:200]}"


def _identity_valid(payload: Mapping[str, Any]) -> bool:
    expected = str(payload.get("performance_id", ""))
    if len(expected) != 64:
        return False
    body = {key: value for key, value in payload.items() if key != "performance_id"}
    actual = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return actual == expected


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _max_drawdown(values: list[float]) -> float:
    peak = 0.0
    result = 0.0
    for value in values:
        peak = max(peak, float(value))
        if peak > 0.0:
            result = min(result, float(value) / peak - 1.0)
    return result


def _relative_period_return(records: list[Mapping[str, Any]], periods: int) -> Optional[float]:
    if len(records) <= periods:
        return None
    current = float(records[-1]["relative_nav"])
    prior = float(records[-periods - 1]["relative_nav"])
    return current / prior - 1.0 if prior > 0.0 else None


def _benchmark_quote_time_valid(value: str) -> bool:
    try:
        parsed = datetime.strptime(str(value)[:8], "%H:%M:%S").time()
    except (TypeError, ValueError):
        return False
    return (
        datetime.strptime("09:30:00", "%H:%M:%S").time()
        <= parsed
        <= datetime.strptime("11:30:00", "%H:%M:%S").time()
        or datetime.strptime("13:00:00", "%H:%M:%S").time()
        <= parsed
        <= datetime.strptime("15:00:00", "%H:%M:%S").time()
    )


def live_performance_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(payload.get("schema_version", 0) or 0) != 1:
        errors.append("LIVE_PERFORMANCE_SCHEMA_MISMATCH")
    if payload.get("benchmark_code") != "510300":
        errors.append("LIVE_PERFORMANCE_BENCHMARK_MISMATCH")
    if not _identity_valid(payload):
        errors.append("LIVE_PERFORMANCE_FINGERPRINT_MISMATCH")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        return errors + ["LIVE_PERFORMANCE_HISTORY_MISSING"]
    try:
        published_count = int(payload.get("observation_count", 0) or 0)
    except (TypeError, ValueError):
        published_count = 0
    if published_count != len(history) or not 1 <= published_count <= 520:
        errors.append("LIVE_PERFORMANCE_OBSERVATION_COUNT_INVALID")
    dates = [
        str(item.get("date", ""))[:10]
        for item in history
        if isinstance(item, Mapping)
    ]
    if len(dates) != len(history) or dates != sorted(set(dates)):
        errors.append("LIVE_PERFORMANCE_DATES_INVALID")

    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping):
        return errors + ["LIVE_PERFORMANCE_BASELINE_INVALID"]
    try:
        baseline_assets = float(baseline.get("strategy_assets", 0.0))
        baseline_benchmark = float(baseline.get("benchmark_price", 0.0))
    except (TypeError, ValueError):
        baseline_assets = 0.0
        baseline_benchmark = 0.0
    if baseline_assets <= 0.0 or baseline_benchmark <= 0.0:
        return errors + ["LIVE_PERFORMANCE_BASELINE_INVALID"]
    if dates and str(baseline.get("date", ""))[:10] > dates[0]:
        errors.append("LIVE_PERFORMANCE_BASELINE_DATE_INVALID")

    strategy_peak = 0.0
    benchmark_peak = 0.0
    relative_peak = 0.0
    strategy_max_drawdown = 0.0
    benchmark_max_drawdown = 0.0
    relative_max_drawdown = 0.0
    recomputed: list[Dict[str, Any]] = []
    numeric_fields = (
        "strategy_nav",
        "benchmark_nav",
        "relative_nav",
        "strategy_return",
        "benchmark_return",
        "excess_return",
        "relative_return",
        "strategy_drawdown",
        "benchmark_drawdown",
        "relative_drawdown",
    )
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            errors.append(f"LIVE_PERFORMANCE_HISTORY_{index}_INVALID")
            continue
        try:
            total_assets = float(item.get("total_assets", 0.0))
            benchmark_price = float(item.get("benchmark_price", 0.0))
        except (TypeError, ValueError):
            total_assets = 0.0
            benchmark_price = 0.0
        if total_assets <= 0.0 or benchmark_price <= 0.0:
            errors.append(f"LIVE_PERFORMANCE_HISTORY_{index}_SOURCE_VALUE_INVALID")
            continue
        state_evidence = str(item.get("portfolio_state_evidence", ""))
        pending_plan_id = str(
            item.get("pending_broker_confirmation_plan_id", "")
        )
        satisfied_plan_id = str(
            item.get("last_execution_satisfied_plan_id", "")
        )
        reconciliation_id = str(item.get("broker_reconciliation_id", ""))
        evidence_valid = bool(
            state_evidence in ALLOWED_PORTFOLIO_STATE_EVIDENCE
            and (
                (
                    state_evidence == "BROKER_RECONCILED"
                    and not pending_plan_id
                    and satisfied_plan_id
                    and reconciliation_id
                )
                or (
                    state_evidence == "MODEL_ESTIMATE_PENDING"
                    and pending_plan_id
                    and not reconciliation_id
                )
                or (
                    state_evidence == "NO_EXECUTION_REQUIRED"
                    and not pending_plan_id
                    and satisfied_plan_id
                    and not reconciliation_id
                )
                or (
                    state_evidence == "INITIAL_STATE"
                    and not pending_plan_id
                    and not satisfied_plan_id
                    and not reconciliation_id
                )
                or state_evidence == "LEGACY_UNVERIFIED"
            )
        )
        if not evidence_valid:
            errors.append(
                f"LIVE_PERFORMANCE_HISTORY_{index}_STATE_EVIDENCE_INVALID"
            )
        quote_source = str(item.get("benchmark_quote_source", ""))
        quote_mode = str(item.get("benchmark_quote_mode", ""))
        quote_date = str(item.get("benchmark_quote_date", ""))[:10]
        quote_time = str(item.get("benchmark_quote_time", ""))[:8]
        quote_fetched_at = str(item.get("benchmark_quote_fetched_at", ""))
        quote_validated_at = str(item.get("benchmark_quote_validated_at", ""))
        quote_tradeable = item.get("benchmark_quote_tradeable") is True
        if (
            quote_source != "SINA_REALTIME"
            or quote_mode != "DAILY_MARK_TO_MARKET"
            or quote_date != str(item.get("date", ""))[:10]
            or not _benchmark_quote_time_valid(quote_time)
            or not quote_fetched_at
            or not quote_validated_at
            or str(quote_validated_at)[:10] != quote_date
            or not quote_tradeable
        ):
            errors.append(
                f"LIVE_PERFORMANCE_HISTORY_{index}_BENCHMARK_QUOTE_EVIDENCE_INVALID"
            )
        strategy_nav = total_assets / baseline_assets
        benchmark_nav = benchmark_price / baseline_benchmark
        relative_nav = strategy_nav / benchmark_nav
        strategy_peak = max(strategy_peak, strategy_nav)
        benchmark_peak = max(benchmark_peak, benchmark_nav)
        relative_peak = max(relative_peak, relative_nav)
        strategy_drawdown = strategy_nav / strategy_peak - 1.0
        benchmark_drawdown = benchmark_nav / benchmark_peak - 1.0
        relative_drawdown = relative_nav / relative_peak - 1.0
        strategy_max_drawdown = min(strategy_max_drawdown, strategy_drawdown)
        benchmark_max_drawdown = min(benchmark_max_drawdown, benchmark_drawdown)
        relative_max_drawdown = min(relative_max_drawdown, relative_drawdown)
        expected = {
            "strategy_nav": round(strategy_nav, 8),
            "benchmark_nav": round(benchmark_nav, 8),
            "relative_nav": round(relative_nav, 8),
            "strategy_return": round(strategy_nav - 1.0, 8),
            "benchmark_return": round(benchmark_nav - 1.0, 8),
            "excess_return": round(strategy_nav - benchmark_nav, 8),
            "relative_return": round(relative_nav - 1.0, 8),
            "strategy_drawdown": round(strategy_drawdown, 8),
            "benchmark_drawdown": round(benchmark_drawdown, 8),
            "relative_drawdown": round(relative_drawdown, 8),
        }
        for field in numeric_fields:
            try:
                published = float(item.get(field))
            except (TypeError, ValueError):
                errors.append(
                    f"LIVE_PERFORMANCE_HISTORY_{index}_{field.upper()}_INVALID"
                )
                continue
            if abs(published - float(expected[field])) > 1e-8:
                errors.append(
                    f"LIVE_PERFORMANCE_HISTORY_{index}_{field.upper()}_MISMATCH"
                )
        recomputed.append(
            {
                **expected,
                "date": str(item.get("date", ""))[:10],
                "total_assets": round(total_assets, 4),
                "model_version": str(item.get("model_version", "")),
                "portfolio_state_evidence": state_evidence,
                "pending_broker_confirmation_plan_id": pending_plan_id,
                "last_execution_satisfied_plan_id": satisfied_plan_id,
                "broker_reconciliation_id": reconciliation_id,
                "benchmark_quote_source": quote_source,
                "benchmark_quote_mode": quote_mode,
                "benchmark_quote_date": quote_date,
                "benchmark_quote_time": quote_time,
                "benchmark_quote_fetched_at": quote_fetched_at,
                "benchmark_quote_validated_at": quote_validated_at,
                "benchmark_quote_tradeable": quote_tradeable,
            }
        )
    if recomputed and len(recomputed) == len(history):
        last = recomputed[-1]
        top_expected = {
            "total_assets": last["total_assets"],
            "strategy_nav": last["strategy_nav"],
            "benchmark_nav": last["benchmark_nav"],
            "relative_nav": last["relative_nav"],
            "strategy_return": last["strategy_return"],
            "benchmark_return": last["benchmark_return"],
            "excess_return": last["excess_return"],
            "relative_return": last["relative_return"],
            "strategy_max_drawdown": round(strategy_max_drawdown, 8),
            "benchmark_max_drawdown": round(benchmark_max_drawdown, 8),
            "relative_max_drawdown": round(relative_max_drawdown, 8),
        }
        for field, expected in top_expected.items():
            try:
                published = float(payload.get(field))
            except (TypeError, ValueError):
                errors.append(f"LIVE_PERFORMANCE_{field.upper()}_INVALID")
                continue
            if abs(published - float(expected)) > 1e-8:
                errors.append(f"LIVE_PERFORMANCE_{field.upper()}_MISMATCH")
        if str(payload.get("data_date", ""))[:10] != last["date"]:
            errors.append("LIVE_PERFORMANCE_LATEST_DATE_MISMATCH")
        if str(payload.get("model_version", "")) != last["model_version"]:
            errors.append("LIVE_PERFORMANCE_LATEST_MODEL_MISMATCH")
        for field in (
            "portfolio_state_evidence",
            "pending_broker_confirmation_plan_id",
            "last_execution_satisfied_plan_id",
            "broker_reconciliation_id",
            "benchmark_quote_source",
            "benchmark_quote_mode",
            "benchmark_quote_date",
            "benchmark_quote_time",
            "benchmark_quote_fetched_at",
            "benchmark_quote_validated_at",
            "benchmark_quote_tradeable",
        ):
            if str(payload.get(field, "")) != str(last.get(field, "")):
                errors.append(f"LIVE_PERFORMANCE_LATEST_{field.upper()}_MISMATCH")
    return list(dict.fromkeys(errors))


def audit_live_performance(
    payload: Optional[Mapping[str, Any]],
    rotation_model: Mapping[str, Any],
    *,
    source_status: str = "LOADED",
    now: Optional[Any] = None,
    expected_latest_data_date: str = "",
    expected_execution: Optional[Mapping[str, Any]] = None,
    previous_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    reference = datetime.fromisoformat(str(now)) if isinstance(now, str) else (now or datetime.now())
    expected_model_version = ""
    expected_execution_date = None
    if isinstance(expected_execution, Mapping) and expected_execution.get("approved") is True:
        expected_model_version = str(
            expected_execution.get("model_version", "")
        ).strip()
        try:
            expected_execution_date = datetime.strptime(
                str(expected_execution.get("execution_date", ""))[:10],
                "%Y-%m-%d",
            ).date()
        except ValueError:
            expected_execution_date = None
    current_version = str(rotation_model.get("version", ""))
    expected_current_model_session = bool(
        expected_execution_date is not None
        and expected_model_version
        and expected_model_version == current_version
    )
    base = {
        "schema_version": 1,
        "policy_version": AUDIT_POLICY_VERSION,
        "generated_at": reference.astimezone().isoformat(timespec="seconds"),
        "source_status": source_status,
        "model_version": str(rotation_model.get("version", "")),
        "rotation_authority_allowed": True,
        "recalibration_required": False,
        "errors": [],
    }
    if payload is None:
        missed_session = bool(
            expected_current_model_session
            and expected_execution_date is not None
            and (reference.date() - expected_execution_date).days
            > MAX_MISSED_SESSION_GRACE_DAYS
        )
        return {
            **base,
            "status": (
                "LIVE_PERFORMANCE_SESSION_MISSED"
                if missed_session
                else "NO_LIVE_PERFORMANCE_EVIDENCE"
            ),
            "rotation_authority_allowed": not missed_session,
            "current_model_observation_count": 0,
            "expected_performance_date": (
                expected_execution_date.isoformat()
                if expected_execution_date is not None
                else ""
            ),
        }

    errors: list[str] = live_performance_errors(payload)
    if int(payload.get("schema_version", 0) or 0) != 1:
        errors.append("LIVE_PERFORMANCE_SCHEMA_MISMATCH")
    if payload.get("benchmark_code") != "510300":
        errors.append("LIVE_PERFORMANCE_BENCHMARK_MISMATCH")
    if not _identity_valid(payload):
        errors.append("LIVE_PERFORMANCE_FINGERPRINT_MISMATCH")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        errors.append("LIVE_PERFORMANCE_HISTORY_MISSING")
        history = []
    try:
        published_count = int(payload.get("observation_count", 0) or 0)
    except (TypeError, ValueError):
        published_count = 0
    if published_count < len(history) or published_count <= 0:
        errors.append("LIVE_PERFORMANCE_OBSERVATION_COUNT_INVALID")
    dates = [str(item.get("date", ""))[:10] for item in history if isinstance(item, Mapping)]
    if len(dates) != len(history) or dates != sorted(set(dates)):
        errors.append("LIVE_PERFORMANCE_DATES_INVALID")
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            continue
        for field in ("strategy_nav", "benchmark_nav", "relative_nav"):
            try:
                value = float(item.get(field))
            except (TypeError, ValueError):
                errors.append(f"LIVE_PERFORMANCE_HISTORY_{index}_{field.upper()}_INVALID")
                continue
            if value <= 0.0:
                errors.append(f"LIVE_PERFORMANCE_HISTORY_{index}_{field.upper()}_INVALID")
    if history:
        last = history[-1]
        if str(payload.get("data_date", ""))[:10] != str(last.get("date", ""))[:10]:
            errors.append("LIVE_PERFORMANCE_LATEST_DATE_MISMATCH")
        for field in ("strategy_nav", "benchmark_nav", "relative_nav"):
            try:
                value = float(last.get(field))
                published = float(payload.get(field))
            except (TypeError, ValueError):
                errors.append(f"LIVE_PERFORMANCE_{field.upper()}_INVALID")
                continue
            if value <= 0.0 or abs(value - published) > 1e-8:
                errors.append(f"LIVE_PERFORMANCE_{field.upper()}_MISMATCH")
    try:
        data_date = datetime.strptime(str(payload.get("data_date", ""))[:10], "%Y-%m-%d")
    except ValueError:
        errors.append("LIVE_PERFORMANCE_DATA_DATE_INVALID")
    else:
        if expected_latest_data_date:
            try:
                expected_date = datetime.strptime(
                    str(expected_latest_data_date)[:10], "%Y-%m-%d"
                )
            except ValueError:
                errors.append("LIVE_PERFORMANCE_EXPECTED_TRADING_DATE_INVALID")
            else:
                if data_date < expected_date:
                    errors.append("LIVE_PERFORMANCE_STALE_TRADING_DATE")
                elif data_date > expected_date:
                    errors.append("LIVE_PERFORMANCE_FUTURE_TRADING_DATE")
        else:
            age_days = (reference.replace(tzinfo=None) - data_date).total_seconds() / 86400.0
            if age_days < -1.0:
                errors.append("LIVE_PERFORMANCE_FUTURE_DATE")
            if age_days > MAX_EVIDENCE_AGE_DAYS:
                errors.append("LIVE_PERFORMANCE_STALE")
    if history and isinstance(payload.get("baseline"), Mapping):
        baseline_id = _canonical_id(payload["baseline"])
        latest_observation_id = _canonical_id(history[-1])
        history_ids = {
            _canonical_id(item)
            for item in history
            if isinstance(item, Mapping)
        }
        history_by_date = {
            str(item.get("date", ""))[:10]: item
            for item in history
            if isinstance(item, Mapping)
        }
        base.update(
            {
                "performance_baseline_id": baseline_id,
                "latest_observation_id": latest_observation_id,
                "history_observation_count": published_count,
                "latest_observation_date": str(history[-1].get("date", ""))[:10],
            }
        )
        prior = dict(previous_audit or {})
        prior_baseline_id = str(prior.get("performance_baseline_id", ""))
        prior_latest_id = str(prior.get("latest_observation_id", ""))
        prior_latest_date = str(prior.get("latest_observation_date", ""))[:10]
        try:
            prior_count = int(prior.get("history_observation_count", 0) or 0)
        except (TypeError, ValueError):
            prior_count = 0
        current_latest_date = str(history[-1].get("date", ""))[:10]
        prior_date_item = history_by_date.get(prior_latest_date)
        prior_date_state_evidence = str(
            (prior_date_item or {}).get("portfolio_state_evidence", "")
        )
        reconciliation_backfill = bool(
            prior_date_item is not None
            and prior_date_state_evidence in ALLOWED_PORTFOLIO_STATE_EVIDENCE
        )
        if prior_baseline_id and prior_baseline_id != baseline_id:
            errors.append("LIVE_PERFORMANCE_BASELINE_RESET")
        if prior_count > 0 and published_count < prior_count:
            errors.append("LIVE_PERFORMANCE_OBSERVATION_COUNT_ROLLBACK")
        if prior_latest_date and current_latest_date < prior_latest_date:
            errors.append("LIVE_PERFORMANCE_HISTORY_DATE_ROLLBACK")
        if (
            prior_latest_id
            and prior_latest_date
            and current_latest_date > prior_latest_date
            and prior_latest_id not in history_ids
            and not reconciliation_backfill
        ):
            errors.append("LIVE_PERFORMANCE_HISTORY_CONTINUITY_BROKEN")
    if errors:
        return {
            **base,
            "status": "LIVE_PERFORMANCE_EVIDENCE_REJECTED",
            "rotation_authority_allowed": False,
            "recalibration_required": False,
            "errors": list(dict.fromkeys(errors)),
            "current_model_observation_count": 0,
        }

    suffix: list[Mapping[str, Any]] = []
    for item in reversed(history):
        if str(item.get("model_version", "")) != current_version:
            break
        suffix.append(item)
    suffix.reverse()
    observation_count = len(suffix)
    if observation_count == 0:
        latest_history_date = None
        if history:
            try:
                latest_history_date = datetime.strptime(
                    str(history[-1].get("date", ""))[:10], "%Y-%m-%d"
                ).date()
            except ValueError:
                latest_history_date = None
        model_session_mismatch = bool(
            expected_current_model_session
            and latest_history_date is not None
            and latest_history_date >= expected_execution_date
        )
        return {
            **base,
            "status": (
                "LIVE_PERFORMANCE_MODEL_SESSION_MISMATCH"
                if model_session_mismatch
                else "WARMUP_CURRENT_MODEL_NOT_OBSERVED"
            ),
            "rotation_authority_allowed": not model_session_mismatch,
            "current_model_observation_count": 0,
            "expected_performance_date": (
                expected_execution_date.isoformat()
                if expected_execution_date is not None
                else ""
            ),
        }

    latest_state_evidence = str(
        suffix[-1].get("portfolio_state_evidence", "")
    )
    if (
        expected_current_model_session
        and expected_execution_date is not None
        and str(suffix[-1].get("date", ""))[:10]
        >= expected_execution_date.isoformat()
        and latest_state_evidence not in FINAL_PORTFOLIO_STATE_EVIDENCE
    ):
        pending = latest_state_evidence == "MODEL_ESTIMATE_PENDING"
        return {
            **base,
            "status": (
                "LIVE_PERFORMANCE_BROKER_RECONCILIATION_PENDING"
                if pending
                else "LIVE_PERFORMANCE_STATE_EVIDENCE_NOT_FINAL"
            ),
            "rotation_authority_allowed": reference.date()
            <= expected_execution_date,
            "recalibration_required": False,
            "current_model_observation_count": observation_count,
            "portfolio_state_evidence": latest_state_evidence,
            "expected_performance_date": expected_execution_date.isoformat(),
        }

    strategy_drawdown = _max_drawdown(
        [float(item["strategy_nav"]) for item in suffix]
    )
    relative_drawdown = _max_drawdown(
        [float(item["relative_nav"]) for item in suffix]
    )
    rolling_20 = _relative_period_return(suffix, 20)
    rolling_60 = _relative_period_return(suffix, 60)
    hard_reasons: list[str] = []
    research_reasons: list[str] = []
    # Drawdown limits are hard risk controls, not statistical acceptance tests.
    # They must revoke authority as soon as the current-model path can breach
    # them; waiting for the 20-observation warmup would leave early live losses
    # unprotected. Rolling relative-return research gates still require their
    # full observation windows below.
    if relative_drawdown <= MAX_RELATIVE_DRAWDOWN:
        hard_reasons.append("LIVE_RELATIVE_DRAWDOWN_BREACH")
    if strategy_drawdown <= MAX_STRATEGY_DRAWDOWN:
        hard_reasons.append("LIVE_STRATEGY_DRAWDOWN_BREACH")
    if observation_count >= MIN_LIVE_OBSERVATIONS:
        if rolling_20 is not None and rolling_20 <= MIN_ROLLING_20_RELATIVE_RETURN:
            research_reasons.append("LIVE_ROLLING_20_RELATIVE_UNDERPERFORMANCE")
    if (
        observation_count >= LONG_LIVE_OBSERVATIONS
        and rolling_60 is not None
        and rolling_60 <= MIN_ROLLING_60_RELATIVE_RETURN
    ):
        research_reasons.append("LIVE_ROLLING_60_RELATIVE_UNDERPERFORMANCE")

    generated = datetime.fromisoformat(
        str(rotation_model.get("generated_at", "")).replace(" ", "T")
    ) if str(rotation_model.get("generated_at", "")) else reference
    cooldown = (reference.replace(tzinfo=None) - generated.replace(tzinfo=None)).days < RECALIBRATION_COOLDOWN_DAYS
    recalibration_required = bool(hard_reasons or research_reasons) and not cooldown
    if hard_reasons:
        status = "LIVE_RISK_LIMIT_BREACH"
    elif research_reasons and cooldown:
        status = "WATCH_RECALIBRATION_COOLDOWN"
    elif research_reasons:
        status = "LIVE_MODEL_RECALIBRATION_REQUIRED"
    elif observation_count < MIN_LIVE_OBSERVATIONS:
        status = "WARMUP"
    else:
        status = "ACTIVE"
    return {
        **base,
        "status": status,
        "performance_id": str(payload.get("performance_id", "")),
        "data_date": str(payload.get("data_date", ""))[:10],
        "current_model_observation_count": observation_count,
        "current_model_strategy_drawdown": round(strategy_drawdown, 8),
        "current_model_relative_drawdown": round(relative_drawdown, 8),
        "current_model_rolling_20_relative_return": (
            round(rolling_20, 8) if rolling_20 is not None else None
        ),
        "current_model_rolling_60_relative_return": (
            round(rolling_60, 8) if rolling_60 is not None else None
        ),
        "rotation_authority_allowed": not bool(hard_reasons),
        "recalibration_required": recalibration_required,
        "recalibration_cooldown_active": cooldown,
        "hard_reasons": hard_reasons,
        "research_reasons": research_reasons,
        "thresholds": {
            "minimum_observations": MIN_LIVE_OBSERVATIONS,
            "long_observations": LONG_LIVE_OBSERVATIONS,
            "maximum_relative_drawdown": MAX_RELATIVE_DRAWDOWN,
            "maximum_strategy_drawdown": MAX_STRATEGY_DRAWDOWN,
            "minimum_rolling_20_relative_return": MIN_ROLLING_20_RELATIVE_RETURN,
            "minimum_rolling_60_relative_return": MIN_ROLLING_60_RELATIVE_RETURN,
            "max_missed_session_grace_days": MAX_MISSED_SESSION_GRACE_DAYS,
        },
    }


def run_live_performance_audit(
    source: str,
    rotation_model: Mapping[str, Any],
    audit_path: Path,
    *,
    timeout: int = 5,
    expected_latest_data_date: str = "",
    expected_execution: Optional[Mapping[str, Any]] = None,
    now: Optional[Any] = None,
) -> Dict[str, Any]:
    previous_audit: Dict[str, Any] = {}
    try:
        prior_value = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        if isinstance(prior_value, dict):
            previous_audit = prior_value
    except (OSError, json.JSONDecodeError):
        previous_audit = {}
    payload, source_status = fetch_live_performance(source, timeout=timeout)
    audit = audit_live_performance(
        payload,
        rotation_model,
        source_status=source_status,
        expected_latest_data_date=expected_latest_data_date,
        expected_execution=expected_execution,
        previous_audit=previous_audit,
        now=now,
    )
    _atomic_json(audit, audit_path)
    return audit


__all__ = [
    "AUDIT_POLICY_VERSION",
    "audit_live_performance",
    "fetch_live_performance",
    "MAX_MISSED_SESSION_GRACE_DAYS",
    "live_performance_errors",
    "run_live_performance_audit",
]
