"""Audit Swing execution feedback without letting estimates validate backtest costs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib import error, parse, request


ALLOWED_EVIDENCE_LEVELS = {
    "MODEL_ESTIMATE_ONLY",
    "BROKER_CONFIRMED",
    "BROKER_EVIDENCE_REJECTED",
    "NO_ORDERS",
}
AUDIT_POLICY_VERSION = "broker-cost-and-execution-session-audit-v2"
MIN_CONFIRMED_SAMPLES = 3
MIN_AGGREGATE_GROSS = 5_000.0
MAX_ACCEPTABLE_COST_RATIO = 1.25
MIN_EXCESS_COST_BPS = 2.0
MAX_LEDGER_SAMPLES = 50
MAX_PENDING_CONFIRMATIONS = 50
MAX_EXPECTED_EXECUTIONS = 50
MAX_UNCONFIRMED_EXECUTION_AGE_DAYS = 7
MAX_MISSED_EXECUTION_AGE_DAYS = 1
RECALIBRATION_SAMPLE_WINDOW = 20
RECALIBRATION_SAFETY_MARGIN_BPS = 1.0
MAX_BASE_SLIPPAGE_INCREMENT_BPS = 20.0
ALLOWED_REMOTE_HOSTS = {
    "raw.githubusercontent.com",
    "github.com",
    "ok1991.github.io",
}


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cost_model(rotation_model: Mapping[str, Any]) -> Dict[str, Any]:
    direct = rotation_model.get("cost_model")
    if isinstance(direct, dict) and direct:
        return dict(direct)
    portfolio = rotation_model.get("portfolio_metrics") or {}
    return dict(portfolio.get("cost_model") or {})


def cost_authority_id(rotation_model: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            "execution_policy_version": str(
                rotation_model.get("execution_policy_version", "")
            ),
            "cost_model": _cost_model(rotation_model),
        }
    )


def _expected_execution_record(
    expected_execution: Optional[Mapping[str, Any]],
    rotation_model: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(expected_execution, Mapping):
        return None
    if expected_execution.get("approved") is not True:
        return None
    model_version = str(expected_execution.get("model_version", "")).strip()
    execution_policy_version = str(
        expected_execution.get("execution_policy_version", "")
    ).strip()
    strategy_fingerprint = str(
        expected_execution.get("strategy_specification_fingerprint", "")
    ).strip()
    execution_date = str(expected_execution.get("execution_date", ""))[:10]
    try:
        datetime.strptime(execution_date, "%Y-%m-%d")
    except ValueError:
        return None
    if not model_version or not execution_policy_version or not strategy_fingerprint:
        return None
    expected_cost_model = dict(
        ((expected_execution.get("walk_forward_metrics") or {}).get("cost_model") or {})
    )
    expected_authority_id = _canonical_hash(
        {
            "execution_policy_version": execution_policy_version,
            "cost_model": expected_cost_model,
        }
    )
    if expected_authority_id != cost_authority_id(rotation_model):
        return None
    execution_key = _canonical_hash(
        {
            "model_version": model_version,
            "execution_policy_version": execution_policy_version,
            "strategy_specification_fingerprint": strategy_fingerprint,
            "execution_date": execution_date,
            "target_weights": dict(expected_execution.get("target_weights") or {}),
            "risk_control_only": bool(
                expected_execution.get("risk_control_only", False)
            ),
        }
    )
    return {
        "execution_key": execution_key,
        "model_version": model_version,
        "execution_policy_version": execution_policy_version,
        "strategy_specification_fingerprint": strategy_fingerprint,
        "execution_date": execution_date,
        "cost_authority_id": expected_authority_id,
    }


def _feedback_matches_expected_execution(
    feedback: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return bool(
        str(feedback.get("model_version", ""))
        == str(expected.get("model_version", ""))
        and str(feedback.get("execution_policy_version", ""))
        == str(expected.get("execution_policy_version", ""))
        and str(feedback.get("strategy_specification_fingerprint", ""))
        == str(expected.get("strategy_specification_fingerprint", ""))
        and str(feedback.get("execution_date", ""))[:10]
        == str(expected.get("execution_date", ""))[:10]
        and str(feedback.get("run_date", ""))[:10]
        == str(expected.get("execution_date", ""))[:10]
        and feedback.get("quote_tradeable") is True
        and feedback.get("state_write_allowed") is True
    )


def _broker_execution_consistency_errors(
    feedback: Mapping[str, Any],
    orders: list[Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    planned: Dict[Tuple[str, str], int] = {}
    for order in orders:
        if not isinstance(order, Mapping):
            continue
        key = (str(order.get("code", "")), str(order.get("side", "")).upper())
        try:
            shares = int(order.get("shares", 0))
        except (TypeError, ValueError):
            shares = 0
        planned[key] = planned.get(key, 0) + shares

    raw_fills = evidence.get("fills")
    raw_outcomes = evidence.get("order_outcomes")
    raw_comparison = evidence.get("comparison")
    if not isinstance(raw_fills, list):
        errors.append("BROKER_FILLS_NOT_LIST")
        raw_fills = []
    if not isinstance(raw_outcomes, list):
        errors.append("BROKER_ORDER_OUTCOMES_NOT_LIST")
        raw_outcomes = []
    if not isinstance(raw_comparison, list):
        errors.append("BROKER_COMPARISON_NOT_LIST")
        raw_comparison = []
    if not str(evidence.get("broker", "")).strip():
        errors.append("BROKER_NAME_MISSING")

    execution_date = str(feedback.get("execution_date", ""))[:10]
    filled: Dict[Tuple[str, str], int] = {}
    for index, fill in enumerate(raw_fills):
        if not isinstance(fill, Mapping):
            errors.append(f"BROKER_FILL_INVALID:{index}")
            continue
        key = (str(fill.get("code", "")), str(fill.get("side", "")).upper())
        try:
            shares = int(fill.get("shares", 0))
            price = float(fill.get("price", 0.0))
            commission = float(fill.get("commission", -1.0))
            other_fees = float(fill.get("other_fees", -1.0))
        except (TypeError, ValueError):
            shares = 0
            price = 0.0
            commission = -1.0
            other_fees = -1.0
        if (
            key not in planned
            or shares <= 0
            or price <= 0.0
            or commission < 0.0
            or other_fees < 0.0
            or str(fill.get("trade_date", ""))[:10] != execution_date
        ):
            errors.append(f"BROKER_FILL_INVALID:{index}")
        filled[key] = filled.get(key, 0) + shares

    outcomes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for index, outcome in enumerate(raw_outcomes):
        if not isinstance(outcome, Mapping):
            errors.append(f"BROKER_OUTCOME_INVALID:{index}")
            continue
        key = (
            str(outcome.get("code", "")),
            str(outcome.get("side", "")).upper(),
        )
        if key in outcomes:
            errors.append(f"BROKER_OUTCOME_DUPLICATE:{key[0]}:{key[1]}")
            continue
        try:
            filled_shares = int(outcome.get("filled_shares", -1))
            unfilled_shares = int(outcome.get("unfilled_shares", -1))
        except (TypeError, ValueError):
            filled_shares = -1
            unfilled_shares = -1
        planned_shares = planned.get(key, 0)
        status = str(outcome.get("status", "")).upper()
        expected_status = (
            "UNFILLED"
            if filled_shares == 0
            else ("FILLED" if unfilled_shares == 0 else "PARTIALLY_FILLED")
        )
        if (
            key not in planned
            or filled_shares < 0
            or unfilled_shares < 0
            or filled_shares + unfilled_shares != planned_shares
            or filled.get(key, 0) != filled_shares
            or status != expected_status
        ):
            errors.append(f"BROKER_OUTCOME_INVALID:{index}")
        outcomes[key] = {
            "filled_shares": filled_shares,
            "unfilled_shares": unfilled_shares,
        }
    if set(outcomes) != set(planned):
        errors.append("BROKER_OUTCOME_SET_MISMATCH")
    if not set(filled).issubset(set(planned)):
        errors.append("BROKER_FILL_SET_MISMATCH")

    comparison_keys: set[Tuple[str, str]] = set()
    for index, row in enumerate(raw_comparison):
        if not isinstance(row, Mapping):
            errors.append(f"BROKER_COMPARISON_INVALID:{index}")
            continue
        key = (str(row.get("code", "")), str(row.get("side", "")).upper())
        outcome = outcomes.get(key)
        try:
            compared_filled = int(row.get("shares", -1))
            compared_planned = int(row.get("planned_shares", -1))
            compared_unfilled = int(row.get("unfilled_shares", -1))
        except (TypeError, ValueError):
            compared_filled = -1
            compared_planned = -1
            compared_unfilled = -1
        expected_status = (
            "UNFILLED"
            if outcome and int(outcome.get("filled_shares", 0)) == 0
            else (
                "FILLED"
                if outcome and int(outcome.get("unfilled_shares", 0)) == 0
                else "PARTIALLY_FILLED"
            )
        )
        if (
            key in comparison_keys
            or outcome is None
            or compared_filled != int(outcome.get("filled_shares", -1))
            or compared_planned != planned.get(key, -1)
            or compared_unfilled != int(outcome.get("unfilled_shares", -1))
            or str(row.get("fill_status", "")).upper() != expected_status
        ):
            errors.append(f"BROKER_COMPARISON_INVALID:{index}")
        comparison_keys.add(key)
    if comparison_keys != set(planned):
        errors.append("BROKER_COMPARISON_SET_MISMATCH")

    total_planned = sum(max(value, 0) for value in planned.values())
    total_filled = sum(
        max(int(item.get("filled_shares", 0)), 0) for item in outcomes.values()
    )
    expected_completion = (
        "UNFILLED"
        if total_filled == 0
        else ("COMPLETE" if total_filled == total_planned else "PARTIAL")
    )
    if str(feedback.get("broker_fill_completion_status", "")) != expected_completion:
        errors.append("BROKER_FILL_COMPLETION_MISMATCH")
    return list(dict.fromkeys(errors))


def feedback_evidence_errors(feedback: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence_level = str(feedback.get("evidence_level", ""))
    orders = feedback.get("orders")
    if not isinstance(orders, list):
        return ["FEEDBACK_ORDERS_NOT_LIST"]
    for index, order in enumerate(orders):
        if not isinstance(order, Mapping):
            errors.append(f"FEEDBACK_ORDER_INVALID:{index}")
            continue
        try:
            shares = int(order.get("shares", 0))
            price = float(order.get("price", 0.0))
            total_cost = float(order.get("total_cost", -1.0))
        except (TypeError, ValueError):
            shares = 0
            price = 0.0
            total_cost = -1.0
        if (
            str(order.get("side", "")).upper() not in {"BUY", "SELL"}
            or not str(order.get("code", "")).strip()
            or shares <= 0
            or price <= 0.0
            or total_cost < 0.0
        ):
            errors.append(f"FEEDBACK_ORDER_INVALID:{index}")
    broker_confirmed = feedback.get("broker_confirmed")
    completion_status = str(
        feedback.get("broker_fill_completion_status", "NOT_APPLICABLE")
    )
    raw_broker_evidence = feedback.get("broker_evidence")
    broker_evidence = (
        dict(raw_broker_evidence)
        if isinstance(raw_broker_evidence, Mapping)
        else {}
    )
    if raw_broker_evidence and not isinstance(raw_broker_evidence, Mapping):
        errors.append("FEEDBACK_BROKER_EVIDENCE_NOT_OBJECT")
    broker_digest = str(feedback.get("broker_evidence_file_sha256", ""))
    if evidence_level == "MODEL_ESTIMATE_ONLY":
        if not orders:
            errors.append("MODEL_ESTIMATE_REQUIRES_ORDERS")
        if broker_confirmed is not False:
            errors.append("MODEL_ESTIMATE_BROKER_CONFIRMATION_INVALID")
        if broker_evidence or broker_digest or completion_status != "NOT_APPLICABLE":
            errors.append("MODEL_ESTIMATE_BROKER_EVIDENCE_INVALID")
        return errors
    if evidence_level == "BROKER_CONFIRMED":
        if not orders:
            errors.append("BROKER_CONFIRMED_REQUIRES_ORDERS")
        if broker_confirmed is not True:
            errors.append("BROKER_CONFIRMATION_FALSE")
        if completion_status not in {"COMPLETE", "PARTIAL", "UNFILLED"}:
            errors.append("BROKER_FILL_COMPLETION_INVALID")
        if orders and broker_evidence:
            errors.extend(
                _broker_execution_consistency_errors(
                    feedback,
                    orders,
                    broker_evidence,
                )
            )
        return errors
    if evidence_level == "BROKER_EVIDENCE_REJECTED":
        if broker_confirmed is not False:
            errors.append("REJECTED_EVIDENCE_BROKER_CONFIRMATION_INVALID")
        if not list(feedback.get("rejection_reasons") or []):
            errors.append("REJECTED_EVIDENCE_REASONS_EMPTY")
        return errors
    if evidence_level != "NO_ORDERS":
        return errors
    if orders:
        errors.append("NO_ORDERS_PAYLOAD_NOT_EMPTY")
    if broker_confirmed is not False:
        errors.append("NO_ORDERS_BROKER_CONFIRMATION_INVALID")
    if broker_evidence or broker_digest or completion_status != "NOT_APPLICABLE":
        errors.append("NO_ORDERS_BROKER_EVIDENCE_INVALID")
    try:
        estimated_cost = float(feedback.get("estimated_execution_cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        estimated_cost = -1.0
    if estimated_cost != 0.0:
        errors.append("NO_ORDERS_ESTIMATED_COST_NONZERO")
    reason_codes = feedback.get("decision_reason_codes")
    reason_set = (
        {str(item) for item in reason_codes}
        if isinstance(reason_codes, list)
        else set()
    )
    rebalance_required = feedback.get("rebalance_required") is True
    valid_reason = bool(
        (reason_set == {"PLAN_ALREADY_APPLIED"} and not rebalance_required)
        or (
            reason_set == {"PORTFOLIO_ALREADY_AT_TARGET"}
            and rebalance_required
        )
    )
    if not valid_reason:
        errors.append("NO_ORDERS_DECISION_REASON_INVALID")
    return errors


def _weighted_quantile(
    values: list[Tuple[float, float]],
    quantile: float,
) -> float:
    valid = sorted(
        (float(value), max(float(weight), 0.0))
        for value, weight in values
        if float(weight) > 0.0
    )
    total = sum(weight for _, weight in valid)
    if not valid or total <= 0.0:
        return 0.0
    threshold = min(max(float(quantile), 0.0), 1.0) * total
    cumulative = 0.0
    for value, weight in valid:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return valid[-1][0]


def build_cost_recalibration_recommendation(
    rotation_model: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    authority_id = cost_authority_id(rotation_model)
    samples = [
        dict(item)
        for item in list(ledger.get("samples") or [])
        if str(item.get("cost_authority_id", "")) == authority_id
    ]
    samples.sort(
        key=lambda item: (str(item.get("execution_date", "")), str(item.get("feedback_id", "")))
    )
    window = samples[-RECALIBRATION_SAMPLE_WINDOW:]
    recent = window[-MIN_CONFIRMED_SAMPLES:]
    aggregate_gross = sum(float(item.get("broker_gross", 0.0)) for item in window)
    current_model = _cost_model(rotation_model)
    result: Dict[str, Any] = {
        "schema_version": 1,
        "policy_version": "conservative-live-cost-candidate-v1",
        "generated_at": (now or datetime.now()).astimezone().isoformat(timespec="seconds"),
        "status": "INSUFFICIENT_EVIDENCE",
        "approved_for_live_use": False,
        "auto_promotion_allowed": False,
        "requires_full_purged_walk_forward": True,
        "cost_authority_id": authority_id,
        "current_execution_policy_version": str(
            rotation_model.get("execution_policy_version", "")
        ),
        "sample_count": len(window),
        "aggregate_broker_gross": round(aggregate_gross, 4),
        "evidence_feedback_ids": [str(item.get("feedback_id", "")) for item in window],
        "current_cost_model": current_model,
        "recommended_cost_model": current_model,
        "candidate_execution_policy_version": "",
        "weighted_p75_excess_cost_bps": None,
        "weighted_p75_actual_to_expected_ratio": None,
        "base_slippage_increment_bps": 0.0,
        "increment_capped": False,
    }
    if len(window) < MIN_CONFIRMED_SAMPLES or aggregate_gross < MIN_AGGREGATE_GROSS:
        return result

    p75_excess = _weighted_quantile(
        [
            (float(item.get("excess_cost_bps", 0.0)), float(item.get("broker_gross", 0.0)))
            for item in window
        ],
        0.75,
    )
    p75_ratio = _weighted_quantile(
        [
            (
                float(item.get("actual_to_expected_cost_ratio", 0.0)),
                float(item.get("broker_gross", 0.0)),
            )
            for item in window
        ],
        0.75,
    )
    sustained = bool(
        len(recent) >= MIN_CONFIRMED_SAMPLES
        and sum(float(item.get("broker_gross", 0.0)) for item in recent)
        >= MIN_AGGREGATE_GROSS
        and all(
            float(item.get("actual_to_expected_cost_ratio", 0.0))
            > MAX_ACCEPTABLE_COST_RATIO
            and float(item.get("excess_cost_bps", 0.0)) >= MIN_EXCESS_COST_BPS
            for item in recent
        )
    )
    result["weighted_p75_excess_cost_bps"] = round(p75_excess, 6)
    result["weighted_p75_actual_to_expected_ratio"] = round(p75_ratio, 8)
    if not sustained:
        result["status"] = "MONITORING_NO_SUSTAINED_DEGRADATION"
        return result

    uncapped_increment = max(p75_excess, 0.0) + RECALIBRATION_SAFETY_MARGIN_BPS
    increment = min(uncapped_increment, MAX_BASE_SLIPPAGE_INCREMENT_BPS)
    recommended = dict(current_model)
    recommended["base_slippage_bps"] = round(
        float(current_model.get("base_slippage_bps", 0.0)) + increment,
        6,
    )
    candidate_fingerprint = _canonical_hash(
        {
            "source_cost_authority_id": authority_id,
            "recommended_cost_model": recommended,
            "method": "broker_gross_weighted_p75_excess_plus_1bp_margin",
        }
    )
    result.update(
        {
            "status": "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION",
            "recommended_cost_model": recommended,
            "candidate_execution_policy_version": (
                f"empirical-cost-candidate-v1-{candidate_fingerprint[:8]}"
            ),
            "candidate_fingerprint": candidate_fingerprint,
            "base_slippage_increment_bps": round(increment, 6),
            "increment_capped": bool(uncapped_increment > increment),
            "method": "broker_gross_weighted_p75_excess_plus_1bp_margin",
        }
    )
    return result


def _empty_ledger() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_version": AUDIT_POLICY_VERSION,
        "samples": [],
        "pending_confirmations": [],
        "expected_executions": [],
        "observed_execution_keys": [],
        "superseded_execution_keys": [],
        "blocked_cost_authority_id": "",
        "blocked_at": "",
    }


def load_ledger(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_ledger()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_ledger()
    if not isinstance(value, dict) or int(value.get("schema_version", 0) or 0) != 1:
        return _empty_ledger()
    value.setdefault("samples", [])
    value.setdefault("pending_confirmations", [])
    value.setdefault("expected_executions", [])
    value.setdefault("observed_execution_keys", [])
    value.setdefault("superseded_execution_keys", [])
    value.setdefault("blocked_cost_authority_id", "")
    value.setdefault("blocked_at", "")
    value["policy_version"] = AUDIT_POLICY_VERSION
    return value


def fetch_feedback(source: str, timeout: int = 5) -> Tuple[Optional[Dict[str, Any]], str]:
    source = str(source or "").strip()
    if not source:
        return None, "NO_FEEDBACK_SOURCE_CONFIGURED"
    try:
        if source.startswith(("https://", "http://")):
            hostname = str(parse.urlparse(source).hostname or "").lower()
            if hostname not in ALLOWED_REMOTE_HOSTS:
                return None, "UNAPPROVED_FEEDBACK_SOURCE"
            req = request.Request(
                source,
                headers={"User-Agent": "etf-main-execution-audit/1.0"},
            )
            with request.urlopen(req, timeout=max(1, int(timeout))) as response:
                raw = response.read()
        else:
            raw = Path(source).read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            return None, "FEEDBACK_NOT_JSON_OBJECT"
        return value, "LOADED"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, error.URLError) as exc:
        return None, f"FEEDBACK_UNAVAILABLE:{str(exc)[:200]}"


def _feedback_hash_matches(feedback: Mapping[str, Any]) -> bool:
    expected = str(feedback.get("feedback_id", ""))
    if len(expected) != 64:
        return False
    actual = _canonical_hash(
        {
            key: value
            for key, value in feedback.items()
            if key
            not in {
                "generated_at",
                "feedback_id",
                "state_reconciliation_applied",
                "state_reconciliation",
            }
        }
    )
    return actual == expected


def _authority_errors(
    feedback: Mapping[str, Any], rotation_model: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    fields = (("execution_policy_version", "execution_policy_version"),)
    for feedback_field, model_field in fields:
        if str(feedback.get(feedback_field, "")) != str(
            rotation_model.get(model_field, "")
        ):
            errors.append(f"AUTHORITY_MISMATCH:{feedback_field}")
    if dict(feedback.get("execution_cost_model") or {}) != _cost_model(rotation_model):
        errors.append("AUTHORITY_MISMATCH:execution_cost_model")
    return errors


def _confirmed_sample(
    feedback: Mapping[str, Any], rotation_model: Mapping[str, Any]
) -> Tuple[Optional[Dict[str, Any]], list[str]]:
    errors = _authority_errors(feedback, rotation_model)
    if feedback.get("broker_confirmed") is not True:
        errors.append("BROKER_CONFIRMATION_FALSE")
    if feedback.get("quote_tradeable") is not True:
        errors.append("QUOTE_AUTHORITY_NOT_TRADEABLE")
    if feedback.get("state_write_allowed") is not True:
        errors.append("STATE_WRITE_NOT_AUTHORISED")
    if str(feedback.get("run_date", ""))[:10] != str(
        feedback.get("execution_date", "")
    )[:10]:
        errors.append("RUN_EXECUTION_DATE_MISMATCH")
    digest = str(feedback.get("broker_evidence_file_sha256", ""))
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
        errors.append("BROKER_FILE_FINGERPRINT_INVALID")
    evidence = feedback.get("broker_evidence") or {}
    completion_status = str(
        feedback.get("broker_fill_completion_status", "COMPLETE")
    )
    if completion_status == "UNFILLED":
        try:
            gross = float(evidence.get("broker_gross", 0.0) or 0.0)
            expected = float(evidence.get("expected_model_cost", 0.0) or 0.0)
            actual = float(evidence.get("actual_total_cost", 0.0) or 0.0)
        except (TypeError, ValueError):
            errors.append("BROKER_UNFILLED_METRICS_INVALID")
            return None, list(dict.fromkeys(errors))
        if any(abs(value) > 1e-8 for value in (gross, expected, actual)):
            errors.append("BROKER_UNFILLED_METRICS_NONZERO")
        return None, list(dict.fromkeys(errors))
    try:
        gross = float(evidence.get("broker_gross"))
        expected = float(evidence.get("expected_model_cost"))
        actual = float(evidence.get("actual_total_cost"))
        ratio = float(evidence.get("actual_to_expected_cost_ratio"))
        excess_bps = float(evidence.get("excess_cost_bps"))
    except (TypeError, ValueError):
        errors.append("BROKER_COST_METRICS_INVALID")
        return None, list(dict.fromkeys(errors))
    if gross <= 0.0 or expected <= 0.0:
        errors.append("BROKER_COST_DENOMINATOR_INVALID")
    if abs(ratio - actual / expected) > 1e-5:
        errors.append("BROKER_COST_RATIO_INCONSISTENT")
    calculated_excess_bps = (actual - expected) / gross * 10_000.0 if gross > 0.0 else 0.0
    if abs(excess_bps - calculated_excess_bps) > 1e-4:
        errors.append("BROKER_EXCESS_BPS_INCONSISTENT")
    if errors:
        return None, list(dict.fromkeys(errors))
    return {
        "feedback_id": str(feedback.get("feedback_id", "")),
        "execution_date": str(feedback.get("execution_date", ""))[:10],
        "model_version": str(feedback.get("model_version", "")),
        "cost_authority_id": cost_authority_id(rotation_model),
        "broker_gross": round(gross, 4),
        "expected_model_cost": round(expected, 4),
        "actual_total_cost": round(actual, 4),
        "actual_to_expected_cost_ratio": round(ratio, 8),
        "excess_cost_bps": round(excess_bps, 6),
    }, []


def audit_feedback(
    feedback: Optional[Mapping[str, Any]],
    rotation_model: Mapping[str, Any],
    ledger: Optional[Mapping[str, Any]] = None,
    *,
    source_status: str = "LOADED",
    now: Optional[datetime] = None,
    expected_execution: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    updated = dict(ledger or _empty_ledger())
    updated["schema_version"] = 1
    updated["policy_version"] = AUDIT_POLICY_VERSION
    updated["samples"] = list(updated.get("samples") or [])[-MAX_LEDGER_SAMPLES:]
    updated["pending_confirmations"] = list(
        updated.get("pending_confirmations") or []
    )[-MAX_PENDING_CONFIRMATIONS:]
    updated["expected_executions"] = list(
        updated.get("expected_executions") or []
    )[-MAX_EXPECTED_EXECUTIONS:]
    updated["observed_execution_keys"] = list(
        updated.get("observed_execution_keys") or []
    )[-MAX_EXPECTED_EXECUTIONS:]
    updated["superseded_execution_keys"] = list(
        updated.get("superseded_execution_keys") or []
    )[-MAX_EXPECTED_EXECUTIONS:]
    authority_id = cost_authority_id(rotation_model)
    errors: list[str] = []
    evidence_level = "NO_FEEDBACK"
    ingested = False
    reference = now or datetime.now()

    expected_record = _expected_execution_record(expected_execution, rotation_model)
    if expected_record is not None:
        execution_key = str(expected_record["execution_key"])
        known_expected = {
            str(item.get("execution_key", ""))
            for item in updated["expected_executions"]
        }
        observed = {str(item) for item in updated["observed_execution_keys"]}
        if execution_key not in known_expected and execution_key not in observed:
            try:
                expected_date = datetime.strptime(
                    str(expected_record.get("execution_date", ""))[:10],
                    "%Y-%m-%d",
                ).date()
            except ValueError:
                expected_date = None
            superseded = []
            if expected_date is not None and reference.date() <= expected_date:
                superseded = [
                    str(item.get("execution_key", ""))
                    for item in updated["expected_executions"]
                    if str(item.get("execution_date", ""))[:10]
                    == str(expected_record.get("execution_date", ""))[:10]
                    and str(item.get("cost_authority_id", ""))
                    == str(expected_record.get("cost_authority_id", ""))
                    and str(item.get("execution_key", "")) != execution_key
                ]
            if superseded:
                superseded_set = set(superseded)
                updated["expected_executions"] = [
                    item
                    for item in updated["expected_executions"]
                    if str(item.get("execution_key", "")) not in superseded_set
                ]
                known_superseded = list(updated["superseded_execution_keys"])
                for old_key in superseded:
                    if old_key not in known_superseded:
                        known_superseded.append(old_key)
                updated["superseded_execution_keys"] = known_superseded[
                    -MAX_EXPECTED_EXECUTIONS:
                ]
            updated["expected_executions"].append(expected_record)
            updated["expected_executions"] = updated["expected_executions"][
                -MAX_EXPECTED_EXECUTIONS:
            ]

    if feedback is not None:
        evidence_level = str(feedback.get("evidence_level", ""))
        if int(feedback.get("schema_version", 0) or 0) != 1:
            errors.append("FEEDBACK_SCHEMA_VERSION_MISMATCH")
        if evidence_level not in ALLOWED_EVIDENCE_LEVELS:
            errors.append("EVIDENCE_LEVEL_INVALID")
        if not _feedback_hash_matches(feedback):
            errors.append("FEEDBACK_FINGERPRINT_MISMATCH")
        errors.extend(feedback_evidence_errors(feedback))
        if evidence_level in ALLOWED_EVIDENCE_LEVELS and not errors:
            errors.extend(_authority_errors(feedback, rotation_model))
        if evidence_level == "BROKER_EVIDENCE_REJECTED":
            errors.append("BROKER_EVIDENCE_REJECTED")
        if evidence_level == "BROKER_CONFIRMED" and not errors:
            sample, sample_errors = _confirmed_sample(feedback, rotation_model)
            errors.extend(sample_errors)
            if sample is not None and not errors:
                known = {
                    str(item.get("feedback_id", "")) for item in updated["samples"]
                }
                if sample["feedback_id"] not in known:
                    updated["samples"].append(sample)
                    updated["samples"] = updated["samples"][-MAX_LEDGER_SAMPLES:]
                    ingested = True
        if not errors and evidence_level == "MODEL_ESTIMATE_ONLY" and list(
            feedback.get("orders") or []
        ):
            plan_id = str(feedback.get("plan_id", ""))
            known_pending = {
                str(item.get("plan_id", ""))
                for item in updated["pending_confirmations"]
            }
            if plan_id and plan_id not in known_pending:
                updated["pending_confirmations"].append(
                    {
                        "plan_id": plan_id,
                        "feedback_id": str(feedback.get("feedback_id", "")),
                        "execution_date": str(feedback.get("execution_date", ""))[:10],
                        "model_version": str(feedback.get("model_version", "")),
                        "cost_authority_id": authority_id,
                    }
                )
                updated["pending_confirmations"] = updated[
                    "pending_confirmations"
                ][-MAX_PENDING_CONFIRMATIONS:]
        if not errors and evidence_level == "BROKER_CONFIRMED":
            confirmed_plan_id = str(feedback.get("plan_id", ""))
            updated["pending_confirmations"] = [
                item
                for item in updated["pending_confirmations"]
                if str(item.get("plan_id", "")) != confirmed_plan_id
            ]
        if not errors and evidence_level in ALLOWED_EVIDENCE_LEVELS:
            matched_execution_keys = [
                str(item.get("execution_key", ""))
                for item in updated["expected_executions"]
                if _feedback_matches_expected_execution(feedback, item)
            ]
            if matched_execution_keys:
                matched = set(matched_execution_keys)
                updated["expected_executions"] = [
                    item
                    for item in updated["expected_executions"]
                    if str(item.get("execution_key", "")) not in matched
                ]
                observed = list(updated["observed_execution_keys"])
                for execution_key in matched_execution_keys:
                    if execution_key not in observed:
                        observed.append(execution_key)
                updated["observed_execution_keys"] = observed[
                    -MAX_EXPECTED_EXECUTIONS:
                ]

    authority_samples = [
        item
        for item in updated["samples"]
        if str(item.get("cost_authority_id", "")) == authority_id
    ]
    authority_samples.sort(
        key=lambda item: (str(item.get("execution_date", "")), str(item.get("feedback_id", "")))
    )
    recent = authority_samples[-MIN_CONFIRMED_SAMPLES:]
    sustained_high = bool(
        len(recent) >= MIN_CONFIRMED_SAMPLES
        and sum(float(item.get("broker_gross", 0.0)) for item in recent)
        >= MIN_AGGREGATE_GROSS
        and all(
            float(item.get("actual_to_expected_cost_ratio", 0.0))
            > MAX_ACCEPTABLE_COST_RATIO
            and float(item.get("excess_cost_bps", 0.0)) >= MIN_EXCESS_COST_BPS
            for item in recent
        )
    )
    if sustained_high:
        updated["blocked_cost_authority_id"] = authority_id
        updated["blocked_at"] = (now or datetime.now()).astimezone().isoformat(
            timespec="seconds"
        )
    blocked = str(updated.get("blocked_cost_authority_id", "")) == authority_id
    overdue_pending: list[Dict[str, Any]] = []
    for item in updated["pending_confirmations"]:
        if str(item.get("cost_authority_id", "")) != authority_id:
            continue
        try:
            execution_date = datetime.strptime(
                str(item.get("execution_date", ""))[:10], "%Y-%m-%d"
            ).date()
        except ValueError:
            overdue_pending.append(dict(item))
            continue
        if (reference.date() - execution_date).days > MAX_UNCONFIRMED_EXECUTION_AGE_DAYS:
            overdue_pending.append(dict(item))
    overdue_executions: list[Dict[str, Any]] = []
    for item in updated["expected_executions"]:
        if str(item.get("cost_authority_id", "")) != authority_id:
            continue
        try:
            execution_date = datetime.strptime(
                str(item.get("execution_date", ""))[:10], "%Y-%m-%d"
            ).date()
        except ValueError:
            overdue_executions.append(dict(item))
            continue
        if (reference.date() - execution_date).days > MAX_MISSED_EXECUTION_AGE_DAYS:
            overdue_executions.append(dict(item))
    if blocked:
        status = "COST_MODEL_RECALIBRATION_REQUIRED"
    elif errors:
        status = "FEEDBACK_REJECTED"
    elif overdue_executions:
        status = "EXECUTION_SESSION_MISSED"
    elif overdue_pending:
        status = "BROKER_CONFIRMATION_OVERDUE"
    elif feedback is None:
        status = "NO_FEEDBACK"
    else:
        status = evidence_level

    audit = {
        "schema_version": 1,
        "policy_version": AUDIT_POLICY_VERSION,
        "generated_at": (now or datetime.now()).astimezone().isoformat(timespec="seconds"),
        "status": status,
        "rotation_authority": {
            "model_version": str(rotation_model.get("version", "")),
            "execution_policy_version": str(
                rotation_model.get("execution_policy_version", "")
            ),
            "cost_authority_id": authority_id,
        },
        "source_status": source_status,
        "evidence_level": evidence_level,
        "feedback_id": str((feedback or {}).get("feedback_id", "")),
        "feedback_ingested": ingested,
        "confirmed_sample_count": len(authority_samples),
        "recent_confirmed_sample_count": len(recent),
        "recent_aggregate_gross": round(
            sum(float(item.get("broker_gross", 0.0)) for item in recent), 4
        ),
        "sustained_cost_degradation": sustained_high,
        "rotation_authority_allowed": not bool(
            blocked or errors or overdue_executions or overdue_pending
        ),
        "expected_execution_count": len(updated["expected_executions"]),
        "superseded_execution_count": len(
            updated["superseded_execution_keys"]
        ),
        "overdue_execution_count": len(overdue_executions),
        "overdue_execution_keys": [
            str(item.get("execution_key", "")) for item in overdue_executions
        ],
        "pending_confirmation_count": len(updated["pending_confirmations"]),
        "overdue_confirmation_count": len(overdue_pending),
        "overdue_plan_ids": [str(item.get("plan_id", "")) for item in overdue_pending],
        "errors": list(dict.fromkeys(errors)),
        "thresholds": {
            "minimum_confirmed_samples": MIN_CONFIRMED_SAMPLES,
            "minimum_aggregate_gross": MIN_AGGREGATE_GROSS,
            "maximum_acceptable_cost_ratio": MAX_ACCEPTABLE_COST_RATIO,
            "minimum_excess_cost_bps": MIN_EXCESS_COST_BPS,
            "maximum_unconfirmed_execution_age_days": (
                MAX_UNCONFIRMED_EXECUTION_AGE_DAYS
            ),
            "maximum_missed_execution_age_days": MAX_MISSED_EXECUTION_AGE_DAYS,
        },
    }
    audit["cost_recalibration_recommendation"] = (
        build_cost_recalibration_recommendation(rotation_model, updated, now=now)
    )
    return audit, updated


def audit_feedback_batch(
    payload: Optional[Mapping[str, Any]],
    rotation_model: Mapping[str, Any],
    ledger: Optional[Mapping[str, Any]] = None,
    *,
    source_status: str = "LOADED",
    now: Optional[datetime] = None,
    expected_execution: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if payload is None:
        return audit_feedback(
            None,
            rotation_model,
            ledger,
            source_status=source_status,
            now=now,
            expected_execution=expected_execution,
        )

    history_errors: list[str] = []
    if "events" in payload:
        events = payload.get("events")
        if int(payload.get("schema_version", 0) or 0) != 1:
            history_errors.append("FEEDBACK_HISTORY_SCHEMA_MISMATCH")
        if not isinstance(events, list):
            history_errors.append("FEEDBACK_HISTORY_EVENTS_INVALID")
            events = []
        if int(payload.get("event_count", -1) or 0) != len(events):
            history_errors.append("FEEDBACK_HISTORY_COUNT_MISMATCH")
        if any(not isinstance(item, dict) for item in events):
            history_errors.append("FEEDBACK_HISTORY_EVENT_NOT_OBJECT")
    else:
        events = [dict(payload)]

    if history_errors:
        audit, updated = audit_feedback(
            None,
            rotation_model,
            ledger,
            source_status=source_status,
            now=now,
            expected_execution=expected_execution,
        )
        audit["status"] = "FEEDBACK_REJECTED"
        cost_blocked = (
            str(updated.get("blocked_cost_authority_id", ""))
            == cost_authority_id(rotation_model)
        )
        audit["rotation_authority_allowed"] = False
        if cost_blocked:
            audit["status"] = "COST_MODEL_RECALIBRATION_REQUIRED"
        audit["errors"] = history_errors
        audit["batch_event_count"] = 0
        audit["batch_ingested_count"] = 0
        audit["batch_rejected_count"] = 1
        return audit, updated

    if not events:
        audit, updated = audit_feedback(
            None,
            rotation_model,
            ledger,
            source_status="FEEDBACK_HISTORY_EMPTY",
            now=now,
            expected_execution=expected_execution,
        )
        audit["batch_event_count"] = 0
        audit["batch_ingested_count"] = 0
        audit["batch_rejected_count"] = 0
        return audit, updated

    updated: Mapping[str, Any] = ledger or _empty_ledger()
    final_audit: Dict[str, Any] = {}
    all_errors: list[str] = []
    ingested_count = 0
    rejected_count = 0
    for event in events:
        final_audit, next_ledger = audit_feedback(
            event,
            rotation_model,
            updated,
            source_status=source_status,
            now=now,
            expected_execution=expected_execution,
        )
        updated = next_ledger
        ingested_count += int(bool(final_audit.get("feedback_ingested")))
        if final_audit.get("errors"):
            rejected_count += 1
            all_errors.extend(str(item) for item in final_audit.get("errors", []))
    final_audit["batch_event_count"] = len(events)
    final_audit["batch_ingested_count"] = ingested_count
    final_audit["batch_rejected_count"] = rejected_count
    final_audit["errors"] = list(dict.fromkeys(all_errors))
    if (
        rejected_count
        and final_audit.get("status") != "COST_MODEL_RECALIBRATION_REQUIRED"
    ):
        final_audit["status"] = "FEEDBACK_BATCH_PARTIAL_REJECTION"
        final_audit["rotation_authority_allowed"] = False
    return final_audit, dict(updated)


def run_execution_feedback_audit(
    source: str,
    rotation_model: Mapping[str, Any],
    ledger_path: Path,
    audit_path: Path,
    recommendation_path: Optional[Path] = None,
    *,
    timeout: int = 5,
    expected_execution: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    feedback_payload, source_status = fetch_feedback(source, timeout=timeout)
    audit, ledger = audit_feedback_batch(
        feedback_payload,
        rotation_model,
        load_ledger(ledger_path),
        source_status=source_status,
        expected_execution=expected_execution,
    )
    _atomic_json(ledger, ledger_path)
    _atomic_json(audit, audit_path)
    if recommendation_path is not None:
        _atomic_json(
            dict(audit.get("cost_recalibration_recommendation") or {}),
            recommendation_path,
        )
    return audit


__all__ = [
    "AUDIT_POLICY_VERSION",
    "audit_feedback",
    "audit_feedback_batch",
    "build_cost_recalibration_recommendation",
    "cost_authority_id",
    "feedback_evidence_errors",
    "fetch_feedback",
    "run_execution_feedback_audit",
]
