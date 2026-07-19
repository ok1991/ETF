"""Producer-side validation for the shared rotation V2 execution contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .rotation import (
    ROTATION_ACCEPTANCE_POLICY_VERSION,
    ROTATION_EXECUTION_POLICY_VERSION,
    ROTATION_SCHEMA_VERSION,
)
from .trading import DEFAULT_ETF_COST_MODEL


def validate_rotation_contract(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if payload.get("schema_version") != ROTATION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {ROTATION_SCHEMA_VERSION}")
    if payload.get("approved") is not True:
        errors.append("rotation target must be approved")
    if not str(payload.get("model_version", "")).strip():
        errors.append("model_version is required")
    if payload.get("execution_policy_version") != ROTATION_EXECUTION_POLICY_VERSION:
        errors.append("execution policy version does not match live execution")
    if payload.get("acceptance_policy_version") != ROTATION_ACCEPTANCE_POLICY_VERSION:
        errors.append("acceptance policy version does not match approved research")
    if payload.get("risk_control_only") is not True:
        specification_fingerprint = str(
            payload.get("strategy_specification_fingerprint", "")
        ).strip().lower()
        if not (
            len(specification_fingerprint) == 64
            and all(char in "0123456789abcdef" for char in specification_fingerprint)
        ):
            errors.append("strategy specification fingerprint must be 64 lowercase hex characters")
        elif not str(payload.get("model_version", "")).endswith(
            specification_fingerprint[:8]
        ):
            errors.append("model version does not match strategy specification fingerprint")
    try:
        data_date = datetime.strptime(str(payload.get("data_date", ""))[:10], "%Y-%m-%d")
        execution_date = datetime.strptime(
            str(payload.get("execution_date", ""))[:10], "%Y-%m-%d"
        )
    except (TypeError, ValueError):
        errors.append("data_date and execution_date must be YYYY-MM-DD")
    else:
        if execution_date <= data_date:
            errors.append("execution_date must be after data_date")

    try:
        max_exposure = float(payload.get("max_exposure_ratio"))
        cash_weight = float(payload.get("cash_weight"))
    except (TypeError, ValueError):
        max_exposure = -1.0
        cash_weight = -1.0
        errors.append("max_exposure_ratio and cash_weight must be numeric")
    else:
        if not 0.0 <= max_exposure <= 1.0:
            errors.append("max_exposure_ratio must be in [0, 1]")
        if not 0.0 <= cash_weight <= 1.0:
            errors.append("cash_weight must be in [0, 1]")
        if abs(max_exposure + cash_weight - 1.0) > 1e-4:
            errors.append("max_exposure_ratio plus cash_weight must equal 1")
    try:
        capacity_reference_capital = float(payload.get("capacity_reference_capital"))
    except (TypeError, ValueError):
        errors.append("capacity_reference_capital must be numeric")
    else:
        if abs(capacity_reference_capital - 10_000.0) > 0.01:
            errors.append("capacity_reference_capital must match the approved backtest")

    weights = payload.get("target_weights")
    if not isinstance(weights, dict):
        errors.append("target_weights must be an object")
    else:
        total = 0.0
        for code, raw_weight in weights.items():
            if not (str(code).isdigit() and len(str(code)) == 6):
                errors.append(f"invalid ETF code: {code}")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                errors.append(f"{code}.weight must be numeric")
                continue
            if weight <= 0.0 or weight > 1.0:
                errors.append(f"{code}.weight must be in (0, 1]")
            total += weight
        if max_exposure > 0.0 and not weights:
            errors.append("positive risk budget requires target weights")
        if abs(total - max(max_exposure, 0.0)) > 1e-4:
            errors.append("target weight sum must equal max_exposure_ratio")

    liquidity = payload.get("execution_liquidity")
    if not isinstance(liquidity, dict):
        errors.append("execution_liquidity must be an object")
    elif isinstance(weights, dict):
        for code in weights:
            item = liquidity.get(str(code))
            if not isinstance(item, dict):
                errors.append(f"execution liquidity missing for target {code}")
                continue
            try:
                average_amount = float(item.get("average_daily_amount_20"))
            except (TypeError, ValueError):
                average_amount = 0.0
            if average_amount <= 0.0:
                errors.append(f"execution liquidity must be positive for target {code}")
            try:
                max_new_risk_amount = float(item.get("max_new_risk_amount"))
                max_participation_rate = float(item.get("max_participation_rate"))
            except (TypeError, ValueError):
                errors.append(f"capacity headroom must be numeric for target {code}")
            else:
                expected_rate = float(DEFAULT_ETF_COST_MODEL.max_participation_rate)
                expected_amount = average_amount * expected_rate
                if abs(max_participation_rate - expected_rate) > 1e-12:
                    errors.append(f"capacity participation rate mismatch for target {code}")
                if abs(max_new_risk_amount - expected_amount) > max(0.01, expected_amount * 1e-10):
                    errors.append(f"capacity headroom mismatch for target {code}")
                required_amount = capacity_reference_capital * float(weights.get(code, 0.0))
                if max_new_risk_amount + 0.01 < required_amount:
                    errors.append(f"capacity headroom cannot carry target weight for {code}")
            if str(item.get("as_of_date", ""))[:10] != str(payload.get("data_date", ""))[:10]:
                errors.append(f"execution liquidity date mismatch for target {code}")

    market_policy = payload.get("market_policy")
    if not isinstance(market_policy, dict):
        errors.append("market_policy must be an object")
    else:
        try:
            policy_exposure = float(market_policy.get("max_exposure_ratio"))
        except (TypeError, ValueError):
            errors.append("market_policy.max_exposure_ratio must be numeric")
        else:
            if abs(policy_exposure - max(max_exposure, 0.0)) > 1e-4:
                errors.append("market policy exposure must match target exposure")

    sleeves = payload.get("sleeves")
    if not isinstance(sleeves, list) or len(sleeves) != 2:
        errors.append("sleeves must contain exactly two staggered sleeves")

    expected_cost = DEFAULT_ETF_COST_MODEL.to_dict()
    walk_forward_metrics = payload.get("walk_forward_metrics") or {}
    capacity_fields = {
        "capacity_truncation_count": int,
        "requested_buy_value": float,
        "executed_buy_value": float,
        "capacity_truncated_buy_value": float,
        "unfilled_buy_value": float,
        "buy_fill_ratio": float,
        "capacity_fill_ratio": float,
    }
    capacity_values: Dict[str, float] = {}
    for field, value_type in capacity_fields.items():
        try:
            value = value_type(walk_forward_metrics.get(field))
        except (TypeError, ValueError):
            errors.append(f"walk_forward_metrics.{field} must be numeric")
            continue
        capacity_values[field] = float(value)
        if value < 0.0:
            errors.append(f"walk_forward_metrics.{field} must be non-negative")
    for field in ("buy_fill_ratio", "capacity_fill_ratio"):
        if field in capacity_values and capacity_values[field] > 1.0:
            errors.append(f"walk_forward_metrics.{field} must be at most 1")
    if (
        "requested_buy_value" in capacity_values
        and "executed_buy_value" in capacity_values
        and capacity_values["executed_buy_value"] > capacity_values["requested_buy_value"] + 0.01
    ):
        errors.append("executed buy value cannot exceed requested buy value")
    if (
        "capacity_truncated_buy_value" in capacity_values
        and "unfilled_buy_value" in capacity_values
        and capacity_values["capacity_truncated_buy_value"] > capacity_values["unfilled_buy_value"] + 0.01
    ):
        errors.append("capacity-truncated value cannot exceed total unfilled value")
    cost_model = walk_forward_metrics.get("cost_model") or {}
    for field, expected in expected_cost.items():
        try:
            recorded = float(cost_model.get(field))
        except (TypeError, ValueError):
            errors.append(f"walk_forward_metrics.cost_model.{field} must be numeric")
            continue
        tolerance = 0.0 if field == "lot_size" else 1e-12
        if abs(recorded - float(expected)) > tolerance:
            errors.append(f"cost model {field} does not match live execution")
    return errors


__all__ = ["validate_rotation_contract"]
