"""Jinja2 based ETF radar report."""

from __future__ import annotations

import shutil
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .paths import PATHS


ENTRY_PERMISSION_LABELS = {
    "TRADEABLE": "可交易",
    "MAINLINE_ONLY": "仅限主线标的",
    "BLOCKED": "禁止开新仓",
    "OPEN": "开放交易",
    "SELECTIVE": "选择性交易",
    "OBSERVE_ONLY": "仅观察",
}

MARKET_STATE_LABELS = {
    "NORMAL": "正常",
    "DEFENSIVE": "防御",
    "RISK_OFF": "风险规避",
    "RISK_ON": "风险偏好",
    "NEUTRAL": "中性",
    "HARD_DEFENSIVE": "强防御",
    "CAUTIOUS": "谨慎",
    "PULSE_FULL": "脉冲满仓",
    "PULSE_CAUTIOUS": "脉冲谨慎",
    "PULSE_HARD": "脉冲防守",
    "PULSE_EARLY": "早期脉冲",
}

MODEL_VERSION_LABELS = {
    "risk-control-cash-v4": "风控现金保护 v4",
}

STATE_RESET_LABELS = {
    "MODEL_AUTHORITY_CHANGED": "模型权威已变更",
}

EXPOSURE_AUTHORITY_LABELS = {
    "v4_market_policy": "V4 市场权限",
    "risk_control_cash": "风控现金",
}

FACTOR_HEALTH_LABELS = {
    "HEALTHY": "健康",
    "SUSPENDED": "已暂停",
    "DEGRADED": "降级",
    "BLOCKED": "已阻断",
}


def _value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _label(value: Any, labels: Dict[str, str]) -> str:
    raw = str(_value(value) or "")
    return labels.get(raw, raw)


def _model_label(value: Any) -> str:
    raw = str(_value(value) or "").strip()
    if not raw:
        return "未提供"
    return MODEL_VERSION_LABELS.get(raw, raw)


def _market_tone(value: Any, bullish: set[str], bearish: set[str]) -> str:
    raw = str(_value(value) or "")
    if raw in bullish:
        return "market-up"
    if raw in bearish:
        return "market-down"
    return "market-neutral"


class HTMLReporter:
    """Render the public dashboard without embedding assets in Python source."""

    @staticmethod
    def _compute_breadth(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "total": 0, "bull": 0, "bear": 0, "neutral": 0,
                "bull_pct": 0.0, "bear_pct": 0.0, "neutral_pct": 0.0,
                "ratio": 0.0, "signal": "无数据", "basis": "raw_status",
                "tone": "market-neutral",
            }
        bullish = {"极强波段多头", "波段多头", "偏多企稳"}
        bearish = {"极弱波段空头", "波段空头", "偏空走弱"}
        values = [str(_value(item.get("raw_status", item.get("status", "")))) for item in results]
        total = len(values)
        bull = sum(value in bullish for value in values)
        bear = sum(value in bearish for value in values)
        neutral = total - bull - bear
        ratio = bull / max(1, bear)
        signal = (
            "极度乐观" if ratio > 3.0 else "偏多" if ratio > 1.5 else
            "中性" if ratio >= 0.67 else "偏空" if ratio >= 0.33 else "极度悲观"
        )
        breadth = {
            "total": total,
            "bull": bull,
            "bear": bear,
            "neutral": neutral,
            "bull_pct": round(bull / total * 100.0, 1),
            "bear_pct": round(bear / total * 100.0, 1),
            "neutral_pct": round(neutral / total * 100.0, 1),
            "ratio": round(ratio, 2),
            "signal": signal,
            "basis": "raw_status",
        }
        breadth["tone"] = _market_tone(
            signal,
            {"偏多", "极度乐观"},
            {"偏空", "极度悲观"},
        )
        return breadth

    @classmethod
    def _rotation_context(cls, rotation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = dict(rotation or {})
        market_policy = dict(payload.get("market_policy") or {})
        target_weights = payload.get("target_weights") or {}
        targets = []
        candidates = {
            str(item.get("code")): item
            for item in (payload.get("top_candidates") or [])
            if item.get("code")
        }
        for code, weight in sorted(
            target_weights.items(),
            key=lambda item: float(item[1] or 0.0),
            reverse=True,
        ):
            meta = candidates.get(str(code), {})
            targets.append(
                {
                    "code": str(code),
                    "name": meta.get("name") or str(code),
                    "weight": float(weight or 0.0),
                    "industry_group": meta.get("industry_group") or "",
                    "rotation_score": float(meta.get("rotation_score") or 0.0),
                }
            )
        model_version = payload.get("model_version", "")
        return {
            "present": bool(payload),
            "data_date": payload.get("data_date", ""),
            "execution_date": payload.get("execution_date", ""),
            "generated_at": payload.get("generated_at", ""),
            "model_version": model_version,
            "model_version_label": _model_label(model_version),
            "approved": bool(payload.get("approved")),
            "risk_control_only": bool(payload.get("risk_control_only")),
            "cash_weight": float(payload.get("cash_weight") or 0.0),
            "max_exposure_ratio": float(
                payload.get("max_exposure_ratio")
                if payload.get("max_exposure_ratio") is not None
                else market_policy.get("max_exposure_ratio") or 0.0
            ),
            "targets": targets,
            "target_count": len(targets),
            "exposure_authority_label": _label(
                payload.get("exposure_authority"),
                EXPOSURE_AUTHORITY_LABELS,
            ),
            "state_reset_reason_label": _label(
                payload.get("state_reset_reason"),
                STATE_RESET_LABELS,
            ),
            "factor_health_label": _label(
                market_policy.get("factor_health_status"),
                FACTOR_HEALTH_LABELS,
            ),
            "policy_state_label": _label(
                market_policy.get("state"),
                MARKET_STATE_LABELS,
            ),
            "entry_permission_label": _label(
                market_policy.get("entry_permission"),
                ENTRY_PERMISSION_LABELS,
            ),
        }

    @classmethod
    def generate(
        cls,
        results: List[Dict[str, Any]],
        env_result: Any,
        filename: str = "index.html",
        rotation: Optional[Dict[str, Any]] = None,
    ) -> None:
        del filename
        PATHS.ensure()
        template_dir = PATHS.web / "templates"
        static_dir = PATHS.web / "static"
        public_static = PATHS.public / "assets"
        if public_static.exists():
            shutil.rmtree(public_static)
        shutil.copytree(static_dir, public_static)

        environment = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = environment.get_template("index.html.j2")
        rows = []
        for item in sorted(
            results,
            key=lambda value: float(value.get("v4_priority", 0.0) or 0.0),
            reverse=True,
        ):
            rows.append({
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "price": float(item.get("price", 0.0) or 0.0),
                "status": str(_value(item.get("raw_status", item.get("status", "")))),
                "priority": float(item.get("v4_priority", 0.0) or 0.0),
                "rps": float(item.get("rps", 0.0) or 0.0),
                "stop_loss": float(item.get("stop_loss", 0.0) or 0.0),
                "stop_dist": float(item.get("stop_dist", 0.0) or 0.0),
                "data_date": item.get("data_date", ""),
            })
        environment_data = env_result.to_dict() if hasattr(env_result, "to_dict") else dict(env_result)
        environment_data["entry_permission_label"] = _label(
            environment_data.get("entry_permission"),
            ENTRY_PERMISSION_LABELS,
        )
        environment_data["entry_permission_tone"] = _market_tone(
            environment_data.get("entry_permission"),
            {"TRADEABLE", "OPEN"},
            {"BLOCKED", "OBSERVE_ONLY"},
        )
        environment_data["regime_level_label"] = _label(
            environment_data.get("regime_level"),
            MARKET_STATE_LABELS,
        )
        environment_data["regime_level_tone"] = _market_tone(
            environment_data.get("regime_level"),
            {"RISK_ON"},
            {"RISK_OFF"},
        )
        rotation_data = cls._rotation_context(rotation)
        if rotation_data["present"] and rotation_data.get("max_exposure_ratio") is not None:
            environment_data["max_exposure_ratio"] = rotation_data["max_exposure_ratio"]
            if rotation_data.get("entry_permission_label"):
                environment_data["entry_permission_label"] = rotation_data["entry_permission_label"]
                environment_data["entry_permission_tone"] = _market_tone(
                    (rotation or {}).get("market_policy", {}).get("entry_permission"),
                    {"TRADEABLE", "OPEN"},
                    {"BLOCKED", "OBSERVE_ONLY"},
                )
            if rotation_data.get("policy_state_label"):
                environment_data["regime_level_label"] = rotation_data["policy_state_label"]
                environment_data["regime_level_tone"] = _market_tone(
                    (rotation or {}).get("market_policy", {}).get("state"),
                    {"RISK_ON", "PULSE_FULL"},
                    {"RISK_OFF", "PULSE_HARD"},
                )
        generated_at = datetime.now()
        document = template.render(
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            asset_version=generated_at.strftime("%Y%m%d%H%M%S"),
            environment=environment_data,
            breadth=cls._compute_breadth(results),
            rows=rows,
            rotation=rotation_data,
        )
        (PATHS.public / "index.html").write_text(document, encoding="utf-8")
