"""Jinja2 based ETF production dashboard."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
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
    "risk_control_fail_closed": "风控失败即现金",
}

FACTOR_HEALTH_LABELS = {
    "HEALTHY": "健康",
    "SUSPENDED": "已暂停",
    "DEGRADED": "降级",
    "BLOCKED": "已阻断",
    "UNKNOWN": "未知",
}

ENTRY_STATE_LABELS = {
    "READY": "可入场",
    "WATCH": "观察",
    "BLOCKED": "已阻断",
}

TREND_STATE_LABELS = {
    "BULL": "波段多头",
    "NEUTRAL": "多空震荡",
    "BEAR": "波段空头",
    "EXTREME_BULL": "极强波段多头",
    "EXTREME_BEAR": "极弱波段空头",
    "WEAK_BULL": "偏多企稳",
    "WEAK_BEAR": "偏空走弱",
}

REASON_CODE_LABELS = {
    "ROTATION_MODEL_NOT_APPROVED": "轮动模型尚未通过验收，已切换现金保护",
    "ROTATION_MODEL_NOT_APPROVED_OR_UNAVAILABLE": "轮动模型未获批或不可用",
    "ROTATION_MODEL_UNAVAILABLE": "轮动模型文件不可用",
    "ROTATION_MODEL_SCHEMA_INVALID": "轮动模型格式无效",
    "ROTATION_MODEL_GENERATED_AT_STALE": "轮动模型生成时间已过期",
    "ROTATION_DATA_FINGERPRINT_MISMATCH": "轮动模型与当前行情指纹不一致，已阻断",
    "ROTATION_SHA256_MISMATCH": "轮动模型校验值不匹配",
    "ROTATION_CONTRACT_INVALID": "轮动合约校验未通过",
    "ROTATION_UNAVAILABLE": "轮动结论暂不可用",
    "ROTATION_ACCEPTANCE_POLICY_MISMATCH": "轮动验收策略版本不一致",
    "ROTATION_MODEL_EXECUTION_POLICY_MISMATCH": "轮动执行策略版本不一致",
    "CALIBRATION_NOT_APPROVED": "事件校准未获批，暂不可开新仓",
    "CALIBRATION_NOT_LOADED": "事件校准尚未加载",
    "CALIBRATION_UNAVAILABLE": "事件校准产物不可用",
    "CALIBRATION_FINGERPRINT_MISMATCH": "事件校准与当前行情指纹不一致",
    "CALIBRATION_GENERATED_AT_STALE": "事件校准生成时间已过期",
    "CALIBRATION_PROMOTED": "校准结果已提升到生产",
    "CALIBRATION_STAGED_NOT_PROMOTED": "校准只在暂存区，尚未提升到生产",
    "CALIBRATION_FAILED_SAFE_FALLBACK": "校准失败，已回退到安全现金保护",
    "CALIBRATION_BUNDLE_INVALID": "校准包无效",
    "CALIBRATION_BUNDLE_UNAVAILABLE": "校准包不可用",
    "FORCED_CALIBRATION": "本次为强制重校准",
    "MARKET_MAINLINE_ONLY": "当前只允许主线标的开仓",
    "MAINLINE_ONLY": "仅限主线标的",
    "WEEKLY_TREND_NOT_CONFIRMED": "周线趋势还没确认",
    "MONTHLY_TREND_NEGATIVE": "月线趋势偏弱",
    "SETUP_NOT_CONFIRMED": "入场形态尚未成立",
    "EARLY_STOP_RISK_HIGH": "早期止损风险偏高",
    "EXPECTED_EXCESS_NOT_POSITIVE": "预期超额收益还没转正",
    "PRIORITY_BELOW_THRESHOLD": "优先级还没达到开仓门槛",
    "RISK_NOT_EXECUTABLE": "止损结构不适合执行",
    "DATA_QUALITY_NOT_VALID": "数据质量未通过",
    "MARKET_BLOCKED": "市场权限已阻断开仓",
    "ADJUSTMENT_FACTOR_CHANGED": "复权因子发生变化，需重新核对",
    "REGISTRY_NOT_APPROVED": "因子注册表未获批",
    "POLICY_SEASONING_INCOMPLETE": "策略还在熟成期",
    "POLICY_CANDIDATE_SPEC_CHANGED_RESET_SEASONING": "候选因子规格已变更，熟成期已重置",
    "FACTOR_SELECTION_GATE_FAILED": "因子选择门控未通过",
    "FACTOR_REGISTRY_DATA_FINGERPRINT_MISMATCH": "因子注册表与当前行情指纹不一致",
    "FACTOR_REGISTRY_CANDIDATE_FINGERPRINT_MISMATCH": "候选因子指纹不一致",
    "FACTOR_REGISTRY_SHA256_MISMATCH": "因子注册表校验值不匹配",
    "FACTOR_HEALTH_REGISTRY_IDENTITY_MISMATCH": "因子健康审计与注册表身份不一致",
    "FACTOR_HEALTH_SHA256_MISMATCH": "因子健康审计校验值不匹配",
    "ADAPTIVE_FACTOR_PROMOTION_NOT_READY": "自适应因子尚未达到可提升条件",
    "FACTOR_PROMOTION_READINESS_UNAVAILABLE": "因子提升就绪状态不可用",
    "FACTOR_PROMOTION_READINESS_STALE_OR_INVALID": "因子提升就绪状态过期或无效",
    "KEEP_CURRENT_ROTATION_AND_REEVALUATE_AFTER_NEW_LABELLED_DATES": "先保持当前轮动，等新样本后再评估",
    "WAITING_FOR_NEW_LABELLED_DATES_AND_STRONGER_CANDIDATES": "还在等待更新的样本和更强候选",
    "RECENT_IC_WEAK": "近期 IC 偏弱",
    "RECENT_IR_WEAK": "近期 IR 偏弱",
    "NEGATIVE_RECENT_IC": "近期 IC 为负",
    "FAST_DECAY": "因子衰减较快",
    "RESEARCH_FAMILY_REDUNDANT": "研究族内因子重复度偏高",
    "SELECTION_IC_HISTORY_INSUFFICIENT": "选择用 IC 历史样本不足",
    "SELECTION_STATUS_NOT_ACTIVE": "选择状态未激活",
    "TRAIN_SELECTION_SIGN_MISMATCH": "训练与选择阶段的方向不一致",
    "TRAIN_STATUS_RETIRED": "训练状态已退役",
    "LLM_REJECTED_EXPRESSION_COOLDOWN": "LLM 提案表达式处于冷却期",
    "PREVIOUS_CANDIDATE_FINGERPRINT_MISMATCH_RESET_SEASONING": "上一候选指纹变化，熟成期已重置",
    "DATA_MANIFEST_NOT_APPROVED": "行情清单未通过验收",
    "ALL_REQUIRED_SERIES_CURRENT": "所需行情序列均为最新",
    "MATCH": "本地与远程结论一致",
    "UP_TO_DATE": "生产闭环已是最新",
    "READY": "已就绪",
    "READY_LOCAL_ONLY": "目前只适合本机执行",
    "READY_FOR_EXTERNAL_PUBLISH": "已具备对外发布条件",
    "READY_FOR_EXECUTION_DATE_QUOTE_REVALIDATION": "待执行日行情复核后可继续",
    "REMOTE_IDENTITY_MISMATCH": "本地与远程发布身份不一致",
    "REMOTE_ONLY_DISTRIBUTION_BLOCKED": "远程只读分发链路暂不可用",
    "REMOTE_UNAVAILABLE": "远程发布暂不可用",
    "REMOTE_RELEASE_READY_FOR_PUBLISH": "远程发布包已可上线",
    "REMOTE_RELEASE_NOT_READY": "远程发布包尚未就绪",
    "REMOTE_CONTRACT_INVALID": "远程合约无效",
    "BUNDLE_MEMBER_HASH_MISMATCH": "产物包成员校验值不匹配",
    "BUNDLE_MEMBER_HASH_MISSING": "产物包成员缺少校验值",
    "BUNDLE_ID_MISMATCH": "产物包 ID 不一致",
    "BUNDLE_MANIFEST_INVALID": "产物包清单无效",
    "BUNDLE_MANIFEST_MISSING": "缺少产物包清单",
    "DISTRIBUTION_AUDIT_UNAVAILABLE": "分发一致性审计不可用",
    "DISTRIBUTION_URL_INVALID": "分发地址无效",
    "LOCAL_DISTRIBUTION_AUTHORITY_BLOCKED": "本机分发权威已被阻断",
    "PUBLIC_ROTATION_AUTHORITY_MISMATCH": "公开轮动权威身份不一致",
    "AUTOMATION_SCHEDULER_NOT_READY": "自动化调度尚未就绪",
    "NOT_INSTALLED": "相关组件未安装",
    "HISTORY_UNAVAILABLE": "历史记录不可用",
    "LATEST_UNAVAILABLE": "最新结果不可用",
    "LIVE_PERFORMANCE_UNAVAILABLE": "还没有可用的实盘绩效证据",
    "NO_LIVE_PERFORMANCE_EVIDENCE": "暂无实盘绩效证据",
    "LIVE_PERFORMANCE_NOT_YET_VALID": "实盘绩效证据尚未生效",
    "LIVE_PERFORMANCE_AUDIT_UNAVAILABLE": "实盘绩效审计不可用",
    "LIVE_PERFORMANCE_AUDIT_FAILED": "实盘绩效审计失败",
    "LIVE_PERFORMANCE_FINGERPRINT_MISMATCH": "实盘绩效指纹不一致",
    "LIVE_PERFORMANCE_AUTHORITY_BLOCKED": "实盘绩效权威已被阻断",
    "LIVE_PERFORMANCE_AUTHORITY_REVOKED": "实盘绩效权威已被撤销",
    "LIVE_PERFORMANCE_EVIDENCE_BLOCKED_SAFE_CASH": "实盘证据异常，已切到安全现金",
    "LIVE_PERFORMANCE_EVIDENCE_REJECTED": "实盘绩效证据被拒绝",
    "LIVE_PERFORMANCE_STALE": "实盘绩效证据已过期",
    "LIVE_PERFORMANCE_SESSION_MISSED": "实盘绩效会话缺失",
    "NO_FEEDBACK": "暂无执行反馈",
    "FEEDBACK_UNAVAILABLE": "执行反馈暂不可用",
    "FEEDBACK_REJECTED": "执行反馈被拒绝",
    "FEEDBACK_FINGERPRINT_MISMATCH": "执行反馈指纹不一致",
    "EXECUTION_FEEDBACK_NOT_YET_VALID": "执行反馈尚未生效",
    "EXECUTION_FEEDBACK_AUDIT_UNAVAILABLE": "执行反馈审计不可用",
    "EXECUTION_FEEDBACK_AUDIT_FAILED": "执行反馈审计失败",
    "EXECUTION_FEEDBACK_AUTHORITY_BLOCKED": "执行反馈权威已被阻断",
    "EXECUTION_FEEDBACK_AUTHORITY_REVOKED": "执行反馈权威已被撤销",
    "EXECUTION_FEEDBACK_EVIDENCE_BLOCKED_SAFE_CASH": "执行反馈异常，已切到安全现金",
    "INSUFFICIENT_EVIDENCE": "证据还不够，不能升级策略",
    "BROKER_CONFIRMATION_OVERDUE": "券商确认已逾期",
    "EXECUTION_SESSION_MISSED": "执行会话缺失",
    "COST_MODEL_RECALIBRATION_REQUIRED": "成本模型需要重新校准",
    "MODEL_ESTIMATE_ONLY": "仅有模型估算，尚未券商确认",
    "BROKER_CONFIRMED": "券商已确认",
    "BROKER_EVIDENCE_REJECTED": "券商证据被拒绝",
    "UNAPPROVED_FEEDBACK_SOURCE": "反馈来源未获批准",
    "UNAPPROVED_LIVE_PERFORMANCE_SOURCE": "实盘绩效来源未获批准",
    "STRATEGY_FINGERPRINT_MISMATCH": "策略指纹不一致",
    "REGISTRY_LIVE_FINGERPRINT_MISMATCH": "注册表与线上指纹不一致",
    "MODEL_VERSION_MISMATCH": "模型版本不一致",
    "ACTIVE_PROVIDER_MODEL_IDENTITY_MISMATCH": "当前服务商模型身份不一致",
    "SWING_ROTATION_CACHE_MISMATCH": "波段端轮动缓存不一致",
    "SWING_ROTATION_CACHE_UNAVAILABLE": "波段端轮动缓存不可用",
    "OBSERVE_ONLY": "仅观察，不开新仓",
    "RISK_OFF": "风险规避",
    "RISK_ON": "风险偏好",
    "PULSE_EARLY": "早期脉冲",
    "DATA_FINGERPRINT_MISMATCH": "与当前行情指纹不一致",
    "LIVE_HEALTH_REGISTRY_MISMATCH": "线上健康状态与因子注册表不一致",
    "LIVE_HEALTH_NOT_APPROVED": "线上因子健康未获批",
    "REGENERATE_LIVE_FACTOR_HEALTH_FOR_CURRENT_REGISTRY": "需按当前注册表重新生成线上因子健康",
    "MONITOR_LIVE_FACTOR_HEALTH": "需持续监控线上因子健康",
    "UNKNOWN": "未知",
}

ARTIFACT_NAME_LABELS = {
    "v4_calibration.json": "事件校准文件",
    "rotation_model.json": "轮动模型文件",
    "adaptive_factor_registry.json": "自适应因子注册表",
    "v4_acceptance_report.json": "验收报告",
    "llm_factor_proposals.json": "LLM 因子提案",
    "calibration_bundle.json": "校准包清单",
    "data_manifest_latest.json": "行情清单",
    "etf_rotation_latest.json": "最新轮动结论",
    "etf_signals_latest.json": "最新信号清单",
    "factor_health_latest.json": "因子健康报告",
    "joint_health_latest.json": "联合健康报告",
    "live_performance_audit_latest.json": "实盘绩效审计",
    "execution_feedback_audit_latest.json": "执行反馈审计",
    "cycle_status_latest.json": "生产闭环状态",
    "distribution_audit_latest.json": "分发一致性审计",
}

POLICY_PULSE_LABELS = {
    "NONE": "无脉冲",
    "MILD": "温和脉冲",
    "STRONG": "强脉冲",
    "FULL": "满仓脉冲",
    "EARLY": "早期脉冲",
    "HARD": "防守脉冲",
    "CAUTIOUS": "谨慎脉冲",
}

TRUST_LABELS = {
    "follow": "可跟随",
    "observe": "仅观察",
    "distrust": "不可信",
}

INDUSTRY_GROUP_LABELS = {
    "advanced_manufacturing": "高端制造",
    "technology": "科技",
    "financials": "金融",
    "materials": "原材料",
    "energy_materials": "能源材料",
    "clean_energy": "清洁能源",
    "utilities": "公用事业",
    "consumer": "消费",
    "healthcare": "医药",
    "precious_metals": "贵金属",
    "broad_market": "宽基",
    "other": "其他",
}


def _value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _label(value: Any, labels: Dict[str, str]) -> str:
    raw = str(_value(value) or "").strip()
    if not raw:
        return "—"
    return labels.get(raw, raw)


def _model_label(value: Any) -> str:
    raw = str(_value(value) or "").strip()
    if not raw:
        return "未提供"
    return MODEL_VERSION_LABELS.get(raw, raw)


def _reason_label(value: Any) -> str:
    raw = str(_value(value) or "").strip()
    if not raw:
        return ""
    if raw in REASON_CODE_LABELS:
        return REASON_CODE_LABELS[raw]
    if raw in ARTIFACT_NAME_LABELS:
        return ARTIFACT_NAME_LABELS[raw]
    if ":" in raw and not raw.startswith("http"):
        prefix, _, detail = raw.partition(":")
        head = REASON_CODE_LABELS.get(
            prefix,
            ARTIFACT_NAME_LABELS.get(prefix, prefix),
        )
        detail = detail.strip()
        if not detail:
            return head
        if detail in REASON_CODE_LABELS:
            tail = REASON_CODE_LABELS[detail]
        elif detail in ARTIFACT_NAME_LABELS:
            tail = ARTIFACT_NAME_LABELS[detail]
        else:
            first, sep, rest = detail.partition(":")
            if sep and first in REASON_CODE_LABELS:
                tail = (
                    f"{REASON_CODE_LABELS[first]}：{rest.strip()}"
                    if rest.strip()
                    else REASON_CODE_LABELS[first]
                )
            elif detail.startswith("HTTP Error ") or detail.startswith("HTTP error "):
                body = detail.split(" ", 2)[-1]
                code, _, _msg = body.partition(":")
                code = code.strip()
                if code == "404":
                    tail = "远程文件不存在（404）"
                elif code.isdigit():
                    tail = f"远程访问失败（HTTP {code}）"
                else:
                    tail = "远程访问失败"
            else:
                tail = detail
        return f"{head}：{tail}"
    return raw





def _market_tone(value: Any, bullish: set[str], bearish: set[str]) -> str:
    raw = str(_value(value) or "")
    if raw in bullish:
        return "tone-up"
    if raw in bearish:
        return "tone-down"
    return "tone-neutral"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _pct(value: Any, digits: int = 0) -> str:
    try:
        number = float(value or 0.0) * 100.0
    except (TypeError, ValueError):
        number = 0.0
    return f"{number:.{digits}f}%"


def _load_enricher():
    try:
        from etf_radar._core import enrich_v4_signal

        return enrich_v4_signal
    except Exception:
        return None



def _file_href(path: Path) -> str:
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return str(path)


def _permission_story(
    environment: Dict[str, Any],
    rotation: Dict[str, Any],
    breadth: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    policy = dict(rotation.get("market_policy") or {})
    score = policy.get("score", environment.get("total_score"))
    raw_score = policy.get("raw_score", environment.get("total_score"))
    atr_pct = environment.get("atr_pct")
    atr_percentile = (
        policy.get("benchmark_natr_percentile")
        if policy.get("benchmark_natr_percentile") is not None
        else environment.get("atr_percentile")
    )
    bull = policy.get("bull_count", breadth.get("bull"))
    bear = policy.get("bear_count", breadth.get("bear"))
    total = policy.get("total_count", breadth.get("total"))
    breadth_balance = policy.get("breadth_balance")
    market_safe = environment.get("market_safe")
    risk_level = environment.get("risk_level") or "—"
    market_status = environment.get("status") or environment.get("regime_level_label") or "—"

    reasons: List[str] = []
    if decision.get("risk_control_only"):
        reasons.append(f"当前发布的是风控现金目标：{decision.get('reason_label')}")
    permission = str(decision.get("permission") or "")
    if permission in {"BLOCKED", "OBSERVE_ONLY"}:
        reasons.append(f"开仓权限为{decision.get('permission_label')}，不允许新的进攻性配置")
    elif permission == "MAINLINE_ONLY":
        reasons.append("开仓权限收缩为仅限主线标的，普通候选不能直接开新仓")
    elif permission in {"TRADEABLE", "OPEN"}:
        reasons.append("开仓权限打开，可按已获批目标执行")

    if market_safe is False:
        reasons.append(f"基准环境偏防守：{market_status}，风险等级 {risk_level}")
    if atr_percentile is not None and float(atr_percentile) >= 90:
        reasons.append(f"基准波动分位高达 {float(atr_percentile):.1f}，属于高压波动区")
    if bull is not None and bear is not None and total:
        if float(bear) > float(bull):
            reasons.append(f"市场宽度偏空：多 {bull} / 空 {bear}（共 {total}）")
    if policy.get("data_manifest_approved") is False:
        reasons.append("行情清单未通过验收，权限会继续收缩")
    if str(policy.get("factor_health_status") or "") in {"SUSPENDED", "BLOCKED", "DEGRADED"}:
        reasons.append(f"因子健康为{_label(policy.get('factor_health_status'), FACTOR_HEALTH_LABELS)}")
    if not reasons:
        reasons.append("当前权限由市场状态、宽度、波动与验收门控共同决定")

    cards = [
        {
            "label": "权限结论",
            "value": decision.get("permission_label") or "—",
            "detail": decision.get("reason_label") or "未提供明确原因",
            "tone": decision.get("trust") or "observe",
            "icon": "lock-keyhole",
        },
        {
            "label": "市场评分",
            "value": f"{float(score or 0):+.2f}",
            "detail": f"原始分 {float(raw_score or 0):+.2f} · 状态 {market_status} · 风险 {risk_level}",
            "tone": "down" if float(score or 0) < 0 else "up" if float(score or 0) > 0 else "neutral",
            "icon": "gauge",
        },
        {
            "label": "波动压力",
            "value": f"{float(atr_percentile or 0):.1f}",
            "detail": f"ATR {float(atr_pct or 0):.2f}% · 分位越高表示波动越极端",
            "tone": "down" if float(atr_percentile or 0) >= 90 else "neutral",
            "icon": "activity",
        },
        {
            "label": "宽度压力",
            "value": f"{bull or 0} / {bear or 0}",
            "detail": (
                f"多头/空头"
                + (f" · 平衡度 {float(breadth_balance):+.2f}" if breadth_balance is not None else "")
                + f" · 宽度信号 {breadth.get('signal') or '—'}"
            ),
            "tone": "down" if float(bear or 0) > float(bull or 0) else "up",
            "icon": "waves",
        },
    ]
    pulse_raw = str(policy.get("policy_pulse") or "").strip()
    pulse = _label(pulse_raw, POLICY_PULSE_LABELS) if pulse_raw else "—"
    vol_ratio = float(environment.get("vol_ratio") or 0.0)
    close_vs_ma60 = float(environment.get("close_vs_ma60_pct") or 0.0)
    close_vs_ma120 = float(environment.get("close_vs_ma120_pct") or 0.0)
    pulse_ret_5 = policy.get("policy_pulse_ret_5")
    safe_label = "安全" if market_safe is True else "防守" if market_safe is False else "未知"
    return {
        "cards": cards,
        "reasons": reasons[:5],
        "benchmark_name": environment.get("index_name") or "沪深300ETF",
        "benchmark_code": environment.get("index_code") or "510300",
        "benchmark_date": environment.get("date") or decision.get("data_date") or "—",
        "price": float(environment.get("price") or 0.0),
        "close_vs_ma20_pct": float(environment.get("close_vs_ma20_pct") or 0.0),
        "close_vs_ma60_pct": close_vs_ma60,
        "close_vs_ma120_pct": close_vs_ma120,
        "authority": rotation.get("exposure_authority_label") or "—",
        "cash_label": decision.get("cash_label") or "—",
        "exposure_label": decision.get("exposure_label") or "—",
        "risk_level": risk_level,
        "market_safe_label": safe_label,
        "market_status": market_status,
        "vol_ratio": vol_ratio,
        "policy_pulse": pulse,
        "policy_pulse_ret_5": float(pulse_ret_5) if pulse_ret_5 is not None else None,
        "data_manifest_approved": bool(policy.get("data_manifest_approved", True)),
    }


class HTMLReporter:
    """Render the public dashboard without embedding assets in Python source."""

    @staticmethod
    def _compute_breadth(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        empty = {
            "total": 0,
            "bull": 0,
            "bear": 0,
            "neutral": 0,
            "bull_pct": 0.0,
            "bear_pct": 0.0,
            "neutral_pct": 0.0,
            "ratio": 0.0,
            "signal": "无数据",
            "basis": "raw_status",
            "tone": "tone-neutral",
        }
        if not results:
            return empty

        bullish = {
            "极强波段多头", "波段多头", "偏多企稳",
            "BULL", "EXTREME_BULL", "WEAK_BULL", "多头",
        }
        bearish = {
            "极弱波段空头", "波段空头", "偏空走弱",
            "BEAR", "EXTREME_BEAR", "WEAK_BEAR", "空头",
        }

        def classify(item: Dict[str, Any]) -> str:
            candidates = [
                item.get("raw_status"),
                item.get("status"),
                ((item.get("trend") or {}) if isinstance(item.get("trend"), dict) else {}).get("state"),
            ]
            for candidate in candidates:
                value = str(_value(candidate) or "").strip()
                if not value:
                    continue
                if value in bullish or "多头" in value or value.upper() == "BULL":
                    return "bull"
                if value in bearish or "空头" in value or value.upper() == "BEAR":
                    return "bear"
            return "neutral"

        labels = [classify(item) for item in results]
        total = len(labels)
        bull = sum(label == "bull" for label in labels)
        bear = sum(label == "bear" for label in labels)
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
            "basis": "status_or_trend",
        }
        breadth["tone"] = _market_tone(
            signal,
            {"偏多", "极度乐观"},
            {"偏空", "极度悲观"},
        )
        return breadth

    @staticmethod
    def _normalize_breadth(payload: Optional[Dict[str, Any]], basis: str) -> Dict[str, Any]:
        data = dict(payload or {})
        total = int(data.get("total") or 0)
        bull = int(data.get("bull") or data.get("bull_count") or 0)
        bear = int(data.get("bear") or data.get("bear_count") or 0)
        neutral = int(data.get("neutral") or data.get("neutral_count") or max(0, total - bull - bear))
        if total <= 0:
            total = bull + bear + neutral
        if total <= 0:
            return {
                "total": 0, "bull": 0, "bear": 0, "neutral": 0,
                "bull_pct": 0.0, "bear_pct": 0.0, "neutral_pct": 0.0,
                "ratio": 0.0, "signal": "无数据", "basis": basis, "tone": "tone-neutral",
            }
        ratio = bull / max(1, bear)
        signal = str(data.get("signal") or (
            "极度乐观" if ratio > 3.0 else "偏多" if ratio > 1.5 else
            "中性" if ratio >= 0.67 else "偏空" if ratio >= 0.33 else "极度悲观"
        ))
        tone = data.get("tone")
        if tone in {"market-up", "market-down", "market-neutral"}:
            tone = {
                "market-up": "tone-up",
                "market-down": "tone-down",
                "market-neutral": "tone-neutral",
            }[tone]
        if tone not in {"tone-up", "tone-down", "tone-neutral"}:
            tone = _market_tone(signal, {"偏多", "极度乐观"}, {"偏空", "极度悲观"})
        return {
            "total": total,
            "bull": bull,
            "bear": bear,
            "neutral": neutral,
            "bull_pct": round(float(data.get("bull_pct") if data.get("bull_pct") is not None else bull / total * 100.0), 1),
            "bear_pct": round(float(data.get("bear_pct") if data.get("bear_pct") is not None else bear / total * 100.0), 1),
            "neutral_pct": round(float(data.get("neutral_pct") if data.get("neutral_pct") is not None else neutral / total * 100.0), 1),
            "ratio": round(float(data.get("ratio") if data.get("ratio") is not None else ratio), 2),
            "signal": signal,
            "basis": basis,
            "tone": tone,
        }

    @classmethod
    def _resolve_breadth(
        cls,
        results: List[Dict[str, Any]],
        rotation: Dict[str, Any],
    ) -> Dict[str, Any]:
        published = _read_json(PATHS.public / "etf_signals_latest.json")
        published_breadth = cls._normalize_breadth(
            published.get("market_breadth") or {},
            "signals_market_breadth",
        )
        if published_breadth.get("total") and (
            published_breadth.get("bull")
            or published_breadth.get("bear")
            or published_breadth.get("neutral")
        ):
            return published_breadth

        computed = cls._compute_breadth(results)
        if computed.get("bull") or computed.get("bear") or computed.get("total"):
            return computed

        policy = dict(rotation.get("market_policy") or {})
        policy_breadth = cls._normalize_breadth(
            {
                "total": policy.get("total_count"),
                "bull": policy.get("bull_count"),
                "bear": policy.get("bear_count"),
            },
            "rotation_market_policy",
        )
        if policy_breadth.get("bull") or policy_breadth.get("bear"):
            return policy_breadth
        return computed

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
                    "industry_group": _label(
                        meta.get("industry_group") or "",
                        INDUSTRY_GROUP_LABELS,
                    ),
                    "industry_group_code": meta.get("industry_group") or "",
                    "rotation_score": float(meta.get("rotation_score") or 0.0),
                }
            )
        model_version = payload.get("model_version", "")
        reason_raw = payload.get("reason") or market_policy.get("risk_reason") or ""
        return {
            "present": bool(payload),
            "data_date": payload.get("data_date", ""),
            "execution_date": payload.get("execution_date", ""),
            "generated_at": payload.get("generated_at", ""),
            "model_version": model_version,
            "model_version_label": _model_label(model_version),
            "approved": bool(payload.get("approved")),
            "alpha_model_approved": bool(payload.get("alpha_model_approved")),
            "risk_control_only": bool(payload.get("risk_control_only")),
            "cash_weight": float(payload.get("cash_weight") or 0.0),
            "max_exposure_ratio": float(
                payload.get("max_exposure_ratio")
                if payload.get("max_exposure_ratio") is not None
                else market_policy.get("max_exposure_ratio") or 0.0
            ),
            "targets": targets,
            "target_count": len(targets),
            "reason_raw": reason_raw,
            "reason_label": _reason_label(reason_raw) or "未提供明确原因",
            "exposure_authority_label": _label(
                payload.get("exposure_authority"),
                EXPOSURE_AUTHORITY_LABELS,
            ),
            "state_reset_reason_label": (
                "" if not payload.get("state_reset_reason")
                else _label(payload.get("state_reset_reason"), STATE_RESET_LABELS)
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
            "entry_permission": str(market_policy.get("entry_permission") or ""),
            "policy_state": str(market_policy.get("state") or ""),
            "market_policy": market_policy,
        }

    @classmethod
    def _health_context(cls) -> Dict[str, Any]:
        factor = _read_json(PATHS.public / "factor_health_latest.json")
        cycle = _read_json(PATHS.public / "cycle_status_latest.json")
        distribution = _read_json(PATHS.public / "distribution_audit_latest.json")
        joint = _read_json(PATHS.public / "joint_health_latest.json")
        live = _read_json(PATHS.public / "live_performance_audit_latest.json")
        feedback = _read_json(PATHS.public / "execution_feedback_audit_latest.json")
        promotion = _read_json(PATHS.public / "factor_promotion_readiness_latest.json")

        factor_status = str(factor.get("status") or "UNKNOWN")
        cycle_status = str(cycle.get("status") or "UNKNOWN")
        dist_status = str(distribution.get("status") or "UNKNOWN")
        joint_status = str(joint.get("status") or "UNKNOWN")
        live_status = str(live.get("status") or "UNKNOWN")
        feedback_status = str(feedback.get("status") or "UNKNOWN")

        lamps = [
            {
                "id": "cycle",
                "label": "生产闭环",
                "status": cycle_status,
                "status_label": _reason_label(cycle_status) or cycle_status,
                "ok": cycle_status in {"UP_TO_DATE", "CALIBRATION_PROMOTED", "READY"},
                "detail": "、".join(_reason_label(item) for item in (cycle.get("reasons") or [])[:2]) or "闭环状态可用",
                "icon": "refresh-cw",
            },
            {
                "id": "distribution",
                "label": "分发一致性",
                "status": dist_status,
                "status_label": _reason_label(dist_status) or dist_status,
                "ok": bool(distribution.get("identity_match")) or dist_status == "MATCH",
                "detail": "本地与远程目标一致" if dist_status == "MATCH" else "需要核对远程发布",
                "icon": "globe-2",
            },
            {
                "id": "factor",
                "label": "因子健康",
                "status": factor_status,
                "status_label": _label(factor_status, FACTOR_HEALTH_LABELS),
                "ok": bool(factor.get("approved_for_live_use")),
                "detail": "、".join(_reason_label(item) for item in (factor.get("reasons") or [])[:2]) or "无线上因子告警",
                "icon": "activity",
            },
            {
                "id": "joint",
                "label": "联合健康",
                "status": joint_status,
                "status_label": _reason_label(joint_status) or joint_status,
                "ok": bool(joint.get("same_host_execution_allowed")),
                "detail": "、".join(_reason_label(item) for item in (joint.get("warnings") or [])[:2]) or "本机链路可用",
                "icon": "link-2",
            },
            {
                "id": "live",
                "label": "实盘证据",
                "status": live_status,
                "status_label": _reason_label(live_status) or live_status,
                "ok": live_status not in {"NO_LIVE_PERFORMANCE_EVIDENCE", "LIVE_PERFORMANCE_UNAVAILABLE", ""},
                "detail": _reason_label(live.get("source_status")) or "等待实盘证据",
                "icon": "line-chart",
            },
            {
                "id": "feedback",
                "label": "执行反馈",
                "status": feedback_status,
                "status_label": _reason_label(feedback_status) or feedback_status,
                "ok": bool(feedback.get("feedback_ingested")),
                "detail": f"确认样本 {int(feedback.get('confirmed_sample_count') or 0)}",
                "icon": "clipboard-check",
            },
        ]
        promotion_status = str(promotion.get("status") or "")
        return {
            "lamps": lamps,
            "promotion_status": promotion_status,
            "promotion_label": _reason_label(promotion_status) or promotion_status or "未提供",
            "promotion_allowed": bool(promotion.get("promotion_allowed")),
            "factor_approved": bool(factor.get("approved_for_live_use")),
            "same_host_ok": bool(joint.get("same_host_execution_allowed", True)),
            "remote_ok": bool(
                joint.get("remote_only_execution_allowed", False)
                or distribution.get("remote_only_execution_allowed")
            ),
        }

    @classmethod
    def _decision_context(
        cls,
        environment: Dict[str, Any],
        rotation: Dict[str, Any],
        health: Dict[str, Any],
    ) -> Dict[str, Any]:
        permission = str(
            rotation.get("entry_permission")
            or environment.get("entry_permission")
            or ""
        )
        risk_only = bool(rotation.get("risk_control_only"))
        exposure = float(
            rotation.get("max_exposure_ratio")
            if rotation.get("present") and rotation.get("max_exposure_ratio") is not None
            else environment.get("max_exposure_ratio") or 0.0
        )
        cash = float(
            rotation.get("cash_weight")
            if rotation.get("present")
            else max(0.0, 1.0 - exposure)
        )
        approved = bool(rotation.get("approved")) if rotation.get("present") else False

        if not health.get("same_host_ok", True) and not health.get("remote_ok", False):
            trust = "distrust"
            headline = "系统暂不可信"
            summary = "联合健康或分发链路异常，先不要跟随今日目标。"
        elif risk_only or exposure <= 0 or permission in {"BLOCKED", "OBSERVE_ONLY"}:
            trust = "observe"
            headline = "现金保护"
            reason = rotation.get("reason_label") or "当前不允许新的进攻性配置"
            if risk_only or exposure <= 0:
                summary = f"这是风控现金保护，不是“没有好行业”。{reason}。"
            else:
                summary = f"{reason}。先观察，不要把雷达排名当成下单清单。"
        elif permission in {"TRADEABLE", "OPEN", "MAINLINE_ONLY", "SELECTIVE"} and approved:
            trust = "follow"
            headline = "可跟随目标"
            summary = "生产端已给出可执行轮动目标，执行端应按权重落地。"
        else:
            trust = "observe"
            headline = "仅观察"
            summary = "权限或验收未完全打开，保留研究视角，不作为下单指令。"

        radar_collapsed = True
        if trust == "follow" and not risk_only and exposure > 0:
            radar_collapsed = False
        return {
            "trust": trust,
            "trust_label": TRUST_LABELS[trust],
            "headline": headline,
            "summary": summary,
            "radar_collapsed": radar_collapsed,
            "permission": permission,
            "permission_label": _label(permission, ENTRY_PERMISSION_LABELS),
            "exposure": exposure,
            "exposure_label": _pct(exposure, 0),
            "cash": cash,
            "cash_label": _pct(cash, 0),
            "approved": approved,
            "risk_control_only": risk_only,
            "execution_date": rotation.get("execution_date") or environment.get("date") or "—",
            "data_date": rotation.get("data_date") or environment.get("date") or "—",
            "model_label": rotation.get("model_version_label") or "未提供",
            "reason_label": rotation.get("reason_label") or "未提供明确原因",
        }

    @classmethod
    def _row_context(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enrich = None
        rows: List[Dict[str, Any]] = []
        for item in sorted(
            results,
            key=lambda value: float(value.get("v4_priority", 0.0) or 0.0),
            reverse=True,
        ):
            # Prefer already-enriched signal fields when present to avoid heavy calibration IO.
            if item.get("entry") is not None and item.get("calibration") is not None:
                signal = item
            else:
                if enrich is None:
                    enrich = _load_enricher()
                signal = enrich(dict(item)) if callable(enrich) else {}
            entry = dict((signal or {}).get("entry") or item.get("entry") or {})
            risk = dict((signal or {}).get("risk") or item.get("risk") or {})
            relative = dict((signal or {}).get("relative_strength") or item.get("relative_strength") or {})
            calibration = dict((signal or {}).get("calibration") or item.get("calibration") or {})
            trend = dict((signal or {}).get("trend") or item.get("trend") or {})
            reasons = [str(reason) for reason in (entry.get("reasons") or [])]
            reason_labels = [_reason_label(reason) for reason in reasons[:2] if reason]
            entry_state = str(entry.get("state") or ("BLOCKED" if reasons else "WATCH"))
            stop_dist = item.get("stop_dist")
            if stop_dist is None:
                stop_dist = risk.get("stop_dist_pct", 0.0)
            status_raw = str(_value(item.get("raw_status", item.get("status", ""))) or "").strip()
            trend_state = str(_value(trend.get("state") or "") or "").strip()
            placeholder_status = {
                "", "x", "X", "n/a", "N/A", "none", "None", "null", "NULL", "-"
            }
            if status_raw in placeholder_status:
                status_label = _label(trend_state, TREND_STATE_LABELS) if trend_state else "—"
            else:
                status_label = status_raw
            if not status_label or status_label == "—":
                status_label = _label(trend_state, TREND_STATE_LABELS) if trend_state else "—"

            rows.append(
                {
                    "code": item.get("code", ""),
                    "name": item.get("name", ""),
                    "price": float(item.get("price", 0.0) or 0.0),
                    "status": status_label,
                    "priority": float(item.get("v4_priority", entry.get("priority", 0.0)) or 0.0),
                    "rps": float(item.get("rps", relative.get("score", 0.0)) or 0.0),
                    "stop_loss": float(item.get("stop_loss", risk.get("stop_loss", 0.0)) or 0.0),
                    "stop_dist": float(stop_dist or 0.0),
                    "data_date": item.get("data_date", ""),
                    "entry_state": entry_state,
                    "entry_state_label": _label(entry_state, ENTRY_STATE_LABELS),
                    "entry_tone": (
                        "tone-up" if entry_state == "READY"
                        else "tone-warn" if entry_state == "WATCH"
                        else "tone-down"
                    ),
                    "reason_labels": reason_labels or ["暂无额外原因"],
                    "trend_state_label": _label(trend.get("state"), TREND_STATE_LABELS),
                    "calibration_approved": bool(calibration.get("approved")),
                    "win_probability": float(calibration.get("win_probability_10d") or 0.0),
                    "expected_excess": float(calibration.get("expected_excess_return_10d") or 0.0),
                }
            )
        return rows

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
        rows = cls._row_context(results)
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
                    {"TRADEABLE", "OPEN", "MAINLINE_ONLY"},
                    {"BLOCKED", "OBSERVE_ONLY"},
                )
            if rotation_data.get("policy_state_label"):
                environment_data["regime_level_label"] = rotation_data["policy_state_label"]
                environment_data["regime_level_tone"] = _market_tone(
                    (rotation or {}).get("market_policy", {}).get("state"),
                    {"RISK_ON", "PULSE_FULL"},
                    {"RISK_OFF", "PULSE_HARD"},
                )
        health = cls._health_context()
        decision = cls._decision_context(environment_data, rotation_data, health)
        breadth = cls._resolve_breadth(results, rotation_data)
        generated_at = datetime.now()
        document = template.render(
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            asset_version=generated_at.strftime("%Y%m%d%H%M%S"),
            environment=environment_data,
            breadth=breadth,
            rows=rows,
            rotation=rotation_data,
            health=health,
            decision=decision,
            bridge={
                "production_label": "生产端 · 权威结论",
                "execution_label": "执行端 · 落地结果",
                "production_href": "https://etf.imlam.com/",
                "execution_href": "https://swing.imlam.com/",
            },
            permission_story=_permission_story(environment_data, rotation_data, breadth, decision),
            ready_count=sum(1 for row in rows if row.get("entry_state") == "READY"),
            watch_count=sum(1 for row in rows if row.get("entry_state") == "WATCH"),
            blocked_count=sum(1 for row in rows if row.get("entry_state") == "BLOCKED"),
        )
        (PATHS.public / "index.html").write_text(document, encoding="utf-8")
