"""Point-in-time market-data freshness and universe consistency manifest."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
APPROVED_DATA_SOURCES = {
    "TENCENT_SINA_VALIDATED",
    "CACHE_TENCENT_SINA_VALIDATED",
}


def shanghai_now(value: Optional[datetime] = None) -> datetime:
    if value is None:
        return datetime.now(SHANGHAI_TZ)
    if value.tzinfo is None:
        return value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(SHANGHAI_TZ)


def expected_latest_completed_date(
    calendar: Iterable[Any],
    now: Optional[datetime] = None,
    close_buffer: time = time(15, 15),
) -> Optional[pd.Timestamp]:
    current = shanghai_now(now)
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna().normalize()
    if dates.empty:
        dates = pd.bdate_range(current.date() - pd.Timedelta(days=40), current.date()).normalize()
    cutoff = pd.Timestamp(current.date())
    if current.time() < close_buffer:
        cutoff -= pd.Timedelta(days=1)
    eligible = dates[dates <= cutoff]
    return pd.Timestamp(eligible.max()).normalize() if len(eligible) else None


def trading_day_lag(
    data_date: Any,
    expected_date: Any,
    calendar: Iterable[Any],
) -> int:
    data = pd.to_datetime(data_date, errors="coerce")
    expected = pd.to_datetime(expected_date, errors="coerce")
    if pd.isna(data) or pd.isna(expected):
        return 999
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna().normalize()
    if dates.empty:
        dates = pd.bdate_range(pd.Timestamp(data).normalize(), pd.Timestamp(expected).normalize())
    return int(((dates > pd.Timestamp(data).normalize()) & (dates <= pd.Timestamp(expected).normalize())).sum())


def next_trading_date(
    calendar: Iterable[Any],
    after_date: Any,
) -> Optional[pd.Timestamp]:
    anchor = pd.to_datetime(after_date, errors="coerce")
    if pd.isna(anchor):
        return None
    dates = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna().normalize()
    eligible = dates[dates > pd.Timestamp(anchor).normalize()]
    return pd.Timestamp(eligible.min()).normalize() if len(eligible) else None


def analyzer_data_record(
    analyzer: Any,
    expected_date: Optional[pd.Timestamp],
    calendar: Iterable[Any],
) -> Dict[str, Any]:
    qfq_date = pd.to_datetime(
        analyzer.df_daily["date"].iloc[-1] if not analyzer.df_daily.empty else None,
        errors="coerce",
    )
    raw_date = pd.to_datetime(
        analyzer.df_raw["date"].iloc[-1] if not analyzer.df_raw.empty else None,
        errors="coerce",
    )
    reasons = []
    if pd.isna(qfq_date):
        reasons.append("QFQ_DATE_MISSING")
    if pd.isna(raw_date):
        reasons.append("RAW_DATE_MISSING")
    if not pd.isna(qfq_date) and not pd.isna(raw_date) and qfq_date.normalize() != raw_date.normalize():
        reasons.append("RAW_QFQ_DATE_MISMATCH")
    lag = trading_day_lag(qfq_date, expected_date, calendar)
    if lag > 0:
        reasons.append("DATA_BEHIND_EXPECTED_TRADING_DATE")
    if (
        not pd.isna(qfq_date)
        and expected_date is not None
        and qfq_date.normalize() > pd.Timestamp(expected_date).normalize()
    ):
        reasons.append("DATA_AHEAD_OF_EXPECTED_TRADING_DATE")
    quality = dict(getattr(analyzer, "data_quality", {}) or {})
    if quality.get("status") == "BLOCKED":
        reasons.append("PRICE_BASIS_BLOCKED")
    if "RAW_PRICE_FALLBACK_TO_QFQ" in quality.get("reasons", []):
        reasons.append("RAW_PRICE_NOT_INDEPENDENT")
    if "FALLBACK_BUSINESS_CALENDAR" in quality.get("reasons", []):
        reasons.append("TRADING_CALENDAR_UNVERIFIED")
    source = str(getattr(analyzer, "data_source", "UNKNOWN"))
    source_audit = dict(getattr(analyzer, "data_source_audit", {}) or {})
    if source not in APPROVED_DATA_SOURCES:
        reasons.append("INDEPENDENT_SOURCE_CROSSCHECK_MISSING")
    elif not bool(source_audit.get("approved", False)):
        reasons.append("SOURCE_VALIDATION_AUDIT_MISSING_OR_FAILED")
    return {
        "code": str(getattr(analyzer, "code", "")),
        "name": str(getattr(analyzer, "name", "")),
        "source": source,
        "source_validation": source_audit,
        "loaded_at": str(getattr(analyzer, "data_loaded_at", "")),
        "qfq_data_date": qfq_date.strftime("%Y-%m-%d") if not pd.isna(qfq_date) else None,
        "raw_data_date": raw_date.strftime("%Y-%m-%d") if not pd.isna(raw_date) else None,
        "qfq_rows": int(len(analyzer.df_daily)),
        "raw_rows": int(len(analyzer.df_raw)),
        "trading_day_lag": int(lag),
        "quality_status": str(quality.get("status", "UNKNOWN")),
        "status": "CURRENT" if not reasons else "BLOCKED",
        "reasons": list(dict.fromkeys(reasons)),
    }


def build_data_manifest(
    analyzers: Sequence[Any],
    expected_codes: Sequence[str],
    benchmark_code: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current = shanghai_now(now)
    benchmark = next(
        (item for item in analyzers if str(getattr(item, "code", "")) == str(benchmark_code)),
        None,
    )
    calendar = getattr(benchmark, "trading_calendar", pd.DatetimeIndex([])) if benchmark else pd.DatetimeIndex([])
    expected_date = expected_latest_completed_date(calendar, current)
    records = [analyzer_data_record(item, expected_date, calendar) for item in analyzers]
    by_code = {item["code"]: item for item in records if item["code"]}
    required = list(dict.fromkeys(str(code) for code in expected_codes))
    missing = [code for code in required if code not in by_code]
    blocked = [code for code in required if code in by_code and by_code[code]["status"] != "CURRENT"]
    observed_dates = sorted(
        {str(by_code[code].get("qfq_data_date")) for code in required if code in by_code and by_code[code].get("qfq_data_date")}
    )
    approved = bool(expected_date is not None and not missing and not blocked and len(observed_dates) == 1)
    return {
        "schema_version": 1,
        "generated_at": current.strftime("%Y-%m-%d %H:%M:%S%z"),
        "timezone": "Asia/Shanghai",
        "benchmark_code": str(benchmark_code),
        "expected_latest_data_date": expected_date.strftime("%Y-%m-%d") if expected_date is not None else None,
        "observed_data_dates": observed_dates,
        "approved": approved,
        "reason": (
            "ALL_REQUIRED_SERIES_CURRENT"
            if approved
            else "MISSING_OR_STALE_OR_INCONSISTENT_MARKET_DATA"
        ),
        "required_count": len(required),
        "current_count": sum(code in by_code and by_code[code]["status"] == "CURRENT" for code in required),
        "missing_codes": missing,
        "blocked_codes": blocked,
        "source_counts": dict(Counter(item["source"] for item in records)),
        "records": sorted(records, key=lambda item: item["code"]),
    }


__all__ = [
    "APPROVED_DATA_SOURCES",
    "analyzer_data_record",
    "build_data_manifest",
    "expected_latest_completed_date",
    "next_trading_date",
    "shanghai_now",
    "trading_day_lag",
]
