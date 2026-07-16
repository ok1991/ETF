"""ETF signal v3 primitives.

This module keeps price-basis validation, confirmed higher-timeframe bars,
empirical calibration, and the public v3 contract independent from the legacy
scoring engine.
"""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


V3_SCHEMA_VERSION = 3
DATA_QUALITY_STATES = {"VALID", "DEGRADED", "BLOCKED"}
TREND_STATES = {"BULL", "NEUTRAL", "BEAR"}
ENTRY_STATES = {"READY", "WATCH", "BLOCKED"}
ENTRY_SETUPS = {"BREAKOUT", "PULLBACK", "REVERSAL", "NONE"}
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
ENTRY_STOP_DIST_MIN = 2.0
ENTRY_STOP_DIST_MAX = 10.0


def fingerprint_price_frames(
    frames: Mapping[str, pd.DataFrame],
    trained_until: str,
) -> str:
    """Fingerprint adjusted history through the calibration cutoff."""
    cutoff = pd.to_datetime(trained_until, errors="raise")
    digest = hashlib.sha256()
    price_columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    for code in sorted(frames):
        frame = frames[code].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["date"].notna() & (frame["date"] <= cutoff)]
        columns = [column for column in price_columns if column in frame.columns]
        frame = frame[columns].sort_values("date").reset_index(drop=True)
        digest.update(f"{code}:{','.join(columns)}\n".encode("utf-8"))
        digest.update(
            frame.to_csv(
                index=False,
                date_format="%Y-%m-%d",
                float_format="%.12g",
                lineterminator="\n",
            ).encode("utf-8")
        )
    return digest.hexdigest()[:20]


def fingerprint_price_directory(data_dir: str, trained_until: str) -> str:
    frames: Dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if "_raw_" in os.path.basename(path):
            continue
        frame = pd.read_csv(path)
        if len(frame) < 275 or "date" not in frame.columns:
            continue
        code = os.path.basename(path).split("_", 1)[0]
        frames[code] = frame
    if not frames:
        return ""
    return fingerprint_price_frames(frames, trained_until)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_prices(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"price frame missing columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    columns = ["date", "open", "high", "low", "close"]
    if "volume" in out.columns:
        columns.append("volume")
    return out[columns].rename(
        columns={column: f"{column}_{suffix}" for column in columns if column != "date"}
    )


def align_price_bases(
    raw_df: pd.DataFrame,
    qfq_df: pd.DataFrame,
    factor_change_threshold: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align executable RAW prices with QFQ analysis prices by trading date."""
    raw = _normalise_prices(raw_df, "raw")
    qfq = _normalise_prices(qfq_df, "qfq")
    aligned = raw.merge(qfq, on="date", how="inner", validate="one_to_one")
    reasons: List[str] = []

    if aligned.empty:
        return aligned, {
            "status": "BLOCKED",
            "price_basis": "RAW",
            "analysis_basis": "QFQ",
            "adjustment_factor": 0.0,
            "adjustment_changed": False,
            "latest_adjustment_change_date": "",
            "reasons": ["NO_ALIGNED_PRICE_DATA"],
        }

    raw_dates = set(raw["date"])
    qfq_dates = set(qfq["date"])
    missing_dates = len(raw_dates.symmetric_difference(qfq_dates))
    if missing_dates:
        reasons.append("PRICE_DATES_MISALIGNED")

    raw_close = aligned["close_raw"].replace(0, np.nan)
    aligned["adjustment_factor"] = aligned["close_qfq"] / raw_close
    aligned = aligned.dropna(subset=["adjustment_factor"])
    factor_changes = aligned["adjustment_factor"].pct_change().abs()
    adjustment_changed = bool((factor_changes > factor_change_threshold).any())
    changed_rows = aligned.loc[factor_changes > factor_change_threshold, "date"]
    latest_adjustment_change_date = (
        pd.Timestamp(changed_rows.iloc[-1]).strftime("%Y-%m-%d")
        if not changed_rows.empty else ""
    )
    if adjustment_changed:
        reasons.append("ADJUSTMENT_FACTOR_CHANGED")

    invalid_ohlc = (
        (aligned["close_raw"] <= 0)
        | (aligned["close_qfq"] <= 0)
        | (aligned["high_raw"] < aligned["low_raw"])
        | (aligned["high_qfq"] < aligned["low_qfq"])
    )
    if bool(invalid_ohlc.any()):
        reasons.append("INVALID_OHLC")

    recent_adjustment = (
        bool(latest_adjustment_change_date)
        and (
            pd.Timestamp(aligned.iloc[-1]["date"])
            - pd.Timestamp(latest_adjustment_change_date)
        ).days <= 10
    )
    if "INVALID_OHLC" in reasons:
        status = "BLOCKED"
    elif "PRICE_DATES_MISALIGNED" in reasons or recent_adjustment:
        status = "DEGRADED"
    else:
        status = "VALID"
    quality = {
        "status": status,
        "price_basis": "RAW",
        "analysis_basis": "QFQ",
        "adjustment_factor": round(_number(aligned.iloc[-1]["adjustment_factor"]), 8),
        "adjustment_changed": adjustment_changed,
        "latest_adjustment_change_date": latest_adjustment_change_date,
        "reasons": reasons,
    }
    return aligned.reset_index(drop=True), quality


def align_return_series(
    etf_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """Return date-aligned close-to-close returns over shared observations."""
    etf = _normalise_prices(etf_df, "etf")[["date", "close_etf"]]
    benchmark = _normalise_prices(benchmark_df, "benchmark")[["date", "close_benchmark"]]
    aligned = etf.merge(benchmark, on="date", how="inner", validate="one_to_one").sort_values("date")
    if window is not None and window > 0:
        aligned = aligned.tail(int(window) + 1)
    aligned["etf_return"] = aligned["close_etf"].pct_change()
    aligned["benchmark_return"] = aligned["close_benchmark"].pct_change()
    return aligned.dropna(subset=["etf_return", "benchmark_return"]).reset_index(drop=True)


def confirmed_resample(
    daily: pd.DataFrame,
    frequency: str,
    as_of: Any,
    trading_calendar: Iterable[Any],
) -> pd.DataFrame:
    """Aggregate only periods whose final exchange trading day has completed."""
    if daily.empty:
        return pd.DataFrame(columns=daily.columns)
    if frequency not in {"W-FRI", "ME", "M"}:
        raise ValueError(f"unsupported confirmed frequency: {frequency}")

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    as_of_ts = pd.Timestamp(as_of).normalize()
    calendar = pd.Series(pd.to_datetime(list(trading_calendar))).dropna()
    calendar = calendar[calendar <= max(as_of_ts + pd.Timedelta(days=40), calendar.max())]

    period_frequency = "W-FRI" if frequency == "W-FRI" else "M"
    frame["_period"] = frame["date"].dt.to_period(period_frequency)
    calendar_periods = pd.DataFrame({"date": calendar})
    calendar_periods["_period"] = calendar_periods["date"].dt.to_period(period_frequency)
    period_ends = calendar_periods.groupby("_period")["date"].max()
    confirmed_periods = set(period_ends[period_ends <= as_of_ts].index)
    frame = frame[frame["_period"].isin(confirmed_periods)]
    if frame.empty:
        return pd.DataFrame(columns=[column for column in daily.columns])

    aggregations = {
        "date": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in frame.columns:
        aggregations["volume"] = "sum"
    return frame.groupby("_period", as_index=False).agg(aggregations).reset_index(drop=True)


def _confidence(sample_count: int) -> str:
    if sample_count >= 100:
        return "HIGH"
    if sample_count >= 30:
        return "MEDIUM"
    return "LOW"


def _calibration_key(
    entry_bin: int,
    setup: str,
    regime: Optional[str] = None,
    stop_bin: Optional[str] = None,
    rps_bin: Optional[int] = None,
) -> str:
    values = [str(int(entry_bin)), str(setup)]
    if regime is not None:
        values.extend([str(regime), str(stop_bin), str(int(rps_bin or 0))])
    return "|".join(values)


@dataclass
class CalibrationTable:
    version: str
    trained_until: str
    prior_strength: int
    global_stats: Dict[str, float]
    exact: Dict[str, Dict[str, Any]]
    entry_setup: Dict[str, Dict[str, Any]]
    data_fingerprint: str = ""
    thresholds: Optional[Dict[str, Any]] = None

    @classmethod
    def fit(
        cls,
        rows: Iterable[Mapping[str, Any]],
        prior_strength: int = 20,
        version: Optional[str] = None,
        trained_until: str = "",
        data_fingerprint: str = "",
    ) -> "CalibrationTable":
        data = pd.DataFrame(list(rows))
        required = {
            "entry_bin",
            "setup",
            "regime",
            "stop_bin",
            "rps_bin",
            "early_stop",
            "win_10d",
            "excess_return_10d",
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"calibration rows missing columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("calibration rows are empty")

        global_stats = {
            "early_stop_probability_3d": _number(data["early_stop"].mean()),
            "win_probability_10d": _number(data["win_10d"].mean()),
            "expected_excess_return_10d": _number(data["excess_return_10d"].mean()),
            "sample_count": int(len(data)),
        }

        def aggregate(group: pd.DataFrame, fallback_level: str) -> Dict[str, Any]:
            count = int(len(group))
            early = (
                _number(group["early_stop"].sum())
                + global_stats["early_stop_probability_3d"] * prior_strength
            ) / (count + prior_strength)
            win = (
                _number(group["win_10d"].sum())
                + global_stats["win_probability_10d"] * prior_strength
            ) / (count + prior_strength)
            excess = (
                _number(group["excess_return_10d"].sum())
                + global_stats["expected_excess_return_10d"] * prior_strength
            ) / (count + prior_strength)
            return {
                "early_stop_probability_3d": round(_clamp(early, 0.0, 1.0), 4),
                "win_probability_10d": round(_clamp(win, 0.0, 1.0), 4),
                "expected_excess_return_10d": round(excess, 6),
                "sample_count": count,
                "confidence": _confidence(count),
                "fallback_level": fallback_level,
            }

        exact: Dict[str, Dict[str, Any]] = {}
        exact_columns = ["entry_bin", "setup", "regime", "stop_bin", "rps_bin"]
        for values, group in data.groupby(exact_columns, dropna=False):
            exact[_calibration_key(*values)] = aggregate(group, "EXACT")

        entry_setup: Dict[str, Dict[str, Any]] = {}
        for values, group in data.groupby(["entry_bin", "setup"], dropna=False):
            entry_setup[_calibration_key(*values)] = aggregate(group, "ENTRY_SETUP")

        generated_version = version or f"empirical-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return cls(
            version=generated_version,
            trained_until=trained_until,
            prior_strength=int(prior_strength),
            global_stats=global_stats,
            exact=exact,
            entry_setup=entry_setup,
            data_fingerprint=data_fingerprint,
            thresholds={},
        )

    def lookup(
        self,
        entry_bin: int,
        setup: str,
        regime: str,
        stop_bin: str,
        rps_bin: int,
    ) -> Dict[str, Any]:
        exact_key = _calibration_key(entry_bin, setup, regime, stop_bin, rps_bin)
        if exact_key in self.exact and int(self.exact[exact_key].get("sample_count", 0)) >= 30:
            result = dict(self.exact[exact_key])
        else:
            setup_key = _calibration_key(entry_bin, setup)
            if setup_key in self.entry_setup:
                result = dict(self.entry_setup[setup_key])
            else:
                count = int(self.global_stats.get("sample_count", 0))
                result = {
                    **self.global_stats,
                    "confidence": _confidence(count),
                    "fallback_level": "GLOBAL",
                }
        result["version"] = self.version
        result.update(dict(self.thresholds or {}))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "version": self.version,
            "trained_until": self.trained_until,
            "prior_strength": self.prior_strength,
            "data_fingerprint": self.data_fingerprint,
            "global_stats": self.global_stats,
            "exact": self.exact,
            "entry_setup": self.entry_setup,
            "thresholds": dict(self.thresholds or {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationTable":
        return cls(
            version=str(value.get("version", "")),
            trained_until=str(value.get("trained_until", "")),
            prior_strength=int(value.get("prior_strength", 20)),
            data_fingerprint=str(value.get("data_fingerprint", "")),
            global_stats=dict(value.get("global_stats", {})),
            exact=dict(value.get("exact", {})),
            entry_setup=dict(value.get("entry_setup", {})),
            thresholds=dict(value.get("thresholds", {})),
        )


def classify_setup(result: Mapping[str, Any]) -> str:
    reason = str(result.get("daily_reason", ""))
    daily_score = _number(result.get("daily_score"))
    weekly_score = _number(result.get("weekly_score"))
    if any(word in reason for word in ("真破位", "冲高回落", "MACD顶背离", "20日新低")):
        return "NONE"
    if any(word in reason for word in ("真突破", "20日新高有效", "收上前高")):
        return "BREAKOUT"
    if weekly_score >= 2.0 and daily_score >= 0.2 and any(
        word in reason for word in ("回踩", "企稳", "收复中轨", "缩量下轨")
    ):
        return "PULLBACK"
    if daily_score >= 1.0 and any(word in reason for word in ("底背离", "RSI超卖", "筑底")):
        return "REVERSAL"
    if weekly_score >= 2.0 and daily_score >= 1.2:
        return "BREAKOUT"
    if weekly_score >= 2.0 and daily_score >= 0.2:
        return "PULLBACK"
    return "NONE"


def calibration_features(result: Mapping[str, Any], regime: str) -> Dict[str, Any]:
    daily_score = _number(result.get("daily_score"))
    stop_dist = _number(result.get("stop_dist", result.get("stop_dist_pct")))
    rps = _clamp(_number(result.get("rps")), 0.0, 100.0)
    entry_bin = int(_clamp(np.floor((daily_score + 5.0) / 2.0), 0.0, 4.0))
    if stop_dist < 2.0:
        stop_bin = "TIGHT"
    elif stop_dist <= 8.0:
        stop_bin = "HEALTHY"
    elif stop_dist <= 12.0:
        stop_bin = "WIDE"
    else:
        stop_bin = "EXTREME"
    return {
        "entry_bin": entry_bin,
        "setup": classify_setup(result),
        "regime": regime if regime in TREND_STATES else "NEUTRAL",
        "stop_bin": stop_bin,
        "rps_bin": int(min(4, np.floor(rps / 20.0))),
    }


def _trend_contract(result: Mapping[str, Any]) -> Dict[str, Any]:
    monthly = _number(result.get("monthly_score"))
    weekly = _number(result.get("weekly_score"))
    directional = monthly * 0.45 + weekly * 0.55
    score = round(_clamp(50.0 + directional / 5.6 * 50.0, 0.0, 100.0), 1)
    state = "BULL" if directional >= 1.5 else "BEAR" if directional <= -1.5 else "NEUTRAL"
    return {
        "state": state,
        "score": score,
        "weekly_confirmed": bool(result.get("weekly_confirmed", True)),
        "monthly_confirmed": bool(result.get("monthly_confirmed", True)),
    }


def build_v3_signal(
    result: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the additive v3 contract without removing any legacy fields."""
    signal = dict(result)
    quality = {
        "status": str(data_quality.get("status", "BLOCKED")),
        "price_basis": "RAW",
        "analysis_basis": "QFQ",
        "adjustment_factor": round(_number(data_quality.get("adjustment_factor")), 8),
        "adjustment_changed": bool(data_quality.get("adjustment_changed", False)),
        "latest_adjustment_change_date": str(data_quality.get("latest_adjustment_change_date", "")),
        "reasons": list(data_quality.get("reasons", [])),
    }
    if quality["status"] not in DATA_QUALITY_STATES:
        quality["status"] = "BLOCKED"
        quality["reasons"].append("INVALID_DATA_QUALITY_STATUS")

    trend = _trend_contract(result)
    setup = classify_setup(result)
    reasons: List[str] = []
    confidence = str(calibration.get("confidence", "LOW"))
    early_stop = _clamp(_number(calibration.get("early_stop_probability_3d"), 1.0), 0.0, 1.0)
    win_10d = _clamp(_number(calibration.get("win_probability_10d"), 0.0), 0.0, 1.0)
    expected_excess = _number(calibration.get("expected_excess_return_10d"))
    stop_dist = _number(result.get("stop_dist", result.get("stop_dist_pct")))
    daily_score = _number(result.get("daily_score"))
    daily_reason = str(result.get("daily_reason", ""))
    early_stop_limit = _clamp(
        _number(calibration.get("early_stop_probability_max"), 0.35),
        0.05,
        0.60,
    )
    entry_score_min = _clamp(_number(calibration.get("entry_score_min"), 0.0), 0.0, 100.0)

    if confidence not in {"MEDIUM", "HIGH"}:
        reasons.append("LOW_CALIBRATION_CONFIDENCE")
    if setup == "NONE":
        reasons.append("NO_CONFIRMED_ENTRY_SETUP")
    if trend["state"] != "BULL":
        reasons.append("TREND_NOT_BULL")
    if daily_score < 0.2:
        reasons.append("DAILY_NOT_CONFIRMED")
    if early_stop > early_stop_limit:
        reasons.append("EARLY_STOP_RISK_HIGH")
    if expected_excess <= 0:
        reasons.append("EXPECTED_EXCESS_NOT_POSITIVE")
    if not ENTRY_STOP_DIST_MIN <= stop_dist <= ENTRY_STOP_DIST_MAX:
        reasons.append("STOP_DISTANCE_UNEXECUTABLE")
    if any(word in daily_reason for word in ("真破位", "冲高回落", "MACD顶背离", "20日新低")):
        reasons.append("DANGER_SETUP")

    blocked = quality["status"] == "BLOCKED" or "STOP_DISTANCE_UNEXECUTABLE" in reasons
    ready = quality["status"] == "VALID" and not reasons
    state = "BLOCKED" if blocked else "READY" if ready else "WATCH"
    daily_norm = _clamp((daily_score + 1.0) / 4.0, 0.0, 1.0)
    excess_norm = _clamp(expected_excess / 0.04, 0.0, 1.0)
    priority = 100.0 * (daily_norm * 0.45 + (1.0 - early_stop) * 0.35 + excess_norm * 0.20)
    entry_score = daily_norm * 100.0
    if entry_score < entry_score_min:
        reasons.append("ENTRY_SCORE_BELOW_CALIBRATED_THRESHOLD")
        ready = False
        state = "BLOCKED" if blocked else "WATCH"
    if state != "READY":
        priority = min(priority, 59.9)

    calibration_contract = {
        "early_stop_probability_3d": round(early_stop, 4),
        "win_probability_10d": round(win_10d, 4),
        "expected_excess_return_10d": round(expected_excess, 6),
        "sample_count": int(calibration.get("sample_count", 0)),
        "confidence": confidence if confidence in CONFIDENCE_LEVELS else "LOW",
        "version": str(calibration.get("version", "")),
        "approved": bool(calibration.get("approved", False)),
        "status_reason": str(calibration.get("status_reason", "CALIBRATION_NOT_APPROVED")),
    }
    signal.update(
        {
            "schema_version": V3_SCHEMA_VERSION,
            "signal_id": (
                f"{result.get('data_date', '')}-{result.get('code', '')}-"
                f"{calibration_contract['version'] or 'uncalibrated'}"
            ),
            "data_quality": quality,
            "trend": trend,
            "entry": {
                "state": state,
                "setup": setup,
                "score": round(entry_score, 1),
                "priority": round(_clamp(priority, 0.0, 100.0), 1),
                "reasons": list(dict.fromkeys(reasons)),
            },
            "calibration": calibration_contract,
        }
    )
    return signal
