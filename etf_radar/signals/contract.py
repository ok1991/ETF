"""ETF V4 signal contract, calibration model, and price-basis utilities."""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


V4_SCHEMA_VERSION = 4
DATA_QUALITY_STATES = {"VALID", "DEGRADED", "BLOCKED"}
TREND_STATES = {"BULL", "NEUTRAL", "BEAR"}
ENTRY_STATES = {"READY", "WATCH", "BLOCKED"}
ENTRY_SETUPS = {"BREAKOUT", "PULLBACK", "NONE"}
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
V4_FEATURE_NAMES = (
    "monthly_trend",
    "weekly_trend",
    "setup_score",
    "relative_strength",
    "risk_quality",
    "market_score",
    "is_breakout",
)


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


def _confidence(sample_count: int) -> str:
    if sample_count >= 100:
        return "HIGH"
    if sample_count >= 30:
        return "MEDIUM"
    return "LOW"


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


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _standardise_features(
    values: np.ndarray,
    mean: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    feature_mean = np.nanmean(matrix, axis=0) if mean is None else np.asarray(mean, dtype=float)
    feature_scale = np.nanstd(matrix, axis=0) if scale is None else np.asarray(scale, dtype=float)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    standardised = np.nan_to_num((matrix - feature_mean) / feature_scale)
    return standardised, feature_mean, feature_scale


def fit_logistic_ridge(
    features: np.ndarray,
    target: Sequence[float],
    regularisation: float = 1.0,
    iterations: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a small L2 logistic model with deterministic IRLS."""
    standardised, mean, scale = _standardise_features(features)
    design = np.column_stack([np.ones(len(standardised)), standardised])
    outcome = np.asarray(target, dtype=float)
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * float(regularisation)
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        probabilities = _sigmoid(design @ coefficients)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-5, None)
        adjusted = design @ coefficients + (outcome - probabilities) / weights
        lhs = design.T @ (design * weights[:, None]) + penalty
        rhs = design.T @ (weights * adjusted)
        updated = np.linalg.solve(lhs, rhs)
        if float(np.max(np.abs(updated - coefficients))) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
    return coefficients, mean, scale


def fit_linear_ridge(
    features: np.ndarray,
    target: Sequence[float],
    regularisation: float = 1.0,
    mean: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardised, feature_mean, feature_scale = _standardise_features(features, mean, scale)
    design = np.column_stack([np.ones(len(standardised)), standardised])
    outcome = np.asarray(target, dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * float(regularisation)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ outcome)
    return coefficients, feature_mean, feature_scale


@dataclass
class V4CalibrationModel:
    version: str
    trained_until: str
    data_fingerprint: str
    feature_mean: List[float]
    feature_scale: List[float]
    early_stop_coefficients: List[float]
    win_coefficients: List[float]
    excess_coefficients: List[float]
    sample_count: int
    thresholds: Dict[str, Any]

    def predict(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        vector = np.asarray(
            [[_number(features.get(name)) for name in V4_FEATURE_NAMES]],
            dtype=float,
        )
        standardised, _, _ = _standardise_features(
            vector,
            np.asarray(self.feature_mean, dtype=float),
            np.asarray(self.feature_scale, dtype=float),
        )
        design = np.column_stack([np.ones(len(standardised)), standardised])
        early = float(_sigmoid(design @ np.asarray(self.early_stop_coefficients, dtype=float))[0])
        win = float(_sigmoid(design @ np.asarray(self.win_coefficients, dtype=float))[0])
        excess = float((design @ np.asarray(self.excess_coefficients, dtype=float))[0])
        return {
            "early_stop_probability_3d": round(_clamp(early, 0.0, 1.0), 6),
            "win_probability_10d": round(_clamp(win, 0.0, 1.0), 6),
            "expected_excess_return_10d": round(excess, 8),
            "sample_count": int(self.sample_count),
            "confidence": _confidence(int(self.sample_count)),
            "version": self.version,
            **dict(self.thresholds or {}),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 4,
            "version": self.version,
            "trained_until": self.trained_until,
            "data_fingerprint": self.data_fingerprint,
            "feature_names": list(V4_FEATURE_NAMES),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "early_stop_coefficients": list(self.early_stop_coefficients),
            "win_coefficients": list(self.win_coefficients),
            "excess_coefficients": list(self.excess_coefficients),
            "sample_count": int(self.sample_count),
            "thresholds": dict(self.thresholds or {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V4CalibrationModel":
        feature_names = tuple(value.get("feature_names", ()))
        if feature_names and feature_names != V4_FEATURE_NAMES:
            raise ValueError("v4 calibration feature contract mismatch")
        return cls(
            version=str(value.get("version", "")),
            trained_until=str(value.get("trained_until", "")),
            data_fingerprint=str(value.get("data_fingerprint", "")),
            feature_mean=[float(x) for x in value.get("feature_mean", [])],
            feature_scale=[float(x) for x in value.get("feature_scale", [])],
            early_stop_coefficients=[float(x) for x in value.get("early_stop_coefficients", [])],
            win_coefficients=[float(x) for x in value.get("win_coefficients", [])],
            excess_coefficients=[float(x) for x in value.get("excess_coefficients", [])],
            sample_count=int(value.get("sample_count", 0)),
            thresholds=dict(value.get("thresholds", {})),
        )


def fit_v4_calibration(
    rows: Iterable[Mapping[str, Any]],
    regularisation: float = 1.0,
    version: str = "",
    trained_until: str = "",
    data_fingerprint: str = "",
    thresholds: Optional[Mapping[str, Any]] = None,
) -> V4CalibrationModel:
    data = pd.DataFrame(list(rows))
    required = set(V4_FEATURE_NAMES) | {"early_stop", "win_10d", "excess_return_10d"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"v4 calibration rows missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("v4 calibration rows are empty")
    features = data[list(V4_FEATURE_NAMES)].astype(float).to_numpy()
    early_coefficients, mean, scale = fit_logistic_ridge(
        features,
        data["early_stop"].astype(float).to_numpy(),
        regularisation=regularisation,
    )
    standardised, _, _ = _standardise_features(features, mean, scale)
    design = np.column_stack([np.ones(len(standardised)), standardised])
    win_coefficients, _, _ = fit_logistic_ridge(
        features,
        data["win_10d"].astype(float).to_numpy(),
        regularisation=regularisation,
    )
    excess_target = data["excess_return_10d"].astype(float).clip(-0.15, 0.15).to_numpy()
    penalty = np.eye(design.shape[1], dtype=float) * float(regularisation)
    penalty[0, 0] = 0.0
    excess_coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ excess_target)
    return V4CalibrationModel(
        version=version or f"v4-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        trained_until=trained_until,
        data_fingerprint=data_fingerprint,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        early_stop_coefficients=early_coefficients.tolist(),
        win_coefficients=win_coefficients.tolist(),
        excess_coefficients=excess_coefficients.tolist(),
        sample_count=int(len(data)),
        thresholds=dict(thresholds or {}),
    )


def v4_calibration_features(result: Mapping[str, Any]) -> Dict[str, float]:
    factors = result.get("v4_factors", {}) or {}
    setup = factors.get("setup", {}) or {}
    relative_strength = result.get("relative_strength", {}) or {}
    market = result.get("v4_market", {}) or {}
    return {
        "monthly_trend": _number((factors.get("monthly") or {}).get("score")),
        "weekly_trend": _number((factors.get("weekly") or {}).get("score")),
        "setup_score": _number(setup.get("score")) / 100.0,
        "relative_strength": _number(relative_strength.get("score")) / 100.0,
        "risk_quality": _number((factors.get("risk") or {}).get("quality")) / 100.0,
        "market_score": _number(market.get("score")),
        "is_breakout": 1.0 if str(setup.get("setup")) == "BREAKOUT" else 0.0,
    }


def build_v4_signal(
    result: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the authoritative V4 signal contract."""
    signal: Dict[str, Any] = {
        "code": str(result.get("code", "")),
        "name": str(result.get("name", "")),
        "data_date": str(result.get("data_date", "")),
        "price": round(_number(result.get("price")), 8),
    }
    factors = dict(result.get("v4_factors", {}) or {})
    monthly = dict(factors.get("monthly", {}) or {})
    weekly = dict(factors.get("weekly", {}) or {})
    setup = dict(factors.get("setup", {}) or {})
    risk = dict(factors.get("risk", {}) or {})
    relative_strength = dict(result.get("relative_strength", {}) or {})
    market = dict(result.get("v4_market", {}) or {})
    quality = dict(result.get("data_quality", {}) or {})
    priority = _clamp(_number(result.get("v4_priority")), 0.0, 100.0)
    setup_score = _clamp(_number(setup.get("score")), 0.0, 100.0)
    monthly_score = _clamp(_number(monthly.get("score")), -1.0, 1.0)
    weekly_score = _clamp(_number(weekly.get("score")), -1.0, 1.0)
    history_ok = bool(monthly.get("history_ok", False))
    has_120 = bool(relative_strength.get("has_120", False))
    setup_name = str(setup.get("setup", "NONE"))
    approved = bool(calibration.get("approved", False))
    early_stop = _clamp(_number(calibration.get("early_stop_probability_3d"), 1.0), 0.0, 1.0)
    win_10d = _clamp(_number(calibration.get("win_probability_10d"), 0.0), 0.0, 1.0)
    expected_excess = _number(calibration.get("expected_excess_return_10d"))
    early_stop_max = _clamp(_number(calibration.get("early_stop_probability_max"), 0.2143), 0.05, 0.60)
    priority_min = _clamp(_number(calibration.get("priority_min"), 65.0), 0.0, 100.0)
    setup_min = _clamp(_number(calibration.get("setup_score_min"), 60.0), 0.0, 100.0)
    market_permission = str(market.get("entry_permission", "BLOCKED"))

    mainline = bool(
        history_ok
        and has_120
        and monthly_score >= 0.20
        and weekly_score >= 0.60
        and _number(relative_strength.get("score")) >= 80.0
        and priority >= 75.0
    )
    channel = "MAINLINE" if mainline else setup_name if setup_name in {"BREAKOUT", "PULLBACK"} else "NORMAL"
    reasons: List[str] = []
    if quality.get("status") != "VALID":
        reasons.append("DATA_QUALITY_NOT_VALID")
    if not approved:
        reasons.append("CALIBRATION_NOT_APPROVED")
    if not bool(risk.get("executable", False)):
        reasons.append("RISK_NOT_EXECUTABLE")
    if market_permission == "BLOCKED":
        reasons.append("MARKET_BLOCKED")
    elif market_permission == "MAINLINE_ONLY" and not mainline:
        reasons.append("MARKET_MAINLINE_ONLY")
    weekly_floor = 0.25 if history_ok else 0.50
    if weekly_score < weekly_floor:
        reasons.append("WEEKLY_TREND_NOT_CONFIRMED")
    if history_ok and monthly_score < -0.15:
        reasons.append("MONTHLY_TREND_NEGATIVE")
    if setup_name == "NONE" or setup_score < setup_min:
        reasons.append("SETUP_NOT_CONFIRMED")
    if early_stop > early_stop_max:
        reasons.append("EARLY_STOP_RISK_HIGH")
    if expected_excess <= 0:
        reasons.append("EXPECTED_EXCESS_NOT_POSITIVE")
    if priority < priority_min:
        reasons.append("PRIORITY_BELOW_THRESHOLD")

    blocked = bool(
        not approved
        or quality.get("status") == "BLOCKED"
        or not bool(risk.get("executable", False))
        or market_permission == "BLOCKED"
    )
    state = "BLOCKED" if blocked else "READY" if not reasons else "WATCH"
    calibration_contract = {
        "early_stop_probability_3d": round(early_stop, 6),
        "win_probability_10d": round(win_10d, 6),
        "expected_excess_return_10d": round(expected_excess, 8),
        "sample_count": int(calibration.get("sample_count", 0)),
        "confidence": str(calibration.get("confidence", "LOW")),
        "version": str(calibration.get("version", "")),
        "approved": approved,
        "status_reason": str(calibration.get("status_reason", "CALIBRATION_NOT_APPROVED")),
    }
    signal.update(
        {
            "schema_version": V4_SCHEMA_VERSION,
            "signal_id": f"{result.get('data_date', '')}-{result.get('code', '')}-{calibration_contract['version'] or 'v4-unapproved'}",
            "data_quality": quality,
            "trend": {
                "state": "BULL" if weekly_score >= weekly_floor else "BEAR" if weekly_score <= -0.25 else "NEUTRAL",
                "monthly": monthly,
                "weekly": weekly,
                "history_ok": history_ok,
            },
            "entry": {
                "state": state,
                "setup": setup_name,
                "setup_score": round(setup_score, 2),
                "score": round(setup_score, 2),
                "priority": round(priority, 2),
                "channel": channel,
                "reasons": list(dict.fromkeys(reasons)),
            },
            "relative_strength": relative_strength,
            "risk": risk,
            "market_policy": market,
            "calibration": calibration_contract,
        }
    )
    return signal
