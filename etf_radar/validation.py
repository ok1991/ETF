"""Leakage-resistant rolling walk-forward split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Calendar-based expanding-window validation configuration."""

    train_months: int = 24
    validation_months: int = 6
    step_months: int = 6
    purge_trading_days: int = 20
    embargo_trading_days: int = 5
    min_train_rows: int = 200
    min_validate_rows: int = 40


def expanding_walk_forward_splits(
    dates: Iterable[Any],
    config: WalkForwardConfig = WalkForwardConfig(),
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Create expanding train/validate windows with purge and fold embargo.

    Each tuple contains ``train_end, validate_start, validate_end, next_start``.
    The training sample is all observations through ``train_end``.  Validation is
    half-open: ``[validate_start, validate_end)``.
    """
    values = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().unique().sort_values()
    if len(values) < 2:
        return []
    first = pd.Timestamp(values[0]).normalize()
    last = pd.Timestamp(values[-1]).normalize()
    validate_start = first + pd.DateOffset(months=max(1, config.train_months))
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    while validate_start <= last:
        train_end = validate_start - pd.offsets.BDay(max(0, config.purge_trading_days))
        validate_end = validate_start + pd.DateOffset(months=max(1, config.validation_months))
        next_start = (
            validate_start
            + pd.DateOffset(months=max(1, config.step_months))
            + pd.offsets.BDay(max(0, config.embargo_trading_days))
        )
        splits.append((pd.Timestamp(train_end), validate_start, pd.Timestamp(validate_end), pd.Timestamp(next_start)))
        validate_start = pd.Timestamp(next_start)
    return splits


def split_report(
    name: str,
    train: pd.DataFrame,
    validate: pd.DataFrame,
    status: str,
) -> Dict[str, Any]:
    def bounds(frame: pd.DataFrame) -> Tuple[str | None, str | None]:
        if frame.empty:
            return None, None
        values = pd.to_datetime(frame["date"])
        return values.min().strftime("%Y-%m-%d"), values.max().strftime("%Y-%m-%d")

    train_start, train_end = bounds(train)
    validate_start, validate_end = bounds(validate)
    return {
        "name": name,
        "train": int(len(train)),
        "validate": int(len(validate)),
        "train_start": train_start,
        "train_end": train_end,
        "validate_start": validate_start,
        "validate_end": validate_end,
        "status": status,
    }


__all__ = ["WalkForwardConfig", "expanding_walk_forward_splits", "split_report"]
