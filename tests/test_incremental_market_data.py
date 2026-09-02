# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from etf_radar import _core

EXPECTED_DATE = pd.Timestamp("2026-09-01")


PRICE_ANCHOR = pd.Timestamp("2026-01-02")


def make_frame(dates):
    """Price by absolute trading day so the same date has identical values
    across overlapping frames (independent of frame length)."""
    dates = pd.to_datetime(dates)
    days = (dates - PRICE_ANCHOR).days.astype(float)
    close = 0.9 + 0.0006 * days
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(dates), 1_000_000.0),
        }
    )


def write_cached(analyzer, qfq, raw, metadata=None):
    last = qfq["date"].iloc[-1].strftime("%Y%m%d")
    qfq.to_csv(Path(analyzer.data_dir) / f"510300_{last}.csv", index=False, encoding="utf-8-sig")
    raw.to_csv(Path(analyzer.data_dir) / f"510300_raw_{last}.csv", index=False, encoding="utf-8-sig")
    payload = {
        "schema_version": 2,
        "code": "510300",
        "data_date": qfq["date"].iloc[-1].strftime("%Y-%m-%d"),
        "source": _core.PRIMARY_MARKET_DATA_SOURCE,
        "validation_policy_version": _core.MARKET_DATA_VALIDATION_POLICY_VERSION,
        "primary_provider": "TENCENT",
        "crosscheck": {"provider": "DISABLED", "approved": True, "reason": "SINA_CROSSCHECK_DISABLED"},
        "cache_integrity": analyzer._cache_integrity(qfq, raw),
    }
    if metadata:
        payload.update(metadata)
    stamp = qfq["date"].iloc[-1].strftime("%Y%m%d")
    path = Path(analyzer.data_dir) / f"510300_source_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")



def latest_metadata(directory, data_date="2026-09-01"):
    """Return (path, metadata) for the metadata file dated data_date; old
    source metadata files linger in the cache directory by design."""
    for path in Path(directory).glob("510300_source_*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("data_date") == data_date:
            return path, meta
    raise AssertionError(f"no source metadata for {data_date}")

class IncrementalMarketDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.analyzer = _core.ETFAnalyzer("510300", "benchmark")
        self.analyzer.data_dir = self.directory.name
        self.cached_dates = pd.bdate_range("2026-05-01", "2026-08-31")
        self.cached_qfq = make_frame(self.cached_dates)
        self.cached_raw = make_frame(self.cached_dates)
        self.tail_dates = pd.bdate_range("2026-07-15", "2026-09-01")
        self.tail_qfq = make_frame(self.tail_dates)
        self.tail_raw = make_frame(self.tail_dates)
        self.full_dates = pd.bdate_range("2026-01-02", "2026-09-01")
        self.full_qfq = make_frame(self.full_dates)
        self.full_raw = make_frame(self.full_dates)
        self.maxDiff = None

    def tearDown(self):
        self.directory.cleanup()

    def _patch_env(self, tencent_side_effect):
        return [
            patch.object(
                _core.ak,
                "stock_zh_a_hist_tx",
                side_effect=tencent_side_effect,
                create=True,
            ),
            patch.object(
                self.analyzer,
                "_load_trading_calendar",
                return_value=pd.bdate_range("2026-01-01", "2026-12-31"),
            ),
            patch.object(
                _core, "expected_latest_completed_date", return_value=EXPECTED_DATE
            ),
        ]

    def test_incremental_append_saves_merged_file(self):
        write_cached(self.analyzer, self.cached_qfq, self.cached_raw)
        calls = []
        def tencent(**kwargs):
            calls.append(kwargs)
            if "start_date" in kwargs:
                assert kwargs["start_date"] == "2026-07-17"
                return self.tail_qfq if kwargs["adjust"] == "qfq" else self.tail_raw
            return None  # must never request full history

        with ExitStack() as stack:
            for guard in self._patch_env(tencent):
                stack.enter_context(guard)
            ok = self.analyzer.fetch_data(max_retries=1)
        self.assertTrue(ok)
        self.assertEqual(_core.PRIMARY_MARKET_DATA_SOURCE, self.analyzer.data_source)
        self.assertTrue(self.analyzer.data_source_audit.get("incremental"))
        files = {path.name for path in Path(self.directory.name).glob("510300_*.csv")}
        self.assertIn("510300_20260901.csv", files)
        self.assertNotIn("510300_20260831.csv", files)  # old qfq cache cleaned
        saved = pd.read_csv(
            Path(self.directory.name) / "510300_20260901.csv", parse_dates=["date"]
        )
        self.assertEqual(EXPECTED_DATE, pd.Timestamp(saved["date"].iloc[-1]))
        self.assertEqual(len(self.cached_qfq) + 1, len(saved))
        _, metadata = latest_metadata(self.directory.name)
        self.assertTrue(metadata["incremental"])
        self.assertEqual(1, metadata["incremental_appends"])
        self.assertEqual(1, metadata["appended_rows"])
        self.assertEqual("2026-07-17", metadata["incremental_tail_start"])

    def test_overlap_mismatch_falls_back_to_full_download(self):
        write_cached(self.analyzer, self.cached_qfq, self.cached_raw)
        revised_tail_qfq = self.tail_qfq.copy()
        revised_tail_qfq.loc[revised_tail_qfq["date"] <= pd.Timestamp("2026-08-31"), "close"] *= 1.05
        def tencent(**kwargs):
            if "start_date" in kwargs:
                return revised_tail_qfq if kwargs["adjust"] == "qfq" else self.tail_raw
            return self.full_qfq if kwargs["adjust"] == "qfq" else self.full_raw

        with ExitStack() as stack:
            for guard in self._patch_env(tencent):
                stack.enter_context(guard)
            ok = self.analyzer.fetch_data(max_retries=1)
        self.assertTrue(ok)
        self.assertEqual(_core.PRIMARY_MARKET_DATA_SOURCE, self.analyzer.data_source)
        self.assertFalse(self.analyzer.data_source_audit.get("incremental", False))
        _, metadata = latest_metadata(self.directory.name)
        self.assertNotIn("incremental", metadata)

    def test_chain_full_refresh_skips_incremental(self):
        write_cached(
            self.analyzer,
            self.cached_qfq,
            self.cached_raw,
            metadata={
                "incremental": True,
                "incremental_since": "2026-08-13",
                "incremental_appends": 3,
            },
        )
        def tencent(**kwargs):
            assert "start_date" not in kwargs
            return self.full_qfq if kwargs["adjust"] == "qfq" else self.full_raw

        with ExitStack() as stack:
            for guard in self._patch_env(tencent):
                stack.enter_context(guard)
            ok = self.analyzer.fetch_data(max_retries=1)
        self.assertTrue(ok)
        self.assertFalse(self.analyzer.data_source_audit.get("incremental", False))
        _, metadata = latest_metadata(self.directory.name)
        self.assertNotIn("incremental", metadata)


if __name__ == "__main__":
    unittest.main()
