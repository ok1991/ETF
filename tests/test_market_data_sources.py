import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from etf_radar import _core


def price_frame(last_close=1.0):
    dates = pd.bdate_range("2026-04-27", periods=60)
    close = np.linspace(0.9, last_close, len(dates))
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


class MarketDataProviderTests(unittest.TestCase):
    def fetch_tencent(self):
        qfq = price_frame(1.0)
        raw = price_frame(1.0)
        with tempfile.TemporaryDirectory() as directory:
            analyzer = _core.ETFAnalyzer("510300", "benchmark", force_download=True)
            analyzer.data_dir = directory
            with (
                patch.object(
                    _core.ak,
                    "stock_zh_a_hist_tx",
                    side_effect=[qfq.copy(), raw.copy()],
                    create=True,
                ) as tencent,
                patch.object(
                    _core.ak,
                    "fund_etf_hist_sina",
                    side_effect=AssertionError("Sina crosscheck should not be called"),
                    create=True,
                ) as sina,
                patch.object(analyzer, "_data_is_current", return_value=True),
                patch.object(
                    analyzer,
                    "_load_trading_calendar",
                    return_value=pd.bdate_range("2026-04-27", "2026-08-31"),
                ),
            ):
                self.assertTrue(analyzer.fetch_data(max_retries=1))
            metadata_files = list(Path(directory).glob("510300_source_*.json"))
            self.assertEqual(1, len(metadata_files))
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
            return analyzer, metadata, tencent, sina

    def test_tencent_primary_source_is_approved_without_sina(self):
        analyzer, metadata, tencent, sina = self.fetch_tencent()
        self.assertEqual(_core.PRIMARY_MARKET_DATA_SOURCE, analyzer.data_source)
        self.assertEqual(_core.PRIMARY_MARKET_DATA_SOURCE, metadata["source"])
        self.assertEqual("DISABLED", metadata["crosscheck"]["provider"])
        self.assertTrue(metadata["crosscheck"]["approved"])
        self.assertEqual(2, metadata["schema_version"])
        self.assertEqual(
            _core.MARKET_DATA_VALIDATION_POLICY_VERSION,
            metadata["validation_policy_version"],
        )
        self.assertEqual(64, len(metadata["cache_integrity"]["qfq_fingerprint"]))
        self.assertEqual(64, len(metadata["cache_integrity"]["raw_fingerprint"]))
        self.assertTrue(analyzer.data_source_audit["approved"])
        self.assertEqual(2, tencent.call_count)
        self.assertEqual(0, sina.call_count)

    def test_cache_fingerprint_detects_tampering_and_legacy_metadata(self):
        analyzer = _core.ETFAnalyzer("510300", "benchmark")
        qfq = price_frame(1.0)
        raw = price_frame(1.0)
        data_date = qfq["date"].iloc[-1].strftime("%Y-%m-%d")
        metadata = {
            "schema_version": 2,
            "code": "510300",
            "data_date": data_date,
            "source": _core.PRIMARY_MARKET_DATA_SOURCE,
            "validation_policy_version": _core.MARKET_DATA_VALIDATION_POLICY_VERSION,
            "primary_provider": "TENCENT",
            "cache_integrity": analyzer._cache_integrity(qfq, raw),
        }
        approved, audit = analyzer._validate_cached_source_metadata(qfq, raw, metadata)
        self.assertTrue(approved)
        self.assertTrue(audit["approved"])

        tampered = qfq.copy()
        tampered.loc[tampered.index[0], "close"] *= 1.10
        approved, audit = analyzer._validate_cached_source_metadata(
            tampered, raw, metadata
        )
        self.assertFalse(approved)
        self.assertIn(
            "CACHE_INTEGRITY_MISMATCH:qfq_fingerprint", audit["reasons"]
        )

        legacy = dict(metadata)
        legacy.pop("validation_policy_version")
        approved, audit = analyzer._validate_cached_source_metadata(qfq, raw, legacy)
        self.assertFalse(approved)
        self.assertIn("VALIDATION_POLICY_VERSION_MISMATCH", audit["reasons"])

        unverified = dict(metadata)
        unverified["source"] = "TENCENT_UNVERIFIED"
        approved, audit = analyzer._validate_cached_source_metadata(qfq, raw, unverified)
        self.assertFalse(approved)
        self.assertIn("SOURCE_NOT_TENCENT_PRIMARY", audit["reasons"])


if __name__ == "__main__":
    unittest.main()
