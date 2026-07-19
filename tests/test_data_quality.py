import unittest
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from etf_radar.data_quality import (
    build_data_manifest,
    expected_latest_completed_date,
    next_trading_date,
)


CALENDAR = pd.to_datetime(
    [
        "2026-07-13",
        "2026-07-14",
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
        "2026-07-20",
    ]
)


def analyzer(
    code,
    data_date="2026-07-17",
    source="TENCENT_SINA_VALIDATED",
    quality_reasons=None,
    source_audit=None,
):
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-16", data_date]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    return SimpleNamespace(
        code=code,
        name=code,
        df_daily=frame,
        df_raw=frame.copy(),
        trading_calendar=CALENDAR,
        data_quality={"status": "VALID", "reasons": list(quality_reasons or [])},
        data_source=source,
        data_source_audit=(
            {"approved": True, "policy_version": "test"}
            if source_audit is None else source_audit
        ),
        data_loaded_at="2026-07-18 09:00:00",
    )


class LatestCompletedTradingDateTests(unittest.TestCase):
    def test_trading_day_before_close_uses_prior_session(self):
        result = expected_latest_completed_date(
            CALENDAR,
            datetime(2026, 7, 17, 14, 59),
        )
        self.assertEqual(pd.Timestamp("2026-07-16"), result)

    def test_trading_day_after_close_uses_same_session(self):
        result = expected_latest_completed_date(
            CALENDAR,
            datetime(2026, 7, 17, 15, 16),
        )
        self.assertEqual(pd.Timestamp("2026-07-17"), result)

    def test_weekend_uses_friday_session(self):
        result = expected_latest_completed_date(
            CALENDAR,
            datetime(2026, 7, 18, 10, 0),
        )
        self.assertEqual(pd.Timestamp("2026-07-17"), result)

    def test_next_execution_session_comes_from_exchange_calendar(self):
        result = next_trading_date(CALENDAR, "2026-07-17")
        self.assertEqual(pd.Timestamp("2026-07-20"), result)


class DataManifestTests(unittest.TestCase):
    def build(self, items, codes=("510300", "512800")):
        return build_data_manifest(
            items,
            list(codes),
            "510300",
            datetime(2026, 7, 18, 10, 0),
        )

    def test_all_required_series_current_is_approved(self):
        manifest = self.build([analyzer("510300"), analyzer("512800")])
        self.assertTrue(manifest["approved"])
        self.assertEqual(2, manifest["current_count"])
        self.assertEqual(["2026-07-17"], manifest["observed_data_dates"])

    def test_missing_code_is_blocked(self):
        manifest = self.build([analyzer("510300")])
        self.assertFalse(manifest["approved"])
        self.assertEqual(["512800"], manifest["missing_codes"])

    def test_mixed_dates_are_blocked(self):
        manifest = self.build(
            [analyzer("510300"), analyzer("512800", data_date="2026-07-16")]
        )
        self.assertFalse(manifest["approved"])
        self.assertEqual(["512800"], manifest["blocked_codes"])

    def test_unverified_source_is_blocked(self):
        manifest = self.build(
            [analyzer("510300"), analyzer("512800", source="TENCENT_UNVERIFIED")]
        )
        self.assertFalse(manifest["approved"])
        record = next(item for item in manifest["records"] if item["code"] == "512800")
        self.assertIn("INDEPENDENT_SOURCE_CROSSCHECK_MISSING", record["reasons"])

    def test_unknown_source_is_blocked(self):
        manifest = self.build(
            [analyzer("510300"), analyzer("512800", source="SOME_OTHER_PROVIDER")]
        )
        self.assertFalse(manifest["approved"])

    def test_validated_source_without_integrity_audit_is_blocked(self):
        manifest = self.build(
            [
                analyzer("510300"),
                analyzer(
                    "512800",
                    source="TENCENT_SINA_VALIDATED",
                    source_audit={},
                ),
            ]
        )
        self.assertFalse(manifest["approved"])
        record = next(item for item in manifest["records"] if item["code"] == "512800")
        self.assertIn("SOURCE_VALIDATION_AUDIT_MISSING_OR_FAILED", record["reasons"])

    def test_future_dated_series_is_blocked(self):
        manifest = self.build(
            [analyzer("510300"), analyzer("512800", data_date="2026-07-20")]
        )
        self.assertFalse(manifest["approved"])
        record = next(item for item in manifest["records"] if item["code"] == "512800")
        self.assertIn("DATA_AHEAD_OF_EXPECTED_TRADING_DATE", record["reasons"])

    def test_unverified_business_day_calendar_is_blocked(self):
        manifest = self.build(
            [
                analyzer("510300", quality_reasons=["FALLBACK_BUSINESS_CALENDAR"]),
                analyzer("512800"),
            ]
        )
        self.assertFalse(manifest["approved"])
        record = next(item for item in manifest["records"] if item["code"] == "510300")
        self.assertIn("TRADING_CALENDAR_UNVERIFIED", record["reasons"])


if __name__ == "__main__":
    unittest.main()
