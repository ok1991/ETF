import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from etf_radar.calibration import pipeline


def price_frames(periods):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    frame = pd.DataFrame({"date": dates, "close": range(1, periods + 1)})
    return {
        pipeline.main.Config.DEFAULT_INDEX_CODE: frame,
        "512800": frame.copy(),
    }


class CalibrationRowCacheTests(unittest.TestCase):
    def test_cache_is_reused_only_while_latest_fully_labelled_sample_is_unchanged(self):
        original_qfq = price_frames(300)
        eligible = pipeline._latest_eligible_signal_date(original_qfq, 5)
        rows = [{"date": eligible, "code": "512800"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            pipeline.save_rows_cache(rows, "fingerprint", 5, str(path), eligible)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(4, saved["schema_version"])
            self.assertEqual(
                pipeline.cost_model_fingerprint(pipeline.DEFAULT_ETF_COST_MODEL),
                saved["cost_model_fingerprint"],
            )
            with (
                patch("etf_radar.calibration.pipeline._load_price_pairs", return_value=(original_qfq, original_qfq)),
                patch("etf_radar.calibration.pipeline.calibration_data_fingerprint", return_value="fingerprint"),
            ):
                cached = pipeline.load_rows_cache("unused", 5, str(path))
            self.assertIsNotNone(cached)

            extended_qfq = price_frames(305)
            self.assertNotEqual(
                eligible,
                pipeline._latest_eligible_signal_date(extended_qfq, 5),
            )
            with (
                patch("etf_radar.calibration.pipeline._load_price_pairs", return_value=(extended_qfq, extended_qfq)),
                patch("etf_radar.calibration.pipeline.calibration_data_fingerprint", return_value="fingerprint"),
            ):
                stale = pipeline.load_rows_cache("unused", 5, str(path))
            self.assertIsNone(stale)

    def test_cost_model_change_invalidates_rows_and_prediction_caches(self):
        qfq = price_frames(300)
        eligible = pipeline._latest_eligible_signal_date(qfq, 5)
        rows = [{"date": eligible, "code": "512800"}]
        changed_cost = pipeline.TradingCostModel(base_slippage_bps=8.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            pipeline.save_rows_cache(rows, "fingerprint", 5, str(path), eligible)
            with patch(
                "etf_radar.calibration.pipeline._load_price_pairs",
                return_value=(qfq, qfq),
            ):
                self.assertIsNone(
                    pipeline.load_rows_cache(
                        "unused", 5, str(path), cost_model=changed_cost
                    )
                )
        prediction_payload = {
            "schema_version": 3,
            "data_fingerprint_policy": pipeline.CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION,
            "data_fingerprint": "fingerprint",
            "calibration_row_count": 1,
            "cost_model": pipeline.DEFAULT_ETF_COST_MODEL.to_dict(),
            "cost_model_fingerprint": pipeline.cost_model_fingerprint(
                pipeline.DEFAULT_ETF_COST_MODEL
            ),
        }
        self.assertTrue(
            pipeline.prediction_cache_compatible(
                prediction_payload, "fingerprint", 1
            )
        )
        self.assertFalse(
            pipeline.prediction_cache_compatible(
                prediction_payload, "fingerprint", 1, changed_cost
            )
        )

    def test_legacy_cache_without_eligible_date_authority_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "data_fingerprint_policy": pipeline.CALIBRATION_DATA_FINGERPRINT_POLICY_VERSION,
                        "sample_step": 5,
                        "trained_until": "2026-06-01",
                        "data_fingerprint": "fingerprint",
                        "row_count": 1,
                        "rows": [{"date": "2026-06-01", "code": "512800"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(pipeline.load_rows_cache("unused", 5, str(path)))


if __name__ == "__main__":
    unittest.main()
