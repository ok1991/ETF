# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from etf_radar import _core
from etf_radar.calibration import pipeline

CODES = ["513120", "515220", "510300"]


def make_frame(seed, n=450, base=1.0, drift=0.0004, adjustment=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    noise = rng.normal(drift, 0.015, n)
    close = base * np.cumprod(1.0 + noise)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * (1.0 + rng.normal(0.0, 0.002, n)),
            "high": close * (1.0 + np.abs(rng.normal(0.0, 0.005, n))),
            "low": close * (1.0 - np.abs(rng.normal(0.0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
        }
    )
    if adjustment is not None:
        frame["close"] = frame["close"] * adjustment
        frame["high"] = frame["high"] * adjustment
        frame["low"] = frame["low"] * adjustment
        frame["open"] = frame["open"] * adjustment
    return frame


def assert_nested_equal(test, actual, expected):
    if isinstance(expected, dict):
        test.assertIsInstance(actual, dict)
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_nested_equal(test, actual[key], expected[key])
        return
    if isinstance(expected, list):
        test.assertIsInstance(actual, list)
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_nested_equal(test, left, right)
        return
    if isinstance(expected, (int, float, np.floating, np.integer)):
        if isinstance(actual, (int, float, np.floating, np.integer)):
            test.assertTrue(
                np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-12),
                f"numeric mismatch: {actual} != {expected}",
            )
            return
    test.assertEqual(actual, expected)


class CalibrationTemplateEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {code: make_frame(seed) for code, seed in zip(CODES, (7, 11, 13))}
        cls.qfq = {}
        for code in CODES:
            qfq = cls.raw[code].copy()
            qfq["close"] = qfq["close"] * 1.25
            qfq["open"] = qfq["open"] * 1.25
            qfq["high"] = qfq["high"] * 1.25
            qfq["low"] = qfq["low"] * 1.25
            cls.qfq[code] = qfq
        cls.calendar = pd.DatetimeIndex(cls.qfq["510300"]["date"])
        cls.templates = pipeline._build_calibration_templates(cls.qfq, cls.raw, cls.calendar)

    def _snapshot(self, template):
        results = {}
        for signal_date in [self.qfq["510300"]["date"].iloc[i] for i in (300, 320, 340, 360)]:
            for code in CODES:
                per_code = template.get(code) if template is not None else None
                snapshot = pipeline._analyse_snapshot(
                    code, self.qfq[code], self.raw[code], signal_date, self.calendar, per_code
                )
                self.assertIsNotNone(snapshot)
                results[(code, signal_date.strftime("%Y-%m-%d"))] = snapshot
        return results

    def test_templated_snapshot_matches_classic(self):
        classic = self._snapshot(None)
        templated = self._snapshot(self.templates)
        for key, (code, analyzer, current) in classic.items():
            _, fast_analyzer, fast_current = templated[key]
            for field in ("df_daily", "df_weekly", "df_monthly", "df_raw"):
                left = getattr(analyzer, field)
                right = getattr(fast_analyzer, field)
                pd.testing.assert_frame_equal(left, right, obj=f"{key} {field}")
            self.assertEqual(analyzer.data_quality, fast_analyzer.data_quality)
            self.assertEqual(analyzer.executable_price, fast_analyzer.executable_price)
            pd.testing.assert_frame_equal(current, fast_current, obj=f"{key} current")
            assert_nested_equal(
                self, dict(analyzer._v4_result), dict(fast_analyzer._v4_result)
            )


if __name__ == "__main__":
    unittest.main()
