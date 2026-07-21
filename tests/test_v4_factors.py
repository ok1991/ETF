import sys
import types
import unittest

import numpy as np
import pandas as pd


sys.modules.setdefault("akshare", types.ModuleType("akshare"))

from v4_signals import (  # noqa: E402
    LEGACY_V4_FEATURE_NAMES,
    V4CalibrationModel,
    V4_FEATURE_NAMES,
    build_v4_signal,
    confirmed_resample,
    fit_v4_calibration,
)
from v4_factors import (  # noqa: E402
    classify_policy_pulse, decide_market_exposure, point_in_time_benchmark_momentum,
    market_policy,
    monthly_trend_factor,
    structural_risk,
    weekly_trend_factor,
)


def price_frame(periods=180, scale=1.0):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    base = np.linspace(10.0, 14.0, periods) + np.sin(np.arange(periods) / 8.0) * 0.15
    close = base * scale
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": np.linspace(1_000_000, 1_300_000, periods),
        }
    )


class V4FactorTests(unittest.TestCase):
    def test_monthly_history_requires_thirteen_confirmed_bars(self):
        monthly = price_frame(13).iloc[:12].copy()
        self.assertFalse(monthly_trend_factor(monthly)["history_ok"])
        self.assertTrue(monthly_trend_factor(price_frame(13))["history_ok"])

    def test_trend_and_risk_are_price_scale_invariant(self):
        weekly = price_frame(80)
        weekly_scaled = price_frame(80, scale=10.0)
        self.assertAlmostEqual(
            weekly_trend_factor(weekly)["score"],
            weekly_trend_factor(weekly_scaled)["score"],
            places=6,
        )
        risk = structural_risk(price_frame(), "PULLBACK", executable_price=14.0, adjustment_factor=1.0)
        scaled_risk = structural_risk(
            price_frame(scale=10.0),
            "PULLBACK",
            executable_price=140.0,
            adjustment_factor=1.0,
        )
        self.assertAlmostEqual(risk["stop_dist_pct"], scaled_risk["stop_dist_pct"], places=5)
        self.assertAlmostEqual(risk["atr_multiple"], scaled_risk["atr_multiple"], places=5)

    def test_confirmed_resample_excludes_incomplete_week(self):
        frame = price_frame(8)
        as_of = frame["date"].iloc[-2]
        calendar = pd.bdate_range(frame["date"].iloc[0], frame["date"].iloc[-1] + pd.Timedelta(days=7))
        weekly = confirmed_resample(frame[frame["date"] <= as_of], "W-FRI", as_of, calendar)
        self.assertTrue(weekly.empty or weekly["date"].max() <= as_of)
        if not weekly.empty:
            self.assertEqual(4, pd.Timestamp(weekly["date"].iloc[-1]).weekday())

    def test_v4_unapproved_blocks_and_approved_can_be_ready(self):
        result = {
            "code": "TEST",
            "data_date": "2026-07-16",
            "data_quality": {"status": "VALID"},
            "v4_priority": 86.0,
            "v4_market": {"entry_permission": "TRADEABLE", "score": 0.5},
            "relative_strength": {"score": 90.0, "has_120": True, "scope": "fixed_pool"},
            "v4_factors": {
                "monthly": {"score": 0.4, "history_ok": True},
                "weekly": {"score": 0.7, "history_ok": True},
                "setup": {"setup": "BREAKOUT", "score": 82.0},
                "risk": {
                    "stop_loss": 9.4,
                    "stop_dist_pct": 6.0,
                    "atr_multiple": 2.2,
                    "quality": 90.0,
                    "executable": True,
                },
            },
        }
        calibration = {
            "early_stop_probability_3d": 0.15,
            "win_probability_10d": 0.55,
            "expected_excess_return_10d": 0.01,
            "sample_count": 500,
            "confidence": "HIGH",
            "version": "test-v4",
            "approved": True,
            "status_reason": "APPROVED",
        }
        ready = build_v4_signal(result, calibration)
        self.assertEqual(4, ready["schema_version"])
        self.assertEqual("READY", ready["entry"]["state"])
        self.assertEqual("MAINLINE", ready["entry"]["channel"])

        blocked = build_v4_signal(result, {**calibration, "approved": False})
        self.assertEqual("BLOCKED", blocked["entry"]["state"])
        self.assertIn("CALIBRATION_NOT_APPROVED", blocked["entry"]["reasons"])

    def test_regularised_calibration_returns_bounded_probabilities(self):
        rows = []
        for index in range(80):
            strength = index / 79.0
            row = {name: strength for name in V4_FEATURE_NAMES}
            row.update(
                {
                    "early_stop": 1 if index < 28 else 0,
                    "win_10d": 1 if index >= 35 else 0,
                    "excess_return_10d": (strength - 0.5) * 0.02,
                }
            )
            rows.append(row)
        model = fit_v4_calibration(rows, regularisation=1.0, version="test")
        prediction = model.predict(rows[-1])
        self.assertGreaterEqual(prediction["early_stop_probability_3d"], 0.0)
        self.assertLessEqual(prediction["early_stop_probability_3d"], 1.0)
        self.assertGreaterEqual(prediction["win_probability_10d"], 0.0)
        self.assertLessEqual(prediction["win_probability_10d"], 1.0)

    def test_legacy_seven_feature_calibration_remains_loadable(self):
        count = len(LEGACY_V4_FEATURE_NAMES)
        model = V4CalibrationModel.from_dict(
            {
                "version": "legacy",
                "trained_until": "2026-06-01",
                "data_fingerprint": "test",
                "feature_names": list(LEGACY_V4_FEATURE_NAMES),
                "feature_mean": [0.0] * count,
                "feature_scale": [1.0] * count,
                "early_stop_coefficients": [0.0] * (count + 1),
                "win_coefficients": [0.0] * (count + 1),
                "excess_coefficients": [0.0] * (count + 1),
                "sample_count": 100,
                "thresholds": {},
            }
        )
        prediction = model.predict({name: 0.0 for name in LEGACY_V4_FEATURE_NAMES})
        self.assertEqual(list(LEGACY_V4_FEATURE_NAMES), model.feature_names)
        self.assertEqual(0.5, prediction["win_probability_10d"])


    def test_market_exposure_ladder_preserves_crisis_cash_and_softens_rebound(self):
        self.assertEqual(("BLOCKED", 0.0, "RISK_OFF"), decide_market_exposure(-0.55, 50.0))
        self.assertEqual(("MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"), decide_market_exposure(0.10, 98.5))
        self.assertEqual(("MAINLINE_ONLY", 0.70, "DEFENSIVE"), decide_market_exposure(0.40, 98.5))
        self.assertEqual(("BLOCKED", 0.0, "RISK_OFF"), decide_market_exposure(-0.10, 98.5))
        self.assertEqual(("MAINLINE_ONLY", 0.50, "HARD_DEFENSIVE"), decide_market_exposure(-0.30, 50.0))
        self.assertEqual(("MAINLINE_ONLY", 0.70, "DEFENSIVE"), decide_market_exposure(0.10, 96.5))
        self.assertEqual(("MAINLINE_ONLY", 0.70, "DEFENSIVE"), decide_market_exposure(-0.10, 50.0))
        self.assertEqual(("MAINLINE_ONLY", 0.90, "CAUTIOUS"), decide_market_exposure(0.10, 50.0))
        self.assertEqual(("TRADEABLE", 1.0, "NORMAL"), decide_market_exposure(0.40, 50.0))

    def test_policy_pulse_classifier_is_point_in_time_and_conservative(self):
        none = classify_policy_pulse({"ret_1": 0.01, "ret_3": 0.01, "ret_5": 0.02, "ret_10": 0.03})
        self.assertEqual("NONE", none["level"])
        # Strong thrust without stress/underinvestment stays off.
        quiet_bull = classify_policy_pulse(
            {"ret_1": 0.05, "ret_3": 0.05, "ret_5": 0.08, "ret_10": 0.10},
            benchmark_natr_percentile=55.0,
            raw_score=0.40,
        )
        self.assertEqual("NONE", quiet_bull["level"])
        mild = classify_policy_pulse(
            {"ret_1": 0.02, "ret_3": 0.03, "ret_5": 0.04, "ret_10": 0.05},
            benchmark_natr_percentile=92.0,
            raw_score=-0.20,
        )
        self.assertEqual("MILD", mild["level"])
        full = classify_policy_pulse(
            {"ret_1": 0.05, "ret_3": 0.05, "ret_5": 0.08, "ret_10": 0.10},
            benchmark_natr_percentile=95.0,
            raw_score=-0.30,
        )
        self.assertEqual("FULL", full["level"])
        early = classify_policy_pulse(
            {"ret_1": 0.003, "ret_3": 0.014, "ret_5": 0.012, "ret_10": -0.01},
            benchmark_natr_percentile=66.0,
            raw_score=-0.46,
        )
        self.assertEqual("EARLY", early["level"])
        self.assertGreaterEqual(early["exposure_floor"], 0.70)

    def test_pulse_keeps_tradable_coverage_under_extreme_stress(self):
        # Extreme NATR without pulse: strong scores stay fully tradable; weak negative still cash.
        self.assertEqual(("MAINLINE_ONLY", 0.70, "DEFENSIVE"), decide_market_exposure(0.40, 99.0, pulse_level="NONE"))
        self.assertEqual(("BLOCKED", 0.0, "RISK_OFF"), decide_market_exposure(-0.10, 99.0, pulse_level="NONE"))
        # Confirmed FULL pulse overrides stress cash trap.
        perm, exp, state = decide_market_exposure(0.40, 99.0, pulse_level="FULL")
        self.assertEqual("TRADEABLE", perm)
        self.assertEqual(1.0, exp)
        self.assertEqual("PULSE_FULL", state)
        # EARLY pulse while underinvested keeps tradable coverage without requiring FULL thrust.
        perm, exp, state = decide_market_exposure(-0.40, 66.0, pulse_level="EARLY")
        self.assertEqual("MAINLINE_ONLY", perm)
        self.assertGreaterEqual(exp, 0.70)
        self.assertTrue(str(state).startswith("PULSE"))
        # Negative score still participates during FULL pulse, not cash.
        perm, exp, state = decide_market_exposure(-0.40, 99.0, pulse_level="FULL")
        self.assertEqual("MAINLINE_ONLY", perm)
        self.assertGreaterEqual(exp, 0.70)

    def test_market_policy_applies_benchmark_momentum_pulse_floor(self):
        weak = [{"v4_factors": {"weekly": {"score": -0.20}}} for _ in range(10)]
        # Without pulse, deep negative weekly stays defensive/cash-ish.
        base = market_policy(weak, benchmark_weekly_score=-0.40, benchmark_natr_percentile=99.0)
        self.assertEqual(0.0, base["max_exposure_ratio"])
        # With confirmed thrust momentum, exposure floor is restored.
        pulsed = market_policy(
            weak,
            benchmark_weekly_score=-0.40,
            benchmark_natr_percentile=99.0,
            benchmark_momentum={"ret_1": 0.05, "ret_3": 0.05, "ret_5": 0.10, "ret_10": 0.12},
        )
        self.assertEqual("FULL", pulsed["policy_pulse"])
        self.assertGreaterEqual(pulsed["max_exposure_ratio"], 0.90)
        self.assertGreaterEqual(pulsed["score"], 0.0)
        self.assertGreaterEqual(pulsed["max_exposure_ratio"], 0.85)

    def test_market_policy_uses_graduated_exposure_instead_of_binary_risk_off(self):
        weak_breadth = [
            {"v4_factors": {"weekly": {"score": -0.10}}}
            for _ in range(10)
        ]
        # Mildly negative score with normal vol should keep partial risk, not cash.
        mild = market_policy(weak_breadth, benchmark_weekly_score=-0.05, benchmark_natr_percentile=60.0)
        self.assertEqual("MAINLINE_ONLY", mild["entry_permission"])
        self.assertGreater(mild["max_exposure_ratio"], 0.0)
        self.assertLess(mild["max_exposure_ratio"], 1.0)
        self.assertIn(mild["state"], {"DEFENSIVE", "HARD_DEFENSIVE", "CAUTIOUS"})

        crisis = market_policy(weak_breadth, benchmark_weekly_score=-0.80, benchmark_natr_percentile=99.0)
        self.assertEqual("BLOCKED", crisis["entry_permission"])
        self.assertEqual(0.0, crisis["max_exposure_ratio"])
        self.assertEqual("RISK_OFF", crisis["state"])

        recovery = market_policy(
            [{"v4_factors": {"weekly": {"score": 0.30}}} for _ in range(10)],
            benchmark_weekly_score=0.20,
            benchmark_natr_percentile=55.0,
        )
        self.assertEqual("TRADEABLE", recovery["entry_permission"])
        self.assertEqual(1.0, recovery["max_exposure_ratio"])
        self.assertEqual("NORMAL", recovery["state"])


if __name__ == "__main__":
    unittest.main()
