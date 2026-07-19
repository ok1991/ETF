import unittest

from etf_radar.factor_evolution import factor_registry_identity
from etf_radar.live_factor_health import assess_live_factor_health


def registry(coefficients=(0.6, 0.4)):
    return {
        "approved": True,
        "trained_until": "2026-06-15",
        "factors": [
            {"name": "factor_a", "status": "ACTIVE", "expression": {"feature": "momentum_20"}},
            {"name": "factor_b", "status": "ACTIVE", "expression": {"feature": "relative_strength"}},
        ],
        "ensemble": {"coefficients": list(coefficients)},
    }


def metrics(ic_mean=0.03, ic_ir=0.5, observations=12, turnover=0.3):
    return {
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "recent_ic_mean": ic_mean,
        "recent_ic_ir": ic_ir,
        "ic_observations": observations,
        "turnover": turnover,
    }


class LiveFactorHealthTests(unittest.TestCase):
    def test_insufficient_post_training_history_stays_in_warmup(self):
        result = assess_live_factor_health(
            registry(),
            {"factor_a": metrics(observations=4), "factor_b": metrics(observations=4)},
            metrics(observations=4),
            signal_date_count=4,
            evaluated_through="2026-06-23",
        )
        self.assertEqual("WARMUP", result["status"])
        self.assertTrue(result["approved_for_live_use"])
        self.assertFalse(result["evidence_mature"])
        self.assertEqual("WARMUP", result["factor_decisions"]["factor_a"])

    def test_positive_post_training_ic_keeps_overlay_active(self):
        result = assess_live_factor_health(
            registry(),
            {"factor_a": metrics(), "factor_b": metrics()},
            metrics(),
            signal_date_count=12,
            evaluated_through="2026-07-03",
        )
        self.assertEqual("ACTIVE", result["status"])
        self.assertTrue(result["approved_for_live_use"])
        self.assertTrue(result["evidence_mature"])
        self.assertEqual(
            factor_registry_identity(registry()),
            {
                key: result[key]
                for key in factor_registry_identity(registry())
            },
        )

    def test_weak_but_non_negative_live_ic_is_watch_not_retired(self):
        result = assess_live_factor_health(
            registry(),
            {"factor_a": metrics(ic_mean=0.0, ic_ir=0.0), "factor_b": metrics()},
            metrics(ic_mean=0.0, ic_ir=0.0),
            signal_date_count=12,
            evaluated_through="2026-07-03",
        )
        self.assertEqual("WATCH", result["status"])
        self.assertTrue(result["approved_for_live_use"])

    def test_negative_live_ensemble_ic_suspends_overlay(self):
        result = assess_live_factor_health(
            registry(),
            {"factor_a": metrics(), "factor_b": metrics()},
            metrics(ic_mean=-0.04, ic_ir=-0.8),
            signal_date_count=12,
            evaluated_through="2026-07-03",
        )
        self.assertEqual("SUSPENDED", result["status"])
        self.assertFalse(result["approved_for_live_use"])
        self.assertIn("LIVE_ENSEMBLE_NEGATIVE_IC", result["reasons"])

    def test_nominal_two_factor_model_with_one_zero_weight_is_suspended(self):
        result = assess_live_factor_health(
            registry((1.0, 0.0)),
            {"factor_a": metrics(), "factor_b": metrics()},
            metrics(),
            signal_date_count=12,
            evaluated_through="2026-07-03",
        )
        self.assertEqual(1, result["effective_factor_count"])
        self.assertIn("EFFECTIVE_FACTOR_COUNT_BELOW_2", result["reasons"])
        self.assertFalse(result["approved_for_live_use"])

    def test_unmonitorable_expression_is_fail_closed(self):
        result = assess_live_factor_health(
            registry(),
            {},
            {},
            signal_date_count=0,
            evaluated_through="",
            unsupported_features=["setup_score"],
        )
        self.assertEqual("SUSPENDED", result["status"])
        self.assertIn("UNSUPPORTED_LIVE_MONITOR_FEATURES", result["reasons"])


if __name__ == "__main__":
    unittest.main()
