import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from etf_radar.factor_promotion_readiness import build_factor_promotion_readiness


class FactorPromotionReadinessTests(unittest.TestCase):
    def test_seasoning_and_selection_failures_remain_separate(self):
        registry = {
            "generated_at": "2026-07-19 18:00:00",
            "trained_until": "2026-06-15",
            "approved": False,
            "approval_reasons": [
                "POLICY_SEASONING_INCOMPLETE",
                "FACTOR_SELECTION_GATE_FAILED",
            ],
            "policy_seasoning_required": True,
            "policy_seasoning_anchor": "2026-06-15",
            "policy_unseen_date_count": 2,
            "policy_seasoning_min_dates": 13,
            "candidate_count": 1,
            "candidate_diagnostics": [
                {
                    "name": "candidate",
                    "candidate_origin": "genetic_or_seeded",
                    "selection_score": 0.01,
                    "accepted": False,
                    "rejection_reasons": ["SELECTION_FDR_ABOVE_0_10"],
                    "train_metrics": {"status": "ACTIVE"},
                    "selection_metrics": {
                        "status": "ACTIVE",
                        "multiple_testing_q_value": 0.5,
                    },
                }
            ],
        }
        health = {
            "status": "SUSPENDED",
            "approved_for_live_use": False,
            "reasons": ["REGISTRY_NOT_APPROVED"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            health_path = root / "health.json"
            output_path = root / "readiness.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            health_path.write_text(json.dumps(health), encoding="utf-8")
            result = build_factor_promotion_readiness(
                registry_path,
                health_path,
                output_path,
                generated_at=datetime(2026, 7, 19, 22, 0),
            )
        self.assertEqual(
            "WAITING_FOR_NEW_LABELLED_DATES_AND_STRONGER_CANDIDATES",
            result["status"],
        )
        self.assertFalse(result["gates"]["policy_seasoned"])
        self.assertFalse(result["gates"]["factor_selection_passed"])
        self.assertEqual(11, result["policy_seasoning"]["remaining_unseen_labelled_dates"])
        self.assertFalse(result["promotion_allowed"])

    def test_registry_and_live_health_must_both_approve_promotion(self):
        registry = {
            "approved": True,
            "approval_reasons": [],
            "policy_seasoned": True,
            "policy_unseen_date_count": 13,
            "policy_seasoning_min_dates": 13,
            "candidate_diagnostics": [],
        }
        health = {"status": "ACTIVE", "approved_for_live_use": True, "reasons": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            health_path = root / "health.json"
            output_path = root / "readiness.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            health_path.write_text(json.dumps(health), encoding="utf-8")
            result = build_factor_promotion_readiness(
                registry_path, health_path, output_path
            )
            self.assertTrue(output_path.is_file())
        self.assertEqual("APPROVED_FOR_LIVE_OVERLAY", result["status"])
        self.assertTrue(result["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
