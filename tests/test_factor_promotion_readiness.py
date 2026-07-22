import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from etf_radar.factor_evolution import factor_registry_identity
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
            "candidate_specification_fingerprint": "a" * 64,
            "policy_candidate_specification_changed": True,
            "previous_candidate_specification_fingerprint_valid": False,
            "candidate_count": 1,
            "candidate_gate_summary": {
                "accepted_count": 0,
                "rejection_counts": {"SELECTION_FDR_ABOVE_0_10": 1},
                "fdr_above_threshold_count": 1,
            },
            "candidate_origin_gate_summary": {
                "genetic_or_seeded": {
                    "candidate_count": 1,
                    "accepted_count": 0,
                    "rejected_count": 1,
                    "fdr_above_threshold_count": 1,
                    "rejection_counts": {"SELECTION_FDR_ABOVE_0_10": 1},
                }
            },
            "candidate_diagnostic_coverage": {
                "stored_count": 1,
                "total_count": 1,
                "complete": True,
            },
            "llm_proposals_submitted": 2,
            "llm_proposals_considered": 1,
            "llm_proposals_skipped_rejected_cooldown": [
                {"name": "old_llm", "reason": "LLM_REJECTED_EXPRESSION_COOLDOWN"}
            ],
            "llm_candidate_trial_history": [
                {"name": "old_llm", "outcome": "SELECTION_REJECTED"}
            ],
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
            **factor_registry_identity(registry),
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
        self.assertTrue(
            result["policy_seasoning"]["candidate_specification_changed"]
        )
        self.assertFalse(
            result["policy_seasoning"][
                "previous_candidate_specification_fingerprint_valid"
            ]
        )
        self.assertFalse(result["promotion_allowed"])
        self.assertTrue(result["gates"]["registry_health_identity_match"])
        self.assertEqual(
            {"SELECTION_FDR_ABOVE_0_10": 1},
            result["candidate_summary"]["rejection_counts"],
        )
        self.assertEqual(
            1,
            result["candidate_summary"]["gate_summary"][
                "fdr_above_threshold_count"
            ],
        )
        self.assertEqual(
            1,
            result["candidate_summary"]["origin_gate_summary"][
                "genetic_or_seeded"
            ]["candidate_count"],
        )
        self.assertTrue(
            result["candidate_summary"]["diagnostic_coverage"]["complete"]
        )
        self.assertEqual(2, result["candidate_summary"]["llm_proposals_submitted"])
        self.assertEqual(1, result["candidate_summary"]["llm_proposals_considered"])
        self.assertEqual(
            1,
            result["candidate_summary"]["llm_candidate_trial_history_count"],
        )

    def test_registry_and_live_health_must_both_approve_promotion(self):
        registry = {
            "approved": True,
            "approval_reasons": [],
            "policy_seasoned": True,
            "policy_unseen_date_count": 13,
            "policy_seasoning_min_dates": 13,
            "candidate_diagnostics": [],
        }
        health = {
            **factor_registry_identity(registry),
            "status": "ACTIVE",
            "approved_for_live_use": True,
            "reasons": [],
        }
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

    def test_stale_live_health_for_different_registry_is_fail_closed(self):
        registry = {
            "approved": True,
            "approval_reasons": [],
            "policy_seasoned": True,
            "factors": [
                {
                    "name": "factor_a",
                    "status": "ACTIVE",
                    "expression": {"feature": "momentum_20"},
                },
                {
                    "name": "factor_b",
                    "status": "ACTIVE",
                    "expression": {"feature": "relative_strength"},
                },
            ],
            "ensemble": {"coefficients": [0.6, 0.4]},
        }
        stale_registry = {
            **registry,
            "ensemble": {"coefficients": [0.1, 0.9]},
        }
        health = {
            **factor_registry_identity(stale_registry),
            "status": "ACTIVE",
            "approved_for_live_use": True,
            "reasons": [],
        }
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
        self.assertEqual("LIVE_HEALTH_REGISTRY_MISMATCH", result["status"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["gates"]["registry_health_identity_match"])
        self.assertIn(
            "REGISTRY_LIVE_FINGERPRINT_MISMATCH",
            result["registry_health_identity"]["errors"],
        )
        self.assertEqual(
            "REGENERATE_LIVE_FACTOR_HEALTH_FOR_CURRENT_REGISTRY",
            result["next_action"],
        )


if __name__ == "__main__":
    unittest.main()
