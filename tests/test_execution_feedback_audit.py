import hashlib
import json
import unittest
from datetime import datetime
from pathlib import Path

import jsonschema

from etf_radar.execution_feedback_audit import (
    audit_feedback,
    audit_feedback_batch,
    cost_authority_id,
    fetch_feedback,
)


def model(cost=None):
    return {
        "version": "rotation-v2-test",
        "execution_policy_version": "single-exposure-authority-v4",
        "acceptance_policy_version": "rolling-excess-stability-v1",
        "strategy_specification_fingerprint": "spec-test",
        "cost_model": cost
        or {
            "commission_rate": 0.00015,
            "minimum_commission": 0.0,
            "exchange_handling_rate": 0.00004,
            "bid_ask_half_spread_bps": 2.0,
            "base_slippage_bps": 3.0,
            "impact_bps_at_full_adv": 18.0,
            "max_participation_rate": 0.1,
            "lot_size": 100,
        },
    }


def with_feedback_id(value):
    value = dict(value)
    value["feedback_id"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "generated_at"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return value


def feedback(index=1, evidence_level="BROKER_CONFIRMED", ratio=2.0, gross=2000.0):
    expected = 1.0
    actual = expected * ratio
    excess = (actual - expected) / gross * 10_000.0
    has_orders = evidence_level != "NO_ORDERS"
    broker_confirmed = evidence_level == "BROKER_CONFIRMED"
    value = {
        "schema_version": 1,
        "generated_at": f"2026-07-{19 + index:02d}T15:10:00+08:00",
        "evidence_level": evidence_level,
        "broker_confirmed": broker_confirmed,
        "plan_id": f"rotation-v2-test:plan-{index}",
        "rebalance_required": False,
        "decision_reason_codes": (
            ["PLAN_ALREADY_APPLIED"] if evidence_level == "NO_ORDERS" else []
        ),
        "model_version": "rotation-v2-test",
        "execution_policy_version": "single-exposure-authority-v4",
        "acceptance_policy_version": "rolling-excess-stability-v1",
        "strategy_specification_fingerprint": "spec-test",
        "data_date": "2026-07-17",
        "execution_date": f"2026-07-{19 + index:02d}",
        "run_date": f"2026-07-{19 + index:02d}",
        "quote_tradeable": True,
        "state_write_allowed": True,
        "orders": (
            [
                {
                    "side": "BUY",
                    "code": "512800",
                    "shares": 100,
                    "price": 20.0,
                    "total_cost": expected,
                }
            ]
            if has_orders
            else []
        ),
        "estimated_execution_cost": expected if has_orders else 0.0,
        "execution_cost_model": model()["cost_model"],
        "capacity_summary": {},
        "unfilled_order_count": 0,
        "rejection_reasons": [],
        "broker_evidence_file_sha256": "a" * 64 if broker_confirmed else "",
        "broker_evidence": (
            {
                "broker": "test-broker",
                "fills": [
                    {
                        "code": "512800",
                        "side": "BUY",
                        "shares": 100,
                        "price": 20.0,
                        "commission": actual,
                        "other_fees": 0.0,
                        "trade_date": f"2026-07-{19 + index:02d}",
                    }
                ],
                "order_outcomes": [
                    {
                        "code": "512800",
                        "side": "BUY",
                        "status": "FILLED",
                        "filled_shares": 100,
                        "unfilled_shares": 0,
                    }
                ],
                "comparison": [
                    {
                        "code": "512800",
                        "side": "BUY",
                        "shares": 100,
                        "planned_shares": 100,
                        "unfilled_shares": 0,
                        "fill_status": "FILLED",
                    }
                ],
                "broker_gross": gross,
                "expected_model_cost": expected,
                "actual_total_cost": actual,
                "actual_to_expected_cost_ratio": ratio,
                "excess_cost_bps": excess,
            }
            if broker_confirmed
            else {}
        ),
        "broker_fill_completion_status": (
            "COMPLETE" if broker_confirmed else "NOT_APPLICABLE"
        ),
    }
    return with_feedback_id(value)


def expected_execution(execution_date="2026-07-20", **overrides):
    value = {
        "schema_version": 2,
        "approved": True,
        "model_version": "rotation-v2-test",
        "execution_policy_version": "single-exposure-authority-v4",
        "acceptance_policy_version": "rolling-excess-stability-v1",
        "strategy_specification_fingerprint": "spec-test",
        "execution_date": execution_date,
        "target_weights": {"512800": 1.0},
        "walk_forward_metrics": {"cost_model": model()["cost_model"]},
    }
    value.update(overrides)
    return value


class ExecutionFeedbackAuditTests(unittest.TestCase):
    def test_no_feedback_does_not_block_rotation(self):
        audit, ledger = audit_feedback(None, model(), source_status="FEEDBACK_UNAVAILABLE")
        self.assertEqual("NO_FEEDBACK", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertEqual([], ledger["samples"])
        self.assertEqual(
            "INSUFFICIENT_EVIDENCE",
            audit["cost_recalibration_recommendation"]["status"],
        )

    def test_model_estimate_is_recorded_but_never_ingested(self):
        value = feedback(evidence_level="MODEL_ESTIMATE_ONLY")
        audit, ledger = audit_feedback(
            value,
            model(),
            now=datetime.fromisoformat("2026-07-20T15:10:00+08:00"),
        )
        self.assertEqual("MODEL_ESTIMATE_ONLY", audit["status"])
        self.assertFalse(audit["feedback_ingested"])
        self.assertEqual([], ledger["samples"])
        self.assertEqual(1, audit["pending_confirmation_count"])
        self.assertTrue(audit["rotation_authority_allowed"])

    def test_empty_model_estimate_is_rejected(self):
        value = feedback(evidence_level="MODEL_ESTIMATE_ONLY")
        value["orders"] = []
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, _ = audit_feedback(value, model())
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertIn("MODEL_ESTIMATE_REQUIRES_ORDERS", audit["errors"])

    def test_empty_broker_confirmation_cannot_clear_expected_execution(self):
        value = feedback(evidence_level="BROKER_CONFIRMED")
        value["orders"] = []
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, ledger = audit_feedback(
            value,
            model(),
            now=datetime.fromisoformat("2026-07-21T09:00:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertIn("BROKER_CONFIRMED_REQUIRES_ORDERS", audit["errors"])
        self.assertEqual(1, len(ledger["expected_executions"]))

    def test_broker_outcome_quantity_mismatch_is_rejected(self):
        value = feedback(evidence_level="BROKER_CONFIRMED")
        value["broker_evidence"]["order_outcomes"][0]["filled_shares"] = 99
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, _ = audit_feedback(value, model())
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertIn("BROKER_OUTCOME_INVALID:0", audit["errors"])

    def test_broker_completion_status_must_match_aggregated_outcomes(self):
        value = feedback(evidence_level="BROKER_CONFIRMED")
        value["broker_evidence"]["fills"][0]["shares"] = 50
        value["broker_evidence"]["order_outcomes"][0].update(
            {
                "status": "PARTIALLY_FILLED",
                "filled_shares": 50,
                "unfilled_shares": 50,
            }
        )
        value["broker_evidence"]["comparison"][0].update(
            {
                "shares": 50,
                "unfilled_shares": 50,
                "fill_status": "PARTIALLY_FILLED",
            }
        )
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, _ = audit_feedback(value, model())
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertIn("BROKER_FILL_COMPLETION_MISMATCH", audit["errors"])

    def test_expected_execution_is_registered_before_its_session(self):
        audit, ledger = audit_feedback(
            None,
            model(),
            source_status="FEEDBACK_UNAVAILABLE",
            now=datetime.fromisoformat("2026-07-19T15:10:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("NO_FEEDBACK", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertEqual(1, audit["expected_execution_count"])
        self.assertEqual(1, len(ledger["expected_executions"]))

    def test_completely_missed_execution_session_revokes_rotation_authority(self):
        _, ledger = audit_feedback(
            None,
            model(),
            now=datetime.fromisoformat("2026-07-19T15:10:00+08:00"),
            expected_execution=expected_execution(),
        )
        audit, ledger = audit_feedback(
            None,
            model(),
            ledger,
            source_status="FEEDBACK_UNAVAILABLE",
            now=datetime.fromisoformat("2026-07-21T09:00:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("EXECUTION_SESSION_MISSED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertEqual(1, audit["overdue_execution_count"])
        self.assertEqual(1, len(audit["overdue_execution_keys"]))

    def test_new_premarket_rotation_supersedes_old_plan_for_same_session(self):
        old = expected_execution(
            model_version="rotation-v2-old",
            strategy_specification_fingerprint="old-spec",
        )
        _, ledger = audit_feedback(
            None,
            model(),
            now=datetime.fromisoformat("2026-07-19T15:10:00+08:00"),
            expected_execution=old,
        )
        new = expected_execution(
            model_version="rotation-v2-new",
            strategy_specification_fingerprint="new-spec",
        )
        audit, ledger = audit_feedback(
            None,
            model(),
            ledger,
            now=datetime.fromisoformat("2026-07-19T15:20:00+08:00"),
            expected_execution=new,
        )
        self.assertEqual(1, audit["expected_execution_count"])
        self.assertEqual("rotation-v2-new", ledger["expected_executions"][0]["model_version"])
        self.assertEqual(1, audit["superseded_execution_count"])

    def test_valid_no_order_session_clears_expected_execution_once(self):
        _, ledger = audit_feedback(
            None,
            model(),
            now=datetime.fromisoformat("2026-07-19T15:10:00+08:00"),
            expected_execution=expected_execution(),
        )
        no_orders = feedback(index=1, evidence_level="NO_ORDERS")
        audit, ledger = audit_feedback(
            no_orders,
            model(),
            ledger,
            now=datetime.fromisoformat("2026-07-20T15:10:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("NO_ORDERS", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertEqual(0, audit["expected_execution_count"])
        self.assertEqual(1, len(ledger["observed_execution_keys"]))

        repeated, repeated_ledger = audit_feedback(
            None,
            model(),
            ledger,
            now=datetime.fromisoformat("2026-07-21T09:00:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("NO_FEEDBACK", repeated["status"])
        self.assertTrue(repeated["rotation_authority_allowed"])
        self.assertEqual([], repeated_ledger["expected_executions"])

    def test_non_tradeable_feedback_cannot_satisfy_expected_execution(self):
        no_orders = feedback(index=1, evidence_level="NO_ORDERS")
        no_orders["quote_tradeable"] = False
        no_orders.pop("feedback_id")
        no_orders = with_feedback_id(no_orders)
        audit, ledger = audit_feedback(
            no_orders,
            model(),
            now=datetime.fromisoformat("2026-07-21T09:00:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("EXECUTION_SESSION_MISSED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertEqual(1, len(ledger["expected_executions"]))

    def test_blocked_no_orders_cannot_satisfy_expected_execution(self):
        no_orders = feedback(index=1, evidence_level="NO_ORDERS")
        no_orders["decision_reason_codes"] = ["SOURCE_BLOCKED"]
        no_orders.pop("feedback_id")
        no_orders = with_feedback_id(no_orders)
        audit, ledger = audit_feedback(
            no_orders,
            model(),
            now=datetime.fromisoformat("2026-07-21T09:00:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertIn("NO_ORDERS_DECISION_REASON_INVALID", audit["errors"])
        self.assertEqual(1, len(ledger["expected_executions"]))

    def test_pending_broker_confirmation_cannot_satisfy_expected_execution(self):
        no_orders = feedback(index=1, evidence_level="NO_ORDERS")
        no_orders["decision_reason_codes"] = [
            "PLAN_AWAITING_BROKER_CONFIRMATION"
        ]
        no_orders.pop("feedback_id")
        no_orders = with_feedback_id(no_orders)
        audit, ledger = audit_feedback(
            no_orders,
            model(),
            now=datetime.fromisoformat("2026-07-21T09:00:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertIn("NO_ORDERS_DECISION_REASON_INVALID", audit["errors"])
        self.assertEqual(1, len(ledger["expected_executions"]))

    def test_aligned_portfolio_no_orders_satisfies_expected_execution(self):
        no_orders = feedback(index=1, evidence_level="NO_ORDERS")
        no_orders["rebalance_required"] = True
        no_orders["decision_reason_codes"] = ["PORTFOLIO_ALREADY_AT_TARGET"]
        no_orders.pop("feedback_id")
        no_orders = with_feedback_id(no_orders)
        audit, ledger = audit_feedback(
            no_orders,
            model(),
            now=datetime.fromisoformat("2026-07-20T15:10:00+08:00"),
            expected_execution=expected_execution(),
        )
        self.assertEqual("NO_ORDERS", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertEqual([], ledger["expected_executions"])

    def test_wrong_cost_policy_broker_evidence_is_rejected(self):
        value = feedback()
        value["execution_policy_version"] = "wrong-cost-policy"
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, ledger = audit_feedback(value, model())
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertIn("AUTHORITY_MISMATCH:execution_policy_version", audit["errors"])
        self.assertEqual([], ledger["samples"])

    def test_history_batch_recovers_confirmed_events_missed_between_etf_runs(self):
        events = [feedback(index=index) for index in range(1, 4)]
        payload = {"schema_version": 1, "event_count": len(events), "events": events}
        audit, ledger = audit_feedback_batch(payload, model())
        self.assertEqual("COST_MODEL_RECALIBRATION_REQUIRED", audit["status"])
        self.assertEqual(3, audit["batch_ingested_count"])
        self.assertEqual(3, audit["confirmed_sample_count"])
        repeated, repeated_ledger = audit_feedback_batch(payload, model(), ledger)
        self.assertEqual(0, repeated["batch_ingested_count"])
        self.assertEqual(3, len(repeated_ledger["samples"]))

    def test_old_model_version_with_same_cost_authority_remains_a_valid_sample(self):
        value = feedback()
        value["model_version"] = "rotation-v2-prior-model"
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, ledger = audit_feedback(value, model())
        self.assertEqual("BROKER_CONFIRMED", audit["status"])
        self.assertTrue(audit["feedback_ingested"])
        self.assertEqual("rotation-v2-prior-model", ledger["samples"][0]["model_version"])

    def test_confirmed_unfilled_order_is_recorded_without_cost_sample(self):
        value = feedback()
        value["broker_fill_completion_status"] = "UNFILLED"
        value["broker_evidence"].update(
            {
                "fills": [],
                "order_outcomes": [
                    {
                        "code": "512800",
                        "side": "BUY",
                        "status": "UNFILLED",
                        "filled_shares": 0,
                        "unfilled_shares": 100,
                    }
                ],
                "comparison": [
                    {
                        "code": "512800",
                        "side": "BUY",
                        "shares": 0,
                        "planned_shares": 100,
                        "unfilled_shares": 100,
                        "fill_status": "UNFILLED",
                    }
                ],
                "broker_gross": 0.0,
                "expected_model_cost": 0.0,
                "actual_total_cost": 0.0,
                "actual_to_expected_cost_ratio": None,
                "excess_cost_bps": None,
            }
        )
        value.pop("feedback_id")
        value = with_feedback_id(value)
        audit, ledger = audit_feedback(value, model())
        self.assertEqual("BROKER_CONFIRMED", audit["status"])
        self.assertFalse(audit["feedback_ingested"])
        self.assertEqual([], ledger["samples"])

    def test_malformed_history_is_rejected_without_partial_ingestion(self):
        payload = {"schema_version": 1, "event_count": 2, "events": [feedback()]}
        audit, ledger = audit_feedback_batch(payload, model())
        self.assertEqual("FEEDBACK_REJECTED", audit["status"])
        self.assertIn("FEEDBACK_HISTORY_COUNT_MISMATCH", audit["errors"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertEqual([], ledger["samples"])

    def test_unconfirmed_order_blocks_after_grace_and_valid_confirmation_clears_it(self):
        estimated = feedback(index=1, evidence_level="MODEL_ESTIMATE_ONLY")
        estimated["orders"] = [
            {
                "side": "BUY",
                "code": "512800",
                "shares": 100,
                "price": 1.0,
                "total_cost": 1.0,
            }
        ]
        estimated.pop("feedback_id")
        estimated = with_feedback_id(estimated)
        audit, ledger = audit_feedback(
            estimated,
            model(),
            now=datetime.fromisoformat("2026-07-20T15:10:00+08:00"),
        )
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertEqual(1, audit["pending_confirmation_count"])

        overdue, ledger = audit_feedback(
            None,
            model(),
            ledger,
            source_status="FEEDBACK_UNAVAILABLE",
            now=datetime.fromisoformat("2026-07-28T15:10:00+08:00"),
        )
        self.assertEqual("BROKER_CONFIRMATION_OVERDUE", overdue["status"])
        self.assertFalse(overdue["rotation_authority_allowed"])
        self.assertEqual([estimated["plan_id"]], overdue["overdue_plan_ids"])

        confirmed = feedback(index=1, evidence_level="BROKER_CONFIRMED", ratio=1.1)
        cleared, ledger = audit_feedback(
            confirmed,
            model(),
            ledger,
            now=datetime.fromisoformat("2026-07-28T15:20:00+08:00"),
        )
        self.assertEqual("BROKER_CONFIRMED", cleared["status"])
        self.assertTrue(cleared["rotation_authority_allowed"])
        self.assertEqual(0, cleared["pending_confirmation_count"])
        self.assertEqual([], ledger["pending_confirmations"])

    def test_three_persistently_expensive_confirmed_samples_latch_block(self):
        ledger = None
        for index in range(1, 4):
            audit, ledger = audit_feedback(feedback(index=index), model(), ledger)
        self.assertEqual("COST_MODEL_RECALIBRATION_REQUIRED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertEqual(3, audit["confirmed_sample_count"])
        self.assertEqual(cost_authority_id(model()), ledger["blocked_cost_authority_id"])
        recommendation = audit["cost_recalibration_recommendation"]
        self.assertEqual(
            "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION",
            recommendation["status"],
        )
        self.assertFalse(recommendation["approved_for_live_use"])
        self.assertFalse(recommendation["auto_promotion_allowed"])
        self.assertTrue(recommendation["requires_full_purged_walk_forward"])
        self.assertGreater(
            recommendation["recommended_cost_model"]["base_slippage_bps"],
            model()["cost_model"]["base_slippage_bps"],
        )
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "contracts" / "execution_cost_recalibration_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.validate(recommendation, schema)

        latched, _ = audit_feedback(None, model(), ledger, source_status="UNAVAILABLE")
        self.assertEqual("COST_MODEL_RECALIBRATION_REQUIRED", latched["status"])
        self.assertFalse(latched["rotation_authority_allowed"])

    def test_changed_cost_policy_does_not_inherit_old_block(self):
        ledger = None
        for index in range(1, 4):
            _, ledger = audit_feedback(feedback(index=index), model(), ledger)
        changed = model({**model()["cost_model"], "base_slippage_bps": 5.0})
        audit, _ = audit_feedback(None, changed, ledger)
        self.assertEqual("NO_FEEDBACK", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])

    def test_single_cost_outlier_cannot_create_recalibration_candidate(self):
        ledger = None
        values = [
            feedback(index=1, ratio=10.0),
            feedback(index=2, ratio=1.1),
            feedback(index=3, ratio=1.1),
        ]
        for value in values:
            audit, ledger = audit_feedback(value, model(), ledger)
        recommendation = audit["cost_recalibration_recommendation"]
        self.assertEqual(
            "MONITORING_NO_SUSTAINED_DEGRADATION",
            recommendation["status"],
        )
        self.assertEqual(
            model()["cost_model"], recommendation["recommended_cost_model"]
        )

    def test_extreme_degradation_recommendation_is_capped(self):
        ledger = None
        for index in range(1, 4):
            audit, ledger = audit_feedback(
                feedback(index=index, ratio=20.0), model(), ledger
            )
        recommendation = audit["cost_recalibration_recommendation"]
        self.assertTrue(recommendation["increment_capped"])
        self.assertEqual(20.0, recommendation["base_slippage_increment_bps"])

    def test_unapproved_remote_feedback_source_is_forbidden(self):
        value, status = fetch_feedback("https://untrusted.invalid/example.json")
        self.assertIsNone(value)
        self.assertEqual("UNAPPROVED_FEEDBACK_SOURCE", status)


if __name__ == "__main__":
    unittest.main()
