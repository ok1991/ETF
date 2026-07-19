import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from etf_radar.joint_health import build_joint_health
from test_runtime_layout import valid_rotation


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_layout(root: Path):
    etf = root / "ETF-main"
    swing = root / "Swing-trading"
    public = etf / "public"
    calibration = etf / "artifacts" / "calibration"
    bundle_id = "bundle-test"
    rotation = valid_rotation()

    files = {}
    for name in [
        "v4_calibration.json",
        "v4_acceptance_report.json",
        "adaptive_factor_registry.json",
        "llm_factor_proposals.json",
        "rotation_model.json",
    ]:
        value = {"artifact_bundle_id": bundle_id}
        if name == "rotation_model.json":
            value.update(
                {
                    "version": rotation["model_version"],
                    "approved": True,
                }
            )
        path = calibration / name
        write_json(path, value)
        files[name] = {"sha256": sha256(path)}
    write_json(
        calibration / "calibration_bundle.json",
        {
            "artifact_bundle_id": bundle_id,
            "valid_purged_fold_count": 18,
            "files": files,
        },
    )
    write_json(public / "etf_rotation_latest.json", rotation)
    write_json(public / "data_manifest_latest.json", {"approved": True})
    factor_health_path = public / "factor_health_latest.json"
    write_json(
        factor_health_path,
        {
            "status": "SUSPENDED",
            "approved_for_live_use": False,
            "reasons": ["REGISTRY_NOT_APPROVED"],
        },
    )
    registry_path = calibration / "adaptive_factor_registry.json"
    write_json(
        public / "factor_promotion_readiness_latest.json",
        {
            "status": "WAITING_FOR_NEW_LABELLED_DATES_AND_STRONGER_CANDIDATES",
            "promotion_allowed": False,
            "rotation_authority_independent": True,
            "registry": {"sha256": sha256(registry_path)},
            "live_health": {"sha256": sha256(factor_health_path)},
            "registry_health_identity": {"match": True, "errors": []},
            "policy_seasoning": {"remaining_unseen_labelled_dates": 13},
            "candidate_summary": {"accepted_candidate_count": 0},
        },
    )
    rotation_path = public / "etf_rotation_latest.json"
    manifest_path = public / "data_manifest_latest.json"
    write_json(
        etf / ".runtime" / "audits" / "pretrade_shadow_20260720.json",
        {
            "status": "READY_FOR_EXECUTION_DATE_QUOTE_REVALIDATION",
            "shadow_only": True,
            "order_submission_allowed": False,
            "state_persisted": False,
            "rotation": {
                "sha256": sha256(rotation_path),
                "model_version": rotation["model_version"],
                "execution_date": rotation["execution_date"],
                "strategy_specification_fingerprint": rotation[
                    "strategy_specification_fingerprint"
                ],
            },
            "market_data_manifest": {"sha256": sha256(manifest_path)},
            "reference_orders": [],
            "portfolio_result": {"estimated_execution_cost": 0.0},
            "errors": [],
        },
    )
    write_json(public / "cycle_status_latest.json", {"status": "UP_TO_DATE"})
    write_json(
        public / "distribution_audit_latest.json",
        {
            "status": "REMOTE_CONTRACT_INVALID",
            "same_host_execution_allowed": True,
            "remote_only_execution_allowed": False,
        },
    )
    write_json(
        public / "execution_feedback_audit_latest.json",
        {"status": "NO_FEEDBACK", "rotation_authority_allowed": True},
    )
    write_json(
        public / "live_performance_audit_latest.json",
        {
            "status": "NO_LIVE_PERFORMANCE_EVIDENCE",
            "rotation_authority_allowed": True,
        },
    )
    write_json(swing / "runtime" / "cache" / "etf_rotation_latest.json", rotation)
    contracts = Path(__file__).resolve().parents[1] / "contracts"
    for name in [
        "etf_rotation_v2.schema.json",
        "execution_feedback_v1.schema.json",
        "live_performance_v1.schema.json",
    ]:
        target_root = etf if name == "etf_rotation_v2.schema.json" else swing
        target = target_root / "contracts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((contracts / name).read_text(encoding="utf-8"), encoding="utf-8")
    return etf, swing, rotation


def valid_feedback(rotation, *, quote_tradeable=True, run_date="2026-07-20"):
    value = {
        "schema_version": 1,
        "feedback_id": "",
        "generated_at": "2026-07-20T09:35:00+08:00",
        "evidence_level": "NO_ORDERS",
        "broker_confirmed": False,
        "plan_id": "",
        "rebalance_required": False,
        "decision_reason_codes": ["PLAN_ALREADY_APPLIED"],
        "model_version": rotation["model_version"],
        "execution_policy_version": rotation["execution_policy_version"],
        "acceptance_policy_version": rotation["acceptance_policy_version"],
        "strategy_specification_fingerprint": rotation[
            "strategy_specification_fingerprint"
        ],
        "data_date": rotation["data_date"],
        "execution_date": rotation["execution_date"],
        "run_date": run_date,
        "quote_tradeable": quote_tradeable,
        "state_write_allowed": True,
        "orders": [],
        "estimated_execution_cost": 0.0,
        "execution_cost_model": {},
        "capacity_summary": {},
        "unfilled_order_count": 0,
        "rejection_reasons": [],
        "broker_evidence_file_sha256": "",
        "broker_evidence": {},
        "broker_fill_completion_status": "NOT_APPLICABLE",
        "state_reconciliation_applied": False,
        "state_reconciliation": {},
    }
    refresh_feedback_id(value)
    return value


def refresh_feedback_id(value):
    payload = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "generated_at",
            "feedback_id",
            "state_reconciliation_applied",
            "state_reconciliation",
        }
    }
    value["feedback_id"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def valid_performance(rotation):
    execution_date = rotation["execution_date"]
    value = {
        "schema_version": 1,
        "generated_at": "2026-07-20T14:55:00+08:00",
        "benchmark_code": "510300",
        "baseline": {
            "date": execution_date,
            "strategy_assets": 10000.0,
            "benchmark_price": 4.0,
            "source": "FIRST_VALID_MARK_TO_MARKET",
        },
        "observation_count": 1,
        "data_date": execution_date,
        "model_version": rotation["model_version"],
        "portfolio_state_evidence": "NO_EXECUTION_REQUIRED",
        "pending_broker_confirmation_plan_id": "",
        "last_execution_satisfied_plan_id": "plan-test",
        "broker_reconciliation_id": "",
        "benchmark_quote_source": "SINA_REALTIME",
        "benchmark_quote_mode": "DAILY_MARK_TO_MARKET",
        "benchmark_quote_date": execution_date,
        "benchmark_quote_time": "14:55:00",
        "benchmark_quote_fetched_at": execution_date + " 14:55:01+0800",
        "benchmark_quote_validated_at": execution_date + " 14:55:02+0800",
        "benchmark_quote_tradeable": True,
        "total_assets": 10000.0,
        "strategy_nav": 1.0,
        "benchmark_nav": 1.0,
        "relative_nav": 1.0,
        "strategy_return": 0.0,
        "benchmark_return": 0.0,
        "excess_return": 0.0,
        "relative_return": 0.0,
        "strategy_max_drawdown": 0.0,
        "benchmark_max_drawdown": 0.0,
        "relative_max_drawdown": 0.0,
        "rolling_20": {},
        "rolling_60": {},
        "history": [
            {
                "date": execution_date,
                "total_assets": 10000.0,
                "benchmark_price": 4.0,
                "model_version": rotation["model_version"],
                "portfolio_state_evidence": "NO_EXECUTION_REQUIRED",
                "pending_broker_confirmation_plan_id": "",
                "last_execution_satisfied_plan_id": "plan-test",
                "broker_reconciliation_id": "",
                "benchmark_quote_source": "SINA_REALTIME",
                "benchmark_quote_mode": "DAILY_MARK_TO_MARKET",
                "benchmark_quote_date": execution_date,
                "benchmark_quote_time": "14:55:00",
                "benchmark_quote_fetched_at": execution_date + " 14:55:01+0800",
                "benchmark_quote_validated_at": execution_date + " 14:55:02+0800",
                "benchmark_quote_tradeable": True,
                "strategy_nav": 1.0,
                "benchmark_nav": 1.0,
                "relative_nav": 1.0,
                "strategy_return": 0.0,
                "benchmark_return": 0.0,
                "excess_return": 0.0,
                "relative_return": 0.0,
                "strategy_drawdown": 0.0,
                "benchmark_drawdown": 0.0,
                "relative_drawdown": 0.0,
            }
        ],
    }
    value["performance_id"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return value


class JointHealthTests(unittest.TestCase):
    def test_pre_execution_same_host_is_ready_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, _ = build_layout(Path(directory))
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        self.assertEqual("READY_LOCAL_ONLY", result["status"])
        self.assertTrue(result["same_host_execution_allowed"])
        self.assertFalse(result["remote_only_execution_allowed"])
        self.assertEqual([], result["blocking_reasons"])
        self.assertIn("REMOTE_ONLY_DISTRIBUTION_BLOCKED", result["warnings"])
        self.assertIn("ADAPTIVE_FACTOR_PROMOTION_NOT_READY", result["warnings"])
        self.assertTrue(result["checks"]["factor_promotion_readiness"]["valid"])
        self.assertTrue(
            result["checks"]["factor_promotion_readiness"][
                "registry_health_identity_match"
            ]
        )
        self.assertEqual(
            "AWAITING_EXECUTION_DATE",
            result["checks"]["direct_execution_evidence"]["phase"],
        )
        self.assertTrue(result["checks"]["pretrade_shadow"]["valid"])
        self.assertFalse(result["automation_execution_ready"])
        self.assertIn("AUTOMATION_SCHEDULER_NOT_AUDITED", result["warnings"])

    def test_factor_health_registry_identity_mismatch_invalidates_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, _ = build_layout(Path(directory))
            path = etf / "public" / "factor_promotion_readiness_latest.json"
            promotion = json.loads(path.read_text(encoding="utf-8"))
            promotion["registry_health_identity"] = {
                "match": False,
                "errors": ["REGISTRY_LIVE_FINGERPRINT_MISMATCH"],
            }
            write_json(path, promotion)
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        readiness = result["checks"]["factor_promotion_readiness"]
        self.assertFalse(readiness["valid"])
        self.assertIn(
            "FACTOR_HEALTH_REGISTRY_IDENTITY_MISMATCH",
            readiness["errors"],
        )
        self.assertIn(
            "FACTOR_PROMOTION_READINESS_STALE_OR_INVALID",
            result["warnings"],
        )

    def test_recent_complete_scheduler_audit_marks_automation_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, _ = build_layout(Path(directory))
            write_json(
                etf / ".runtime" / "audits" / "windows_scheduler_latest.json",
                {
                    "schema_version": 1,
                    "policy_version": "windows-closed-loop-scheduler-audit-v1",
                    "generated_at": "2026-07-19T19:55:00+08:00",
                    "status": "READY",
                    "automation_execution_ready": True,
                    "expected_task_count": 3,
                    "installed_task_count": 3,
                    "enabled_task_count": 3,
                    "tasks": [],
                },
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        self.assertTrue(result["automation_execution_ready"])
        self.assertTrue(
            result["checks"]["automation_scheduler"]["automation_execution_ready"]
        )
        self.assertNotIn("AUTOMATION_SCHEDULER_NOT_READY", result["warnings"])

    def test_missing_pretrade_shadow_blocks_same_host_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, _ = build_layout(Path(directory))
            (etf / ".runtime" / "audits" / "pretrade_shadow_20260720.json").unlink()
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        self.assertIn("PRETRADE_SHADOW_UNAVAILABLE", result["blocking_reasons"])

    def test_tampered_rotation_invalidates_bound_pretrade_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            rotation["generated_at"] = "2026-07-19 22:00:00"
            write_json(etf / "public" / "etf_rotation_latest.json", rotation)
            write_json(swing / "runtime" / "cache" / "etf_rotation_latest.json", rotation)
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        self.assertIn("PRETRADE_SHADOW_INVALID", result["blocking_reasons"])
        self.assertIn(
            "ROTATION_SHA256_MISMATCH",
            result["checks"]["pretrade_shadow"]["errors"],
        )

    def test_swing_cache_identity_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            rotation["target_weights"] = {"510300": 0.4}
            write_json(
                swing / "runtime" / "cache" / "etf_rotation_latest.json",
                rotation,
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        self.assertIn("SWING_ROTATION_CACHE_MISMATCH", result["blocking_reasons"])

    def test_public_rotation_not_bound_to_calibration_bundle_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, _ = build_layout(Path(directory))
            model_path = etf / "artifacts" / "calibration" / "rotation_model.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["version"] = "rotation-v2-other-bbbbbbbb"
            write_json(model_path, model)
            manifest_path = etf / "artifacts" / "calibration" / "calibration_bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["rotation_model.json"]["sha256"] = sha256(model_path)
            write_json(manifest_path, manifest)
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 19, 20, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        self.assertIn("PUBLIC_ROTATION_AUTHORITY_MISMATCH", result["blocking_reasons"])

    def test_post_execution_missing_direct_evidence_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, _ = build_layout(Path(directory))
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        self.assertIn(
            "POST_EXECUTION_FEEDBACK_MISSING_OR_INVALID",
            result["blocking_reasons"],
        )
        self.assertIn(
            "POST_EXECUTION_PERFORMANCE_MISSING_OR_INVALID",
            result["blocking_reasons"],
        )

    def test_post_execution_uses_valid_history_when_latest_is_later_valuation(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            write_json(
                swing / "public" / "execution_feedback_history.json",
                {
                    "schema_version": 1,
                    "generated_at": "2026-07-21T14:55:00+08:00",
                    "event_count": 1,
                    "events": [valid_feedback(rotation)],
                },
            )
            write_json(
                swing / "public" / "execution_feedback_latest.json",
                valid_feedback(
                    rotation,
                    quote_tradeable=False,
                    run_date="2026-07-21",
                ),
            )
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("READY_LOCAL_ONLY", result["status"])
        evidence = result["checks"]["direct_execution_evidence"]
        self.assertTrue(evidence["feedback_valid"])
        self.assertEqual("history", evidence["feedback_source"])
        self.assertTrue(evidence["performance_valid"])

    def test_post_execution_model_estimate_with_orders_is_not_direct_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            feedback = valid_feedback(rotation)
            feedback.update(
                {
                    "evidence_level": "MODEL_ESTIMATE_ONLY",
                    "plan_id": "plan-test",
                    "orders": [
                        {
                            "side": "BUY",
                            "code": "512800",
                            "shares": 100,
                            "price": 1.0,
                            "total_cost": 1.0,
                        }
                    ],
                    "estimated_execution_cost": 1.0,
                }
            )
            refresh_feedback_id(feedback)
            write_json(swing / "public" / "execution_feedback_latest.json", feedback)
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        self.assertFalse(
            result["checks"]["direct_execution_evidence"]["feedback_valid"]
        )
        self.assertIn(
            "POST_EXECUTION_FEEDBACK_MISSING_OR_INVALID",
            result["blocking_reasons"],
        )

    def test_post_execution_blocked_no_orders_is_not_direct_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            feedback = valid_feedback(rotation)
            feedback["decision_reason_codes"] = ["SOURCE_BLOCKED"]
            refresh_feedback_id(feedback)
            write_json(swing / "public" / "execution_feedback_latest.json", feedback)
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        evidence = result["checks"]["direct_execution_evidence"]
        self.assertFalse(evidence["feedback_valid"])
        self.assertIn("ZERO_ORDER_DECISION_REASON_INVALID", evidence["feedback_errors"])

    def test_post_execution_pending_broker_confirmation_is_not_direct_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            feedback = valid_feedback(rotation)
            feedback["decision_reason_codes"] = [
                "PLAN_AWAITING_BROKER_CONFIRMATION"
            ]
            refresh_feedback_id(feedback)
            write_json(swing / "public" / "execution_feedback_latest.json", feedback)
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        evidence = result["checks"]["direct_execution_evidence"]
        self.assertFalse(evidence["feedback_valid"])
        self.assertIn("ZERO_ORDER_DECISION_REASON_INVALID", evidence["feedback_errors"])

    def test_post_execution_aligned_portfolio_no_orders_is_direct_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            feedback = valid_feedback(rotation)
            feedback["rebalance_required"] = True
            feedback["decision_reason_codes"] = ["PORTFOLIO_ALREADY_AT_TARGET"]
            refresh_feedback_id(feedback)
            write_json(swing / "public" / "execution_feedback_latest.json", feedback)
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("READY_LOCAL_ONLY", result["status"])
        self.assertTrue(
            result["checks"]["direct_execution_evidence"]["feedback_valid"]
        )

    def test_post_execution_broker_confirmed_order_is_direct_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            feedback = valid_feedback(rotation)
            feedback.update(
                {
                    "evidence_level": "BROKER_CONFIRMED",
                    "broker_confirmed": True,
                    "plan_id": "plan-test",
                    "orders": [
                        {
                            "side": "BUY",
                            "code": "512800",
                            "shares": 100,
                            "price": 1.0,
                            "total_cost": 1.0,
                        }
                    ],
                    "estimated_execution_cost": 1.0,
                    "broker_evidence_file_sha256": "c" * 64,
                    "broker_evidence": {
                        "broker": "test-broker",
                        "fills": [
                            {
                                "code": "512800",
                                "side": "BUY",
                                "shares": 100,
                                "price": 1.0,
                                "commission": 1.0,
                                "other_fees": 0.0,
                                "trade_date": "2026-07-20",
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
                    },
                    "broker_fill_completion_status": "COMPLETE",
                }
            )
            refresh_feedback_id(feedback)
            write_json(swing / "public" / "execution_feedback_latest.json", feedback)
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("READY_LOCAL_ONLY", result["status"])
        self.assertTrue(
            result["checks"]["direct_execution_evidence"]["feedback_valid"]
        )

    def test_post_execution_tampered_broker_outcome_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            feedback = valid_feedback(rotation)
            feedback.update(
                {
                    "evidence_level": "BROKER_CONFIRMED",
                    "broker_confirmed": True,
                    "plan_id": "plan-test",
                    "orders": [
                        {
                            "side": "BUY",
                            "code": "512800",
                            "shares": 100,
                            "price": 1.0,
                            "total_cost": 1.0,
                        }
                    ],
                    "estimated_execution_cost": 1.0,
                    "broker_evidence_file_sha256": "c" * 64,
                    "broker_evidence": {
                        "broker": "test-broker",
                        "fills": [
                            {
                                "code": "512800",
                                "side": "BUY",
                                "shares": 100,
                                "price": 1.0,
                                "commission": 1.0,
                                "other_fees": 0.0,
                                "trade_date": "2026-07-20",
                            }
                        ],
                        "order_outcomes": [
                            {
                                "code": "512800",
                                "side": "BUY",
                                "status": "FILLED",
                                "filled_shares": 99,
                                "unfilled_shares": 0,
                            }
                        ],
                        "comparison": [
                            {
                                "code": "512800",
                                "side": "BUY",
                                "shares": 99,
                                "planned_shares": 100,
                                "unfilled_shares": 0,
                                "fill_status": "FILLED",
                            }
                        ],
                    },
                    "broker_fill_completion_status": "COMPLETE",
                }
            )
            refresh_feedback_id(feedback)
            write_json(swing / "public" / "execution_feedback_latest.json", feedback)
            write_json(
                swing / "public" / "live_performance_latest.json",
                valid_performance(rotation),
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        errors = result["checks"]["direct_execution_evidence"]["feedback_errors"]
        self.assertIn("BROKER_OUTCOME_INVALID:0", errors)

    def test_post_execution_arithmetically_false_performance_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            write_json(
                swing / "public" / "execution_feedback_latest.json",
                valid_feedback(rotation),
            )
            performance = valid_performance(rotation)
            performance["history"][0]["benchmark_price"] = 8.0
            performance["performance_id"] = hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in performance.items()
                        if key != "performance_id"
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            write_json(
                swing / "public" / "live_performance_latest.json",
                performance,
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        evidence = result["checks"]["direct_execution_evidence"]
        self.assertFalse(evidence["performance_valid"])
        self.assertIn(
            "LIVE_PERFORMANCE_HISTORY_0_BENCHMARK_NAV_MISMATCH",
            evidence["performance_errors"],
        )

    def test_post_execution_model_estimate_performance_is_not_final(self):
        with tempfile.TemporaryDirectory() as directory:
            etf, swing, rotation = build_layout(Path(directory))
            write_json(
                swing / "public" / "execution_feedback_latest.json",
                valid_feedback(rotation),
            )
            performance = valid_performance(rotation)
            performance.update(
                {
                    "portfolio_state_evidence": "MODEL_ESTIMATE_PENDING",
                    "pending_broker_confirmation_plan_id": "plan-test",
                    "last_execution_satisfied_plan_id": "",
                    "broker_reconciliation_id": "",
                }
            )
            performance["history"][0].update(
                {
                    "portfolio_state_evidence": "MODEL_ESTIMATE_PENDING",
                    "pending_broker_confirmation_plan_id": "plan-test",
                    "last_execution_satisfied_plan_id": "",
                    "broker_reconciliation_id": "",
                }
            )
            performance["performance_id"] = hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in performance.items()
                        if key != "performance_id"
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            write_json(
                swing / "public" / "live_performance_latest.json",
                performance,
            )
            result = build_joint_health(
                etf,
                swing,
                now=datetime(2026, 7, 21, 18, 0),
            )
        self.assertEqual("BLOCKED", result["status"])
        evidence = result["checks"]["direct_execution_evidence"]
        self.assertFalse(evidence["performance_valid"])
        self.assertIn(
            "LIVE_PERFORMANCE_STATE_EVIDENCE_NOT_FINAL",
            evidence["performance_errors"],
        )


if __name__ == "__main__":
    unittest.main()
