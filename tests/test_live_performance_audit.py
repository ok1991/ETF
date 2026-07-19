import hashlib
import json
import unittest
from datetime import date, timedelta

from etf_radar.live_performance_audit import audit_live_performance


def rotation_model():
    return {
        "version": "rotation-v2-live-test",
        "generated_at": "2026-01-01 09:00:00",
    }


def expected_execution(execution_date="2026-07-20"):
    return {
        "approved": True,
        "model_version": "rotation-v2-live-test",
        "execution_date": execution_date,
    }


def performance_payload(relative_values, *, model_version="rotation-v2-live-test", end_date=date(2026, 7, 18)):
    history = []
    start = end_date - timedelta(days=len(relative_values) - 1)
    strategy_peak = 0.0
    benchmark_peak = 0.0
    relative_peak = 0.0
    strategy_max_drawdown = 0.0
    benchmark_max_drawdown = 0.0
    relative_max_drawdown = 0.0
    for index, relative_nav in enumerate(relative_values):
        strategy_nav = float(relative_nav)
        benchmark_nav = 1.0
        relative_nav = strategy_nav / benchmark_nav
        strategy_peak = max(strategy_peak, strategy_nav)
        benchmark_peak = max(benchmark_peak, benchmark_nav)
        relative_peak = max(relative_peak, relative_nav)
        strategy_drawdown = strategy_nav / strategy_peak - 1.0
        benchmark_drawdown = benchmark_nav / benchmark_peak - 1.0
        relative_drawdown = relative_nav / relative_peak - 1.0
        strategy_max_drawdown = min(strategy_max_drawdown, strategy_drawdown)
        benchmark_max_drawdown = min(benchmark_max_drawdown, benchmark_drawdown)
        relative_max_drawdown = min(relative_max_drawdown, relative_drawdown)
        history.append(
            {
                "date": (start + timedelta(days=index)).isoformat(),
                "total_assets": round(10000.0 * relative_nav, 4),
                "benchmark_price": 4.0,
                "model_version": model_version,
                "strategy_nav": round(relative_nav, 8),
                "benchmark_nav": 1.0,
                "relative_nav": round(relative_nav, 8),
                "strategy_return": round(relative_nav - 1.0, 8),
                "benchmark_return": 0.0,
                "excess_return": round(relative_nav - 1.0, 8),
                "relative_return": round(relative_nav - 1.0, 8),
                "strategy_drawdown": round(strategy_drawdown, 8),
                "benchmark_drawdown": round(benchmark_drawdown, 8),
                "relative_drawdown": round(relative_drawdown, 8),
            }
        )
    last = history[-1]
    payload = {
        "schema_version": 1,
        "generated_at": "2026-07-18T15:00:00+08:00",
        "benchmark_code": "510300",
        "baseline": {"date": history[0]["date"], "strategy_assets": 10000.0, "benchmark_price": 4.0},
        "observation_count": len(history),
        "data_date": last["date"],
        "model_version": last["model_version"],
        "total_assets": last["total_assets"],
        "strategy_nav": last["strategy_nav"],
        "benchmark_nav": last["benchmark_nav"],
        "relative_nav": last["relative_nav"],
        "strategy_return": last["strategy_return"],
        "benchmark_return": last["benchmark_return"],
        "excess_return": last["excess_return"],
        "relative_return": last["relative_return"],
        "strategy_max_drawdown": round(strategy_max_drawdown, 8),
        "benchmark_max_drawdown": round(benchmark_max_drawdown, 8),
        "relative_max_drawdown": round(relative_max_drawdown, 8),
        "rolling_20": {},
        "rolling_60": {},
        "history": history,
    }
    payload["performance_id"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


class LivePerformanceAuditTests(unittest.TestCase):
    def test_expected_performance_session_allows_premarket_absence(self):
        audit = audit_live_performance(
            None,
            rotation_model(),
            source_status="UNAVAILABLE",
            now="2026-07-20 09:00:00",
            expected_execution=expected_execution(),
        )
        self.assertEqual("NO_LIVE_PERFORMANCE_EVIDENCE", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertEqual("2026-07-20", audit["expected_performance_date"])

    def test_missing_performance_after_expected_session_revokes_authority(self):
        audit = audit_live_performance(
            None,
            rotation_model(),
            source_status="UNAVAILABLE",
            now="2026-07-21 09:00:00",
            expected_execution=expected_execution(),
        )
        self.assertEqual("LIVE_PERFORMANCE_SESSION_MISSED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])

    def test_latest_session_attributed_to_wrong_model_revokes_authority(self):
        payload = performance_payload(
            [1.0, 1.001],
            model_version="old-model",
            end_date=date(2026, 7, 20),
        )
        audit = audit_live_performance(
            payload,
            rotation_model(),
            now="2026-07-20 16:00:00",
            expected_latest_data_date="2026-07-20",
            expected_execution=expected_execution(),
        )
        self.assertEqual(
            "LIVE_PERFORMANCE_MODEL_SESSION_MISMATCH", audit["status"]
        )
        self.assertFalse(audit["rotation_authority_allowed"])

    def test_missing_and_short_history_remain_warmup_without_revoking_authority(self):
        missing = audit_live_performance(
            None, rotation_model(), source_status="UNAVAILABLE", now="2026-07-19 12:00:00"
        )
        self.assertEqual("NO_LIVE_PERFORMANCE_EVIDENCE", missing["status"])
        self.assertTrue(missing["rotation_authority_allowed"])
        short = audit_live_performance(
            performance_payload([1.0 + index * 0.001 for index in range(10)]),
            rotation_model(),
            now="2026-07-19 12:00:00",
        )
        self.assertEqual("WARMUP", short["status"])
        self.assertFalse(short["recalibration_required"])

    def test_rolling_underperformance_triggers_research_but_not_immediate_revocation(self):
        values = [1.0 - index * 0.003 for index in range(25)]
        audit = audit_live_performance(
            performance_payload(values),
            rotation_model(),
            now="2026-07-19 12:00:00",
        )
        self.assertEqual("LIVE_MODEL_RECALIBRATION_REQUIRED", audit["status"])
        self.assertTrue(audit["recalibration_required"])
        self.assertTrue(audit["rotation_authority_allowed"])
        self.assertIn(
            "LIVE_ROLLING_20_RELATIVE_UNDERPERFORMANCE",
            audit["research_reasons"],
        )

    def test_relative_drawdown_breach_revokes_rotation_authority(self):
        values = [1.0] * 20 + [0.98, 0.95, 0.92, 0.88, 0.85]
        audit = audit_live_performance(
            performance_payload(values),
            rotation_model(),
            now="2026-07-19 12:00:00",
        )
        self.assertEqual("LIVE_RISK_LIMIT_BREACH", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertTrue(audit["recalibration_required"])
        self.assertIn("LIVE_RELATIVE_DRAWDOWN_BREACH", audit["hard_reasons"])

    def test_warmup_does_not_delay_hard_drawdown_revocation(self):
        audit = audit_live_performance(
            performance_payload([1.0, 1.01, 1.0, 0.94, 0.89]),
            rotation_model(),
            now="2026-07-19 12:00:00",
        )
        self.assertEqual(5, audit["current_model_observation_count"])
        self.assertEqual("LIVE_RISK_LIMIT_BREACH", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        self.assertTrue(audit["recalibration_required"])
        self.assertIn("LIVE_RELATIVE_DRAWDOWN_BREACH", audit["hard_reasons"])

    def test_only_contiguous_current_model_observations_are_used(self):
        payload = performance_payload([1.0] * 30, model_version="old-model")
        for index in range(5):
            payload["history"][-5 + index]["model_version"] = "rotation-v2-live-test"
        payload["model_version"] = "rotation-v2-live-test"
        payload["performance_id"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in payload.items() if key != "performance_id"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        audit = audit_live_performance(
            payload, rotation_model(), now="2026-07-19 12:00:00"
        )
        self.assertEqual(5, audit["current_model_observation_count"])
        self.assertEqual("WARMUP", audit["status"])

    def test_tampered_or_stale_published_evidence_fails_closed(self):
        tampered = performance_payload([1.0] * 20)
        tampered["relative_nav"] = 9.0
        audit = audit_live_performance(
            tampered, rotation_model(), now="2026-07-19 12:00:00"
        )
        self.assertEqual("LIVE_PERFORMANCE_EVIDENCE_REJECTED", audit["status"])
        self.assertFalse(audit["rotation_authority_allowed"])
        stale = audit_live_performance(
            performance_payload([1.0] * 20, end_date=date(2026, 6, 1)),
            rotation_model(),
            now="2026-07-19 12:00:00",
        )
        self.assertIn("LIVE_PERFORMANCE_STALE", stale["errors"])
        self.assertFalse(stale["rotation_authority_allowed"])

    def test_self_hashed_but_arithmetically_false_nav_is_rejected(self):
        payload = performance_payload([1.0, 1.02])
        payload["history"][-1]["benchmark_price"] = 8.0
        payload["performance_id"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in payload.items() if key != "performance_id"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        audit = audit_live_performance(
            payload, rotation_model(), now="2026-07-19 12:00:00"
        )
        self.assertEqual("LIVE_PERFORMANCE_EVIDENCE_REJECTED", audit["status"])
        self.assertIn(
            "LIVE_PERFORMANCE_HISTORY_1_BENCHMARK_NAV_MISMATCH",
            audit["errors"],
        )

    def test_verified_latest_trading_date_avoids_long_holiday_false_staleness(self):
        payload = performance_payload(
            [1.0] * 10,
            end_date=date(2026, 2, 13),
        )
        audit = audit_live_performance(
            payload,
            rotation_model(),
            now="2026-02-25 12:00:00",
            expected_latest_data_date="2026-02-13",
        )
        self.assertEqual("WARMUP", audit["status"])
        self.assertTrue(audit["rotation_authority_allowed"])

    def test_lagging_or_future_trading_date_fails_closed(self):
        lagging = audit_live_performance(
            performance_payload([1.0] * 10, end_date=date(2026, 7, 17)),
            rotation_model(),
            now="2026-07-19 12:00:00",
            expected_latest_data_date="2026-07-18",
        )
        self.assertIn("LIVE_PERFORMANCE_STALE_TRADING_DATE", lagging["errors"])
        future = audit_live_performance(
            performance_payload([1.0] * 10, end_date=date(2026, 7, 18)),
            rotation_model(),
            now="2026-07-19 12:00:00",
            expected_latest_data_date="2026-07-17",
        )
        self.assertIn("LIVE_PERFORMANCE_FUTURE_TRADING_DATE", future["errors"])
        self.assertFalse(future["rotation_authority_allowed"])


if __name__ == "__main__":
    unittest.main()
