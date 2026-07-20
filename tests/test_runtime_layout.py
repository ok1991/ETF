import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar import pipeline, reporting
from etf_radar.config import _resolve_swing_source
from etf_radar.calibration.pipeline import rotation_model_identity
from etf_radar.paths import RuntimePaths
from etf_radar.trading import DEFAULT_ETF_COST_MODEL
from test_signal_contracts import v4_signal


def temporary_paths(root: Path) -> RuntimePaths:
    runtime = root / ".runtime"
    artifacts = root / "artifacts"
    return RuntimePaths(
        root=root,
        runtime=runtime,
        data=runtime / "data",
        state=runtime / "state",
        logs=runtime / "logs",
        artifacts=artifacts,
        calibration=artifacts / "calibration",
        public=root / "public",
        web=Path(__file__).resolve().parents[1] / "web",
    )


def valid_rotation():
    return {
        "schema_version": 2,
        "data_date": "2026-07-17",
        "execution_date": "2026-07-20",
        "generated_at": "2026-07-17 17:00:00",
        "last_rebalance_week": "2026-W29",
        "approved": True,
        "model_version": "rotation-v2-test-aaaaaaaa",
        "execution_policy_version": "single-exposure-authority-v4",
        "acceptance_policy_version": "rolling-excess-stability-v1",
        "exposure_authority": "v4_market_policy",
        "strategy_specification_fingerprint": "a" * 64,
        "sleeves": [["510300"], ["510300"]],
        "target_weights": {"510300": 0.5},
        "execution_liquidity": {
            "510300": {
                "average_daily_amount_20": 100000000.0,
                "max_new_risk_amount": 10000000.0,
                "max_participation_rate": 0.1,
                "as_of_date": "2026-07-17",
            }
        },
        "max_exposure_ratio": 0.5,
        "cash_weight": 0.5,
        "capacity_reference_capital": 10000.0,
        "market_policy": {
            "state": "RISK_OFF",
            "entry_permission": "MAINLINE_ONLY",
            "max_exposure_ratio": 0.5,
        },
        "walk_forward_metrics": {
            "information_ratio": 0.5,
            "capacity_truncation_count": 0,
            "requested_buy_value": 10000.0,
            "executed_buy_value": 10000.0,
            "capacity_truncated_buy_value": 0.0,
            "unfilled_buy_value": 0.0,
            "buy_fill_ratio": 1.0,
            "capacity_fill_ratio": 1.0,
            "cost_model": DEFAULT_ETF_COST_MODEL.to_dict(),
        },
    }


class RuntimeLayoutTests(unittest.TestCase):
    def test_local_swing_evidence_is_preferred_without_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "execution_feedback_history.json"
            local.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"SWING_EXECUTION_FEEDBACK_SOURCE": ""},
            ):
                self.assertEqual(
                    str(local),
                    _resolve_swing_source(
                        "SWING_EXECUTION_FEEDBACK_SOURCE",
                        local,
                        "https://example.invalid/remote.json",
                    ),
                )

    def test_explicit_swing_evidence_source_overrides_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "execution_feedback_history.json"
            local.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"SWING_EXECUTION_FEEDBACK_SOURCE": "D:/override/history.json"},
            ):
                self.assertEqual(
                    "D:/override/history.json",
                    _resolve_swing_source(
                        "SWING_EXECUTION_FEEDBACK_SOURCE",
                        local,
                        "https://example.invalid/remote.json",
                    ),
                )

    def test_rotation_model_identity_changes_with_executable_specification(self):
        portfolio = {
            **valid_rotation()["walk_forward_metrics"],
            "top_n": 3,
            "sleeve_count": 2,
            "holding_period_trading_days": 10,
            "weekly_trend_min": -0.25,
            "exposure_authority": "v4_market_policy",
            "rank_buffer": 2,
            "factor_weights": {"relative_strength": 0.5},
            "industry_constraint": "one_per_group",
        }
        first = rotation_model_identity(
            "abcdef0123456789", portfolio, "2026-07-19 01:00:00"
        )
        changed = json.loads(json.dumps(portfolio))
        changed["cost_model"]["max_participation_rate"] = 0.05
        second = rotation_model_identity(
            "abcdef0123456789", changed, "2026-07-19 01:00:00"
        )
        self.assertEqual(
            first,
            rotation_model_identity(
                "abcdef0123456789", portfolio, "2026-07-19 12:00:00"
            ),
        )
        self.assertNotEqual(first, second)

    def test_jinja_report_writes_public_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            original = reporting.PATHS
            reporting.PATHS = paths
            try:
                reporting.HTMLReporter.generate(
                    [{"code": "TEST", "name": "测试ETF", "price": 10, "stop_loss": 9.5, "stop_dist": 5.0, "rps": 80, "v4_priority": 90, "data_date": "2026-07-16", "status": "波段多头"}],
                    {"entry_permission": "TRADEABLE", "regime_level": "RISK_ON", "max_exposure_ratio": 0.8},
                )
                self.assertTrue((paths.public / "index.html").exists())
                self.assertTrue((paths.public / "assets" / "style.css").exists())
                document = (paths.public / "index.html").read_text(encoding="utf-8")
                self.assertIn('href="assets/style.css?v=', document)
                self.assertIn('src="assets/app.js?v=', document)
                self.assertIn('市场权限：<strong class="market-up">可交易</strong>', document)
                self.assertIn('<span>市场状态</span><strong class="market-up">风险偏好</strong>', document)
                self.assertIn('<span class="market-up">1</span> / <span class="market-down">0</span>', document)
                self.assertIn("<span>仓位上限</span>", document)
                self.assertIn("<th>损距</th>", document)
                self.assertIn("<td>5.0%</td>", document)
            finally:
                reporting.PATHS = original

    def test_bearish_market_metrics_render_in_green(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            original = reporting.PATHS
            reporting.PATHS = paths
            try:
                results = [
                    {"code": "BULL", "status": "波段多头"},
                    {"code": "BEAR1", "status": "波段空头"},
                    {"code": "BEAR2", "status": "偏空走弱"},
                ]
                reporting.HTMLReporter.generate(
                    results,
                    {"entry_permission": "BLOCKED", "regime_level": "RISK_OFF"},
                )
                document = (paths.public / "index.html").read_text(encoding="utf-8")
                self.assertIn('市场权限：<strong class="market-down">禁止开新仓</strong>', document)
                self.assertIn('<span>市场状态</span><strong class="market-down">风险规避</strong>', document)
                self.assertIn('<span>市场宽度</span><strong class="market-down">偏空</strong>', document)
                self.assertIn('<span class="market-up">1</span> / <span class="market-down">2</span>', document)
            finally:
                reporting.PATHS = original

    def test_public_contract_validation_accepts_v4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = temporary_paths(root)
            paths.ensure()
            contracts = root / "contracts"
            contracts.mkdir()
            source_schema = Path(__file__).resolve().parents[1] / "contracts" / "etf_signal_v4.schema.json"
            (contracts / "etf_signal_v4.schema.json").write_text(source_schema.read_text(encoding="utf-8"), encoding="utf-8")
            item = v4_signal()
            item["market_policy"]["max_exposure_ratio"] = 0.8
            payload = {
                "schema_version": 4,
                "update_time": "2026-07-17 17:00:00",
                "market_policy": item["market_policy"],
                "market_breadth": {},
                "signals": [item],
            }
            (paths.public / "etf_signals_latest.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                pipeline.verify_public_signal()
            finally:
                pipeline.PATHS = original

    def test_public_contract_validation_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = temporary_paths(root)
            paths.ensure()
            contracts = root / "contracts"
            contracts.mkdir()
            source_schema = Path(__file__).resolve().parents[1] / "contracts" / "etf_signal_v4.schema.json"
            (contracts / "etf_signal_v4.schema.json").write_text(source_schema.read_text(encoding="utf-8"), encoding="utf-8")
            (paths.public / "etf_signals_latest.json").write_text(
                json.dumps({"schema_version": 3, "signals": []}), encoding="utf-8"
            )
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                with self.assertRaises(RuntimeError):
                    pipeline.verify_public_signal()
            finally:
                pipeline.PATHS = original

    def test_rotation_contract_validation_accepts_shared_v2_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = temporary_paths(root)
            paths.ensure()
            contracts = root / "contracts"
            contracts.mkdir()
            source_schema = Path(__file__).resolve().parents[1] / "contracts" / "etf_rotation_v2.schema.json"
            (contracts / "etf_rotation_v2.schema.json").write_text(
                source_schema.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (paths.public / "etf_rotation_latest.json").write_text(
                json.dumps(valid_rotation(), ensure_ascii=False), encoding="utf-8"
            )
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                pipeline.verify_rotation_target()
            finally:
                pipeline.PATHS = original

    def test_rotation_contract_validation_rejects_weight_budget_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = temporary_paths(root)
            paths.ensure()
            contracts = root / "contracts"
            contracts.mkdir()
            source_schema = Path(__file__).resolve().parents[1] / "contracts" / "etf_rotation_v2.schema.json"
            (contracts / "etf_rotation_v2.schema.json").write_text(
                source_schema.read_text(encoding="utf-8"), encoding="utf-8"
            )
            payload = valid_rotation()
            payload["target_weights"] = {"510300": 0.4}
            (paths.public / "etf_rotation_latest.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                with self.assertRaises(RuntimeError):
                    pipeline.verify_rotation_target()
            finally:
                pipeline.PATHS = original

    def test_rotation_contract_validation_rejects_execution_cost_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = temporary_paths(root)
            paths.ensure()
            contracts = root / "contracts"
            contracts.mkdir()
            source_schema = Path(__file__).resolve().parents[1] / "contracts" / "etf_rotation_v2.schema.json"
            (contracts / "etf_rotation_v2.schema.json").write_text(
                source_schema.read_text(encoding="utf-8"), encoding="utf-8"
            )
            payload = valid_rotation()
            payload["walk_forward_metrics"]["cost_model"]["base_slippage_bps"] = 9.0
            (paths.public / "etf_rotation_latest.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                with self.assertRaises(RuntimeError):
                    pipeline.verify_rotation_target()
            finally:
                pipeline.PATHS = original

    def test_rotation_contract_validation_rejects_old_acceptance_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = temporary_paths(root)
            paths.ensure()
            contracts = root / "contracts"
            contracts.mkdir()
            source_schema = Path(__file__).resolve().parents[1] / "contracts" / "etf_rotation_v2.schema.json"
            (contracts / "etf_rotation_v2.schema.json").write_text(
                source_schema.read_text(encoding="utf-8"), encoding="utf-8"
            )
            payload = valid_rotation()
            payload["acceptance_policy_version"] = "aggregate-only-v0"
            (paths.public / "etf_rotation_latest.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                with self.assertRaises(RuntimeError):
                    pipeline.verify_rotation_target()
            finally:
                pipeline.PATHS = original

    def test_factor_health_artifact_validation_accepts_suspension(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.public / "factor_health_latest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-07-19 01:00:00",
                        "status": "SUSPENDED",
                        "approved_for_live_use": False,
                        "reasons": ["EFFECTIVE_FACTOR_COUNT_BELOW_2"],
                    }
                ),
                encoding="utf-8",
            )
            original = pipeline.PATHS
            pipeline.PATHS = paths
            try:
                pipeline.verify_factor_health()
            finally:
                pipeline.PATHS = original


if __name__ == "__main__":
    unittest.main()
