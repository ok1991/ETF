import json
import tempfile
import unittest
from pathlib import Path

from etf_radar import pipeline, reporting
from etf_radar.paths import RuntimePaths
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


class RuntimeLayoutTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
