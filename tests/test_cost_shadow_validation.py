import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar.calibration import pipeline


def candidate_payload():
    recommended = {
        **pipeline.DEFAULT_ETF_COST_MODEL.to_dict(),
        "base_slippage_bps": 8.0,
    }
    return {
        "schema_version": 1,
        "status": "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION",
        "approved_for_live_use": False,
        "auto_promotion_allowed": False,
        "requires_full_purged_walk_forward": True,
        "candidate_execution_policy_version": "empirical-cost-candidate-v1-12345678",
        "candidate_fingerprint": "a" * 64,
        "current_cost_model": pipeline.DEFAULT_ETF_COST_MODEL.to_dict(),
        "recommended_cost_model": recommended,
    }


class CostShadowValidationTests(unittest.TestCase):
    def test_ready_candidate_loads_as_typed_cost_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(candidate_payload()), encoding="utf-8")
            model, metadata = pipeline.load_cost_model_candidate(str(path))
        self.assertEqual(8.0, model.base_slippage_bps)
        self.assertFalse(metadata["auto_promotion_allowed"])

    def test_unready_or_live_approved_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            value = candidate_payload()
            value["status"] = "INSUFFICIENT_EVIDENCE"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not ready"):
                pipeline.load_cost_model_candidate(str(path))
            value = candidate_payload()
            value["approved_for_live_use"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "governance flags"):
                pipeline.load_cost_model_candidate(str(path))

    def test_shadow_cli_forces_all_outputs_into_isolated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.json"
            shadow = root / "shadow"
            candidate.write_text(json.dumps(candidate_payload()), encoding="utf-8")
            report = {
                "strategy_approved": True,
                "rotation_acceptance_gates": {"test_gate": True},
            }
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "calibrate_v4.py",
                        "--cost-model-candidate",
                        str(candidate),
                        "--shadow-output-dir",
                        str(shadow),
                        "--reuse-rows-cache",
                    ],
                ),
                patch(
                    "etf_radar.calibration.pipeline.load_rows_cache",
                    return_value=([{"date": "2026-06-15", "code": "512800"}], "fingerprint", {}),
                ) as cache_load,
                patch(
                    "etf_radar.calibration.pipeline.build_artifacts",
                    return_value=report,
                ) as build,
            ):
                pipeline.main_cli()
            manifest = json.loads(
                (shadow / "shadow_cost_validation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertTrue(manifest["shadow_only"])
        self.assertFalse(manifest["promotion_allowed"])
        self.assertEqual(8.0, manifest["cost_model"]["base_slippage_bps"])
        self.assertEqual(8.0, cache_load.call_args.kwargs["cost_model"].base_slippage_bps)
        self.assertEqual(8.0, build.call_args.kwargs["cost_model"].base_slippage_bps)
        self.assertTrue(str(build.call_args.args[3]).startswith(str(shadow)))
        self.assertEqual(
            "empirical-cost-candidate-v1-12345678",
            build.call_args.kwargs["execution_policy_version"],
        )

    def test_shadow_cli_rejects_production_artifact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text(json.dumps(candidate_payload()), encoding="utf-8")
            production_dir = Path(
                pipeline.main.Config.V4_CALIBRATION_FILE
            ).resolve().parent
            with patch.object(
                sys,
                "argv",
                [
                    "calibrate_v4.py",
                    "--cost-model-candidate",
                    str(candidate),
                    "--shadow-output-dir",
                    str(production_dir),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "production artifact"):
                    pipeline.main_cli()


if __name__ == "__main__":
    unittest.main()
