import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar import _core
from etf_radar.factor_evolution import load_factor_registry_with_status
from etf_radar.model_governance import validate_artifact_time, validate_bundle_member
from etf_radar.rotation import load_rotation_model_with_status


NOW = "2026-07-19 12:00:00"


class ModelTimeGovernanceTests(unittest.TestCase):
    def tearDown(self):
        _core._V4_CALIBRATION_CACHE = None
        _core._V4_CALIBRATION_LOADED = False
        _core._V4_CALIBRATION_STATUS_REASON = "CALIBRATION_NOT_LOADED"

    def test_generation_age_and_forward_label_lag_are_governed_separately(self):
        status = validate_artifact_time(
            {
                "generated_at": "2026-07-19 07:12:55",
                "trained_until": "2026-06-15",
            },
            now=NOW,
        )
        self.assertTrue(status.approved)
        self.assertEqual(status.generated_age_days, 0)
        self.assertEqual(status.training_lag_days, 34)

    def test_stale_generation_timestamp_is_fail_closed(self):
        status = validate_artifact_time(
            {
                "generated_at": "2026-06-20 07:12:55",
                "trained_until": "2026-07-15",
            },
            now=NOW,
        )
        self.assertFalse(status.approved)
        self.assertEqual(status.reason, "GENERATED_AT_STALE")

    def test_training_lag_has_its_own_fail_closed_limit(self):
        status = validate_artifact_time(
            {
                "generated_at": "2026-07-19 07:12:55",
                "trained_until": "2026-05-01",
            },
            now=NOW,
        )
        self.assertFalse(status.approved)
        self.assertEqual(status.reason, "TRAINED_UNTIL_STALE")

    def test_missing_or_future_generation_timestamp_is_rejected(self):
        missing = validate_artifact_time(
            {"trained_until": "2026-06-15"},
            now=NOW,
        )
        future = validate_artifact_time(
            {
                "generated_at": "2026-07-20 07:12:55",
                "trained_until": "2026-06-15",
            },
            now=NOW,
        )
        self.assertEqual(missing.reason, "GENERATED_AT_MISSING_OR_INVALID")
        self.assertEqual(future.reason, "GENERATED_AT_IN_FUTURE")

    def test_rotation_loader_returns_explicit_stale_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approved": True,
                        "generated_at": "2026-06-20 07:12:55",
                        "trained_until": "2026-07-15",
                    }
                ),
                encoding="utf-8",
            )
            model, reason = load_rotation_model_with_status(str(path), now=NOW)
        self.assertIsNone(model)
        self.assertEqual(reason, "ROTATION_MODEL_GENERATED_AT_STALE")

    def test_rotation_loader_rejects_old_execution_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approved": True,
                        "generated_at": "2026-07-19 07:12:55",
                        "trained_until": "2026-06-15",
                        "execution_policy_version": "adv-capacity-audit-authority-v3",
                    }
                ),
                encoding="utf-8",
            )
            model, reason = load_rotation_model_with_status(str(path), now=NOW)
        self.assertIsNone(model)
        self.assertEqual(reason, "ROTATION_MODEL_EXECUTION_POLICY_MISMATCH")

    def test_rotation_loader_rejects_legacy_exposure_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approved": True,
                        "generated_at": "2026-07-19 07:12:55",
                        "trained_until": "2026-06-15",
                        "execution_policy_version": "single-exposure-authority-v4",
                        "risk_budget_profile": {"RISK_OFF": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            model, reason = load_rotation_model_with_status(str(path), now=NOW)
        self.assertIsNone(model)
        self.assertEqual(reason, "ROTATION_MODEL_LEGACY_EXPOSURE_AUTHORITY")

    def test_factor_registry_loader_suspends_stale_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "approved": True,
                        "generated_at": "2026-07-19 07:12:55",
                        "trained_until": "2026-05-01",
                    }
                ),
                encoding="utf-8",
            )
            registry, reason = load_factor_registry_with_status(str(path), now=NOW)
        self.assertIsNone(registry)
        self.assertEqual(reason, "FACTOR_REGISTRY_TRAINED_UNTIL_STALE")

    def test_factor_registry_loader_rejects_candidate_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "approved": True,
                        "generated_at": "2026-07-19 07:12:55",
                        "trained_until": "2026-06-15",
                        "candidate_specification_fingerprint": "f" * 64,
                        "factors": [
                            {
                                "name": "momentum",
                                "status": "ACTIVE",
                                "expression": {"feature": "momentum_20"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry, reason = load_factor_registry_with_status(str(path), now=NOW)
        self.assertIsNone(registry)
        self.assertEqual(
            "FACTOR_REGISTRY_CANDIDATE_FINGERPRINT_MISMATCH", reason
        )

    def test_factor_registry_loader_computes_missing_legacy_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "approved": True,
                        "generated_at": "2026-07-19 07:12:55",
                        "trained_until": "2026-06-15",
                        "factors": [
                            {
                                "name": "momentum",
                                "status": "ACTIVE",
                                "expression": {"feature": "momentum_20"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry, reason = load_factor_registry_with_status(str(path), now=NOW)
        self.assertEqual("APPROVED", reason)
        self.assertIsNotNone(registry)
        self.assertEqual(64, len(registry["candidate_specification_fingerprint"]))
        self.assertEqual(
            "COMPUTED_LEGACY_REGISTRY",
            registry["candidate_specification_fingerprint_source"],
        )

    def test_v4_loader_rejects_stale_generated_at_before_using_model(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "calibration"
            / "v4_calibration.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["generated_at"] = "2020-01-01 00:00:00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v4.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "etf_radar._core.fingerprint_joint_price_directory",
                return_value=payload["data_fingerprint"],
            ), patch("etf_radar._core.validate_bundle_member", return_value="APPROVED"):
                model = _core.load_v4_calibration(str(path))
        self.assertIsNone(model)
        self.assertEqual(
            _core.v4_calibration_status_reason(),
            "CALIBRATION_GENERATED_AT_STALE",
        )

    def test_bundle_hash_detects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "rotation_model.json"
            artifact = {"artifact_bundle_id": "bundle-test", "approved": True}
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            (root / "calibration_bundle.json").write_text(
                json.dumps(
                    {
                        "artifact_bundle_id": "bundle-test",
                        "files": {"rotation_model.json": {"sha256": digest}},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                "APPROVED", validate_bundle_member(artifact_path, artifact)
            )
            artifact_path.write_text(json.dumps({**artifact, "approved": False}), encoding="utf-8")
            self.assertEqual(
                "BUNDLE_MEMBER_HASH_MISMATCH",
                validate_bundle_member(artifact_path, artifact),
            )


if __name__ == "__main__":
    unittest.main()
