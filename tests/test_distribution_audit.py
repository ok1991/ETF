import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar.distribution_audit import audit_rotation_distribution
from test_runtime_layout import valid_rotation


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class DistributionAuditTests(unittest.TestCase):
    @property
    def schema_path(self):
        return Path(__file__).resolve().parents[1] / "contracts" / "etf_rotation_v2.schema.json"

    def run_audit(self, remote):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        local_path = root / "local.json"
        output_path = root / "audit.json"
        local_path.write_text(json.dumps(valid_rotation()), encoding="utf-8")
        with patch(
            "etf_radar.distribution_audit.urllib.request.urlopen",
            return_value=FakeResponse(remote),
        ):
            result = audit_rotation_distribution(
                local_path,
                output_path,
                self.schema_path,
                url="https://distribution.example.invalid/rotation.json",
            )
        persisted = json.loads(output_path.read_text(encoding="utf-8"))
        directory.cleanup()
        self.assertEqual(result, persisted)
        return result

    def test_matching_remote_grants_remote_distribution_readiness(self):
        result = self.run_audit(valid_rotation())
        self.assertEqual("MATCH", result["status"])
        self.assertTrue(result["same_host_execution_allowed"])
        self.assertTrue(result["remote_only_execution_allowed"])
        self.assertEqual([], result["mismatched_fields"])

    def test_valid_but_different_remote_is_reported_without_blocking_same_host(self):
        remote = valid_rotation()
        remote["model_version"] = "rotation-v2-other-bbbbbbbb"
        remote["strategy_specification_fingerprint"] = "b" * 64
        result = self.run_audit(remote)
        self.assertEqual("REMOTE_IDENTITY_MISMATCH", result["status"])
        self.assertTrue(result["same_host_execution_allowed"])
        self.assertFalse(result["remote_only_execution_allowed"])
        self.assertIn("model_version", result["mismatched_fields"])

    def test_invalid_remote_contract_is_fail_closed_remotely(self):
        result = self.run_audit({"approved": True})
        self.assertEqual("REMOTE_CONTRACT_INVALID", result["status"])
        self.assertTrue(result["same_host_execution_allowed"])
        self.assertFalse(result["remote_only_execution_allowed"])
        self.assertTrue(result["remote_contract_errors"])

    def test_unavailable_remote_is_fail_closed_remotely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_path = root / "local.json"
            output_path = root / "audit.json"
            local_path.write_text(json.dumps(valid_rotation()), encoding="utf-8")
            with patch(
                "etf_radar.distribution_audit.urllib.request.urlopen",
                side_effect=RuntimeError("offline"),
            ):
                result = audit_rotation_distribution(
                    local_path,
                    output_path,
                    self.schema_path,
                    url="https://distribution.example.invalid/rotation.json",
                )
        self.assertEqual("REMOTE_UNAVAILABLE", result["status"])
        self.assertTrue(result["same_host_execution_allowed"])
        self.assertFalse(result["remote_only_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
