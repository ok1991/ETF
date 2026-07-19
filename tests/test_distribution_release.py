import json
import tempfile
import unittest
from pathlib import Path

from etf_radar.distribution_release import prepare_distribution_release
from test_runtime_layout import valid_rotation


class DistributionReleaseTests(unittest.TestCase):
    def test_valid_rotation_is_staged_with_exact_hashes(self):
        payload = valid_rotation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rotation_path = root / "rotation.json"
            rotation_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            schema = (
                Path(__file__).resolve().parents[1]
                / "contracts"
                / "etf_rotation_v2.schema.json"
            )
            manifest = prepare_distribution_release(
                rotation_path,
                schema,
                {
                    "status": "REMOTE_CONTRACT_INVALID",
                    "distribution_url": "https://distribution.example/etf_rotation_latest.json",
                    "remote_only_execution_allowed": False,
                },
                root / "releases",
            )
            staged = Path(manifest["payload_path"])
            latest = root / "releases" / "distribution_release_latest.json"
            self.assertEqual("READY_FOR_EXTERNAL_PUBLISH", manifest["status"])
            self.assertEqual(rotation_path.read_bytes(), staged.read_bytes())
            self.assertTrue(latest.is_file())
            self.assertEqual(manifest, json.loads(latest.read_text(encoding="utf-8")))

    def test_invalid_rotation_cannot_be_prepared_for_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rotation_path = root / "rotation.json"
            rotation_path.write_text(json.dumps({"approved": True}), encoding="utf-8")
            schema = (
                Path(__file__).resolve().parents[1]
                / "contracts"
                / "etf_rotation_v2.schema.json"
            )
            with self.assertRaisesRegex(ValueError, "release source is invalid"):
                prepare_distribution_release(
                    rotation_path,
                    schema,
                    {},
                    root / "releases",
                )
            self.assertFalse((root / "releases" / "distribution_release_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
