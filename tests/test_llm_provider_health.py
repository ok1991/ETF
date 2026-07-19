import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar.llm_provider_health import run_provider_health_check


class LLMProviderHealthTests(unittest.TestCase):
    def _result(self, *, fallback_used=False, model="gemini-3.5-flash"):
        return {
            "status": "OK",
            "provider": "OPENAI_CHAT_COMPATIBLE",
            "model": model,
            "endpoint_fingerprint": "a" * 64,
            "fallback_used": fallback_used,
            "provider_attempts": [
                {
                    "provider": "OPENAI_CHAT_COMPATIBLE",
                    "model": model,
                    "status": "OK",
                }
            ],
            "proposals": [{"name": "candidate"}],
            "rejected": [],
        }

    def test_primary_success_writes_isolated_ok_health_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_path = root / "shadow" / "proposals.json"
            artifact_path = root / "audits" / "health.json"

            def generate(*_args, **_kwargs):
                proposal_path.parent.mkdir(parents=True, exist_ok=True)
                proposal_path.write_text(
                    json.dumps({"proposals": [{"name": "candidate"}]}),
                    encoding="utf-8",
                )
                return self._result()

            with patch(
                "etf_radar.llm_provider_health.load_or_generate_llm_proposals",
                side_effect=generate,
            ):
                health = run_provider_health_check(
                    artifact_path=artifact_path,
                    proposal_path=proposal_path,
                    proposal_count=1,
                )

            self.assertEqual("OK", health["status"])
            self.assertFalse(health["fallback_used"])
            self.assertEqual("PRIMARY", health["health_mode"])
            self.assertTrue(health["primary_provider_healthy"])
            self.assertTrue(health["refresh_allowed"])
            self.assertFalse(health["credential_value_persisted"])
            self.assertEqual(64, len(health["cache_sha256"]))
            self.assertEqual(health, json.loads(artifact_path.read_text("utf-8")))

    def test_unexpected_fallback_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_path = root / "shadow" / "proposals.json"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text("{}", encoding="utf-8")
            with patch(
                "etf_radar.llm_provider_health.load_or_generate_llm_proposals",
                return_value=self._result(fallback_used=True),
            ):
                health = run_provider_health_check(
                    artifact_path=root / "health.json",
                    proposal_path=proposal_path,
                )

            self.assertEqual("FAILED", health["status"])
            self.assertEqual("PRIMARY", health["health_mode"])
            self.assertFalse(health["primary_provider_healthy"])
            self.assertFalse(health["refresh_allowed"])
            self.assertEqual(
                "ACTIVE_PROVIDER_MODEL_IDENTITY_MISMATCH",
                health["error_code"],
            )

    def test_health_check_restores_refresh_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_path = root / "proposals.json"
            proposal_path.write_text("{}", encoding="utf-8")
            before = os.environ.get("LLM_FACTOR_PROPOSALS_REFRESH")
            with patch(
                "etf_radar.llm_provider_health.load_or_generate_llm_proposals",
                return_value=self._result(),
            ):
                run_provider_health_check(
                    artifact_path=root / "health.json",
                    proposal_path=proposal_path,
                )
            self.assertEqual(
                before, os.environ.get("LLM_FACTOR_PROPOSALS_REFRESH")
            )


if __name__ == "__main__":
    unittest.main()
