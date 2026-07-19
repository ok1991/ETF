import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GitHubLLMWorkflowTests(unittest.TestCase):
    def _workflow(self, name: str) -> str:
        return (ROOT / ".github" / "workflows" / name).read_text(
            encoding="utf-8"
        )

    def test_daily_cycle_cannot_be_redirected_by_openai_secret(self):
        workflow = self._workflow("etf-daily-analysis.yml")
        self.assertNotIn("OPENAI_API_KEY", workflow)
        self.assertIn('LLM_FACTOR_PROVIDER: "local"', workflow)
        self.assertIn('LLM_LOCAL_ENDPOINT: "https://ai.imlam.com/v1"', workflow)
        self.assertIn('LLM_LOCAL_MODEL: "gemini-3.5-flash"', workflow)

    def test_calibration_refresh_requires_primary_health_success(self):
        workflow = self._workflow("calibrate-v4.yml")
        self.assertNotIn("OPENAI_API_KEY", workflow)
        self.assertIn("python -m etf_radar.llm_provider_health", workflow)
        self.assertIn("continue-on-error: true", workflow)
        self.assertIn(
            "LLM_FACTOR_PROPOSALS_REFRESH: ${{ steps.llm_health.outcome == 'success' && 'true' || 'false' }}",
            workflow,
        )
        self.assertIn('LLM_FACTOR_PROPOSAL_COUNT: "2"', workflow)


if __name__ == "__main__":
    unittest.main()
