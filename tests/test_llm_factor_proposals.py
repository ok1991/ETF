import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar.llm_factor_proposals import (
    BUILTIN_CHAT_API_KEY,
    BUILTIN_CHAT_ENDPOINT,
    BUILTIN_CHAT_MODEL,
    load_or_generate_llm_proposals,
    normalise_proposals,
    parse_functional_expression,
    request_chat_compatible_proposals,
    request_llm_proposals,
    validate_expression,
)


FEATURES = ("relative_strength", "momentum_20", "volatility_20")


def proposal_payload():
    return {
        "proposals": [
            {
                "name": "quality momentum",
                "expression": {
                    "op": "div",
                    "args": [
                        {"feature": "relative_strength"},
                        {"feature": "volatility_20"},
                    ],
                },
                "economic_logic": "Relative strength scaled by volatility prefers persistent but less crowded trends.",
                "hypothesis": "Industry leadership with lower realised risk should persist over the next ten sessions.",
                "expected_horizon_days": 10,
                "failure_modes": ["Sharp market reversal", "Volatility regime discontinuity"],
            }
        ]
    }


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class LLMFactorProposalTests(unittest.TestCase):
    def test_expression_validator_rejects_unknown_feature_and_excess_complexity(self):
        valid, _, complexity = validate_expression(
            {"op": "mul", "args": [{"feature": "relative_strength"}, {"feature": "momentum_20"}]},
            FEATURES,
        )
        self.assertTrue(valid)
        self.assertEqual(3, complexity)
        valid, reason, _ = validate_expression({"feature": "future_return"}, FEATURES)
        self.assertFalse(valid)
        self.assertEqual("FEATURE_NOT_ALLOWED", reason)

    def test_normalised_proposal_is_auditable_candidate(self):
        accepted, rejected = normalise_proposals(proposal_payload(), FEATURES, model="test-model")
        self.assertEqual([], rejected)
        self.assertEqual(1, len(accepted))
        candidate = accepted[0]
        self.assertEqual("llm_structured_proposal", candidate["candidate_origin"])
        self.assertEqual("test-model", candidate["proposal_metadata"]["model"])
        self.assertEqual(10, candidate["proposal_metadata"]["expected_horizon_days"])
        self.assertTrue(candidate["proposal_metadata"]["expression_signature"])

    def test_failure_modes_must_be_a_bounded_array_of_strings(self):
        invalid_values = [
            ("single failure mode string", "FAILURE_MODES_NOT_ARRAY"),
            ([], "FAILURE_MODES_COUNT_OUT_OF_RANGE"),
            (["one", "two", "three", "four", "five", "six"], "FAILURE_MODES_COUNT_OUT_OF_RANGE"),
            (["x"], "INVALID_FAILURE_MODE"),
            ([123], "INVALID_FAILURE_MODE"),
        ]
        for failure_modes, reason in invalid_values:
            with self.subTest(reason=reason, failure_modes=failure_modes):
                payload = proposal_payload()
                payload["proposals"][0]["failure_modes"] = failure_modes
                accepted, rejected = normalise_proposals(
                    payload, FEATURES, model="test-model"
                )
                self.assertEqual([], accepted)
                self.assertEqual(reason, rejected[0]["reason"])

    def test_invalid_metadata_does_not_reserve_expression_signature(self):
        invalid = proposal_payload()["proposals"][0]
        invalid["failure_modes"] = "not an array"
        valid = proposal_payload()["proposals"][0]
        accepted, rejected = normalise_proposals(
            {"proposals": [invalid, valid]}, FEATURES, model="test-model"
        )
        self.assertEqual(1, len(accepted))
        self.assertEqual("FAILURE_MODES_NOT_ARRAY", rejected[0]["reason"])

    def test_functional_expression_text_is_parsed_without_eval(self):
        expression = parse_functional_expression(
            "div(relative_strength, max(volatility_20, momentum_20))"
        )
        valid, reason, _ = validate_expression(expression, FEATURES)
        self.assertTrue(valid, reason)
        payload = proposal_payload()
        payload["proposals"][0]["expression"] = (
            "div(relative_strength, volatility_20)"
        )
        accepted, rejected = normalise_proposals(
            payload, FEATURES, model="test-model"
        )
        self.assertEqual([], rejected)
        self.assertEqual(
            "functional_text_normalised",
            accepted[0]["proposal_metadata"]["expression_source_format"],
        )
        self.assertEqual(
            "div(relative_strength, volatility_20)",
            accepted[0]["proposal_metadata"]["original_expression"],
        )

    def test_missing_key_is_fail_closed_and_writes_audit_artifact(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                "LLM_FACTOR_PROVIDER": "openai",
                "OPENAI_MODEL": "test-model",
            },
            clear=True,
        ):
            path = Path(directory) / "llm.json"
            result = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("MISSING_API_KEY", result["status"])
            self.assertEqual([], result["proposals"])
            self.assertTrue(path.is_file())
            self.assertEqual("OPENAI_RESPONSES", result["provider"])
            self.assertIn("OPENAI_RESPONSES:test-model", result["model_identity"])

    def test_valid_recent_cache_is_reused_offline_without_being_overwritten(self):
        provider = "OPENAI_CHAT_COMPATIBLE"
        model = "cached-compatible-model"
        identity = f"{provider}:{model}:cached"
        accepted, rejected = normalise_proposals(
            proposal_payload(),
            FEATURES,
            model=model,
            provider=provider,
            model_identity=identity,
        )
        self.assertEqual([], rejected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            cached = {
                "status": "OK",
                "model": model,
                "provider": provider,
                "model_identity": identity,
                "endpoint_fingerprint": "a" * 64,
                "generated_at": "2026-07-19 12:00:00",
                "prompt_version": "llm-factor-proposal-v2-static-context",
                "historical_safe_context": True,
                "proposals": accepted,
                "rejected": [],
            }
            path.write_text(json.dumps(cached), encoding="utf-8")
            before = path.read_bytes()
            with patch.dict(
                os.environ,
                {
                    "LLM_FACTOR_PROPOSALS_ENABLED": "auto",
                    "LLM_BUILTIN_PROVIDER_ENABLED": "false",
                },
                clear=True,
            ):
                result = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("CACHED_OFFLINE", result["status"])
            self.assertEqual(1, len(result["proposals"]))
            self.assertTrue(result["cache_artifact_preserved"])
            self.assertEqual(before, path.read_bytes())

    def test_invalid_cached_rationale_is_not_reused_offline(self):
        provider = "OPENAI_CHAT_COMPATIBLE"
        model = "cached-compatible-model"
        identity = f"{provider}:{model}:cached"
        accepted, rejected = normalise_proposals(
            proposal_payload(),
            FEATURES,
            model=model,
            provider=provider,
            model_identity=identity,
        )
        self.assertEqual([], rejected)
        accepted[0]["proposal_metadata"]["failure_modes"] = "not an array"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "OK",
                        "model": model,
                        "provider": provider,
                        "model_identity": identity,
                        "endpoint_fingerprint": "a" * 64,
                        "generated_at": "2026-07-19 12:00:00",
                        "prompt_version": "llm-factor-proposal-v2-static-context",
                        "historical_safe_context": True,
                        "proposals": accepted,
                        "rejected": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "LLM_FACTOR_PROPOSALS_ENABLED": "auto",
                    "LLM_FACTOR_PROVIDER": "openai",
                },
                clear=True,
            ):
                result = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("MISSING_API_KEY", result["status"])
            self.assertEqual([], result["proposals"])

    def test_missing_key_artifact_does_not_block_later_authenticated_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            with patch.dict(
                os.environ,
                {
                    "LLM_FACTOR_PROPOSALS_ENABLED": "auto",
                    "LLM_FACTOR_PROVIDER": "openai",
                    "OPENAI_MODEL": "test-model",
                },
                clear=True,
            ):
                first = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("MISSING_API_KEY", first["status"])
            generated = {
                "status": "OK",
                "model": "test-model",
                "generated_at": "2026-07-18 23:30:00",
                "prompt_version": "llm-factor-proposal-v1",
                "proposals": [],
                "rejected": [],
            }
            with patch.dict(
                os.environ,
                {
                    "LLM_FACTOR_PROPOSALS_ENABLED": "auto",
                    "LLM_FACTOR_PROVIDER": "openai",
                    "OPENAI_MODEL": "test-model",
                    "OPENAI_API_KEY": "later-key",
                },
                clear=True,
            ), patch(
                "etf_radar.llm_factor_proposals.request_llm_proposals",
                return_value=generated,
            ) as mocked:
                second = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("OK", second["status"])
            mocked.assert_called_once()

    def test_tampered_cached_expression_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_FACTOR_PROPOSALS_ENABLED": "auto",
                "LLM_FACTOR_PROVIDER": "openai",
                "OPENAI_MODEL": "test-model",
            },
            clear=True,
        ):
            path = Path(directory) / "llm.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "OK",
                        "model": "test-model",
                        "generated_at": "2026-07-18 23:30:00",
                        "prompt_version": "llm-factor-proposal-v1",
                        "proposals": [
                            {
                                "name": "tampered",
                                "candidate_origin": "llm_structured_proposal",
                                "expression": {"feature": "future_return"},
                                "proposal_metadata": {"model": "test-model"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("MISSING_API_KEY", result["status"])

    def test_clear_environment_uses_builtin_chat_provider_defaults(self):
        generated = {
            "status": "OK",
            "provider": "OPENAI_CHAT_COMPATIBLE",
            "model": BUILTIN_CHAT_MODEL,
            "proposals": [],
            "rejected": [],
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LLM_FACTOR_PROPOSALS_ENABLED": "true"},
            clear=True,
        ), patch(
            "etf_radar.llm_factor_proposals.request_chat_compatible_proposals",
            return_value=generated,
        ) as mocked:
            result = load_or_generate_llm_proposals(
                FEATURES, {}, Path(directory) / "llm.json"
            )
        self.assertEqual("OK", result["status"])
        mocked.assert_called_once()
        call = mocked.call_args
        self.assertEqual(BUILTIN_CHAT_API_KEY, call.kwargs["api_key"])
        self.assertEqual(BUILTIN_CHAT_MODEL, call.kwargs["model"])
        self.assertEqual(
            f"{BUILTIN_CHAT_ENDPOINT}/chat/completions",
            call.kwargs["endpoint"],
        )

    def test_builtin_provider_failure_does_not_use_a_secondary_provider(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"LLM_FACTOR_PROPOSALS_ENABLED": "true"},
            clear=True,
        ), patch(
            "etf_radar.llm_factor_proposals.request_chat_compatible_proposals",
            side_effect=RuntimeError("provider unavailable"),
        ) as mocked:
            result = load_or_generate_llm_proposals(
                FEATURES, {}, Path(directory) / "llm.json"
            )
        self.assertEqual("PROVIDER_REQUEST_FAILED", result["status"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(1, len(result["provider_attempts"]))
        self.assertEqual(1, mocked.call_count)

    def test_custom_remote_provider_failure_never_borrows_builtin_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                "LLM_FACTOR_PROVIDER": "local",
                "LLM_LOCAL_ENDPOINT": "https://custom.example.invalid/v1",
                "LLM_LOCAL_MODEL": "custom-model",
                "LLM_LOCAL_API_KEY": "custom-key",
            },
            clear=True,
        ), patch(
            "etf_radar.llm_factor_proposals.request_chat_compatible_proposals",
            side_effect=RuntimeError("custom unavailable"),
        ) as mocked:
            result = load_or_generate_llm_proposals(
                FEATURES, {}, Path(directory) / "llm.json"
            )
        self.assertEqual("PROVIDER_REQUEST_FAILED", result["status"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(1, mocked.call_count)
        self.assertEqual(1, len(result["provider_attempts"]))

    def test_builtin_provider_failure_preserves_recent_valid_cache(self):
        provider = "OPENAI_CHAT_COMPATIBLE"
        model = "cached-model"
        identity = f"{provider}:{model}:cached"
        accepted, rejected = normalise_proposals(
            proposal_payload(),
            FEATURES,
            model=model,
            provider=provider,
            model_identity=identity,
        )
        self.assertEqual([], rejected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "OK",
                        "model": model,
                        "provider": provider,
                        "model_identity": identity,
                        "endpoint_fingerprint": "a" * 64,
                        "generated_at": "2026-07-19 12:00:00",
                        "prompt_version": "llm-factor-proposal-v2-static-context",
                        "historical_safe_context": True,
                        "proposals": accepted,
                        "rejected": [],
                    }
                ),
                encoding="utf-8",
            )
            before = path.read_bytes()
            with patch.dict(
                os.environ,
                {
                    "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                    "LLM_FACTOR_PROPOSALS_REFRESH": "true",
                },
                clear=True,
            ), patch(
                "etf_radar.llm_factor_proposals.request_chat_compatible_proposals",
                side_effect=RuntimeError("provider unavailable"),
            ) as mocked:
                result = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("CACHED_PROVIDER_FAILURE", result["status"])
            self.assertEqual(1, mocked.call_count)
            self.assertTrue(result["cache_artifact_preserved"])
            self.assertEqual(before, path.read_bytes())

    def test_responses_api_structured_output_is_parsed_and_validated(self):
        api_response = {
            "id": "resp_test",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": json.dumps(proposal_payload())}
                    ],
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        with patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            return_value=FakeResponse(api_response),
        ) as mocked:
            result = request_llm_proposals(
                FEATURES,
                {
                    "factors": [{"name": "DO_NOT_LEAK_ACTIVE_FACTOR"}],
                    "retired_factors": [{"name": "DO_NOT_LEAK_RETIREMENT", "reasons": ["secret"]}],
                },
                api_key="not-a-real-key",
                model="test-model",
                endpoint="https://example.invalid/v1/responses",
                proposal_count=3,
            )
        self.assertEqual("OK", result["status"])
        self.assertEqual("resp_test", result["request_id"])
        self.assertEqual(1, len(result["proposals"]))
        request = mocked.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual("json_schema", body["text"]["format"]["type"])
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertNotIn("not-a-real-key", request.data.decode("utf-8"))
        self.assertNotIn("DO_NOT_LEAK_ACTIVE_FACTOR", request.data.decode("utf-8"))
        self.assertNotIn("DO_NOT_LEAK_RETIREMENT", request.data.decode("utf-8"))

    def test_local_chat_compatible_provider_generates_auditable_candidates_without_cloud_key(self):
        api_response = {
            "id": "local-response",
            "choices": [
                {"message": {"content": json.dumps(proposal_payload())}}
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 40},
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                "LLM_FACTOR_PROVIDER": "local",
                "LLM_LOCAL_ENDPOINT": "http://127.0.0.1:11434/v1/chat/completions",
                "LLM_LOCAL_MODEL": "qwen-local-test",
            },
            clear=True,
        ), patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            return_value=FakeResponse(api_response),
        ) as mocked:
            result = load_or_generate_llm_proposals(
                FEATURES, {}, Path(directory) / "llm.json"
            )
        self.assertEqual("OK", result["status"])
        self.assertEqual("OPENAI_CHAT_COMPATIBLE", result["provider"])
        self.assertEqual("qwen-local-test", result["model"])
        self.assertIn("OPENAI_CHAT_COMPATIBLE:qwen-local-test", result["model_identity"])
        self.assertEqual(
            "OPENAI_CHAT_COMPATIBLE",
            result["proposals"][0]["proposal_metadata"]["provider"],
        )
        request = mocked.call_args.args[0]
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(
            "etf-main-llm-factor-research/1.0",
            request.headers["User-agent"],
        )
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual("json_schema", body["response_format"]["type"])

    def test_chat_compatible_base_v1_url_is_normalised(self):
        api_response = {
            "id": "compatible-response",
            "choices": [{"message": {"content": json.dumps(proposal_payload())}}],
        }
        with patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            return_value=FakeResponse(api_response),
        ) as mocked:
            result = request_chat_compatible_proposals(
                FEATURES,
                {},
                api_key="remote-test-key",
                model="compatible-model",
                endpoint="https://llm.example.invalid/v1",
            )
        self.assertEqual("OK", result["status"])
        self.assertEqual(
            "https://llm.example.invalid/v1/chat/completions",
            mocked.call_args.args[0].full_url,
        )

    def test_chat_compatible_markdown_response_gets_one_strict_json_retry(self):
        repaired = {
            "id": "repaired-response",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(proposal_payload()["proposals"])
                    }
                }
            ],
        }
        markdown = {
            "id": "ignored-schema-response",
            "choices": [{"message": {"content": "## Candidate factors\nnot json"}}],
        }
        with patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            side_effect=[FakeResponse(markdown), FakeResponse(repaired)],
        ) as mocked:
            result = request_chat_compatible_proposals(
                FEATURES,
                {},
                api_key="remote-test-key",
                model="compatible-model",
                endpoint="https://llm.example.invalid/v1",
            )
        self.assertEqual("OK", result["status"])
        self.assertTrue(result["compatibility_fallback_used"])
        self.assertEqual(2, mocked.call_count)
        retry_body = json.loads(mocked.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual("json_object", retry_body["response_format"]["type"])

    def test_chat_compatible_empty_schema_response_gets_one_json_object_retry(self):
        class EmptyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b""

        repaired = {
            "id": "repaired-empty-response",
            "choices": [{"message": {"content": json.dumps(proposal_payload())}}],
        }
        with patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            side_effect=[EmptyResponse(), FakeResponse(repaired)],
        ) as mocked:
            result = request_chat_compatible_proposals(
                FEATURES,
                {},
                api_key="remote-test-key",
                model="compatible-model",
                endpoint="https://llm.example.invalid/v1",
            )
        self.assertEqual("OK", result["status"])
        self.assertTrue(result["compatibility_fallback_used"])
        self.assertEqual(2, mocked.call_count)

    def test_chat_compatible_invalid_metadata_gets_one_bounded_repair(self):
        invalid = proposal_payload()
        invalid["proposals"][0]["failure_modes"] = "Abrupt regime reversal"
        first = {
            "id": "invalid-metadata-response",
            "choices": [{"message": {"content": json.dumps(invalid)}}],
        }
        repaired = {
            "id": "repaired-metadata-response",
            "choices": [
                {"message": {"content": json.dumps(proposal_payload())}}
            ],
        }
        with patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            side_effect=[FakeResponse(first), FakeResponse(repaired)],
        ) as mocked:
            result = request_chat_compatible_proposals(
                FEATURES,
                {},
                api_key="remote-test-key",
                model="compatible-model",
                endpoint="https://llm.example.invalid/v1",
            )
        self.assertEqual("OK", result["status"])
        self.assertTrue(result["validation_repair_used"])
        self.assertEqual(2, mocked.call_count)
        retry_body = json.loads(mocked.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual("json_object", retry_body["response_format"]["type"])
        self.assertIn(
            "FAILURE_MODES_NOT_ARRAY",
            retry_body["messages"][-1]["content"],
        )

    def test_chat_compatible_repeated_single_failure_mode_is_normalised(self):
        invalid = proposal_payload()
        invalid["proposals"][0]["failure_modes"] = "Abrupt regime reversal"
        response = {
            "id": "invalid-metadata-response",
            "choices": [{"message": {"content": json.dumps(invalid)}}],
        }
        with patch(
            "etf_radar.llm_factor_proposals.urllib.request.urlopen",
            side_effect=[FakeResponse(response), FakeResponse(response)],
        ) as mocked:
            result = request_chat_compatible_proposals(
                FEATURES,
                {},
                api_key="remote-test-key",
                model="compatible-model",
                endpoint="https://llm.example.invalid/v1",
            )
        self.assertEqual("OK", result["status"])
        self.assertTrue(result["validation_repair_used"])
        self.assertTrue(result["compatibility_metadata_normalised"])
        self.assertEqual(2, mocked.call_count)
        self.assertEqual(
            ["Abrupt regime reversal"],
            result["proposals"][0]["proposal_metadata"]["failure_modes"],
        )

    def test_remote_chat_compatible_endpoint_requires_authentication(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                "LLM_FACTOR_PROVIDER": "local",
                "LLM_LOCAL_ENDPOINT": "https://llm.example.invalid/v1/chat/completions",
                "LLM_LOCAL_MODEL": "remote-model",
            },
            clear=True,
        ):
            result = load_or_generate_llm_proposals(
                FEATURES, {}, Path(directory) / "llm.json"
            )
        self.assertEqual("LOCAL_ENDPOINT_REQUIRES_API_KEY", result["status"])
        self.assertEqual([], result["proposals"])

    def test_local_provider_cache_is_isolated_by_model_identity(self):
        api_response = {
            "id": "local-response",
            "choices": [{"message": {"content": json.dumps(proposal_payload())}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            base_env = {
                "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                "LLM_FACTOR_PROVIDER": "local",
                "LLM_LOCAL_ENDPOINT": "http://localhost:11434/v1/chat/completions",
                "LLM_LOCAL_MODEL": "model-a",
            }
            with patch.dict(os.environ, base_env, clear=True), patch(
                "etf_radar.llm_factor_proposals.urllib.request.urlopen",
                return_value=FakeResponse(api_response),
            ):
                first = load_or_generate_llm_proposals(FEATURES, {}, path)
            self.assertEqual("OK", first["status"])
            with patch.dict(
                os.environ,
                {**base_env, "LLM_LOCAL_MODEL": "model-b"},
                clear=True,
            ), patch(
                "etf_radar.llm_factor_proposals.request_chat_compatible_proposals",
                return_value={
                    **first,
                    "model": "model-b",
                    "model_identity": "OPENAI_CHAT_COMPATIBLE:model-b:new",
                },
            ) as regenerated:
                second = load_or_generate_llm_proposals(FEATURES, {}, path)
            regenerated.assert_called_once()
            self.assertEqual("model-b", second["model"])


if __name__ == "__main__":
    unittest.main()
