import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etf_radar import cycle
from etf_radar.paths import RuntimePaths


ROOT = Path(__file__).resolve().parents[1]
# Synthetic fixture identity. Tests no longer depend on production artifacts or
# the gitignored local market-data cache (which is empty on GitHub Actions).
FINGERPRINT = "test-fingerprint-5y-fixture-000000000000000000000000000000000001"
TRAINED_UNTIL = "2026-06-15"
BUNDLE_ID = "test-bundle-5y-fixture"


def _synthetic_folds(count: int = 8) -> list:
    """Enough purged OK folds for validate_staged_bundle without production data."""
    folds = []
    for index in range(count):
        year = 2018 + index
        folds.append(
            {
                "name": f"fold_{index + 1}",
                "train": f"{year}-01-01..{year}-06-30",
                "validate": f"{year}-08-01..{year}-09-30",
                "train_start": f"{year}-01-01",
                "train_end": f"{year}-06-30",
                # >= 28 calendar days after train_end for purge gap.
                "validate_start": f"{year}-08-01",
                "validate_end": f"{year}-09-30",
                "status": "OK",
                "factor_purge_method": "28_calendar_day_approx_20_trading_day_purge",
            }
        )
    return folds


def write_fixture_market_data(data_dir: Path) -> str:
    """Write minimal QFQ+RAW history and return its joint fingerprint."""
    import numpy as np
    import pandas as pd
    from etf_radar.signals.contract import fingerprint_joint_price_directory

    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2024-01-02", periods=700)
    close = np.cumprod(np.full(len(dates), 1.001))
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(dates), 1_000_000.0),
            "amount": close * 1_000_000.0,
        }
    )
    # One paired QFQ/RAW series is enough for joint fingerprinting.
    stamp = dates[-1].strftime("%Y%m%d")
    frame.to_csv(data_dir / f"510300_{stamp}.csv", index=False)
    frame.to_csv(data_dir / f"510300_raw_{stamp}.csv", index=False)
    return fingerprint_joint_price_directory(
        str(data_dir),
        TRAINED_UNTIL,
        policy=cycle.CALIBRATION_FINGERPRINT_POLICY,
    )


def temporary_paths(root: Path) -> RuntimePaths:
    runtime = root / ".runtime"
    data = runtime / "data"
    return RuntimePaths(
        root=root,
        runtime=runtime,
        data=data,
        state=runtime / "state",
        logs=runtime / "logs",
        artifacts=root / "artifacts",
        calibration=root / "artifacts" / "calibration",
        public=root / "public",
        web=ROOT / "web",
    )


def build_staged_bundle(
    directory: Path,
    *,
    generated_at="2026-07-19 07:12:55",
    data_dir=None,
    fingerprint=None,
) -> str:
    """Build a complete, self-contained staged calibration bundle for tests.

    Returns the data fingerprint used by the fixture. If data_dir is provided,
    synthetic market history is written there and the real joint fingerprint is
    used so validate_staged_bundle / calibration_due match on CI.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if fingerprint is None:
        if data_dir is not None:
            fingerprint = write_fixture_market_data(data_dir)
        else:
            fingerprint = FINGERPRINT
    if not fingerprint:
        raise RuntimeError("fixture market fingerprint is empty")

    version = f"v4-test-{fingerprint[:8]}"
    rotation_version = f"rotation-v2-test-{fingerprint[:8]}"
    generated_at = str(generated_at)

    v4 = {
        "schema_version": 4,
        "version": version,
        "trained_until": TRAINED_UNTIL,
        "data_fingerprint": fingerprint,
        "feature_names": ["f1", "f2", "f3", "f4", "f5", "f6", "f7"],
        "feature_mean": [0.0] * 7,
        "feature_scale": [1.0] * 7,
        "early_stop_coefficients": [0.0] * 7,
        "win_coefficients": [0.0] * 7,
        "excess_coefficients": [0.0] * 7,
        "sample_count": 100,
        "thresholds": {"approved": True},
        "generated_at": generated_at,
        "artifact_bundle_id": BUNDLE_ID,
    }
    report = {
        "schema_version": 4,
        "generated_at": generated_at,
        "trained_until": TRAINED_UNTIL,
        "data_fingerprint": fingerprint,
        "calibration_version": version,
        "walk_forward_method": "expanding_calendar_windows_with_20d_purge_and_5d_embargo",
        "strategy_approved": True,
        "factor_registry_approved": False,
        "folds": _synthetic_folds(8),
        "rotation_acceptance_gates": {"test_gate": True},
        "artifact_bundle_id": BUNDLE_ID,
    }
    registry = {
        "schema_version": 2,
        "evolution_policy_version": cycle.FACTOR_EVOLUTION_POLICY_VERSION,
        "generated_at": generated_at,
        "trained_until": TRAINED_UNTIL,
        "approved": False,
        "data_fingerprint": fingerprint,
        "artifact_bundle_id": BUNDLE_ID,
        "active_factors": [],
        "retired_factors": [],
        "new_replacements": [],
        "research_challengers": [],
    }
    llm = {
        "status": "OK",
        "model": "test-model",
        "provider": "TEST",
        "model_identity": "test-model-identity",
        "endpoint_fingerprint": "test-endpoint",
        "generated_at": generated_at,
        "prompt_version": "test",
        "proposals": [],
        "rejected": [],
    }
    rotation = {
        "schema_version": 1,
        "artifact_bundle_id": BUNDLE_ID,
        "version": rotation_version,
        "generated_at": generated_at,
        "trained_until": TRAINED_UNTIL,
        "data_fingerprint": fingerprint,
        "execution_policy_version": cycle.ROTATION_EXECUTION_POLICY_VERSION,
        "acceptance_policy_version": cycle.ROTATION_ACCEPTANCE_POLICY_VERSION,
        "factor_evolution_policy_version": cycle.FACTOR_EVOLUTION_POLICY_VERSION,
        "strategy_specification_fingerprint": "test-spec",
        "approved": True,
        "approval_gates": {"test_gate": True},
        "portfolio_metrics": {},
        "top_n": 3,
        "sleeve_count": 2,
        "holding_period_trading_days": 10,
        "weekly_trend_min": -0.25,
        "exposure_authority": "v4_market_policy",
        "rank_buffer": 3,
        "selection_protocol": {},
        "factor_weights": {},
        "factor_economic_logic": {},
        "industry_constraint": None,
        "cost_model": {},
        "capacity_reference_capital": 10000.0,
        "capacity_selection_policy": None,
    }

    payloads = {
        "v4_calibration.json": v4,
        "v4_acceptance_report.json": report,
        "adaptive_factor_registry.json": registry,
        "llm_factor_proposals.json": llm,
        "rotation_model.json": rotation,
    }
    for name, value in payloads.items():
        (directory / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    files = {
        name: {"sha256": cycle._sha256(directory / name)}
        for name in cycle.CALIBRATION_FILES
    }
    (directory / "calibration_bundle.json").write_text(
        json.dumps({"artifact_bundle_id": BUNDLE_ID, "files": files}),
        encoding="utf-8",
    )
    return fingerprint


class ProductionCycleTests(unittest.TestCase):
    def test_explicit_llm_cache_source_is_seeded_without_touching_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            staging = root / "staging"
            calibration.mkdir()
            staging.mkdir()
            production_llm = calibration / "llm_factor_proposals.json"
            production_llm.write_text('{"status":"MISSING_API_KEY"}', encoding="utf-8")
            cache = root / "research_llm_cache.json"
            cache.write_text('{"status":"OK","proposals":[1]}', encoding="utf-8")
            before = production_llm.read_bytes()
            with patch.dict(
                "os.environ",
                {"LLM_FACTOR_CACHE_SOURCE": str(cache)},
            ):
                cycle._seed_staging(staging, calibration)
            self.assertEqual(
                cache.read_bytes(),
                (staging / "llm_factor_proposals.json").read_bytes(),
            )
            self.assertEqual(before, production_llm.read_bytes())

    def test_healthy_gemini_refresh_replaces_only_staging_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            runtime = root / ".runtime"
            staging.mkdir()
            target = staging / "llm_factor_proposals.json"
            target.write_text('{"status":"CACHED"}', encoding="utf-8")

            def healthy_refresh(*, artifact_path, proposal_path, proposal_count):
                self.assertEqual(6, proposal_count)
                proposal_path.parent.mkdir(parents=True, exist_ok=True)
                proposal_path.write_text(
                    '{"status":"OK","proposals":[{"name":"fresh"}]}',
                    encoding="utf-8",
                )
                return {
                    "status": "OK",
                    "refresh_allowed": True,
                    "provider": "OPENAI_CHAT_COMPATIBLE",
                    "model": "gemini-3.5-flash",
                    "proposal_count": 1,
                    "error_code": "",
                    "cache_sha256": cycle._sha256(proposal_path),
                }

            with patch.dict(
                "os.environ",
                {
                    "LLM_FACTOR_CACHE_SOURCE": "",
                    "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                    "LLM_CYCLE_PROVIDER_REFRESH": "true",
                    "LLM_FACTOR_PROPOSAL_COUNT": "6",
                },
            ), patch(
                "etf_radar.cycle.run_provider_health_check",
                side_effect=healthy_refresh,
            ):
                result = cycle.refresh_llm_staging_cache(staging, runtime)
            refreshed_document = target.read_text(encoding="utf-8")
        self.assertTrue(result["refreshed"])
        self.assertEqual("REFRESHED_FROM_HEALTHY_GEMINI", result["status"])
        self.assertIn('"fresh"', refreshed_document)

    def test_unhealthy_gemini_preserves_seeded_staging_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            runtime = root / ".runtime"
            staging.mkdir()
            target = staging / "llm_factor_proposals.json"
            target.write_text('{"status":"CACHED"}', encoding="utf-8")
            before = target.read_bytes()
            with patch.dict(
                "os.environ",
                {
                    "LLM_FACTOR_CACHE_SOURCE": "",
                    "LLM_FACTOR_PROPOSALS_ENABLED": "true",
                    "LLM_CYCLE_PROVIDER_REFRESH": "true",
                },
            ), patch(
                "etf_radar.cycle.run_provider_health_check",
                return_value={
                    "status": "FAILED",
                    "refresh_allowed": False,
                    "provider": "OPENAI_CHAT_COMPATIBLE",
                    "model": "gemini-3.5-flash",
                    "proposal_count": 0,
                    "error_code": "PROVIDER_REQUEST_FAILED",
                },
            ):
                result = cycle.refresh_llm_staging_cache(staging, runtime)
            self.assertEqual(before, target.read_bytes())
        self.assertFalse(result["refreshed"])
        self.assertEqual("PROVIDER_UNHEALTHY_CACHE_PRESERVED", result["status"])

    def test_explicit_llm_cache_pin_disables_cycle_refresh(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"LLM_FACTOR_CACHE_SOURCE": str(Path(directory) / "pinned.json")},
        ), patch("etf_radar.cycle.run_provider_health_check") as health:
            result = cycle.refresh_llm_staging_cache(
                Path(directory) / "staging",
                Path(directory) / ".runtime",
            )
        health.assert_not_called()
        self.assertEqual("EXPLICIT_CACHE_PINNED", result["status"])

    def test_transactional_calibration_refreshes_research_cache_before_subprocess(self):
        refresh = {"status": "REFRESHED_FROM_HEALTHY_GEMINI", "refreshed": True}
        with tempfile.TemporaryDirectory() as directory, patch(
            "etf_radar.cycle.refresh_llm_staging_cache",
            return_value=refresh,
        ) as refresh_call, patch("etf_radar.cycle.subprocess.run") as subprocess_run:
            result = cycle._run_calibration(Path(directory), 5, 6)
        refresh_call.assert_called_once()
        subprocess_run.assert_called_once()
        self.assertEqual(refresh, result)

    def test_ready_cost_candidate_runs_shadow_once_then_reuses_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recommendation = root / "candidate.json"
            runtime = root / "runtime"
            data = root / "data"
            fingerprint = "a" * 64
            recommendation.write_text(
                json.dumps(
                    {
                        "status": "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION",
                        "candidate_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )

            def create_manifest(candidate_path, output_dir, data_dir, sample_step, workers):
                del candidate_path, data_dir, sample_step, workers
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "shadow_cost_validation_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "SHADOW_VALIDATION_COMPLETE",
                            "shadow_only": True,
                            "promotion_allowed": False,
                            "candidate_fingerprint": fingerprint,
                            "rotation_strategy_approved_under_candidate_costs": True,
                        }
                    ),
                    encoding="utf-8",
                )

            with patch(
                "etf_radar.cycle._run_cost_shadow_validation",
                side_effect=create_manifest,
            ) as runner:
                first = cycle.ensure_cost_shadow_validation(
                    recommendation, runtime, data
                )
                second = cycle.ensure_cost_shadow_validation(
                    recommendation, runtime, data
                )
        self.assertEqual("COMPLETED", first["status"])
        self.assertTrue(first["attempted"])
        self.assertEqual("REUSED", second["status"])
        self.assertFalse(second["attempted"])
        runner.assert_called_once()

    def test_unready_cost_candidate_never_runs_shadow_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recommendation = root / "candidate.json"
            recommendation.write_text(
                json.dumps(
                    {
                        "status": "INSUFFICIENT_EVIDENCE",
                        "candidate_fingerprint": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with patch("etf_radar.cycle._run_cost_shadow_validation") as runner:
                result = cycle.ensure_cost_shadow_validation(
                    recommendation, root / "runtime", root / "data"
                )
        self.assertEqual("CANDIDATE_NOT_READY", result["status"])
        runner.assert_not_called()

    def test_mature_hard_failure_of_approved_factor_registry_triggers_evolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            public = root / "public"
            calibration.mkdir()
            public.mkdir()
            (calibration / "adaptive_factor_registry.json").write_text(
                json.dumps(
                    {
                        "approved": True,
                        "generated_at": "2026-06-01 09:00:00",
                    }
                ),
                encoding="utf-8",
            )
            health_path = public / "factor_health_latest.json"
            health_path.write_text(
                json.dumps(
                    {
                        "status": "SUSPENDED",
                        "evidence_mature": True,
                        "reasons": [
                            "LIVE_ENSEMBLE_NEGATIVE_IC",
                            "LIVE_FACTOR_NEGATIVE_IC:failed_alpha",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            reasons = cycle.factor_health_recalibration_due(
                calibration,
                health_path,
                now="2026-07-19 12:00:00",
            )
        self.assertIn(
            "FACTOR_LIVE_HEALTH_HARD_FAILURE:LIVE_ENSEMBLE_NEGATIVE_IC",
            reasons,
        )
        self.assertIn(
            "FACTOR_LIVE_HEALTH_HARD_FAILURE:LIVE_FACTOR_NEGATIVE_IC:failed_alpha",
            reasons,
        )

    def test_unapproved_warmup_and_recent_registries_do_not_create_recalibration_storm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            public = root / "public"
            calibration.mkdir()
            public.mkdir()
            registry_path = calibration / "adaptive_factor_registry.json"
            health_path = public / "factor_health_latest.json"
            health_path.write_text(
                json.dumps(
                    {
                        "status": "SUSPENDED",
                        "evidence_mature": True,
                        "reasons": ["LIVE_ENSEMBLE_NEGATIVE_IC"],
                    }
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps({"approved": False, "generated_at": "2026-06-01"}),
                encoding="utf-8",
            )
            self.assertEqual(
                [],
                cycle.factor_health_recalibration_due(
                    calibration, health_path, now="2026-07-19"
                ),
            )

            registry_path.write_text(
                json.dumps({"approved": True, "generated_at": "2026-07-18"}),
                encoding="utf-8",
            )
            self.assertEqual(
                [],
                cycle.factor_health_recalibration_due(
                    calibration, health_path, now="2026-07-19"
                ),
            )
            health_path.write_text(
                json.dumps(
                    {
                        "status": "WARMUP",
                        "evidence_mature": False,
                        "reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps({"approved": True, "generated_at": "2026-06-01"}),
                encoding="utf-8",
            )
            self.assertEqual(
                [],
                cycle.factor_health_recalibration_due(
                    calibration, health_path, now="2026-07-19"
                ),
            )

    def test_structural_factor_failure_recalibrates_without_statistical_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            public = root / "public"
            calibration.mkdir()
            public.mkdir()
            (calibration / "adaptive_factor_registry.json").write_text(
                json.dumps(
                    {
                        "approved": True,
                        "generated_at": "2026-06-01 09:00:00",
                    }
                ),
                encoding="utf-8",
            )
            health_path = public / "factor_health_latest.json"
            health_path.write_text(
                json.dumps(
                    {
                        "status": "SUSPENDED",
                        "evidence_mature": False,
                        "reasons": ["UNSUPPORTED_LIVE_MONITOR_FEATURES"],
                    }
                ),
                encoding="utf-8",
            )
            reasons = cycle.factor_health_recalibration_due(
                calibration,
                health_path,
                now="2026-07-19 12:00:00",
            )
        self.assertEqual(
            [
                "FACTOR_LIVE_HEALTH_HARD_FAILURE:"
                "UNSUPPORTED_LIVE_MONITOR_FEATURES"
            ],
            reasons,
        )

    def test_immature_statistical_decay_does_not_recalibrate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            public = root / "public"
            calibration.mkdir()
            public.mkdir()
            (calibration / "adaptive_factor_registry.json").write_text(
                json.dumps(
                    {
                        "approved": True,
                        "generated_at": "2026-06-01 09:00:00",
                    }
                ),
                encoding="utf-8",
            )
            health_path = public / "factor_health_latest.json"
            health_path.write_text(
                json.dumps(
                    {
                        "status": "SUSPENDED",
                        "evidence_mature": False,
                        "reasons": ["LIVE_ENSEMBLE_NEGATIVE_IC"],
                    }
                ),
                encoding="utf-8",
            )
            reasons = cycle.factor_health_recalibration_due(
                calibration,
                health_path,
                now="2026-07-19 12:00:00",
            )
        self.assertEqual([], reasons)

    def test_github_schedules_use_transactional_cycle_entrypoint(self):
        daily = (ROOT / ".github" / "workflows" / "etf-daily-analysis.yml").read_text(
            encoding="utf-8"
        )
        calibration = (ROOT / ".github" / "workflows" / "calibrate-v4.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("python run_cycle.py", daily)
        self.assertIn("python run_cycle.py --check-last-status", daily)
        self.assertIn("python run_cycle.py --force-calibration --sample-step 5", calibration)
        self.assertIn("python run_cycle.py --check-last-status", calibration)
        self.assertNotIn("python calibrate_v4.py", calibration)

    def test_complete_current_bundle_is_not_due_before_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            data = root / "data"
            build_staged_bundle(calibration, data_dir=data)
            reasons = cycle.calibration_due(
                calibration,
                data,
                now="2026-07-19 12:00:00",
            )
        self.assertEqual([], reasons)

    def test_stale_generation_time_triggers_full_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            data = root / "data"
            build_staged_bundle(
                calibration,
                generated_at="2026-06-20 07:12:55",
                data_dir=data,
            )
            registry_path = calibration / "adaptive_factor_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["generated_at"] = "2026-06-20 07:12:55"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            reasons = cycle.calibration_due(
                calibration,
                data,
                now="2026-07-19 12:00:00",
            )
        self.assertTrue(any("GENERATED_AT_STALE" in item for item in reasons))

    def test_staged_bundle_requires_matching_authority_and_purged_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            data = root / "data"
            build_staged_bundle(staging, data_dir=data)
            manifest = cycle.validate_staged_bundle(
                staging,
                data,
                now="2026-07-19 12:00:00",
            )
            self.assertEqual(BUNDLE_ID, manifest["artifact_bundle_id"])
            self.assertEqual(8, manifest["valid_purged_fold_count"])

            report_path = staging / "v4_acceptance_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["folds"][1]["validate_start"] = report["folds"][1]["train_end"]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "purge gap"):
                cycle.validate_staged_bundle(
                    staging,
                    data,
                    now="2026-07-19 12:00:00",
                )

    def test_successful_llm_artifact_requires_provider_identity(self):
        for status in ("OK", "CACHED", "CACHED_OFFLINE", "CACHED_PROVIDER_FAILURE"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                staging = root / "staging"
                data = root / "data"
                build_staged_bundle(staging, data_dir=data)
                llm_path = staging / "llm_factor_proposals.json"
                llm = json.loads(llm_path.read_text(encoding="utf-8"))
                llm["status"] = status
                for field in ("provider", "model_identity", "endpoint_fingerprint"):
                    llm.pop(field, None)
                llm_path.write_text(json.dumps(llm), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "provider identity"):
                    cycle.validate_staged_bundle(
                        staging,
                        data,
                        now="2026-07-19 12:00:00",
                    )

    def test_promotion_rolls_back_every_replaced_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            target = root / "target"
            build_staged_bundle(staging)
            target.mkdir()
            old_contents = {}
            for name in cycle.CALIBRATION_FILES:
                old_contents[name] = f"old-{name}"
                (target / name).write_text(old_contents[name], encoding="utf-8")
            manifest = {"artifact_bundle_id": "test-bundle-5y-fixture"}
            real_replace = cycle.os.replace
            failed = {"value": False}

            def fail_once(source, destination):
                if str(destination).endswith("rotation_model.json") and not failed["value"]:
                    failed["value"] = True
                    raise OSError("injected promotion failure")
                return real_replace(source, destination)

            with patch("etf_radar.cycle.os.replace", side_effect=fail_once):
                with self.assertRaisesRegex(OSError, "injected"):
                    cycle.promote_staged_bundle(staging, target, manifest)
            for name, expected in old_contents.items():
                self.assertEqual(expected, (target / name).read_text(encoding="utf-8"))
            self.assertFalse((target / "calibration_bundle.json").exists())

    def test_calibration_failure_keeps_existing_bundle_and_writes_safe_status(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run"),
                patch.object(cycle, "calibration_due", return_value=["FORCED_TEST"]),
                patch.object(cycle, "_run_calibration", side_effect=RuntimeError("boom")),
            ):
                result = cycle.run_cycle()
            self.assertEqual("CALIBRATION_FAILED_SAFE_FALLBACK", result["status"])
            public_status = json.loads(
                (paths.public / "cycle_status_latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], public_status["status"])

    def test_up_to_date_cycle_runs_production_once_without_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.public / "distribution_audit_latest.json").write_text(
                json.dumps(
                    {
                        "status": "REMOTE_CONTRACT_INVALID",
                        "same_host_execution_allowed": True,
                        "remote_only_execution_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run") as production,
                patch.object(cycle, "calibration_due", return_value=[]),
                patch.object(cycle, "_run_calibration") as calibrate,
            ):
                result = cycle.run_cycle()
            self.assertEqual("UP_TO_DATE", result["status"])
            self.assertEqual(
                "REMOTE_CONTRACT_INVALID",
                result["remote_distribution_status"],
            )
            self.assertTrue(result["same_host_execution_allowed"])
            self.assertFalse(result["remote_only_execution_allowed"])
            production.assert_called_once_with()
            calibrate.assert_not_called()

    def test_live_cost_recalibration_status_stops_automatic_bundle_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.public / "execution_feedback_audit_latest.json").write_text(
                json.dumps({"status": "COST_MODEL_RECALIBRATION_REQUIRED"}),
                encoding="utf-8",
            )
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run") as production,
                patch.object(cycle, "calibration_due") as calibration_due,
                patch.object(cycle, "_run_calibration") as calibrate,
            ):
                result = cycle.run_cycle()
            self.assertEqual("COST_MODEL_RECALIBRATION_REQUIRED", result["status"])
            self.assertEqual(
                "CANDIDATE_MISSING", result["shadow_validation"]["status"]
            )
            production.assert_called_once_with()
            calibration_due.assert_not_called()
            calibrate.assert_not_called()

    def test_shadow_cost_validation_failure_keeps_cash_and_fails_cycle_health(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.public / "execution_feedback_audit_latest.json").write_text(
                json.dumps({"status": "COST_MODEL_RECALIBRATION_REQUIRED"}),
                encoding="utf-8",
            )
            (paths.public / "execution_cost_recalibration_latest.json").write_text(
                json.dumps(
                    {
                        "status": "READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION",
                        "candidate_fingerprint": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run"),
                patch.object(
                    cycle,
                    "ensure_cost_shadow_validation",
                    side_effect=RuntimeError("shadow boom"),
                ),
            ):
                result = cycle.run_cycle()
            self.assertEqual(
                "COST_MODEL_SHADOW_VALIDATION_FAILED_SAFE_CASH",
                result["status"],
            )
            self.assertFalse(result["rotation_authority_allowed"])
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
            ):
                with self.assertRaisesRegex(RuntimeError, "shadow validation failed"):
                    cycle.assert_last_cycle_healthy()

    def test_health_check_fails_when_cost_authority_is_revoked(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.state / "cycle_status_latest.json").write_text(
                json.dumps({"status": "COST_MODEL_RECALIBRATION_REQUIRED"}),
                encoding="utf-8",
            )
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
            ):
                with self.assertRaisesRegex(RuntimeError, "cost authority"):
                    cycle.assert_last_cycle_healthy()

    def test_successful_cycle_promotes_then_republishes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            source_bundle = Path(directory) / "source-bundle"
            build_staged_bundle(source_bundle, data_dir=paths.data)

            def create_calibration(staging_dir, sample_step, workers):
                for name in cycle.CALIBRATION_FILES:
                    shutil.copy2(source_bundle / name, staging_dir / name)

            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run") as production,
                patch.object(cycle, "calibration_due", return_value=["DUE_TEST"]),
                patch.object(cycle, "_run_calibration", side_effect=create_calibration),
            ):
                result = cycle.run_cycle()
            self.assertEqual("CALIBRATION_PROMOTED", result["status"])
            self.assertEqual(2, production.call_count)
            self.assertTrue((paths.calibration / "calibration_bundle.json").exists())
            for name in cycle.CALIBRATION_FILES:
                self.assertTrue((paths.calibration / name).exists())

    def test_factor_health_trigger_runs_full_transactional_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            source_bundle = Path(directory) / "source-bundle"
            build_staged_bundle(source_bundle, data_dir=paths.data)

            def create_calibration(staging_dir, sample_step, workers):
                for name in cycle.CALIBRATION_FILES:
                    shutil.copy2(source_bundle / name, staging_dir / name)

            factor_reason = "FACTOR_LIVE_HEALTH_HARD_FAILURE:LIVE_ENSEMBLE_NEGATIVE_IC"
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run") as production,
                patch.object(cycle, "factor_health_recalibration_due", return_value=[factor_reason]),
                patch.object(cycle, "calibration_due", return_value=[]),
                patch.object(cycle, "_run_calibration", side_effect=create_calibration),
            ):
                result = cycle.run_cycle()
            self.assertEqual("CALIBRATION_PROMOTED", result["status"])
            self.assertIn(factor_reason, result["reasons"])
            self.assertEqual(2, production.call_count)

    def test_live_performance_degradation_triggers_full_transactional_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            source_bundle = Path(directory) / "source-bundle"
            build_staged_bundle(source_bundle, data_dir=paths.data)
            (paths.public / "live_performance_audit_latest.json").write_text(
                json.dumps(
                    {
                        "status": "LIVE_MODEL_RECALIBRATION_REQUIRED",
                        "recalibration_required": True,
                    }
                ),
                encoding="utf-8",
            )

            def create_calibration(staging_dir, sample_step, workers):
                for name in cycle.CALIBRATION_FILES:
                    shutil.copy2(source_bundle / name, staging_dir / name)

            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run") as production,
                patch.object(cycle, "factor_health_recalibration_due", return_value=[]),
                patch.object(cycle, "calibration_due", return_value=[]),
                patch.object(cycle, "_run_calibration", side_effect=create_calibration),
            ):
                result = cycle.run_cycle()
            self.assertEqual("CALIBRATION_PROMOTED", result["status"])
            self.assertIn(
                "LIVE_PERFORMANCE_RECALIBRATION_REQUIRED:LIVE_MODEL_RECALIBRATION_REQUIRED",
                result["reasons"],
            )
            self.assertEqual(2, production.call_count)

    def test_invalid_live_performance_evidence_keeps_cash_and_fails_health(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.public / "live_performance_audit_latest.json").write_text(
                json.dumps(
                    {
                        "status": "LIVE_PERFORMANCE_EVIDENCE_REJECTED",
                        "rotation_authority_allowed": False,
                        "recalibration_required": False,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run"),
                patch.object(cycle, "_run_calibration") as calibrate,
            ):
                result = cycle.run_cycle()
            self.assertEqual(
                "LIVE_PERFORMANCE_EVIDENCE_BLOCKED_SAFE_CASH",
                result["status"],
            )
            calibrate.assert_not_called()
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
            ):
                with self.assertRaisesRegex(RuntimeError, "performance evidence"):
                    cycle.assert_last_cycle_healthy()

    def test_invalid_execution_feedback_keeps_cash_and_fails_health(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = temporary_paths(Path(directory))
            paths.ensure()
            (paths.public / "execution_feedback_audit_latest.json").write_text(
                json.dumps(
                    {
                        "status": "FEEDBACK_REJECTED",
                        "rotation_authority_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
                patch.object(cycle, "_production_run"),
                patch.object(cycle, "calibration_due") as calibration_due,
                patch.object(cycle, "_run_calibration") as calibrate,
            ):
                result = cycle.run_cycle()
            self.assertEqual(
                "EXECUTION_FEEDBACK_EVIDENCE_BLOCKED_SAFE_CASH",
                result["status"],
            )
            calibration_due.assert_not_called()
            calibrate.assert_not_called()
            with (
                patch.object(cycle, "PATHS", paths),
                patch.object(cycle, "configure_runtime_paths"),
            ):
                with self.assertRaisesRegex(RuntimeError, "execution feedback"):
                    cycle.assert_last_cycle_healthy()


if __name__ == "__main__":
    unittest.main()
