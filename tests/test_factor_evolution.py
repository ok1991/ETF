import unittest
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

import etf_radar.factor_evolution as factor_evolution_module

from etf_radar.factor_evolution import (
    evaluate_expression,
    evolve_factor_registry,
    factor_metrics,
    industry_neutralise,
    primitive_factor_specs,
    sanitize_factor_registry,
    seeded_factor_specs,
)
from etf_radar.calibration.pipeline import (
    _apply_approved_adaptive_priority,
    _atomic_json,
    _rotation_acceptance_gates,
    _selected,
    choose_rotation_rank_buffer,
    calibration_data_fingerprint,
    select_thresholds,
    simulate_portfolio,
)
from etf_radar.signals.labeling import build_forward_label
from etf_radar.signals.contract import (
    fingerprint_joint_price_directory,
    fingerprint_joint_price_frames,
)
from etf_radar.rotation import (
    _exposure_ratio,
    _prepared_frames,
    _rebalance_sleeve,
    _rolling_rotation_stability,
    build_cash_rotation_target,
    load_rotation_state,
    score_rotation_candidates,
    select_rotation_targets,
    simulate_staggered_rotation,
    update_live_rotation_state,
)
from etf_radar.trading import TradingCostModel
from etf_radar.validation import WalkForwardConfig, expanding_walk_forward_splits


def labelled_panel(periods=45, assets=8):
    rng = np.random.default_rng(20260718)
    rows = []
    codes = [f"ETF{i}" for i in range(assets)]
    groups = ["growth" if index < assets // 2 else "value" for index in range(assets)]
    for date in pd.bdate_range("2022-01-03", periods=periods, freq="5B"):
        latent = rng.normal(size=assets)
        for index, code in enumerate(codes):
            momentum = latent[index] + rng.normal(scale=0.10)
            excess = 0.025 * momentum + rng.normal(scale=0.008)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry_group": groups[index],
                    "monthly_trend": rng.normal(),
                    "weekly_trend": rng.normal(),
                    "setup_score": rng.random(),
                    "relative_strength": momentum,
                    "risk_quality": rng.random(),
                    "market_score": rng.normal(),
                    "momentum_20": momentum,
                    "momentum_60": 0.8 * momentum + rng.normal(scale=0.15),
                    "reversal_5": rng.normal(),
                    "trend_efficiency_20": 0.8 * momentum + rng.normal(scale=0.15),
                    "volatility_20": abs(rng.normal(0.20, 0.04)),
                    "downside_volatility_60": abs(rng.normal(0.15, 0.03)),
                    "volume_confirmation": rng.normal(),
                    "liquidity_log": rng.normal(18.0, 0.5),
                    "excess_return_5d": 0.65 * excess + rng.normal(scale=0.004),
                    "excess_return_10d": excess,
                    "excess_return_20d": 0.45 * excess + rng.normal(scale=0.008),
                }
            )
    return pd.DataFrame(rows)


class FactorEvolutionTests(unittest.TestCase):
    def test_benjamini_hochberg_adjustment_controls_candidate_family(self):
        adjusted = factor_evolution_module._benjamini_hochberg_adjust(
            [0.001, 0.02, 0.20]
        )
        self.assertEqual([0.003, 0.03, 0.20], [round(value, 3) for value in adjusted])

    def test_short_history_never_falls_back_to_unpurged_holdouts(self):
        frame = labelled_panel(periods=10, assets=6)
        with self.assertRaises(factor_evolution_module.PurgedHoldoutInsufficientError):
            evolve_factor_registry(
                frame.to_dict("records"),
                population_size=8,
                generations=1,
                max_active=3,
            )

    def test_unapproved_challenger_cannot_retire_or_replace_incumbent(self):
        cleaned = sanitize_factor_registry(
            {
                "approved": False,
                "new_replacements": ["challenger"],
                "factors": [{"name": "challenger", "status": "ACTIVE"}],
                "retired_factors": [
                    {
                        "name": "incumbent",
                        "reasons": ["OUTPERFORMED_BY_REPLACEMENT"],
                    }
                ],
                "replacement_events": [
                    {
                        "new_factor": "challenger",
                        "replaces": "incumbent",
                        "approved_on_independent_holdout": False,
                    }
                ],
            }
        )
        self.assertEqual([], cleaned["new_replacements"])
        self.assertEqual(["challenger"], cleaned["research_challengers"])
        self.assertEqual([], cleaned["retired_factors"])
        self.assertEqual([], cleaned["replacement_events"])
        self.assertEqual("RESEARCH", cleaned["factors"][0]["status"])

    def test_unapproved_event_cannot_erase_prior_approved_retirement(self):
        cleaned = sanitize_factor_registry(
            {
                "approved": False,
                "trained_until": "2026-06-15",
                "factors": [],
                "retired_factors": [
                    {
                        "name": "volume_confirmed_trend",
                        "reasons": ["OUTPERFORMED_BY_REPLACEMENT"],
                    }
                ],
                "replacement_events": [
                    {
                        "event_type": "registry_v2_migration_snapshot",
                        "retired_factors": ["volume_confirmed_trend"],
                        "approved_on_independent_holdout": True,
                    },
                    {
                        "new_factor": "failed_challenger",
                        "replaces": "volume_confirmed_trend",
                        "approved_on_independent_holdout": False,
                    },
                ],
            }
        )
        self.assertEqual(
            ["volume_confirmed_trend"],
            [item["name"] for item in cleaned["retired_factors"]],
        )

    def test_rotation_authority_is_independent_from_adaptive_overlay_registry(self):
        portfolio = {
            "excess_return": 0.20,
            "information_ratio": 0.50,
            "max_drawdown": 0.15,
            "benchmark_max_drawdown": 0.30,
            "positive_year_ratio": 0.60,
            "rolling_12m_observations": 150,
            "rolling_12m_positive_excess_ratio": 0.65,
            "rolling_12m_worst_excess_return": -0.20,
            "max_relative_drawdown": 0.25,
            "longest_relative_underwater_periods": 200,
            "capacity_fill_ratio": 1.0,
            "year_checks": [{"year": year} for year in range(2019, 2025)],
        }
        holdout = {
            "excess_return": 0.10,
            "information_ratio": 0.60,
            "max_drawdown": 0.10,
            "positive_year_ratio": 0.50,
            "rolling_12m_observations": 40,
            "rolling_12m_positive_excess_ratio": 0.70,
            "rolling_12m_worst_excess_return": -0.10,
            "max_relative_drawdown": 0.20,
            "longest_relative_underwater_periods": 80,
            "capacity_fill_ratio": 1.0,
            "year_checks": [{"year": 2024}, {"year": 2025}],
        }
        gates = _rotation_acceptance_gates(portfolio, holdout)
        self.assertTrue(all(gates.values()))
        self.assertNotIn("factor_registry_oos", gates)

    def test_rotation_stability_gate_rejects_aggregate_winner_with_weak_rolling_excess(self):
        portfolio = {
            "excess_return": 0.30,
            "information_ratio": 0.60,
            "max_drawdown": 0.15,
            "benchmark_max_drawdown": 0.30,
            "positive_year_ratio": 0.80,
            "rolling_12m_observations": 150,
            "rolling_12m_positive_excess_ratio": 0.45,
            "rolling_12m_worst_excess_return": -0.30,
            "max_relative_drawdown": 0.35,
            "longest_relative_underwater_periods": 300,
            "capacity_fill_ratio": 1.0,
            "year_checks": [{"year": year} for year in range(2019, 2025)],
        }
        holdout = {
            "excess_return": 0.20,
            "information_ratio": 0.70,
            "max_drawdown": 0.10,
            "positive_year_ratio": 1.0,
            "rolling_12m_observations": 40,
            "rolling_12m_positive_excess_ratio": 0.80,
            "rolling_12m_worst_excess_return": -0.10,
            "max_relative_drawdown": 0.15,
            "longest_relative_underwater_periods": 60,
            "capacity_fill_ratio": 1.0,
            "year_checks": [{"year": 2024}, {"year": 2025}],
        }
        gates = _rotation_acceptance_gates(portfolio, holdout)
        self.assertFalse(gates["rolling_12m_positive_excess_ratio_min_0_60"])
        self.assertFalse(gates["rolling_12m_worst_excess_at_least_minus_0_25"])
        self.assertFalse(gates["max_relative_drawdown_at_most_0_30"])
        self.assertFalse(gates["relative_underwater_periods_at_most_260"])

    def test_rolling_rotation_stability_measures_persistent_outperformance(self):
        dates = pd.bdate_range("2020-01-03", periods=120, freq="5B")
        strategy = pd.Series(0.012, index=dates)
        benchmark = pd.Series(0.004, index=dates)
        metrics = _rolling_rotation_stability(strategy, benchmark, rolling_periods=52)
        self.assertEqual(69, metrics["rolling_12m_observations"])
        self.assertEqual(1.0, metrics["rolling_12m_positive_excess_ratio"])
        self.assertGreater(metrics["rolling_12m_median_excess_return"], 0.0)
        self.assertEqual(0.0, metrics["max_relative_drawdown"])
        self.assertEqual(0, metrics["longest_relative_underwater_periods"])

    def test_unapproved_adaptive_registry_cannot_change_base_priority(self):
        frame = pd.DataFrame(labelled_panel(periods=3, assets=4))
        frame["priority"] = np.linspace(10.0, 90.0, len(frame))
        original = frame["priority"].astype(float).tolist()
        scored = _apply_approved_adaptive_priority(
            frame,
            {"approved": False, "factors": [], "ensemble": {}},
        )
        self.assertEqual(original, scored["priority"].astype(float).tolist())
        self.assertFalse(bool(scored["adaptive_factor_applied"].any()))

    def test_joint_search_rejects_redundant_zero_value_factor_for_complement(self):
        rng = np.random.default_rng(20260719)
        rows = []
        for date in pd.bdate_range("2024-01-02", periods=30):
            for asset in range(8):
                first = rng.normal()
                complement = rng.normal()
                rows.append(
                    {
                        "date": date,
                        "code": f"ETF{asset}",
                        "industry_group": "g1" if asset < 4 else "g2",
                        "momentum_20": first,
                        "momentum_60": first + rng.normal(scale=0.001),
                        "reversal_5": complement,
                        "excess_return_10d": 0.03 * first + 0.03 * complement + rng.normal(scale=0.003),
                    }
                )
        frame = pd.DataFrame(rows)
        for name in factor_evolution_module.PRIMITIVE_FEATURES:
            if name not in frame.columns:
                frame[name] = 0.0
        dates = sorted(pd.to_datetime(frame["date"]).unique())
        train = frame[frame["date"] <= dates[19]].copy()
        selection = frame[frame["date"] > dates[19]].copy()
        evaluated = [
            {"name": "primary", "expression": {"feature": "momentum_20"}, "accepted": True},
            {"name": "redundant", "expression": {"feature": "momentum_60"}, "accepted": True},
            {"name": "complement", "expression": {"feature": "reversal_5"}, "accepted": True},
        ]
        selected, diagnostics = factor_evolution_module._select_complementary_factor_set(
            train,
            selection,
            evaluated,
            max_active=3,
        )
        names = {item["name"] for item in selected}
        self.assertIn("complement", names)
        self.assertFalse({"primary", "redundant"}.issubset(names))
        self.assertEqual("SELECTED", diagnostics["status"])
        self.assertFalse(diagnostics["approval_holdout_used"])
        self.assertEqual(
            len(selected),
            len(diagnostics["selected_ensemble"]["coefficients"]),
        )

    def test_joint_search_accepts_correlated_factors_only_with_incremental_stable_value(self):
        rng = np.random.default_rng(7)
        rows = []
        dates = pd.bdate_range("2023-01-02", periods=75)
        for date in dates:
            for asset in range(12):
                first = rng.normal()
                independent = rng.normal()
                second = 0.9 * first + 0.43589 * independent
                target = 0.03 * first + 0.03 * second + rng.normal(scale=0.004)
                rows.append(
                    {
                        "date": date,
                        "code": f"ETF{asset}",
                        "industry_group": "g1" if asset < 6 else "g2",
                        "momentum_20": first,
                        "trend_efficiency_20": second,
                        "excess_return_5d": target,
                        "excess_return_10d": target,
                        "excess_return_20d": target,
                    }
                )
        frame = pd.DataFrame(rows)
        for name in factor_evolution_module.PRIMITIVE_FEATURES:
            if name not in frame.columns:
                frame[name] = 0.0
        train = frame[frame["date"] <= dates[44]].copy()
        selection = frame[frame["date"] > dates[44]].copy()
        evaluated = []
        for name, feature in (
            ("first", "momentum_20"),
            ("second", "trend_efficiency_20"),
        ):
            evaluated.append(
                {
                    "name": name,
                    "expression": {"feature": feature},
                    "accepted": True,
                    "selection_metrics": factor_metrics(selection, selection[feature]),
                }
            )
        selected, diagnostics = factor_evolution_module._select_complementary_factor_set(
            train,
            selection,
            evaluated,
            max_active=2,
        )
        self.assertEqual({"first", "second"}, {item["name"] for item in selected})
        self.assertGreater(
            diagnostics["complementarity"]["max_pairwise_correlation"],
            0.85,
        )
        self.assertGreater(
            diagnostics["complementarity"]["incremental_ic_gain"],
            0.0005,
        )
        self.assertGreaterEqual(
            diagnostics["complementarity"]["selection_subperiod_stability"][
                "positive_block_ratio"
            ],
            0.67,
        )
        self.assertEqual(
            "train_only_frozen_before_selection",
            diagnostics["ensemble_fit_scope"],
        )
        self.assertIn("selected_ensemble", diagnostics)
        self.assertEqual(
            "benjamini_hochberg",
            diagnostics["discovery_control"]["method"],
        )
        self.assertLessEqual(
            diagnostics["discovery_control"]["selected_q_value"],
            factor_evolution_module.DISCOVERY_FDR_MAX,
        )
        self.assertEqual(
            len(selected),
            len(diagnostics["selected_ensemble"]["coefficients"]),
        )

    def test_two_of_three_positive_subperiods_pass_without_float_rounding_error(self):
        self.assertTrue(
            factor_evolution_module._passes_selection_subperiod_stability(
                {
                    "block_count": 3,
                    "positive_block_count": 2,
                    "positive_block_ratio": 0.666667,
                    "worst_block_ic": -0.019,
                }
            )
        )

    def test_changed_factor_policy_requires_new_unseen_dates_before_approval(self):
        frame = labelled_panel(periods=90, assets=8)
        registry = evolve_factor_registry(
            frame.to_dict("records"),
            previous_registry={"evolution_policy_version": "old-policy"},
            population_size=8,
            generations=1,
            max_active=3,
            require_policy_seasoning=True,
        )
        self.assertFalse(registry["policy_seasoned"])
        self.assertEqual(0, registry["policy_unseen_date_count"])
        self.assertIn("POLICY_SEASONING_INCOMPLETE", registry["approval_reasons"])

        registry["candidate_specification_fingerprint"] = ""
        registry["factors"] = []
        extended = labelled_panel(periods=110, assets=8)
        seasoned = evolve_factor_registry(
            extended.to_dict("records"),
            previous_registry=registry,
            population_size=8,
            generations=1,
            max_active=3,
            require_policy_seasoning=True,
        )
        self.assertEqual(
            registry["policy_seasoning_anchor"],
            seasoned["policy_seasoning_anchor"],
        )
        self.assertTrue(seasoned["policy_seasoned"])
        self.assertGreaterEqual(
            seasoned["policy_unseen_date_count"],
            seasoned["policy_seasoning_min_dates"],
        )

    def test_changed_candidate_specification_resets_policy_seasoning_anchor(self):
        frame = labelled_panel(periods=90, assets=8)
        previous = {
            "evolution_policy_version": factor_evolution_module.FACTOR_EVOLUTION_POLICY_VERSION,
            "policy_seasoning_anchor": "2020-01-01",
            "candidate_specification_fingerprint": "f" * 64,
        }
        registry = evolve_factor_registry(
            frame.to_dict("records"),
            previous_registry=previous,
            population_size=8,
            generations=1,
            max_active=3,
            require_policy_seasoning=True,
        )
        self.assertTrue(registry["policy_candidate_specification_changed"])
        self.assertEqual(registry["trained_until"], registry["policy_seasoning_anchor"])
        self.assertEqual(0, registry["policy_unseen_date_count"])
        self.assertFalse(registry["policy_seasoned"])
        self.assertIn(
            "POLICY_CANDIDATE_SPEC_CHANGED_RESET_SEASONING",
            registry["approval_reasons"],
        )
        self.assertFalse(
            registry["previous_candidate_specification_fingerprint_valid"]
        )
        self.assertIn(
            "PREVIOUS_CANDIDATE_FINGERPRINT_MISMATCH_RESET_SEASONING",
            registry["approval_reasons"],
        )

    def test_legacy_registry_without_stored_fingerprint_uses_computed_identity(self):
        expression = {"feature": "momentum_20"}
        previous = {
            "evolution_policy_version": factor_evolution_module.FACTOR_EVOLUTION_POLICY_VERSION,
            "policy_seasoning_anchor": "2020-01-01",
            "factors": [{"name": "legacy", "expression": expression}],
        }
        expected = factor_evolution_module._factor_specification_fingerprint(
            previous["factors"]
        )
        registry = evolve_factor_registry(
            labelled_panel(periods=90, assets=8).to_dict("records"),
            previous_registry=previous,
            population_size=8,
            generations=1,
            max_active=3,
            require_policy_seasoning=True,
        )
        self.assertTrue(
            registry["previous_candidate_specification_fingerprint_valid"]
        )
        self.assertEqual(
            expected, registry["previous_candidate_specification_fingerprint"]
        )
        self.assertNotIn(
            "PREVIOUS_CANDIDATE_FINGERPRINT_MISMATCH_RESET_SEASONING",
            registry["approval_reasons"],
        )

    def test_factor_specification_fingerprint_ignores_names_but_not_expressions(self):
        momentum = {"feature": "momentum_20"}
        trend = {"feature": "trend_efficiency_20"}
        first = factor_evolution_module._factor_specification_fingerprint(
            [{"name": "old-name", "expression": momentum}]
        )
        renamed = factor_evolution_module._factor_specification_fingerprint(
            [{"name": "new-name", "expression": momentum}]
        )
        changed = factor_evolution_module._factor_specification_fingerprint(
            [{"name": "old-name", "expression": trend}]
        )
        sign_reversed = factor_evolution_module._factor_specification_fingerprint(
            [
                {
                    "name": "old-name",
                    "expression": {"op": "neg", "args": [momentum]},
                }
            ]
        )
        self.assertEqual(first, renamed)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, sign_reversed)

    def test_industry_neutral_scores_have_zero_group_mean(self):
        frame = labelled_panel(periods=1)
        neutral = industry_neutralise(frame, frame["momentum_20"])
        means = neutral.groupby(frame["industry_group"]).mean()
        self.assertTrue((means.abs() < 1e-10).all())

    def test_monitor_reports_ic_ir_decay_and_turnover(self):
        frame = labelled_panel()
        metrics = factor_metrics(frame, frame["momentum_20"])
        self.assertGreater(metrics["ic_mean"], 0.25)
        self.assertGreater(metrics["ic_ir"], 0.0)
        self.assertGreater(metrics["ic_t_stat"], 0.0)
        self.assertLessEqual(metrics["ic_p_value"], 0.10)
        self.assertGreater(metrics["ic_positive_ratio"], 0.50)
        self.assertIn("5", metrics["ic_decay"])
        self.assertIn("20", metrics["ic_decay"])
        self.assertGreaterEqual(metrics["turnover"], 0.0)
        self.assertLessEqual(metrics["turnover"], 1.0)

    def test_transparent_primitive_challengers_are_always_available(self):
        challengers = {item["name"]: item for item in primitive_factor_specs()}
        self.assertIn("primitive_relative_strength", challengers)
        self.assertIn("primitive_momentum_20", challengers)
        self.assertTrue(all(item["candidate_origin"] == "primitive_challenger" for item in challengers.values()))

    def test_llm_candidate_enters_research_pool_without_bypassing_approval(self):
        frame = labelled_panel(periods=70, assets=10)
        registry = evolve_factor_registry(
            frame.to_dict("records"),
            llm_candidates=[
                {
                    "name": "llm_test_candidate",
                    "expression": {
                        "op": "max",
                        "args": [
                            {"feature": "relative_strength"},
                            {"feature": "liquidity_log"},
                        ],
                    },
                    "economic_logic": "测试LLM候选只能进入研究池，不能直接获得批准。",
                    "candidate_origin": "llm_structured_proposal",
                    "proposal_metadata": {"model": "test-model"},
                }
            ],
            population_size=8,
            generations=1,
            max_active=3,
        )
        self.assertGreaterEqual(registry["llm_proposals_considered"], 1)
        self.assertEqual(1, registry["llm_proposals_submitted"])
        self.assertEqual([], registry["llm_proposals_skipped_rejected_cooldown"])
        trial = next(
            item
            for item in registry["llm_candidate_trial_history"]
            if item["name"] == "llm_test_candidate"
        )
        self.assertEqual(1, trial["trial_count"])
        self.assertIn(
            trial["outcome"],
            {"SELECTION_ACCEPTED", "SELECTION_REJECTED"},
        )
        self.assertIn("llm_structured_proposal", registry["candidate_origins"])
        if not registry["approved"]:
            self.assertEqual([], registry["llm_proposals_selected"])
            self.assertTrue(
                set(registry["llm_research_challengers"]).issubset(
                    set(registry["research_challengers"])
                )
            )

    def test_rejected_llm_expression_is_skipped_during_cooldown(self):
        expression = {
            "op": "max",
            "args": [
                {"feature": "relative_strength"},
                {"feature": "liquidity_log"},
            ],
        }
        family_key = factor_evolution_module._expression_family_key(expression)
        previous = {
            "approved": False,
            "factors": [],
            "llm_candidate_trial_history": [
                {
                    "name": "prior_rejected_llm",
                    "expression_family_key": family_key,
                    "outcome": "SELECTION_REJECTED",
                    "cooldown_until": "2099-12-31",
                    "trial_count": 2,
                }
            ],
        }
        registry = evolve_factor_registry(
            labelled_panel(periods=70, assets=10).to_dict("records"),
            previous_registry=previous,
            llm_candidates=[
                {
                    "name": "repeated_llm_candidate",
                    "expression": expression,
                    "economic_logic": "A repeated LLM research hypothesis must wait for its cooldown.",
                    "candidate_origin": "llm_structured_proposal",
                }
            ],
            population_size=8,
            generations=1,
            max_active=3,
        )
        self.assertEqual(1, registry["llm_proposals_submitted"])
        self.assertEqual(0, registry["llm_proposals_considered"])
        self.assertEqual(
            "LLM_REJECTED_EXPRESSION_COOLDOWN",
            registry["llm_proposals_skipped_rejected_cooldown"][0]["reason"],
        )
        self.assertEqual(
            2,
            registry["llm_candidate_trial_history"][0]["trial_count"],
        )

    def test_rejected_llm_cooldown_expires_on_newer_training_date(self):
        history = [
            {
                "expression_family_key": "active",
                "outcome": "SELECTION_REJECTED",
                "cooldown_until": "2026-07-01",
            },
            {
                "expression_family_key": "expired",
                "outcome": "SELECTION_REJECTED",
                "cooldown_until": "2026-05-01",
            },
            {
                "expression_family_key": "passed",
                "outcome": "SELECTION_ACCEPTED",
                "cooldown_until": "2099-12-31",
            },
        ]
        active = factor_evolution_module._llm_rejected_cooldown_keys(
            history,
            "2026-06-15",
        )
        self.assertEqual({"active"}, set(active))

    def test_unapproved_registry_cannot_dampen_base_priority(self):
        frame = labelled_panel(periods=2, assets=4)
        frame["priority"] = [20.0, 40.0, 60.0, 80.0] * 2
        scored = _apply_approved_adaptive_priority(
            frame,
            {"approved": False, "factors": []},
        )
        self.assertEqual(
            list(frame["priority"].astype(float)),
            list(scored["priority"].astype(float)),
        )
        self.assertFalse(scored["adaptive_factor_applied"].any())

    def test_gp_retires_failed_incumbent_and_keeps_economic_logic(self):
        frame = labelled_panel()
        previous = {
            "approved": True,
            "factors": [
                {
                    "name": "failed_negative_momentum",
                    "status": "ACTIVE",
                    "expression": {"op": "neg", "args": [{"feature": "momentum_20"}]},
                    "economic_logic": "故意构造的失效反向动量。",
                }
            ]
        }
        registry = evolve_factor_registry(
            frame.to_dict("records"),
            previous_registry=previous,
            population_size=8,
            generations=1,
            max_active=3,
        )
        retired = {item["name"] for item in registry["retired_factors"]}
        self.assertIn("failed_negative_momentum", retired)
        self.assertEqual(2, registry["schema_version"])
        self.assertEqual(
            "train_70_selection_15_approval_15_with_purged_boundaries",
            registry["validation_method"],
        )
        self.assertEqual(
            "28_calendar_day_approx_20_trading_day_purge",
            registry["purge_method"],
        )
        self.assertGreater(registry["selection_rows"], 0)
        self.assertGreater(registry["approval_rows"], 0)
        self.assertIn("ensemble_approval_metrics", registry)
        self.assertTrue(registry["factors"])
        self.assertEqual(
            "benjamini_hochberg",
            registry["candidate_discovery_control"]["method"],
        )
        self.assertEqual(
            registry["candidate_count"],
            registry["candidate_discovery_control"]["family_size"],
        )
        self.assertTrue(
            all(
                float(item["selection_metrics"]["multiple_testing_q_value"])
                <= factor_evolution_module.DISCOVERY_FDR_MAX
                for item in registry["candidate_diagnostics"]
                if item["accepted"]
            )
        )
        self.assertTrue(all(item["economic_logic"] for item in registry["factors"]))
        for item in registry["factors"]:
            values = evaluate_expression(item["expression"], frame)
            self.assertEqual(len(values), len(frame))

    def test_independent_approval_holdout_can_veto_selected_factors(self):
        frame = labelled_panel(periods=80, assets=10)
        dates = sorted(pd.to_datetime(frame["date"]).unique())
        approval_start = dates[int(len(dates) * 0.85)]
        approval_mask = frame["date"] >= approval_start
        for target in ("excess_return_5d", "excess_return_10d", "excess_return_20d"):
            frame.loc[approval_mask, target] *= -1.0
        candidates = [
            {
                "name": "momentum_candidate",
                "expression": {"feature": "momentum_20"},
                "economic_logic": "测试动量。",
                "generation": 0,
            },
            {
                "name": "trend_candidate",
                "expression": {"feature": "trend_efficiency_20"},
                "economic_logic": "测试趋势效率。",
                "generation": 0,
            },
        ]
        with patch("etf_radar.factor_evolution.genetic_candidates", return_value=candidates), patch(
            "etf_radar.factor_evolution.seeded_factor_specs", return_value=[]
        ):
            registry = evolve_factor_registry(
                frame.to_dict("records"),
                population_size=8,
                generations=1,
                max_active=2,
            )
        self.assertFalse(registry["approved"])
        self.assertIn(
            "INDEPENDENT_ENSEMBLE_HOLDOUT_GATE_FAILED",
            registry["approval_reasons"],
        )

    def test_zero_weight_factor_cannot_satisfy_two_factor_approval_gate(self):
        def one_effective_factor(_frame, matrix, **_kwargs):
            count = len(matrix.columns)
            return {
                "intercept": 0.0,
                "coefficients": [1.0] + [0.0] * max(0, count - 1),
                "feature_mean": [0.0] * count,
                "feature_scale": [1.0] * count,
                "training_half_life_dates": 52.0,
                "non_negative_coefficients": True,
            }

        with patch("etf_radar.factor_evolution._ridge_ensemble", side_effect=one_effective_factor):
            value = evolve_factor_registry(
                labelled_panel(periods=90, assets=8).to_dict("records"),
                seed=9,
                max_active=3,
                population_size=12,
                generations=1,
            )
        self.assertFalse(value["approved"])
        self.assertLess(value["effective_factor_count"], 2)
        self.assertIn("EFFECTIVE_FACTOR_COUNT_BELOW_2", value["approval_reasons"])
        self.assertEqual(
            {item["name"] for item in value["factors"]},
            set(value["research_challengers"]),
        )

    def test_retired_expression_cannot_reenter_during_cooldown(self):
        frame = labelled_panel(periods=70, assets=10)
        blocked = seeded_factor_specs()[0]
        expression_key = hashlib.sha1(
            json.dumps(blocked["expression"], sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        previous = {
            "factors": [],
            "retired_factors": [
                {
                    "name": blocked["name"],
                    "expression": blocked["expression"],
                    "expression_key": expression_key,
                    "cooldown_until": "2099-12-31",
                    "reasons": ["TEST_COOLDOWN"],
                }
            ],
        }
        registry = evolve_factor_registry(
            frame.to_dict("records"),
            previous_registry=previous,
            population_size=8,
            generations=1,
            max_active=3,
        )
        blocked_family = factor_evolution_module._expression_family_key(blocked["expression"])
        selected_families = {
            factor_evolution_module._expression_family_key(item["expression"])
            for item in registry["factors"]
        }
        self.assertNotIn(blocked_family, selected_families)
        self.assertNotIn(
            factor_evolution_module._expression_family_key(
                {"op": "neg", "args": [blocked["expression"]]}
            ),
            selected_families,
        )

    def test_replacement_history_survives_next_registry_iteration(self):
        frame = labelled_panel(periods=70, assets=10)
        prior_event = {
            "new_factor": "prior_alpha",
            "replaces": "old_alpha",
            "approved_on_independent_holdout": True,
            "event_date": "2025-01-01",
        }
        previous = {
            "approved": False,
            "factors": [],
            "retired_factors": [],
            "replacement_events": [prior_event],
        }
        registry = evolve_factor_registry(
            frame.to_dict("records"),
            previous_registry=previous,
            population_size=8,
            generations=1,
            max_active=3,
        )
        self.assertIn(prior_event, registry["replacement_events"])

    def test_walk_forward_is_purged_and_calendar_driven(self):
        dates = pd.bdate_range("2020-01-01", "2025-12-31")
        config = WalkForwardConfig(train_months=18, validation_months=4, step_months=4)
        splits = expanding_walk_forward_splits(dates, config)
        self.assertGreater(len(splits), 3)
        for train_end, validate_start, validate_end, next_start in splits:
            self.assertLess(train_end, validate_start)
            self.assertGreaterEqual((validate_start - train_end).days, 20)
            self.assertGreater(validate_end, validate_start)
            self.assertGreater(next_start, validate_start)

    def test_realistic_cost_model_uses_1_5_bps_without_minimum_and_charges_impact(self):
        model = TradingCostModel()
        small = model.estimate("BUY", price=1.0, shares=100, average_daily_amount=10_000_000)
        large = model.estimate("BUY", price=1.0, shares=100_000, average_daily_amount=1_000_000)
        self.assertEqual(0.00015, model.commission_rate)
        self.assertEqual(0.0, model.minimum_commission)
        expected_small_fees = 100.0 * (model.commission_rate + model.exchange_handling_rate)
        self.assertAlmostEqual(expected_small_fees, small["fees"], places=10)
        self.assertLess(small["fees"], 5.0)
        self.assertGreater(large["slippage"], small["slippage"])
        self.assertLess(small["cash_delta"], 0.0)
        self.assertEqual(100_000, model.capacity_lot(price=1.0, average_daily_amount=1_000_000))
        oversized = model.estimate(
            "BUY", price=1.0, shares=200_000, average_daily_amount=1_000_000
        )
        self.assertTrue(oversized["capacity_exceeded"])
        self.assertAlmostEqual(0.20, oversized["requested_participation_rate"])
        self.assertAlmostEqual(0.10, oversized["participation_rate"])

    def test_labels_and_portfolio_include_cost_after_benchmark_metrics(self):
        dates = pd.bdate_range("2026-01-01", periods=35)

        def price_frame(start, daily_return):
            close = start * np.cumprod(np.full(len(dates), 1.0 + daily_return))
            return pd.DataFrame(
                {
                    "date": dates,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.full(len(dates), 5_000_000.0),
                    "amount": close * 5_000_000.0,
                }
            )

        etf = price_frame(1.0, 0.002)
        benchmark = price_frame(1.0, 0.0005)
        label = build_forward_label(
            etf,
            benchmark,
            signal_date=dates[5],
            stop_price=float(etf.iloc[5]["close"] * 0.92),
            cost_model=TradingCostModel(),
        )
        self.assertIn("excess_return_5d", label)
        self.assertIn("excess_return_20d", label)
        self.assertGreater(label["estimated_round_trip_cost"], 0.0)

        rows = pd.DataFrame(
            [
                {
                    "entry_date": dates[6],
                    "code": "ETF1",
                    "name": "测试行业ETF",
                    "industry_group": "growth",
                    "priority": 90.0,
                    "stop_loss": float(etf.iloc[5]["close"] * 0.90),
                }
            ]
        )
        metrics = simulate_portfolio(
            rows,
            {"ETF1": etf, "510300": benchmark},
            initial_capital=100_000.0,
        )
        self.assertGreater(metrics["total_cost"], 0.0)
        self.assertIsNotNone(metrics["benchmark_return"])
        self.assertIsNotNone(metrics["information_ratio"])
        self.assertIn("industry_constraint", metrics)

    def test_threshold_selection_marks_candidate_approved(self):
        predictions = []
        for index in range(40):
            selected = index < 20
            predictions.append(
                {
                    "baseline_candidate": True,
                    "priority": 70.0 if selected else 55.0,
                    "setup_score_raw": 70.0 if selected else 50.0,
                    "risk_executable": True,
                    "early_stop_probability_3d": 0.10 if selected else 0.50,
                    "expected_excess_return_10d": 0.02 if selected else -0.01,
                    "early_stop": 0 if selected else 1,
                    "excess_return_10d": 0.02 if selected else -0.01,
                }
            )
        thresholds = select_thresholds(predictions)
        self.assertTrue(thresholds["approved"])
        chosen = _selected(pd.DataFrame(predictions), thresholds)
        self.assertEqual(len(chosen), 20)

    def test_atomic_json_serialises_pandas_and_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            _atomic_json(
                {
                    "timestamp": pd.Timestamp("2026-07-18"),
                    "integer": np.int64(3),
                    "array": np.asarray([1.0, 2.0]),
                },
                str(path),
            )
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(3, value["integer"])
            self.assertEqual([1.0, 2.0], value["array"])

    def test_calibration_fingerprint_binds_qfq_and_raw_price_bases(self):
        dates = pd.bdate_range("2026-01-05", periods=10)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": np.linspace(1.0, 1.1, len(dates)),
                "high": np.linspace(1.01, 1.11, len(dates)),
                "low": np.linspace(0.99, 1.09, len(dates)),
                "close": np.linspace(1.0, 1.1, len(dates)),
                "volume": np.full(len(dates), 1_000_000.0),
            }
        )
        cutoff = dates[-1].strftime("%Y-%m-%d")
        base = calibration_data_fingerprint({"A": frame}, {"A": frame}, cutoff)
        self.assertEqual(
            base,
            fingerprint_joint_price_frames({"A": frame}, {"A": frame}, cutoff),
        )
        raw_changed = frame.copy()
        raw_changed.loc[raw_changed.index[-2], "close"] *= 1.01
        qfq_changed = frame.copy()
        qfq_changed.loc[qfq_changed.index[-3], "open"] *= 1.01
        self.assertNotEqual(
            base,
            calibration_data_fingerprint({"A": frame}, {"A": raw_changed}, cutoff),
        )
        self.assertNotEqual(
            base,
            calibration_data_fingerprint({"A": qfq_changed}, {"A": frame}, cutoff),
        )
        with tempfile.TemporaryDirectory() as directory:
            frame.to_csv(Path(directory) / "A_20260116.csv", index=False)
            frame.to_csv(Path(directory) / "A_raw_20260116.csv", index=False)
            directory_base = fingerprint_joint_price_directory(
                directory, cutoff, minimum_rows=1
            )
            self.assertEqual(base, directory_base)
            raw_changed.to_csv(
                Path(directory) / "A_raw_20260116.csv", index=False
            )
            self.assertNotEqual(
                directory_base,
                fingerprint_joint_price_directory(
                    directory, cutoff, minimum_rows=1
                ),
            )

    def test_staggered_rotation_outperforms_synthetic_benchmark_after_costs(self):
        dates = pd.bdate_range("2023-01-02", periods=180)

        def prices(daily_return):
            close = np.cumprod(np.full(len(dates), 1.0 + daily_return))
            return pd.DataFrame(
                {
                    "date": dates,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.full(len(dates), 10_000_000.0),
                    "amount": close * 10_000_000.0,
                }
            )

        rows = []
        for date in dates[20:-10:5]:
            for code, group, strength in (
                ("A", "growth", 0.95),
                ("B", "value", 0.80),
                ("C", "growth", 0.20),
            ):
                rows.append(
                    {
                        "entry_date": date,
                        "date": date,
                        "code": code,
                        "industry_group": group,
                        "weekly_trend": 0.5,
                        "relative_strength": strength,
                        "trend_efficiency_20": strength,
                        "volume_confirmation": strength,
                        "priority": strength * 100.0,
                        "max_exposure_ratio": 1.0,
                    }
                )
        metrics = simulate_staggered_rotation(
            pd.DataFrame(rows),
            {
                "A": prices(0.0030),
                "B": prices(0.0015),
                "C": prices(-0.0010),
                "510300": prices(0.0005),
            },
            top_n=2,
            initial_capital=200_000.0,
        )
        self.assertGreater(metrics["excess_return"], 0.0)
        self.assertGreater(metrics["information_ratio"], 0.25)
        self.assertGreater(metrics["total_cost"], 0.0)
        self.assertLess(metrics["max_drawdown"], metrics["benchmark_max_drawdown"] + 0.10)
        self.assertEqual(0, metrics["capacity_truncation_count"])
        self.assertEqual(1.0, metrics["capacity_fill_ratio"])
        self.assertLessEqual(metrics["executed_buy_value"], metrics["requested_buy_value"])

        illiquid_frames = {
            code: frame.assign(amount=np.full(len(frame), 100_000.0))
            for code, frame in {
                "A": prices(0.0030),
                "B": prices(0.0015),
                "C": prices(-0.0010),
                "510300": prices(0.0005),
            }.items()
        }
        constrained = simulate_staggered_rotation(
            pd.DataFrame(rows),
            illiquid_frames,
            top_n=2,
            initial_capital=10_000_000.0,
        )
        self.assertEqual(0, constrained["capacity_truncation_count"])
        self.assertEqual(0.0, constrained["requested_buy_value"])
        self.assertEqual(1.0, constrained["capacity_fill_ratio"])

    def test_rotation_exposure_authority_is_complete_and_consistent_per_date(self):
        with self.assertRaisesRegex(ValueError, "required"):
            _exposure_ratio(pd.DataFrame({"code": ["A", "B"]}))
        with self.assertRaisesRegex(ValueError, "not numeric"):
            _exposure_ratio(
                pd.DataFrame({"max_exposure_ratio": [0.5, "missing"]})
            )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _exposure_ratio(pd.DataFrame({"max_exposure_ratio": [0.5, 1.0]}))
        self.assertEqual(
            0.5,
            _exposure_ratio(pd.DataFrame({"max_exposure_ratio": [0.5, 0.5]})),
        )

    def test_daily_capacity_is_shared_across_both_sleeves(self):
        dates = pd.bdate_range("2026-01-05", periods=20)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": np.ones(len(dates)),
                "high": np.ones(len(dates)),
                "low": np.ones(len(dates)),
                "close": np.ones(len(dates)),
                "volume": np.full(len(dates), 1_000_000.0),
                "amount": np.full(len(dates), 1_000_000.0),
            }
        )
        frames = _prepared_frames({"A": frame})
        usage = {}
        first = {"cash": 500_000.0, "positions": {}}
        second = {"cash": 500_000.0, "positions": {}}
        _, _, first_audit = _rebalance_sleeve(
            first, ["A"], frames, dates[-1], TradingCostModel(), 1.0, usage
        )
        _, _, second_audit = _rebalance_sleeve(
            second, ["A"], frames, dates[-1], TradingCostModel(), 1.0, usage
        )
        self.assertEqual(100_000, first["positions"]["A"])
        self.assertEqual({}, second["positions"])
        self.assertEqual(100_000, usage["A"])
        self.assertEqual(1, int(first_audit["capacity_truncation_count"]))
        self.assertEqual(1, int(second_audit["capacity_truncation_count"]))
        self.assertLessEqual(first_audit["max_executed_participation_rate"], 0.10)

    def test_live_rotation_updates_only_one_sleeve_per_week(self):
        frame = pd.DataFrame(
            [
                {"date": "2026-07-17", "code": "A", "industry_group": "g1", "weekly_trend": 0.5, "relative_strength": 0.9, "trend_efficiency_20": 0.8, "volume_confirmation": 0.7, "priority": 90},
                {"date": "2026-07-17", "code": "B", "industry_group": "g2", "weekly_trend": 0.5, "relative_strength": 0.8, "trend_efficiency_20": 0.7, "volume_confirmation": 0.6, "priority": 80},
                {"date": "2026-07-17", "code": "C", "industry_group": "g3", "weekly_trend": 0.5, "relative_strength": 0.7, "trend_efficiency_20": 0.6, "volume_confirmation": 0.5, "priority": 70},
            ]
        )
        frame["average_daily_amount_20"] = 10_000_000.0
        scored = score_rotation_candidates(frame)
        self.assertEqual(3, len(select_rotation_targets(scored, top_n=3)))
        first = update_live_rotation_state(
            frame,
            None,
            "2026-07-17",
            market_policy={"state": "NORMAL", "entry_permission": "TRADEABLE", "max_exposure_ratio": 1.0},
            execution_date="2026-07-20",
        )
        second = update_live_rotation_state(
            frame,
            first,
            "2026-07-18",
            market_policy={"state": "NORMAL", "entry_permission": "TRADEABLE", "max_exposure_ratio": 1.0},
            execution_date="2026-07-20",
        )
        self.assertEqual(first["sleeves"], second["sleeves"])
        self.assertAlmostEqual(1.0, sum(second["target_weights"].values()), places=5)
        self.assertEqual("2026-07-20", second["execution_date"])

    def test_saved_rotation_v2_state_is_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation_state.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "data_date": "2026-07-17",
                        "execution_date": "2026-07-20",
                        "sleeves": [["A"], ["B"]],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_rotation_state(str(path))
        self.assertIsNotNone(loaded)
        self.assertEqual([["A"], ["B"]], loaded["sleeves"])

    def test_live_rotation_resets_same_week_state_when_model_authority_changes(self):
        first_frame = pd.DataFrame(
            [
                {"date": "2026-07-17", "code": "A", "industry_group": "g1", "weekly_trend": 0.5, "relative_strength": 0.9, "trend_efficiency_20": 0.9, "volume_confirmation": 0.9, "priority": 90, "average_daily_amount_20": 10_000_000.0},
                {"date": "2026-07-17", "code": "B", "industry_group": "g2", "weekly_trend": 0.5, "relative_strength": 0.8, "trend_efficiency_20": 0.8, "volume_confirmation": 0.8, "priority": 80, "average_daily_amount_20": 10_000_000.0},
            ]
        )
        first = update_live_rotation_state(
            first_frame,
            None,
            "2026-07-17",
            top_n=2,
            market_policy={"state": "NORMAL", "entry_permission": "TRADEABLE", "max_exposure_ratio": 1.0},
            model_authority={
                "model_version": "rotation-old-aaaaaaaa",
                "execution_policy_version": "single-exposure-authority-v4",
                "acceptance_policy_version": "rolling-excess-stability-v0",
                "strategy_specification_fingerprint": "a" * 64,
            },
        )
        second_frame = pd.DataFrame(
            [
                {"date": "2026-07-18", "code": "C", "industry_group": "g3", "weekly_trend": 0.5, "relative_strength": 0.95, "trend_efficiency_20": 0.95, "volume_confirmation": 0.95, "priority": 95, "average_daily_amount_20": 10_000_000.0},
                {"date": "2026-07-18", "code": "D", "industry_group": "g4", "weekly_trend": 0.5, "relative_strength": 0.85, "trend_efficiency_20": 0.85, "volume_confirmation": 0.85, "priority": 85, "average_daily_amount_20": 10_000_000.0},
            ]
        )
        second = update_live_rotation_state(
            second_frame,
            first,
            "2026-07-18",
            top_n=2,
            market_policy={"state": "NORMAL", "entry_permission": "TRADEABLE", "max_exposure_ratio": 1.0},
            model_authority={
                "model_version": "rotation-new-bbbbbbbb",
                "execution_policy_version": "single-exposure-authority-v4",
                "acceptance_policy_version": "rolling-excess-stability-v1",
                "strategy_specification_fingerprint": "b" * 64,
            },
        )
        self.assertEqual([["C", "D"], ["C", "D"]], second["sleeves"])
        self.assertEqual("MODEL_AUTHORITY_CHANGED", second["state_reset_reason"])
        self.assertIn("model_version", second["state_reset_fields"])
        self.assertIn("acceptance_policy_version", second["state_reset_fields"])
        self.assertIn("strategy_specification_fingerprint", second["state_reset_fields"])

    def test_rotation_score_prefers_absolute_momentum_over_defensive_relative_strength(self):
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-09-30",
                    "code": "GOLD",
                    "industry_group": "precious_metals",
                    "weekly_trend": -0.05,
                    "relative_strength": 0.95,
                    "momentum_20": -0.02,
                    "trend_efficiency_20": 0.10,
                    "volume_confirmation": 0.10,
                    "priority": 70,
                    "market_score": 0.15,
                },
                {
                    "date": "2024-09-30",
                    "code": "TECH",
                    "industry_group": "technology",
                    "weekly_trend": 0.35,
                    "relative_strength": 0.55,
                    "momentum_20": 0.12,
                    "trend_efficiency_20": 0.40,
                    "volume_confirmation": 0.30,
                    "priority": 75,
                    "market_score": 0.15,
                },
                {
                    "date": "2024-09-30",
                    "code": "BANK",
                    "industry_group": "financials",
                    "weekly_trend": 0.05,
                    "relative_strength": 0.80,
                    "momentum_20": 0.01,
                    "trend_efficiency_20": 0.15,
                    "volume_confirmation": 0.15,
                    "priority": 65,
                    "market_score": 0.15,
                },
            ]
        )
        scored = score_rotation_candidates(frame)
        ordered = scored.sort_values("rotation_score", ascending=False)["code"].tolist()
        self.assertEqual("TECH", ordered[0])
        self.assertEqual({"recovery"}, set(scored["rotation_regime"]))
        selected = select_rotation_targets(scored, top_n=2)
        self.assertEqual(["TECH", "BANK"], [row["code"] for row in selected])
        # Gold stays out when weekly trend is negative in recovery regime.
        self.assertNotIn("GOLD", [row["code"] for row in selected])

    def test_rank_buffer_retains_incumbent_until_it_falls_outside_buffer(self):

        frame = pd.DataFrame(
            [
                {"date": "2026-07-17", "code": "A", "industry_group": "g1", "weekly_trend": 0.5, "rotation_score": 1.00},
                {"date": "2026-07-17", "code": "B", "industry_group": "g2", "weekly_trend": 0.5, "rotation_score": 0.90},
                {"date": "2026-07-17", "code": "C", "industry_group": "g3", "weekly_trend": 0.5, "rotation_score": 0.80},
                {"date": "2026-07-17", "code": "D", "industry_group": "g4", "weekly_trend": 0.5, "rotation_score": 0.70},
            ]
        )
        kept = select_rotation_targets(
            frame,
            top_n=2,
            incumbent_codes=["A", "C"],
            rank_buffer=1,
        )
        self.assertEqual(["A", "C"], [row["code"] for row in kept])
        replaced = select_rotation_targets(
            frame,
            top_n=2,
            incumbent_codes=["A", "D"],
            rank_buffer=1,
        )
        self.assertEqual(["A", "B"], [row["code"] for row in replaced])

    def test_rank_buffer_selection_uses_development_ir_not_holdout(self):
        selected = choose_rotation_rank_buffer(
            {
                0: {"excess_return": 0.10, "information_ratio": 0.10, "max_drawdown": 0.20, "turnover": 100},
                1: {"excess_return": 0.08, "information_ratio": 0.08, "max_drawdown": 0.18, "turnover": 80},
                2: {"excess_return": 0.18, "information_ratio": 0.19, "max_drawdown": 0.19, "turnover": 66},
                3: {"excess_return": 0.09, "information_ratio": 0.11, "max_drawdown": 0.20, "turnover": 60},
            }
        )
        self.assertEqual(2, selected)

    def test_live_rotation_scales_targets_to_market_risk_budget(self):
        frame = pd.DataFrame(
            [
                {"date": "2026-07-17", "code": "A", "industry_group": "g1", "weekly_trend": 0.5, "relative_strength": 0.9, "trend_efficiency_20": 0.8, "volume_confirmation": 0.7, "priority": 90},
                {"date": "2026-07-17", "code": "B", "industry_group": "g2", "weekly_trend": 0.5, "relative_strength": 0.8, "trend_efficiency_20": 0.7, "volume_confirmation": 0.6, "priority": 80},
            ]
        )
        frame["average_daily_amount_20"] = 10_000_000.0
        defensive = update_live_rotation_state(
            frame,
            None,
            "2026-07-17",
            top_n=2,
            market_policy={"state": "DEFENSIVE", "entry_permission": "MAINLINE_ONLY", "max_exposure_ratio": 0.5},
        )
        self.assertEqual(2, defensive["schema_version"])
        self.assertAlmostEqual(0.5, sum(defensive["target_weights"].values()), places=5)
        self.assertAlmostEqual(0.5, defensive["cash_weight"], places=5)

        risk_off = update_live_rotation_state(
            frame,
            defensive,
            "2026-07-18",
            top_n=2,
            market_policy={"state": "RISK_OFF", "entry_permission": "BLOCKED", "max_exposure_ratio": 0.0},
        )
        self.assertEqual({}, risk_off["target_weights"])
        self.assertEqual(0.0, risk_off["market_policy"]["max_exposure_ratio"])
        self.assertEqual("v4_market_policy", risk_off["exposure_authority"])

        blocked = update_live_rotation_state(
            frame,
            defensive,
            "2026-07-18",
            top_n=2,
            market_policy={"state": "RISK_OFF", "entry_permission": "BLOCKED", "max_exposure_ratio": 0.0},
        )
        self.assertEqual({}, blocked["target_weights"])
        self.assertEqual(1.0, blocked["cash_weight"])

        with self.assertRaisesRegex(ValueError, "required exposure authority"):
            update_live_rotation_state(frame, defensive, "2026-07-18", top_n=2)

    def test_unapproved_alpha_publishes_actionable_cash_target(self):
        target = build_cash_rotation_target(
            "2026-07-17",
            market_policy={
                "state": "DEFENSIVE",
                "entry_permission": "MAINLINE_ONLY",
                "max_exposure_ratio": 0.5,
            },
        )
        self.assertEqual(2, target["schema_version"])
        self.assertTrue(target["approved"])
        self.assertFalse(target["alpha_model_approved"])
        self.assertTrue(target["risk_control_only"])
        self.assertEqual({}, target["target_weights"])
        self.assertEqual(1.0, target["cash_weight"])


if __name__ == "__main__":
    unittest.main()
