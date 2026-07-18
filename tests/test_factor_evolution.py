import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from etf_radar.factor_evolution import (
    evaluate_expression,
    evolve_factor_registry,
    factor_metrics,
    industry_neutralise,
)
from etf_radar.calibration.pipeline import _atomic_json, _selected, select_thresholds, simulate_portfolio
from etf_radar.signals.labeling import build_forward_label
from etf_radar.rotation import (
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
        self.assertIn("5", metrics["ic_decay"])
        self.assertIn("20", metrics["ic_decay"])
        self.assertGreaterEqual(metrics["turnover"], 0.0)
        self.assertLessEqual(metrics["turnover"], 1.0)

    def test_gp_retires_failed_incumbent_and_keeps_economic_logic(self):
        frame = labelled_panel()
        previous = {
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
        self.assertTrue(registry["factors"])
        self.assertTrue(all(item["economic_logic"] for item in registry["factors"]))
        for item in registry["factors"]:
            values = evaluate_expression(item["expression"], frame)
            self.assertEqual(len(values), len(frame))

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

    def test_live_rotation_updates_only_one_sleeve_per_week(self):
        frame = pd.DataFrame(
            [
                {"date": "2026-07-17", "code": "A", "industry_group": "g1", "weekly_trend": 0.5, "relative_strength": 0.9, "trend_efficiency_20": 0.8, "volume_confirmation": 0.7, "priority": 90},
                {"date": "2026-07-17", "code": "B", "industry_group": "g2", "weekly_trend": 0.5, "relative_strength": 0.8, "trend_efficiency_20": 0.7, "volume_confirmation": 0.6, "priority": 80},
                {"date": "2026-07-17", "code": "C", "industry_group": "g3", "weekly_trend": 0.5, "relative_strength": 0.7, "trend_efficiency_20": 0.6, "volume_confirmation": 0.5, "priority": 70},
            ]
        )
        scored = score_rotation_candidates(frame)
        self.assertEqual(3, len(select_rotation_targets(scored, top_n=3)))
        first = update_live_rotation_state(frame, None, "2026-07-17")
        second = update_live_rotation_state(frame, first, "2026-07-18")
        self.assertEqual(first["sleeves"], second["sleeves"])
        self.assertAlmostEqual(1.0, sum(second["target_weights"].values()), places=5)


if __name__ == "__main__":
    unittest.main()
