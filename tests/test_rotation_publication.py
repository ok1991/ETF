import unittest

from etf_radar.rotation import stabilize_rotation_publication


def rotation(generated_at="2026-07-20 05:00:00"):
    return {
        "schema_version": 2,
        "generated_at": generated_at,
        "data_date": "2026-07-17",
        "execution_date": "2026-07-20",
        "model_version": "rotation-v2-test",
        "strategy_specification_fingerprint": "a" * 64,
        "target_weights": {"510300": 0.5},
        "execution_liquidity": {
            "510300": {
                "average_daily_amount_20": 1_000_000.0,
                "max_new_risk_amount": 100_000.0,
            }
        },
        "cash_weight": 0.5,
    }


class RotationPublicationTests(unittest.TestCase):
    def test_identical_economic_payload_preserves_publication_time(self):
        previous = rotation("2026-07-19 18:45:49")
        candidate = rotation("2026-07-20 05:03:26")
        result = stabilize_rotation_publication(candidate, previous)
        self.assertEqual("2026-07-19 18:45:49", result["generated_at"])
        self.assertEqual(previous, result)

    def test_target_change_creates_new_publication_identity(self):
        previous = rotation("2026-07-19 18:45:49")
        candidate = rotation("2026-07-20 05:03:26")
        candidate["target_weights"] = {"510300": 0.4}
        candidate["cash_weight"] = 0.6
        result = stabilize_rotation_publication(candidate, previous)
        self.assertEqual("2026-07-20 05:03:26", result["generated_at"])

    def test_liquidity_change_creates_new_publication_identity(self):
        previous = rotation("2026-07-19 18:45:49")
        candidate = rotation("2026-07-20 05:03:26")
        candidate["execution_liquidity"]["510300"][
            "average_daily_amount_20"
        ] = 900_000.0
        result = stabilize_rotation_publication(candidate, previous)
        self.assertEqual("2026-07-20 05:03:26", result["generated_at"])

    def test_future_previous_timestamp_is_not_reused(self):
        previous = rotation("2026-07-21 05:03:26")
        candidate = rotation("2026-07-20 05:03:26")
        result = stabilize_rotation_publication(candidate, previous)
        self.assertEqual("2026-07-20 05:03:26", result["generated_at"])


if __name__ == "__main__":
    unittest.main()
