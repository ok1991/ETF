import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("akshare", types.ModuleType("akshare"))

import main as etf_main  # noqa: E402
from v4_signals import build_v4_signal  # noqa: E402


def v4_signal(approved=True):
    result = {
        "code": "TEST",
        "name": "测试ETF",
        "data_date": "2026-07-16",
        "price": 10.0,
        "data_quality": {"status": "VALID"},
        "v4_priority": 86.0,
        "v4_market": {"entry_permission": "TRADEABLE", "score": 0.5},
        "relative_strength": {"score": 90.0, "has_120": True, "scope": "fixed_pool"},
        "v4_factors": {
            "monthly": {"score": 0.4, "history_ok": True},
            "weekly": {"score": 0.7, "history_ok": True},
            "setup": {"setup": "BREAKOUT", "score": 82.0},
            "risk": {
                "stop_loss": 9.4,
                "stop_dist_pct": 6.0,
                "atr_multiple": 2.2,
                "quality": 90.0,
                "executable": True,
            },
        },
    }
    calibration = {
        "early_stop_probability_3d": 0.15,
        "win_probability_10d": 0.55,
        "expected_excess_return_10d": 0.01,
        "sample_count": 500,
        "confidence": "HIGH",
        "version": "test-v4",
        "approved": approved,
        "status_reason": "APPROVED" if approved else "CALIBRATION_NOT_APPROVED",
    }
    return build_v4_signal(result, calibration)


class SignalContractTests(unittest.TestCase):
    def test_only_schema_v4_is_accepted(self):
        signal = v4_signal()
        payload = {"schema_version": 4, "signals": [signal]}
        self.assertEqual([], etf_main.validate_signal_contract(payload))

        signal["schema_version"] = 3
        errors = etf_main.validate_signal_contract(payload)
        self.assertTrue(any("schema_version" in error for error in errors))

    def test_unapproved_calibration_is_fail_closed(self):
        signal = v4_signal(approved=False)
        self.assertEqual("BLOCKED", signal["entry"]["state"])
        self.assertFalse(signal["calibration"]["approved"])

if __name__ == "__main__":
    unittest.main()
