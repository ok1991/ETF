import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = [
    ROOT / "etf_radar",
    ROOT / ".github",
    ROOT / "main.py",
    ROOT / "calibrate_v4.py",
    ROOT / "run_cycle.py",
]


class ForbiddenDataSourceTests(unittest.TestCase):
    def test_production_code_never_reintroduces_eastmoney_interfaces(self):
        forbidden = [
            re.compile(r"eastmoney", re.IGNORECASE),
            re.compile(r"fund_etf_hist_em", re.IGNORECASE),
            re.compile(r"\b[a-zA-Z0-9_]+_em\s*\("),
        ]
        violations = []
        for item in PRODUCTION_PATHS:
            files = [item] if item.is_file() else list(item.rglob("*"))
            for path in files:
                if not path.is_file() or path.suffix.lower() not in {".py", ".yml", ".yaml"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(pattern.search(text) for pattern in forbidden):
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
