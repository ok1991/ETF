"""Jinja2 based ETF radar report."""

from __future__ import annotations

import shutil
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .paths import PATHS


def _value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


class HTMLReporter:
    """Render the public dashboard without embedding assets in Python source."""

    @staticmethod
    def _compute_breadth(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "total": 0, "bull": 0, "bear": 0, "neutral": 0,
                "bull_pct": 0.0, "bear_pct": 0.0, "neutral_pct": 0.0,
                "ratio": 0.0, "signal": "无数据", "basis": "raw_status",
            }
        bullish = {"极强波段多头", "波段多头", "偏多企稳"}
        bearish = {"极弱波段空头", "波段空头", "偏空走弱"}
        values = [str(_value(item.get("raw_status", item.get("status", "")))) for item in results]
        total = len(values)
        bull = sum(value in bullish for value in values)
        bear = sum(value in bearish for value in values)
        neutral = total - bull - bear
        ratio = bull / max(1, bear)
        signal = (
            "极度乐观" if ratio > 3.0 else "偏多" if ratio > 1.5 else
            "中性" if ratio >= 0.67 else "偏空" if ratio >= 0.33 else "极度悲观"
        )
        return {
            "total": total,
            "bull": bull,
            "bear": bear,
            "neutral": neutral,
            "bull_pct": round(bull / total * 100.0, 1),
            "bear_pct": round(bear / total * 100.0, 1),
            "neutral_pct": round(neutral / total * 100.0, 1),
            "ratio": round(ratio, 2),
            "signal": signal,
            "basis": "raw_status",
        }

    @classmethod
    def generate(cls, results: List[Dict[str, Any]], env_result: Any, filename: str = "index.html") -> None:
        del filename
        PATHS.ensure()
        template_dir = PATHS.web / "templates"
        static_dir = PATHS.web / "static"
        public_static = PATHS.public / "assets"
        if public_static.exists():
            shutil.rmtree(public_static)
        shutil.copytree(static_dir, public_static)

        environment = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = environment.get_template("index.html.j2")
        rows = []
        for item in sorted(
            results,
            key=lambda value: float(value.get("v4_priority", 0.0) or 0.0),
            reverse=True,
        ):
            rows.append({
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "price": float(item.get("price", 0.0) or 0.0),
                "status": str(_value(item.get("raw_status", item.get("status", "")))),
                "priority": float(item.get("v4_priority", 0.0) or 0.0),
                "rps": float(item.get("rps", 0.0) or 0.0),
                "stop_loss": float(item.get("stop_loss", 0.0) or 0.0),
                "data_date": item.get("data_date", ""),
            })
        document = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            environment=env_result.to_dict() if hasattr(env_result, "to_dict") else dict(env_result),
            breadth=cls._compute_breadth(results),
            rows=rows,
        )
        (PATHS.public / "index.html").write_text(document, encoding="utf-8")
