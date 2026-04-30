#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF实战波段交易雷达 (Pro 版) - 多周期加权评分框架
(多维混合RPS + 大盘基准Alpha锚定 + 历史拐点捕捉 + 多线程并发)
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from typing import Optional, Dict, List
import os
import json
import concurrent.futures

HISTORY_FILE = "etf_history_state.json"


class DataNormalizer:
    PRIORITY_MAP = {
        'date': ['date', 'trade_date', '交易日期', '日期', 'index'],
        'open': ['open_price', 'open', '开盘价', '开盘'],
        'high': ['high_price', 'high', '最高价', '最高'],
        'low': ['low_price', 'low', '最低价', '最低'],
        'close': ['close_price', 'close', '收盘价', '收盘'],
        'amount': ['amount', '成交额'],
        'volume': ['volume', '成交量', 'vol'],
    }

    @classmethod
    def normalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        if isinstance(df.index, pd.DatetimeIndex): df = df.reset_index()

        existing_cols = {str(col).lower(): col for col in df.columns}
        rename_dict, drop_cols = {}, set()

        for std_name, aliases in cls.PRIORITY_MAP.items():
            primary_col = None
            for alias in aliases:
                if alias.lower() in existing_cols:
                    original_col = existing_cols[alias.lower()]
                    if primary_col is None:
                        primary_col = original_col
                        if original_col != std_name: rename_dict[original_col] = std_name
                    else:
                        drop_cols.add(original_col)

        df = df.drop(columns=list(drop_cols)).rename(columns=rename_dict)
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing: raise KeyError(f"缺少核心列: {missing}")
        df['date'] = pd.to_datetime(df['date'])

        if 'amount' in df.columns:
            df['volume'] = df['amount']
            df = df.drop(columns=['amount'])
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        return df


class TechnicalIndicators:
    @staticmethod
    def ma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        for p in periods: df[f'MA{p}'] = df['close'].rolling(window=p).mean()
        return df

    @staticmethod
    def ma_slope(df: pd.DataFrame, period: int = 20, lookback: int = 3) -> pd.DataFrame:
        if f'MA{period}' not in df.columns: return df
        ma_col = df[f'MA{period}']
        df[f'MA{period}_slope'] = (ma_col - ma_col.shift(lookback)) / ma_col.shift(lookback) * 100
        return df

    @staticmethod
    def macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        df['MACD_hist'] = dif - dif.ewm(span=signal, adjust=False).mean()
        return df

    @staticmethod
    def boll(df: pd.DataFrame, period=20, std_dev=2) -> pd.DataFrame:
        df['BOLL_mid'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['BOLL_upper'] = df['BOLL_mid'] + (std_dev * std)
        df['BOLL_lower'] = df['BOLL_mid'] - (std_dev * std)
        return df

    @staticmethod
    def volume_ma(df: pd.DataFrame, period=20) -> pd.DataFrame:
        df['VMA'] = df['volume'].rolling(window=period).mean()
        return df

    @staticmethod
    def atr(df: pd.DataFrame, period=14) -> pd.DataFrame:
        high, low, prev_close = df['high'], df['low'], df['close'].shift()
        tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        return df


class MarketAnalyzer:
    def __init__(self, code: str, name: str, market_safe: bool = True):
        self.code = code
        self.name = name
        self.data_dir = "etf_data"
        self.market_safe = market_safe
        self.df_daily = pd.DataFrame()
        self.df_weekly = pd.DataFrame()
        self.df_monthly = pd.DataFrame()
        self.rps = 0.0
        self.stop_loss_price = 0.0

    def _add_market_prefix(self, code: str) -> str:
        if code.startswith(('5', '6')): return f"sh{code}"
        if code.startswith(('1', '0', '3')): return f"sz{code}"
        return code

    def fetch_data(self) -> bool:
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        today_str = datetime.now().strftime('%Y%m%d')
        df = pd.DataFrame()

        existing = sorted(
            [f for f in os.listdir(self.data_dir) if f.startswith(f"{self.code}_") and f.endswith('.csv')],
            reverse=True)
        if existing and today_str in existing[0]:
            try:
                df = pd.read_csv(os.path.join(self.data_dir, existing[0]), parse_dates=['date'])
            except:
                pass

        if df.empty or len(df) < 500:
            try:
                df_net = ak.stock_zh_a_hist_tx(symbol=self._add_market_prefix(self.code), adjust="qfq")
                if df_net is not None and not df_net.empty:
                    df = DataNormalizer.normalize(df_net).sort_values('date').reset_index(drop=True)
                    new_file = f"{self.code}_{df['date'].iloc[-1].strftime('%Y%m%d')}.csv"
                    df.to_csv(os.path.join(self.data_dir, new_file), index=False, encoding='utf-8-sig')
                    for f_name in existing:
                        if f_name != new_file:
                            try:
                                os.remove(os.path.join(self.data_dir, f_name))
                            except:
                                pass
            except Exception as e:
                if existing: df = pd.read_csv(os.path.join(self.data_dir, existing[0]), parse_dates=['date'])

        if df.empty: return False

        df['date'] = pd.to_datetime(df['date'])
        self.df_daily = df.sort_values('date').reset_index(drop=True)
        self.df_weekly = self._resample('W')
        self.df_monthly = self._resample('ME')
        return True

    def _resample(self, freq: str) -> pd.DataFrame:
        if self.df_daily.empty: return pd.DataFrame()
        df = self.df_daily.set_index('date')
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        return df.resample(freq).agg(agg_dict).dropna(subset=['close']).reset_index()

    def calculate_indicators(self):
        TechnicalIndicators.ma(self.df_monthly, [5, 10])
        TechnicalIndicators.macd(self.df_monthly)
        TechnicalIndicators.ma(self.df_weekly, [5, 10, 20])
        TechnicalIndicators.ma_slope(self.df_weekly, period=20, lookback=3)
        TechnicalIndicators.macd(self.df_weekly)
        TechnicalIndicators.boll(self.df_daily)
        TechnicalIndicators.macd(self.df_daily)
        TechnicalIndicators.volume_ma(self.df_daily)
        TechnicalIndicators.atr(self.df_daily)

        if len(self.df_daily) >= 20:
            highest_20 = self.df_daily['high'].rolling(20).max().iloc[-1]
            atr_val = self.df_daily['ATR'].iloc[-1]
            atr_multiplier = 2.0 if self.market_safe else 1.5
            self.stop_loss_price = highest_20 - (atr_multiplier * atr_val)

    def _get_value(self, df: pd.DataFrame, col: str) -> float:
        return df[col].iloc[-1] if not df.empty and col in df.columns else np.nan

    @staticmethod
    def _s_format(value: float, precision: int = 3) -> str:
        return 'N/A' if pd.isna(value) else f'{value:.{precision}f}'

    def _log_details(self, monthly_score: float, weekly_score: float, daily_score: float):
        p_m, h_m = (self._get_value(self.df_monthly, k) for k in ['close', 'MACD_hist'])
        print(f"   [月线] Score={monthly_score:>5.1f} | Price={self._s_format(p_m)} | "
              f"MA5/10={self._s_format(self._get_value(self.df_monthly, 'MA5'))}/"
              f"{self._s_format(self._get_value(self.df_monthly, 'MA10'))} | Hist={self._s_format(h_m, 3)}")

        p_w, h_w, s_w = (self._get_value(self.df_weekly, k) for k in ['close', 'MACD_hist', 'MA20_slope'])
        print(f"   [周线] Score={weekly_score:>5.1f} | Price={self._s_format(p_w)} | "
              f"MA5/20={self._s_format(self._get_value(self.df_weekly, 'MA5'))}/"
              f"{self._s_format(self._get_value(self.df_weekly, 'MA20'))} | "
              f"Hist={self._s_format(h_w, 3)} | Slope={self._s_format(s_w, 2)}%")

        p_d, b_u, b_l, b_m, h_d, vol, vma = (self._get_value(self.df_daily, k) for k in
                                             ['close', 'BOLL_upper', 'BOLL_lower', 'BOLL_mid',
                                              'MACD_hist', 'volume', 'VMA'])
        vol_ratio = vol / vma if pd.notna(vma) and vma > 0 else 0
        print(f"   [日线] Score={daily_score:>5.1f} | Price={self._s_format(p_d)} | "
              f"BOLL={self._s_format(b_l)}~{self._s_format(b_u)} | Hist={self._s_format(h_d, 3)} | "
              f"Vol={vol_ratio:.2f}x")

    # ====================== 多周期评分框架 ======================
    def _analyze_monthly(self) -> tuple[float, str]:
        ma5, ma10, price, hist = (self._get_value(self.df_monthly, k)
                                  for k in ['MA5', 'MA10', 'close', 'MACD_hist'])
        if pd.isna(ma10) or pd.isna(hist):
            return 0.0, "数据不足：等待月线走势成型"

        ma_gap = (ma5 - ma10) / ma10 * 100 if ma10 != 0 else 0
        price_to_ma5 = (price - ma5) / ma5 * 100 if ma5 != 0 else 0

        if ma5 > ma10 and price > ma5 and hist > 0:
            score = 4.0 if (ma_gap > 3.0 and hist > 0.6 and price_to_ma5 > 1.5) else 3.5
            return score, "强多：月线金叉 + 站稳MA5 + 趋势加速"

        if price > ma5 and ma5 > ma10 * 0.97:
            return 2.0, "偏多：价格运行于月线MA5上方"

        if ma5 < ma10 and price < ma5 and hist < 0:
            score = -4.0 if (ma_gap < -3.0 and hist < -0.6) else -3.5
            return score, "强空：月线死叉 + 受压MA5 + 趋势加速"

        if price < ma5:
            return -2.0, "偏空：价格运行于月线MA5下方"

        return 0.0, "震荡：月线无明显方向趋势"

    def _analyze_weekly(self) -> tuple[float, str]:
        ma5, ma10, ma20, slope, price, hist = (self._get_value(self.df_weekly, k)
                                               for k in ['MA5', 'MA10', 'MA20', 'MA20_slope', 'close', 'MACD_hist'])
        if pd.isna(ma20) or pd.isna(slope):
            return 0.0, "数据不足"

        if abs(slope) < 0.8 and abs((price - ma20) / ma20) < 0.035:
            return 0.0, "周线震荡：20周线走平 + 价格纠缠"

        ma_bull = ma5 > ma10 > ma20 * 0.985
        ma_bear = ma5 < ma10 < ma20 * 1.015

        if ma_bull and price > ma20 and hist > 0 and slope > 0.6:
            return 6.0, "极强多：周线多头排列 + 斜率向上 + MACD柱正"

        if price > ma20 and hist > 0 and slope > 0.3:
            return 4.5, "多头趋势：站稳20周线 + MACD柱正"

        if ma_bear and price < ma20 and hist < 0 and slope < -0.6:
            return -6.0, "极弱空：周线空头排列 + 向下发散"

        if price < ma20 and hist < 0 and slope < -0.3:
            return -4.5, "空头趋势：跌破20周线 + MACD柱负"

        return 0.5 if price > ma20 else -0.5, "弱势震荡：围绕20周线波动"

    def _analyze_daily(self) -> tuple[float, str]:
        price, mid, upper, lower, hist, vol, vma = (self._get_value(self.df_daily, k)
                                                    for k in ['close', 'BOLL_mid', 'BOLL_upper',
                                                              'BOLL_lower', 'MACD_hist', 'volume', 'VMA'])
        if pd.isna(upper) or pd.isna(vma) or vma <= 0:
            return 0.0, "数据不足"

        vol_ratio = vol / vma

        if price >= upper * 0.99:
            if vol_ratio > 1.55:
                return 3.0, "真突破：放量站上布林上轨"
            else:
                return 1.0, "滞涨触顶：缩量上轨（警惕诱多）"

        if price <= lower * 1.015:
            if vol_ratio < 0.65:
                return 2.8, "极佳洗盘：缩量回踩下轨（低吸良机）"
            else:
                return -3.0, "真破位：放量跌破布林下轨（危险信号）"

        if price >= mid * 0.99 and hist > 0:
            return 1.3, "企稳：守住布林中轨 + MACD红柱"

        if price < mid * 1.01 and hist < -0.08:
            return -1.6, "走弱：跌破布林中轨 + MACD绿柱"

        return 0.0, "日线震荡：布林带中轨内盘整"

    def _apply_resonance_and_conflict(self, m: float, w: float, d: float) -> tuple[float, float, list]:
        bonus = 0.0
        penalty = 0.0
        tags = []

        if m >= 3.0 and w >= 4.5 and d >= 1.5:
            bonus += 2.2
            tags.append("🌟 多周期共振")

        if m >= 2.5 and w >= 5.0 and "极佳洗盘" in self._analyze_daily()[1]:
            bonus += 1.8
            tags.append("💎 月周强 + 黄金坑")

        if m >= 2.0 and w <= -4.0:
            penalty -= 3.5
            tags.append("⚠️ 月周严重冲突")

        if w >= 4.0 and d <= -2.5:
            penalty -= 2.5
            tags.append("🪤 周强日破")

        if d >= 2.5 and w <= -3.0:
            penalty -= 2.8
            tags.append("❌ 日线诱多")

        return bonus, penalty, tags

    def _generate_tags(self, m: float, w: float, d: float, total: float,
                       prev_score: Optional[float], stop_dist: float) -> tuple[list, float]:
        tags = []
        final_score = total

        if final_score >= 11.0 and self.rps >= 85:
            tags.append("👑 领涨龙头")
        elif final_score >= 9.0:
            tags.append("🚀 主升浪")
        elif final_score <= -9.0:
            tags.append("❄️ 主跌崩盘")

        if "极佳洗盘" in self._analyze_daily()[1] and w >= 4.0:
            tags.append("💎 黄金坑低吸")

        if stop_dist < 0:
            tags.append("🚨 破位止损离场")

        if prev_score is not None and prev_score <= 0 and final_score >= 7.0:
            tags.append("🔥 底部拐点 / 新晋多头")

        return tags, final_score

    def analyze(self, prev_score: Optional[float]) -> Optional[Dict]:
        self.calculate_indicators()

        monthly_score, monthly_reason = self._analyze_monthly()
        weekly_score, weekly_reason = self._analyze_weekly()
        daily_score, daily_reason = self._analyze_daily()

        raw_total = monthly_score * 1.25 + weekly_score * 1.40 + daily_score * 1.00
        bonus, penalty, extra_tags = self._apply_resonance_and_conflict(monthly_score, weekly_score, daily_score)

        final_score = round(raw_total + bonus + penalty, 1)
        status = self._determine_status(final_score)

        price = self._get_value(self.df_daily, 'close')
        stop_dist = ((price - self.stop_loss_price) / price * 100) if price and price > 0 else 0

        tags, final_score = self._generate_tags(monthly_score, weekly_score, daily_score, final_score, prev_score,
                                                stop_dist)
        all_tags = extra_tags + tags

        print(f"▶️ {self.name} ({self.code})")
        self._log_details(monthly_score, weekly_score, daily_score)

        tag_str = f" → 标签: [{', '.join(all_tags)}]" if all_tags else ""
        print(f"   └── 📊 Alpha-RPS: {self.rps:>5.1f} | "
              f"总分: {final_score:>5.1f} | 止损距: {stop_dist:>.1f}% → {status}{tag_str}\n")

        return {
            "code": self.code, "name": self.name,
            "monthly_score": monthly_score, "weekly_score": weekly_score, "daily_score": daily_score,
            "total_score": final_score, "status": status, "tags": all_tags,
            "monthly_reason": monthly_reason, "weekly_reason": weekly_reason, "daily_reason": daily_reason,
            "price": price, "stop_loss": self.stop_loss_price, "rps": self.rps, "stop_dist": stop_dist
        }

    @staticmethod
    def _determine_status(score: float) -> str:
        if score >= 10.0:   return "极强波段多头"
        if score >= 6.0:    return "波段多头"
        if score >= 2.0:    return "偏多企稳"
        if score <= -10.0:  return "极弱波段空头"
        if score <= -6.0:   return "波段空头"
        if score <= -2.0:   return "偏空走弱"
        return "多空震荡"


class HTMLReporter:
    STYLE = {
        "极强波段多头": {"cls": "badge-bull-super", "icon": "🔥"},
        "波段多头": {"cls": "badge-bull-strong", "icon": "📈"},
        "偏多企稳": {"cls": "badge-bull-weak", "icon": "↗️"},
        "多空震荡": {"cls": "badge-neutral", "icon": "⚖️"},
        "偏空走弱": {"cls": "badge-bear-weak", "icon": "↘️"},
        "波段空头": {"cls": "badge-bear-strong", "icon": "📉"},
        "极弱波段空头": {"cls": "badge-bear-super", "icon": "❄️"}
    }

    @classmethod
    def generate(cls, results: List[Dict], market_safe: bool, filename="index.html"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        css, js = cls._get_assets()
        stats = cls._generate_stats(results)
        rows = cls._generate_rows(results)

        market_env_html = '''
        <div class="market-env env-safe">
            <h3>✅ 大盘环境评估：安全多头期</h3>
            <p>沪深300指数站稳20日均线，市场情绪稳定。个股防守底线维持标准 <strong>2.0倍 ATR</strong>。</p>
        </div>
        ''' if market_safe else '''
        <div class="market-env env-danger">
            <h3>🚨 大盘环境评估：空头防守期</h3>
            <p>沪深300指数跌破20日均线，系统性风险增加！止损已收紧至 <strong>1.5倍 ATR</strong>。</p>
        </div>
        '''

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF 实战波段雷达 (Alpha-RPS 版)</title><style>{css}</style>
</head>
<body>
    <div class="dashboard">
        <header class="hero">
            <h1>🎯 ETF 实战波段交易雷达</h1>
            <p>多周期加权评分框架 | 大盘风控 | 数据更新: <strong>{timestamp}</strong></p>
        </header>
        {market_env_html}
        <section class="stats-grid">{stats}</section>
        <section class="table-card">
            <table id="radarTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">标的 / Alpha-RPS ⇅</th>
                        <th onclick="sortTable(1)">月线(±4) ⇅</th>
                        <th onclick="sortTable(2)">周线(±6) ⇅</th>
                        <th onclick="sortTable(3)">日线(±3) ⇅</th>
                        <th onclick="sortTable(4)" class="text-center">操作与防守距 ⇅</th>
                        <th onclick="sortTable(5)" class="text-center">总分 ⇅</th>
                        <th onclick="sortTable(6)" class="text-center">状态信号 ⇅</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </section>
        <footer class="footer">💡 <strong>交易铁律：</strong>顺大势、看节奏、抓时机。只做 Alpha-RPS 高 + 多周期共振的多头标的，破位必须止损！</footer>
    </div>
    <script>{js}</script>
</body></html>"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n🎉 交互看板已生成: {os.path.abspath(filename)}")

    @classmethod
    def _generate_rows(cls, results: List[Dict]) -> str:
        rows = []
        for r in sorted(results, key=lambda x: x['total_score'], reverse=True):
            style = cls.STYLE.get(r['status'], cls.STYLE["多空震荡"])

            tag_html = ""
            for t in r['tags']:
                t_cls = "tag-mark "
                if "龙头" in t or "主升" in t:
                    t_cls += "tag-king"
                elif "黄金坑" in t:
                    t_cls += "tag-pit"
                elif "止损" in t or "破位" in t:
                    t_cls += "tag-danger"
                elif "诱多" in t or "冲突" in t:
                    t_cls += "tag-trap"
                elif "拐点" in t:
                    t_cls += "tag-new-bull"
                else:
                    t_cls += "tag-fire"
                tag_html += f'<span class="{t_cls}">{t}</span> '

            if tag_html: tag_html = f'<div style="margin-top:6px">{tag_html}</div>'

            def sc_cls(s):
                return "score-pos" if s > 0 else ("score-neg" if s < 0 else "score-zero")

            def sc_str(s):
                return f"+{s:.1f}" if s > 0 else f"{s:.1f}"

            dist_color = "#dc2626" if r['stop_dist'] < 0 else ("#ca8a04" if r['stop_dist'] < 3 else "#16a34a")
            rps_color = "#d97706" if r['rps'] >= 50 else "#64748b"

            rows.append(f"""
                <tr>
                    <td data-label="标的/Alpha-RPS" data-sort="{r['rps']}">
                        <div class="code-title">
                            <span class="code-name">{r['name']}</span>
                            <span class="code-num">{r['code']} | RPS: <span style="color:{rps_color};font-weight:bold">{r['rps']:.1f}</span></span>
                        </div>
                    </td>
                    <td data-label="月线评分" data-sort="{r['monthly_score']}"><div class="signal-box"><span class="score-pill {sc_cls(r['monthly_score'])}">{sc_str(r['monthly_score'])}</span><span class="signal-text">{r['monthly_reason']}</span></div></td>
                    <td data-label="周线评分" data-sort="{r['weekly_score']}"><div class="signal-box"><span class="score-pill {sc_cls(r['weekly_score'])}">{sc_str(r['weekly_score'])}</span><span class="signal-text">{r['weekly_reason']}</span></div></td>
                    <td data-label="日线评分" data-sort="{r['daily_score']}"><div class="signal-box"><span class="score-pill {sc_cls(r['daily_score'])}">{sc_str(r['daily_score'])}</span><span class="signal-text">{r['daily_reason']}</span></div></td>
                    <td data-label="防守位置" class="text-center" data-sort="{r['stop_dist']}">
                        <div style="font-size:0.9rem; color:#4b5563">底线 <strong style="font-family:monospace">{r['stop_loss']:.3f}</strong></div>
                        <div style="font-size:0.85rem; color:{dist_color}; font-weight:700">离现价 {r['stop_dist']:.1f}%</div>
                    </td>
                    <td data-label="总分" class="text-center" data-sort="{r['total_score']}"><span class="total-score" style="color: {'#dc2626' if r['total_score'] > 0 else '#16a34a' if r['total_score'] < 0 else '#475569'}">{sc_str(r['total_score'])}</span></td>
                    <td data-label="状态信号" class="text-center" data-sort="{r['total_score']}"><span class="status-badge {style['cls']}">{style['icon']} {r['status']}</span>{tag_html}</td>
                </tr>
            """)
        return "".join(rows)

    @classmethod
    def _generate_stats(cls, results: List[Dict]) -> str:
        if not results: return ""
        counts = {
            "total": len(results),
            "bull": sum(1 for r in results if '多' in r['status']),
            "king": sum(1 for r in results if r['rps'] >= 85),
            "new_bull": sum(1 for r in results if any("拐点" in t for t in r['tags']))
        }
        return f"""
            <div class="stat-card"><div class="stat-label">标的池总数</div><div class="stat-val" style="color:#3b82f6">{counts['total']}</div></div>
            <div class="stat-card"><div class="stat-label">波段多头数量</div><div class="stat-val" style="color:#dc2626">{counts['bull']}</div></div>
            <div class="stat-card"><div class="stat-label">Alpha领涨龙头</div><div class="stat-val" style="color:#7c3aed">{counts['king']}</div></div>
            <div class="stat-card"><div class="stat-label">🔥 今日新晋拐点</div><div class="stat-val" style="color:#ea580c">{counts['new_bull']}</div></div>
        """

    @staticmethod
    def _get_assets():
        css = """
        :root { --primary: #3b82f6; --bg-color: #f8fafc; --text-main: #0f172a; --text-muted: #64748b; --border: #e2e8f0; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: var(--bg-color); color: var(--text-main); line-height: 1.5; padding: 20px; }
        .dashboard { max-width: 1450px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
        .hero { background: linear-gradient(135deg, #0f172a 0%, #312e81 100%); color: white; padding: 24px; border-radius: 16px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0, 0.1); }
        .hero p { color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;}
        .market-env { padding: 16px 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .env-safe { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
        .env-danger { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
        .stat-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid var(--border); text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .stat-label { font-size: 0.85rem; color: var(--text-muted); font-weight: 700; margin-bottom: 4px; }
        .stat-val { font-size: 2rem; font-weight: 800; }
        .table-card { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid var(--border); }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th { background: #f1f5f9; padding: 14px; font-weight: 700; color: #334155; font-size: 0.9rem; border-bottom: 2px solid #cbd5e1; cursor: pointer; }
        td { padding: 14px; border-bottom: 1px solid var(--border); font-size: 0.95rem; vertical-align: middle; }
        tr:hover { background-color: #f8fafc; }
        .code-title { display: flex; flex-direction: column; }
        .code-name { font-weight: 700; font-size: 1.05rem; }
        .code-num { font-size: 0.8rem; color: var(--text-muted); font-family: monospace; }
        .signal-box { display: flex; align-items: center; gap: 8px; }
        .score-pill { min-width: 40px; padding: 2px 6px; border-radius: 6px; text-align: center; font-weight: 800; font-size: 0.85rem; font-family: monospace;}
        .score-pos { background: #fee2e2; color: #b91c1c; }
        .score-neg { background: #dcfce7; color: #15803d; }
        .score-zero { background: #f1f5f9; color: #64748b; }
        .signal-text { font-size: 0.85rem; color: #475569; }
        .text-center { text-align: center; }
        .total-score { font-size: 1.6rem; font-weight: 800; font-family: monospace; }
        .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
        .badge-bull-super { background: #fef2f2; color: #991b1b; border: 2px solid #fecaca; }
        .badge-bull-strong { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .badge-bull-weak { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
        .badge-neutral { background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; }
        .badge-bear-weak { background: #f0fdfa; color: #0f766e; border: 1px solid #ccfbf1; }
        .badge-bear-strong { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
        .badge-bear-super { background: #166534; color: #fef2f2; border: 2px solid #86efac; }
        .tag-mark { display: inline-block; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right:4px; margin-bottom:4px; }
        .tag-king { background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
        .tag-pit { background: #fef08a; color: #854d0e; border: 1px solid #fde047; }
        .tag-new-bull { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
        .tag-danger { background: #166534; color: #fef2f2; border: 1px solid #fecaca; }
        .tag-trap { background: #f59e0b; color: white; border: 1px solid #fbbf24; }
        .tag-fire { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
        .footer { text-align: center; padding: 16px; color: var(--text-muted); font-size: 0.9rem; background: white; border-radius: 12px; border: 1px solid var(--border); }
        """
        js = """
        let sortStates = [0,0,0,0,0,0,0]; 
        function sortTable(colIndex) {
            const table = document.getElementById("radarTable");
            const tbody = table.querySelector("tbody");
            const rows = Array.from(tbody.querySelectorAll("tr"));
            let isAsc = sortStates[colIndex] === 1;
            sortStates[colIndex] = isAsc ? 0 : 1; 
            rows.sort((a, b) => {
                let valA = parseFloat(a.cells[colIndex].getAttribute("data-sort")) || 0;
                let valB = parseFloat(b.cells[colIndex].getAttribute("data-sort")) || 0;
                return isAsc ? (valA - valB) : (valB - valA);
            });
            rows.forEach(row => tbody.appendChild(row));
        }
        """
        return css, js


# ====================== 核心启动逻辑 ======================
def get_etf_name_map() -> dict:
    print("🔄 初始化名称映射...")
    try:
        spot = ak.fund_etf_spot_ths()
        return dict(zip(spot['基金代码'], spot['基金名称']))
    except:
        return {}


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def save_history(results: List[Dict]):
    hist = {r['code']: {'total_score': r['total_score']} for r in results}
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def evaluate_market_environment() -> tuple:
    print("🌐 正在评估大盘系统风控 (沪深300 - 510300)...")
    bm = MarketAnalyzer('510300', '沪深300ETF', market_safe=True)
    is_safe = True

    if bm.fetch_data():
        bm.calculate_indicators()
        price = bm._get_value(bm.df_daily, 'close')
        ma20 = bm._get_value(bm.df_daily, 'MA20')  # 注意：大盘这里使用日线MA20
        if price and ma20 and price < ma20:
            print("🚨【警告】沪深300跌破20日线，触发【系统防守模式】！\n")
            is_safe = False
        else:
            print("✅ 大盘环境安全（站在20日线之上）。\n")
    else:
        print("⚠️ 大盘数据获取失败，默认严格防守模式。\n")
        is_safe = False

    return is_safe, bm


def fetch_single_etf(code: str, name: str, market_safe: bool) -> Optional[MarketAnalyzer]:
    a = MarketAnalyzer(code, name, market_safe=market_safe)
    if a.fetch_data():
        return a
    return None


def calc_blended_return(df: pd.DataFrame) -> float:
    if len(df) < 21: return -999.0
    p_now = df['close'].iloc[-1]
    ret20 = (p_now - df['close'].iloc[-21]) / df['close'].iloc[-21]
    ret60 = (p_now - df['close'].iloc[-61]) / df['close'].iloc[-61] if len(df) >= 61 else ret20
    ret120 = (p_now - df['close'].iloc[-121]) / df['close'].iloc[-121] if len(df) >= 121 else ret60
    return 0.3 * ret20 + 0.3 * ret60 + 0.4 * ret120


def main():
    codes = [
        '159326', '512400', '512480', '512880', '159206', '159870', '515880', '159869', '516150',
        '159852', '515220', '159201', '515790', '512660', '159755', '515210', '159611', '512690',
        '512800', '159851', '561360', '560710', '159766', '512200', '518880', '562500', '513120',
        '513050', '513520', '159941', '159667', '159825', '560280'
    ]

    print(f"🚀 [ETF波段交易雷达] 启动! 标的池数量: {len(codes)}")
    name_map = get_etf_name_map()
    prev_history = load_history()

    market_safe, bm_analyzer = evaluate_market_environment()
    bm_blended_ret = calc_blended_return(
        bm_analyzer.df_daily) if bm_analyzer and not bm_analyzer.df_daily.empty else 0.0
    print(f"📈 沪深300基准混合收益率: {bm_blended_ret * 100:.2f}%\n")

    analyzers = []
    print("⏳ 正在并发拉取 K线数据...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {
            executor.submit(fetch_single_etf, code, name_map.get(code, f"ETF_{code}"), market_safe): code
            for code in codes
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_code):
            completed += 1
            analyzer = future.result()
            if analyzer:
                analyzers.append(analyzer)
            print(f"\r拉取进度: {completed}/{len(codes)}", end="", flush=True)

    print("\n\n🧮 正在计算 Alpha-Anchored RPS...")

    alphas = {}
    for a in analyzers:
        etf_ret = calc_blended_return(a.df_daily)
        alphas[a.code] = etf_ret - bm_blended_ret if etf_ret != -999.0 else -999.0

    rps_map = {}
    pos_alphas = {k: v for k, v in alphas.items() if v >= 0}
    neg_alphas = {k: v for k, v in alphas.items() if v < 0 and v != -999.0}

    if pos_alphas:
        sorted_pos = sorted(pos_alphas.keys(), key=lambda k: pos_alphas[k])
        for i, code in enumerate(sorted_pos):
            rps_map[code] = 50.0 + (i / max(1, len(sorted_pos) - 1)) * 50.0

    if neg_alphas:
        sorted_neg = sorted(neg_alphas.keys(), key=lambda k: neg_alphas[k])
        for i, code in enumerate(sorted_neg):
            rps_map[code] = (i / max(1, len(sorted_neg) - 1)) * 49.9

    print("🧠 执行多周期波段评分引擎...\n")
    results = []
    for a in analyzers:
        a.rps = rps_map.get(a.code, 0.0)
        prev_score = prev_history.get(a.code, {}).get('total_score')
        res = a.analyze(prev_score)
        if res: results.append(res)

    if results:
        save_history(results)
        HTMLReporter.generate(results, market_safe, "index.html")
    else:
        print("\n❌ 未获取到有效数据。")


if __name__ == "__main__":
    main()
