#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF实战波段交易雷达 - 多周期加权评分框架
优化: JSON序列化修复 + 手机响应式 + 波动层ATR百分位 + 维度原因显示修复
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, fields
import os
import json
import concurrent.futures

# ====================== 配置文件路径 ======================
HISTORY_FILE = "etf_history_state.json"
MARKET_ENV_HISTORY_FILE = "market_env_history.json"
MARKET_ENV_LATEST_FILE = "market_env_latest.json"
ETF_SIGNALS_LATEST_FILE = "etf_signals_latest.json"


# ====================================================================
#  MarketEnvResult - 修复JSON序列化 + 扩展波动层
# ====================================================================
@dataclass
class MarketEnvResult:
    """大盘环境评估结构化结果

    量化交易调用示例:
        with open('market_env_latest.json') as f:
            env = json.load(f)
        if env['market_safe']:
            weight = env['position_ratio']
            atr_m = env['atr_multiplier']
        else:
            weight = env['position_ratio']
            atr_m = env['atr_multiplier']
    """
    date: str
    index_code: str
    index_name: str
    price: float
    ma20: float
    ma60: float
    ma120: float
    close_vs_ma20_pct: float
    close_vs_ma60_pct: float
    close_vs_ma120_pct: float
    macd_hist: float
    vol_ratio: float
    atr_pct: float
    atr_percentile: float          # 【新增】ATR历史百分位(0~100)
    trend_score: float
    momentum_score: float
    volume_score: float
    volatility_score: float        # 【扩展】±2分
    total_score: float
    trend_details: dict
    momentum_details: dict
    volume_details: dict
    volatility_details: dict
    status: str
    market_safe: bool
    atr_multiplier: float
    position_ratio: float
    risk_level: str
    score_change: float
    status_changed: bool

    def to_dict(self) -> dict:
        """转为字典，【修复】确保所有值可JSON序列化

        核心修复: numpy.bool_ / numpy.integer / numpy.floating
        均无法被json.dumps序列化，需强制转为Python原生类型
        """
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            # 【修复】处理dict中的numpy类型
            if isinstance(val, dict):
                converted = {}
                for k, v in val.items():
                    if isinstance(v, (np.bool_,)):
                        converted[k] = bool(v)          # numpy.bool_ → Python bool
                    elif isinstance(v, (np.integer,)):
                        converted[k] = int(v)           # numpy.int64 → Python int
                    elif isinstance(v, (np.floating,)):
                        converted[k] = float(v)         # numpy.float64 → Python float
                    else:
                        converted[k] = v
                result[f.name] = converted
            # 【修复】处理顶层的numpy类型
            elif isinstance(val, (np.bool_,)):
                result[f.name] = bool(val)
            elif isinstance(val, (np.integer,)):
                result[f.name] = int(val)
            elif isinstance(val, (np.floating,)):
                result[f.name] = float(val)
            else:
                result[f.name] = val
        return result


# ====================================================================
#  DataNormalizer - 与原版相同
# ====================================================================
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
        if df.empty:
            return df
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        existing_cols = {str(col).lower(): col for col in df.columns}
        rename_dict, drop_cols = {}, set()
        for std_name, aliases in cls.PRIORITY_MAP.items():
            primary_col = None
            for alias in aliases:
                if alias.lower() in existing_cols:
                    original_col = existing_cols[alias.lower()]
                    if primary_col is None:
                        primary_col = original_col
                        if original_col != std_name:
                            rename_dict[original_col] = std_name
                    else:
                        drop_cols.add(original_col)
        df = df.drop(columns=list(drop_cols)).rename(columns=rename_dict)
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"缺少核心列: {missing}")
        df['date'] = pd.to_datetime(df['date'])
        if 'amount' in df.columns:
            df['volume'] = df['amount']
            df = df.drop(columns=['amount'])
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        return df


# ====================================================================
#  TechnicalIndicators - 与原版相同
# ====================================================================
class TechnicalIndicators:
    @staticmethod
    def ma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        for p in periods:
            df[f'MA{p}'] = df['close'].rolling(window=p).mean()
        return df

    @staticmethod
    def ma_slope(df: pd.DataFrame, period: int = 20, lookback: int = 3) -> pd.DataFrame:
        if f'MA{period}' not in df.columns:
            return df
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

    @staticmethod
    def yesterday_macd_cross(df: pd.DataFrame) -> Optional[str]:
        if 'MACD_hist' not in df.columns or len(df) < 3:
            return None
        h_today = df['MACD_hist'].iloc[-1]
        h_yesterday = df['MACD_hist'].iloc[-2]
        if pd.isna(h_today) or pd.isna(h_yesterday):
            return None
        if h_yesterday <= 0 < h_today:
            return 'golden_cross'
        if h_yesterday >= 0 > h_today:
            return 'death_cross'
        return None


# ====================================================================
#  MarketAnalyzer - ETF评分引擎
# ====================================================================
class MarketAnalyzer:
    def __init__(self, code: str, name: str, market_safe: bool = True,
                 atr_multiplier: float = 2.0):
        self.code = code
        self.name = name
        self.data_dir = "etf_data"
        self.market_safe = market_safe
        self.atr_multiplier = atr_multiplier
        self.df_daily = pd.DataFrame()
        self.df_weekly = pd.DataFrame()
        self.df_monthly = pd.DataFrame()
        self.rps = 0.0
        self.stop_loss_price = 0.0

    def _add_market_prefix(self, code: str) -> str:
        if code.startswith(('5', '6')):
            return f"sh{code}"
        if code.startswith(('1', '0', '3')):
            return f"sz{code}"
        return code

    def fetch_data(self) -> bool:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        today_str = datetime.now().strftime('%Y%m%d')
        df = pd.DataFrame()
        existing = sorted(
            [f for f in os.listdir(self.data_dir) if f.startswith(f"{self.code}_") and f.endswith('.csv')],
            reverse=True)
        if existing and today_str in existing[0]:
            try:
                df = pd.read_csv(os.path.join(self.data_dir, existing[0]), parse_dates=['date'])
            except Exception:
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
                            except Exception:
                                pass
            except Exception:
                if existing:
                    df = pd.read_csv(os.path.join(self.data_dir, existing[0]), parse_dates=['date'])
        if df.empty:
            return False
        df['date'] = pd.to_datetime(df['date'])
        self.df_daily = df.sort_values('date').reset_index(drop=True)
        self.df_weekly = self._resample('W')
        self.df_monthly = self._resample('ME')
        return True

    def _resample(self, freq: str) -> pd.DataFrame:
        if self.df_daily.empty:
            return pd.DataFrame()
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
            self.stop_loss_price = highest_20 - (self.atr_multiplier * atr_val)

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
        vol_ratio_d = vol / vma if pd.notna(vma) and vma > 0 else 0
        print(f"   [日线] Score={daily_score:>5.1f} | Price={self._s_format(p_d)} | "
              f"BOLL={self._s_format(b_l)}~{self._s_format(b_u)} | Hist={self._s_format(h_d, 3)} | "
              f"Vol={vol_ratio_d:.2f}x")

    def _analyze_monthly(self) -> Tuple[float, str]:
        ma5, ma10, price, hist = (self._get_value(self.df_monthly, k)
                                  for k in ['MA5', 'MA10', 'close', 'MACD_hist'])
        if pd.isna(ma10) or pd.isna(hist):
            return 0.0, "数据不足：等待月线走势成型"
        ma_gap = (ma5 - ma10) / ma10 * 100 if ma10 != 0 else 0
        price_to_ma5 = (price - ma5) / ma5 * 100 if ma5 != 0 else 0
        # 1. 优先定义清晰的震荡区 (Explicitly define consolidation zone)
        if abs(ma_gap) < 1.0 and abs(price_to_ma5) < 2.0:
            return 0.0, "震荡：月线均线黏合，方向不明"
        # 2. 处理多头趋势 (ma5 > ma10)
        if ma5 > ma10:
            # 2.1 强多头：趋势延续且价格强势
            if price > ma5 and hist > 0:
                # 区分极强和普通强
                is_accelerating = (ma_gap > 3.0 and hist > 0.6 and price_to_ma5 > 1.5)
                score = 4.0 if is_accelerating else 3.5
                reason = "极强多头：月线趋势加速" if is_accelerating else "多头趋势：金叉+站稳MA5"
                return score, reason

            # 2.2 多头回调 (!!! 核心改进 !!!)
            elif price < ma5:
                # 如果回调很深，比如跌破了MA10，风险加大
                if price < ma10:
                    return -1.0, "警惕：多头趋势但价格跌破MA10"
                return 1.5, "多头回调：趋势向上，价格回踩MA5"

            # 2.3 价格在MA5附近，但趋势向上
            else:  # price is very close to ma5
                return 2.0, "偏多：价格运行于月线MA5上方"
        # 3. 处理空头趋势 (ma5 < ma10)
        elif ma5 < ma10:
            # 3.1 强空头：趋势延续且价格弱势
            if price < ma5 and hist < 0:
                is_accelerating = (ma_gap < -3.0 and hist < -0.6)
                score = -4.0 if is_accelerating else -3.5
                reason = "极强空头：月线趋势加速" if is_accelerating else "空头趋势：死叉+受压MA5"
                return score, reason
            # 3.2 空头反弹 (!!! 核心改进 !!!)
            elif price > ma5:
                # 如果反弹很强，比如站上了MA10，可能转势
                if price > ma10:
                    return 1.0, "注意：空头趋势但价格突破MA10"
                return -1.5, "空头反弹：趋势向下，价格触碰MA5"

            # 3.3 价格在MA5附近，但趋势向下
            else:  # price is very close to ma5
                return -2.0, "偏空：价格运行于月线MA5下方"

        # 4. 最后的默认情况
        return 0.0, "震荡：月线无明显方向"

    def _analyze_weekly(self) -> Tuple[float, str]:
        ma5, ma10, ma20, slope, price, hist = (self._get_value(self.df_weekly, k)
                                               for k in ['MA5', 'MA10', 'MA20', 'MA20_slope', 'close', 'MACD_hist'])
        if pd.isna(ma20) or pd.isna(slope):
            return 0.0, "数据不足"

        # 1. 主动识别震荡
        if abs(slope) < 0.8 and abs((price - ma20) / ma20) < 0.035:
            return 0.0, "周线震荡：20周线走平 + 价格纠缠"

        is_price_strong = price > ma20 and slope > 0.2
        is_price_weak = price < ma20 and slope < -0.3  # 定义一个弱势的flag
        is_ma_bullish = ma5 > ma10 > ma20 * 0.985
        is_ma_bearish = ma5 < ma10 < ma20 * 1.015
        # 2. 多头市场判断
        if is_ma_bullish and is_price_strong and hist > 0:
            return 6.0, "极强多：周线多头排列 + 斜率向上 + MACD金叉"
        if is_price_strong and hist < 0:
            return 2.0, "多头回调：趋势向上，但MACD死叉减速"
        if is_price_strong and hist > 0:
            return 4.5, "多头趋势：站稳20周线 + MACD柱为正"
        # 3. 空头市场判断
        if is_ma_bearish and is_price_weak and hist < 0 and slope < -0.6:  # 增强极弱空的条件
            return -6.0, "极弱空：周线空头排列 + 向下发散"

        # 【新增】识别"空头反弹"场景 (核心改进)
        if is_price_weak and hist > 0:
            return -2.5, "空头反弹：趋势向下，MACD暂现金叉(警惕诱多)"
        if is_price_weak and hist < 0:  # 将原空头趋势判断条件收紧
            return -4.5, "空头趋势：跌破20周线 + MACD柱为负"

        # 4. 最后的弱势/模糊地带
        return 0.5 if price > ma20 else -0.5, "弱势震荡：围绕20周线波动"

    def _analyze_daily(self) -> Tuple[float, str]:
        price, mid, upper, lower, hist, vol, vma = (self._get_value(self.df_daily, k)
                                                    for k in ['close', 'BOLL_mid', 'BOLL_upper',
                                                              'BOLL_lower', 'MACD_hist', 'volume', 'VMA'])
        if pd.isna(upper) or pd.isna(vma) or vma <= 0 or pd.isna(mid) or mid <= 0:
            return 0.0, "数据不足"

        vol_ratio = vol / vma

        # 【新增】1. 优先判断布林带Squeeze状态
        # 计算布林带带宽 (Bollinger Bandwidth)
        bbw = (upper - lower) / mid
        # 假设我们有一个方法 _is_squeezing(bbw) 来判断是否处于历史低位
        # 这里用一个简化的阈值代替，例如带宽小于4%
        if bbw < 0.04:
            return 0.5, "蓄力：布林带收口，等待方向选择"
        # 2. 判断触及上下轨的极端情况
        if price >= upper * 0.99:
            if vol_ratio > 1.55:
                return 3.0, "真突破：放量站上布林上轨"
            return 1.0, "滞涨触顶：缩量上轨（警惕诱多）"

        if price <= lower * 1.015:
            if vol_ratio < 0.65:
                return 2.8, "极佳洗盘：缩量回踩下轨（低吸良机）"
            return -3.0, "真破位：放量跌破布林下轨（危险信号）"
        # 3. 判断在中轨附近的攻防
        if abs((price - mid) / mid) < 0.01:  # 价格紧贴中轨1%以内
            if hist > 0:
                return 1.3, "企稳：守住布林中轨 + MACD红柱"
            if hist < -0.08:
                return -1.6, "走弱：跌破布林中轨 + MACD绿柱"
            return 0.0, "盘整：价格缠绕布林中轨"

        # 【改进】4. 判断在通道“无人区”的运行状态
        if price > mid and price < upper:
            return 0.8, "多头通道：价格运行于布林带中上轨"

        if price < mid and price > lower:
            return -0.8, "空头通道：价格运行于布林带中下轨"
        # 5. 最后的默认情况（理论上很少触发）
        return 0.0, "日线震荡：布林带中轨内盘整"

    def _apply_resonance_and_conflict(self, m, w, d, daily_reason):
        bonus, penalty = 0.0, 0.0
        tags = []
        # 多周期共振
        if m > 0 and w > 0 and d > 0:
            bonus += min(m, w, d) * 0.5  # 取三者最小分的一半作为奖励
            tags.append("📈 三周期共振")
        # 顶背离/冲突
        if w >= 4.0 and d <= -1.0:  # 周线强多，日线走弱
            penalty -= 1.5
            tags.append("⚠️ 周日顶背离")
        # 底背离/冲突
        if w <= -4.0 and d >= 1.0:  # 周线强空，日线走强
            penalty -= 1.0  # 熊市反弹风险更大
            tags.append("⚠️ 周日底背离(诱多)")
        return bonus, penalty, tags

    def _generate_tags(self, m: float, w: float, d: float, total: float,
                       prev_score: Optional[float], stop_dist: float, daily_reason: str) -> Tuple[list, float]:
        tags = []
        if total >= 11.0 and self.rps >= 85:
            tags.append("👑 领涨龙头")
        elif total >= 9.0:
            tags.append("🚀 主升浪")
        elif total <= -9.0:
            tags.append("❄️ 主跌崩盘")
        if "极佳洗盘" in daily_reason and w >= 4.0:
            tags.append("💎 黄金坑低吸")
        if stop_dist < 0:
            tags.append("🚨 破位止损离场")
        if prev_score is not None and prev_score <= 0 and total >= 7.0:
            tags.append("🔥 底部拐点 / 新晋多头")
        return tags, total

    def analyze(self, prev_score: Optional[float]) -> Optional[Dict]:
        self.calculate_indicators()
        monthly_score, monthly_reason = self._analyze_monthly()
        weekly_score, weekly_reason = self._analyze_weekly()
        daily_score, daily_reason = self._analyze_daily()
        raw_total = monthly_score * 1.20 + weekly_score * 1.50 + daily_score * 0.80
        bonus, penalty, extra_tags = self._apply_resonance_and_conflict(monthly_score, weekly_score, daily_score,
                                                                        daily_reason)
        final_score = round(raw_total + bonus + penalty, 1)
        status = self._determine_status(final_score)
        price = self._get_value(self.df_daily, 'close')
        stop_dist = ((price - self.stop_loss_price) / price * 100) if price and price > 0 else 0
        tags, final_score = self._generate_tags(monthly_score, weekly_score, daily_score, final_score, prev_score,
                                                stop_dist, daily_reason)
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


# ====================================================================
#  MarketEnvironment - 大盘环境评估引擎
#  修复: JSON序列化 + 波动层扩展 ±2 + ATR百分位
# ====================================================================
class MarketEnvironment:
    """大盘环境评估引擎

    评估维度:
        趋势层 (±4分): 价格与MA20/MA60/MA120位置
        动能层 (±3分): MACD状态及变化
        量价层 (±2分): 量比与价格方向
        波动层 (±2分): ATR绝对水平 + 历史百分位  ← 【扩展】
    """

    SCORE_THRESHOLDS = {
        'strong_bull': 4.5,
        'weak_bull': 1.5,
        'neutral_low': -1.5,
        'weak_bear': -4.5,
    }

    WEIGHTS = {
        'trend': 1.0,
        'momentum': 0.8,
        'volume': 0.6,
        'volatility': 0.4,       # 【微调】波动层权重从0.3升到0.4
    }

    def __init__(self, index_code: str = '510300', index_name: str = '沪深300ETF'):
        self.index_code = index_code
        self.index_name = index_name
        self.analyzer: Optional[MarketAnalyzer] = None
        self.result: Optional[MarketEnvResult] = None

    def evaluate(self) -> MarketEnvResult:
        print(f"🌐 正在评估大盘系统风控 ({self.index_name} - {self.index_code})...")
        self.analyzer = MarketAnalyzer(self.index_code, self.index_name, market_safe=True)

        if not self.analyzer.fetch_data():
            print("⚠️ 大盘数据获取失败，默认严格防守模式。\n")
            self.result = self._default_danger_result("数据获取失败")
            self._save_result()
            return self.result

        TechnicalIndicators.ma(self.analyzer.df_daily, [20, 60, 120])
        TechnicalIndicators.macd(self.analyzer.df_daily)
        TechnicalIndicators.volume_ma(self.analyzer.df_daily)
        TechnicalIndicators.atr(self.analyzer.df_daily)

        df = self.analyzer.df_daily
        if df.empty or len(df) < 120:
            print("⚠️ 大盘数据不足120日，默认严格防守模式。\n")
            self.result = self._default_danger_result("数据不足120日")
            self._save_result()
            return self.result

        price = self.analyzer._get_value(df, 'close')
        ma20 = self.analyzer._get_value(df, 'MA20')
        ma60 = self.analyzer._get_value(df, 'MA60')
        ma120 = self.analyzer._get_value(df, 'MA120')
        macd_hist = self.analyzer._get_value(df, 'MACD_hist')
        vol = self.analyzer._get_value(df, 'volume')
        vma = self.analyzer._get_value(df, 'VMA')
        atr = self.analyzer._get_value(df, 'ATR')

        close_vs_ma20 = (price - ma20) / ma20 * 100 if pd.notna(ma20) and ma20 != 0 else 0.0
        close_vs_ma60 = (price - ma60) / ma60 * 100 if pd.notna(ma60) and ma60 != 0 else 0.0
        close_vs_ma120 = (price - ma120) / ma120 * 100 if pd.notna(ma120) and ma120 != 0 else 0.0
        vol_ratio = vol / vma if pd.notna(vma) and vma > 0 else 1.0
        atr_pct = atr / price * 100 if price > 0 and pd.notna(atr) else 0.0
        # 【新增】计算ATR历史百分位
        atr_percentile = self._calc_atr_percentile(df)
        # ======== 四维度评估 ========
        trend_score, trend_reason, trend_details = self._assess_trend(price, ma20, ma60, ma120)
        momentum_score, momentum_reason, momentum_details = self._assess_momentum(df, macd_hist)
        volume_score, volume_reason, volume_details = self._assess_volume(df, price, vol_ratio)
        volatility_score, volatility_reason, volatility_details = self._assess_volatility(
            atr_pct, atr_percentile, df)
        # ======== 综合评分 ========
        total_score = round(
            trend_score * self.WEIGHTS['trend'] +
            momentum_score * self.WEIGHTS['momentum'] +
            volume_score * self.WEIGHTS['volume'] +
            volatility_score * self.WEIGHTS['volatility'],
            1
        )
        # ======== 综合判定 ========
        status = self._determine_status(total_score)
        market_safe = total_score >= self.SCORE_THRESHOLDS['weak_bull']
        atr_multiplier = self._calc_atr_multiplier(total_score)
        position_ratio = self._calc_position_ratio(total_score)
        risk_level = self._calc_risk_level(total_score)
        # ======== 与上次记录对比 ========
        prev_record = self._load_last_record()
        score_change = 0.0
        status_changed = False
        if prev_record:
            score_change = round(total_score - prev_record.get('total_score', 0), 1)
            status_changed = (status != prev_record.get('status', ''))
        # ======== 日志输出 ========
        self._log_assessment(
            trend_score, trend_reason, momentum_score, momentum_reason,
            volume_score, volume_reason, volatility_score, volatility_reason,
            total_score, status, market_safe, position_ratio, atr_multiplier,
            score_change, status_changed
        )
        # ======== 构建结果 ========
        self.result = MarketEnvResult(
            date=datetime.now().strftime('%Y-%m-%d'),
            index_code=self.index_code,
            index_name=self.index_name,
            price=round(float(price), 3) if pd.notna(price) else 0.0,
            ma20=round(float(ma20), 3) if pd.notna(ma20) else 0.0,
            ma60=round(float(ma60), 3) if pd.notna(ma60) else 0.0,
            ma120=round(float(ma120), 3) if pd.notna(ma120) else 0.0,
            close_vs_ma20_pct=round(float(close_vs_ma20), 2),
            close_vs_ma60_pct=round(float(close_vs_ma60), 2),
            close_vs_ma120_pct=round(float(close_vs_ma120), 2),
            macd_hist=round(float(macd_hist), 4) if pd.notna(macd_hist) else 0.0,
            vol_ratio=round(float(vol_ratio), 2),
            atr_pct=round(float(atr_pct), 2),
            atr_percentile=round(float(atr_percentile), 1),
            trend_score=float(trend_score),
            momentum_score=float(momentum_score),
            volume_score=float(volume_score),
            volatility_score=float(volatility_score),
            total_score=float(total_score),
            trend_details=trend_details,
            momentum_details=momentum_details,
            volume_details=volume_details,
            volatility_details=volatility_details,
            status=status,
            market_safe=bool(market_safe),
            atr_multiplier=float(atr_multiplier),
            position_ratio=float(position_ratio),
            risk_level=risk_level,
            score_change=float(score_change),
            status_changed=bool(status_changed),
        )
        self._save_result()
        return self.result
        # ====================== ATR历史百分位 ======================

    @staticmethod
    def _calc_atr_percentile(df: pd.DataFrame) -> float:
        """计算当前ATR在过去120个交易日中的百分位
        Returns:
            0~100 的百分位值, 50表示当前ATR处于历史中位数水平
        """
        if 'ATR' not in df.columns or len(df) < 30:
            return 50.0
        atr_series = df['ATR'].dropna()
        if len(atr_series) < 20:
            return 50.0
        # 取最近120日(或全部可用数据)
        lookback = min(120, len(atr_series))
        recent_atr = atr_series.iloc[-lookback:]
        current_atr = recent_atr.iloc[-1]
        if pd.isna(current_atr):
            return 50.0
        # 百分位 = 当前值在历史序列中的排名位置
        percentile = (recent_atr < current_atr).sum() / len(recent_atr) * 100
        return float(percentile)

    # ====================== 趋势层评估 (±4分) ======================
    def _assess_trend(self, price: float, ma20: float, ma60: float, ma120: float) -> Tuple[float, str, dict]:
        """趋势层: 价格与MA20/MA60/MA120的位置关系
        判断依据:
            +4: 完美多头排列 (Price > MA20 > MA60 > MA120)
            +3: 中期多头 (Price > MA60 > MA120, 但Price < MA20或MA20<MA60)
            +2: 短期偏多 (Price > MA20 且 Price > MA60)
            +1: 微弱偏多 (仅 Price > MA20)
             0: 纠缠震荡 (价格在MA20±1.5%内)
            -1: 微弱偏空 (仅 Price < MA20)
            -2: 短期偏空 (Price < MA20 且 Price < MA60)
            -3: 中期空头 (Price < MA60 < MA120)
            -4: 完美空头排列 (Price < MA20 < MA60 < MA120)
        """
        has_ma20 = pd.notna(ma20) and ma20 != 0
        has_ma60 = pd.notna(ma60) and ma60 != 0
        has_ma120 = pd.notna(ma120) and ma120 != 0
        details = {
            'above_ma20': bool(price > ma20) if has_ma20 else False,
            'above_ma60': bool(price > ma60) if has_ma60 else False,
            'above_ma120': bool(price > ma120) if has_ma120 else False,
            'ma20_above_ma60': bool(ma20 > ma60) if (has_ma20 and has_ma60) else False,
            'ma60_above_ma120': bool(ma60 > ma120) if (has_ma60 and has_ma120) else False,
            'bullish_alignment': False,
            'bearish_alignment': False,
        }
        if has_ma20 and has_ma60 and has_ma120:
            if price > ma20 > ma60 > ma120:
                score, reason = 4.0, "完美多头排列: Price>MA20>MA60>MA120"
                details['bullish_alignment'] = True
            elif price > ma60 > ma120 and price > ma20:
                score, reason = 3.5, "强多头: 站稳所有均线, MA60>MA120"
                details['bullish_alignment'] = True
            elif price > ma60 > ma120:
                score, reason = 2.5, "中期多头: Price>MA60>MA120 (但跌破MA20)"
            elif price > ma20 and price > ma60:
                score, reason = 2.0, "短期偏多: 站稳MA20和MA60"
            elif price > ma20:
                dev_pct = abs(price - ma20) / ma20 * 100
                if dev_pct < 1.5:
                    score, reason = 0.0, "纠缠震荡: 价格紧贴MA20 (<±1.5%)"
                else:
                    score, reason = 1.0, "微弱偏多: 仅站稳MA20"
            elif price < ma20 < ma60 < ma120:
                score, reason = -4.0, "完美空头排列: Price<MA20<MA60<MA120"
                details['bearish_alignment'] = True
            elif price < ma60 < ma120 and price < ma20:
                score, reason = -3.5, "强空头: 跌破所有均线, MA60<MA120"
                details['bearish_alignment'] = True
            elif price < ma60 < ma120:
                score, reason = -2.5, "中期空头: Price<MA60<MA120 (但站上MA20)"
            elif price < ma20 and price < ma60:
                score, reason = -2.0, "短期偏空: 跌破MA20和MA60"
            elif price < ma20:
                dev_pct = abs(price - ma20) / ma20 * 100
                if dev_pct < 1.5:
                    score, reason = 0.0, "纠缠震荡: 价格紧贴MA20 (<±1.5%)"
                else:
                    score, reason = -1.0, "微弱偏空: 仅跌破MA20"
            else:
                score, reason = 0.0, "纠缠震荡: 均线交叉区"
        elif has_ma20 and has_ma60:
            if price > ma20 > ma60:
                score, reason = 3.0, "短期多头排列: Price>MA20>MA60"
            elif price > ma20:
                score, reason = 1.0, "站稳MA20"
            elif price < ma20 < ma60:
                score, reason = -3.0, "短期空头排列: Price<MA20<MA60"
            elif price < ma20:
                score, reason = -1.0, "跌破MA20"
            else:
                score, reason = 0.0, "震荡"
        elif has_ma20:
            if price > ma20:
                score, reason = 1.0, "站稳MA20"
            elif price < ma20:
                score, reason = -1.0, "跌破MA20"
            else:
                score, reason = 0.0, "价格=MA20"
        else:
            score, reason = 0.0, "趋势数据不足(缺失均线)"
        return score, reason, details

    # ====================== 动能层评估 (±3分) ======================
    def _assess_momentum(self, df: pd.DataFrame, macd_hist: float) -> Tuple[float, str, dict]:
        """动能层: MACD状态及变化方向
        判断依据:
            +3: MACD金叉 + 红柱连续放大(3日以上)
            +2: MACD金叉 + 红柱(刚金叉或红柱稳定)
            +1: MACD红柱(但可能缩小)
             0: MACD柱≈0 (纠缠)
            -1: MACD绿柱(但可能缩小)
            -2: MACD死叉 + 绿柱(刚死叉或绿柱稳定)
            -3: MACD死叉 + 绿柱连续放大(3日以上)
        """
        details = {
            'golden_cross': False,
            'death_cross': False,
            'hist_positive': False,
            'hist_expanding': False,
            'hist_shrinking': False,
        }
        if pd.isna(macd_hist) or len(df) < 5:
            return 0.0, "动能数据不足", details
        details['hist_positive'] = bool(macd_hist > 0)
        cross = TechnicalIndicators.yesterday_macd_cross(df)
        if cross == 'golden_cross':
            details['golden_cross'] = True
        elif cross == 'death_cross':
            details['death_cross'] = True
        recent_hist = df['MACD_hist'].tail(4).values
        hist_expanding_pos = False
        hist_expanding_neg = False
        if len(recent_hist) >= 4 and all(pd.notna(v) for v in recent_hist):
            if all(recent_hist[i] > 0 and recent_hist[i] >= recent_hist[i - 1]
                   for i in range(1, len(recent_hist))):
                hist_expanding_pos = True
            if all(recent_hist[i] < 0 and recent_hist[i] <= recent_hist[i - 1]
                   for i in range(1, len(recent_hist))):
                hist_expanding_neg = True
        details['hist_expanding'] = bool(hist_expanding_pos or hist_expanding_neg)
        if len(recent_hist) >= 3 and pd.notna(recent_hist[-1]) and pd.notna(recent_hist[-2]):
            if recent_hist[-1] > 0 and recent_hist[-1] < recent_hist[-2]:
                details['hist_shrinking'] = True
            elif recent_hist[-1] < 0 and recent_hist[-1] > recent_hist[-2]:
                details['hist_shrinking'] = True
        if details['golden_cross'] and hist_expanding_pos:
            return 3.0, "强动能: MACD金叉 + 红柱放大", details
        if details['golden_cross']:
            return 2.5, "动能转多: MACD刚金叉", details
        if macd_hist > 0 and hist_expanding_pos:
            return 2.0, "动能偏多: MACD红柱放大", details
        if macd_hist > 0:
            if details['hist_shrinking']:
                return 1.0, "动能减弱: MACD红柱缩小", details
            return 1.5, "动能中性偏多: MACD红柱稳定", details
        if details['death_cross'] and hist_expanding_neg:
            return -3.0, "弱动能: MACD死叉 + 绿柱放大", details
        if details['death_cross']:
            return -2.5, "动能转空: MACD刚死叉", details
        if macd_hist < 0 and hist_expanding_neg:
            return -2.0, "动能偏空: MACD绿柱放大", details
        if macd_hist < 0:
            if details['hist_shrinking']:
                return -1.0, "动能回暖: MACD绿柱缩小", details
            return -1.5, "动能中性偏空: MACD绿柱稳定", details
        return 0.0, "动能中性: MACD柱≈0", details

    # ====================== 量价层评估 (±2分) ======================
    def _assess_volume(self, df: pd.DataFrame, price: float, vol_ratio: float) -> Tuple[float, str, dict]:
        """量价层: 量比与价格方向配合
        判断依据:
            +2: 放量上涨 (vol_ratio>1.5 且 涨)
            +1: 温和放量上涨 (vol_ratio>1.0 且 涨)
             0: 平量盘整
            -1: 缩量下跌 (vol_ratio<0.7 且 跌) → 洗盘特征
            -2: 放量下跌 (vol_ratio>1.5 且 跌) → 恐慌出逃
        """
        details = {
            'vol_ratio': round(float(vol_ratio), 2),
            'price_up': False,
            'price_down': False,
            'high_volume': bool(vol_ratio > 1.5),
            'low_volume': bool(vol_ratio < 0.7),
            'price_change_pct': 0.0,
        }
        if len(df) < 2:
            return 0.0, "量价数据不足", details
        prev_close = df['close'].iloc[-2]
        if pd.notna(prev_close) and prev_close > 0:
            price_change = (price - prev_close) / prev_close * 100
            details['price_up'] = bool(price_change > 0)
            details['price_down'] = bool(price_change < 0)
            details['price_change_pct'] = round(float(price_change), 2)
        else:
            price_change = 0.0
        if details['high_volume'] and details['price_up']:
            return 2.0, "放量上涨: 资金积极入场", details
        if vol_ratio > 1.0 and details['price_up']:
            return 1.0, "温和放量上涨: 资金稳步入场", details
        if details['price_up'] and details['low_volume']:
            return 0.5, "缩量上涨: 上涨动能不足", details
        if details['high_volume'] and details['price_down']:
            return -2.0, "放量下跌: 恐慌出逃/主力出货", details
        if vol_ratio > 1.0 and details['price_down']:
            return -1.0, "放量下跌: 卖压增加", details
        if details['price_down'] and details['low_volume']:
            return -0.5, "缩量下跌: 洗盘特征(关注支撑)", details
        return 0.0, "平量盘整: 量价中性", details

    # ====================== 波动层评估 (±2分) ======================
    # 【扩展】原版±1分太窄，正常波动区间(0.8%~1.5%)占80%+时间 → 永远0分
    # 改进: ±2分 + ATR历史百分位双维度，即使绝对值正常，
    #        如果百分位>80(相对历史偏高)也会微调扣分
    def _assess_volatility(self, atr_pct: float, atr_percentile: float,
                           df: pd.DataFrame) -> Tuple[float, str, dict]:
        """波动层: ATR绝对水平 + 历史百分位双维度
        ... (docstring) ...
        """
        # 【关键】确保 details 字典在使用前被定义
        details = {
            'atr_pct': round(float(atr_pct), 2),
            'atr_percentile': round(float(atr_percentile), 1),
            'very_low_volatility': bool(atr_pct < 0.5),
            'low_volatility': bool(0.5 <= atr_pct < 0.8),
            'normal_volatility': bool(0.8 <= atr_pct <= 1.2),
            'high_volatility': bool(1.2 < atr_pct <= 2.0),
            'extreme_volatility': bool(atr_pct > 2.0),
            'percentile_high': bool(atr_percentile > 75),
            'percentile_low': bool(atr_percentile < 25),
        }
        # 绝对水平基础分
        if details['extreme_volatility']:
            base_score, base_reason = -2.0, "极端波动"
        elif details['high_volatility']:
            base_score, base_reason = -1.0, "波动偏高"
        elif details['normal_volatility']:
            base_score, base_reason = 0.0, "正常波动"
        elif details['low_volatility']:
            base_score, base_reason = 1.0, "低波动"
        else:  # very_low
            base_score, base_reason = 2.0, "极低波动/蓄力"
        # 百分位修正
        bonus = 0.0
        final_reason = base_reason  # 先将最终原因设置为基础原因
        if details['percentile_high'] and base_score >= 0:
            # 绝对值正常/偏低，但相对历史偏高 → 警惕波动回升
            bonus = -0.5
            final_reason += " [相对偏高]"  # 在这里拼接原因
        elif details['percentile_low'] and base_score < 0:
            # 绝对值偏高，但相对历史低位 → 可能回归正常
            bonus = 0.5
            final_reason += " [相对偏低]"  # 在这里拼接原因
        final_score = max(-2.0, min(2.0, base_score + bonus))
        return final_score, final_reason, details

    # ====================== 综合判定方法 ======================
    def _determine_status(self, total: float) -> str:
        th = self.SCORE_THRESHOLDS
        if total >= th['strong_bull']:
            return "强多头"
        if total >= th['weak_bull']:
            return "偏多"
        if total >= th['neutral_low']:
            return "震荡"
        if total >= th['weak_bear']:
            return "偏空"
        return "强空头"

    def _calc_atr_multiplier(self, score: float) -> float:
        """根据评分计算建议ATR止损倍数
        范围: 1.2 (极紧) ~ 2.5 (极宽)
        """
        score_clamped = max(-6.0, min(6.0, score))
        multiplier = 1.2 + (score_clamped + 6.0) / 12.0 * 1.3
        return round(float(multiplier), 1)

    def _calc_position_ratio(self, score: float) -> float:
        """根据评分计算建议仓位比例
        范围: 0.0 (空仓) ~ 1.0 (满仓)
        【优化】仓位曲线改为S型, 避免极端评分下仓位变化过激
        """
        score_clamped = max(-6.0, min(6.0, score))
        # S型映射: 用sigmoid变体, 中间区间更敏感
        x = score_clamped / 6.0  # 归一化到 [-1, 1]
        ratio = 1.0 / (1.0 + np.exp(-3.0 * x))  # sigmoid, k=3
        # 确保输出在 [0.05, 0.95] 之间, 留一点安全边际
        ratio = 0.05 + ratio * 0.9
        return round(float(ratio), 2)

    def _calc_risk_level(self, score: float) -> str:
        if score >= 3.0:
            return "低"
        if score >= -1.5:
            return "中"
        return "高"

    # ====================== 历史记录管理 ======================
    def _save_result(self):
        if self.result is None:
            return
        # 【修复】使用to_dict()确保所有值可JSON序列化
        result_dict = self.result.to_dict()
        # 1. 保存最新结果
        try:
            with open(MARKET_ENV_LATEST_FILE, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存最新市场环境失败: {e}")
        # 2. 追加到历史记录
        history = []
        if os.path.exists(MARKET_ENV_HISTORY_FILE):
            try:
                with open(MARKET_ENV_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []
        today = self.result.date
        history = [h for h in history if h.get('date') != today]
        history.append(result_dict)
        history.sort(key=lambda x: x.get('date', ''))
        if len(history) > 365:
            history = history[-365:]
        try:
            with open(MARKET_ENV_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存市场环境历史失败: {e}")

    @staticmethod
    def _load_last_record() -> Optional[dict]:
        if not os.path.exists(MARKET_ENV_HISTORY_FILE):
            return None
        try:
            with open(MARKET_ENV_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            today = datetime.now().strftime('%Y-%m-%d')
            prev_records = [h for h in history if h.get('date', '') != today]
            return prev_records[-1] if prev_records else None
        except Exception:
            return None

    def _default_danger_result(self, reason: str = "") -> MarketEnvResult:
        return MarketEnvResult(
            date=datetime.now().strftime('%Y-%m-%d'),
            index_code=self.index_code,
            index_name=self.index_name,
            price=0.0, ma20=0.0, ma60=0.0, ma120=0.0,
            close_vs_ma20_pct=0.0, close_vs_ma60_pct=0.0, close_vs_ma120_pct=0.0,
            macd_hist=0.0, vol_ratio=0.0, atr_pct=0.0, atr_percentile=50.0,
            trend_score=0.0, momentum_score=0.0, volume_score=0.0, volatility_score=0.0,
            total_score=-6.0,
            trend_details={}, momentum_details={}, volume_details={}, volatility_details={},
            status="强空头", market_safe=False,
            atr_multiplier=1.2, position_ratio=0.05, risk_level="高",
            score_change=0.0, status_changed=False,
        )

    # ====================== 日志输出 ======================
    def _log_assessment(self, trend_score, trend_reason, momentum_score, momentum_reason,
                        volume_score, volume_reason, volatility_score, volatility_reason,
                        total_score, status, market_safe, position_ratio, atr_multiplier,
                        score_change, status_changed):
        """打印结构化评估日志"""
        safety_icon = "✅" if market_safe else "🚨"
        change_icon = "📈" if score_change > 0 else ("📉" if score_change < 0 else "➡️")
        risk_level = self._calc_risk_level(total_score)  # 【修复】直接调用方法
        print(f"┌────────────────────────────────────────────────────────")
        print(f"│ 🌐 大盘环境评估: {self.index_name} ({self.index_code})")
        print(f"├────────────────────────────────────────────────────────")
        print(f"│ 【趋势层】 {trend_score:>+5.1f}分 │ {trend_reason}")
        print(f"│ 【动能层】 {momentum_score:>+5.1f}分 │ {momentum_reason}")
        print(f"│ 【量价层】 {volume_score:>+5.1f}分 │ {volume_reason}")
        print(f"│ 【波动层】 {volatility_score:>+5.1f}分 │ {volatility_reason}")
        print(f"├────────────────────────────────────────────────────────")
        print(f"│ 综合评分: {total_score:>+5.1f} │ 状态: {status}")
        print(f"│ 评分变动: {change_icon} {score_change:>+5.1f} │ 状态切换: {'是 ⚡' if status_changed else '否'}")
        print(f"├────────────────────────────────────────────────────────")
        print(f"│ {safety_icon} 安全判定: {'是' if market_safe else '否'}")
        print(f"│ 🛡️ 建议ATR止损倍数: {atr_multiplier}倍")
        print(f"│ 💰 建议仓位比例: {position_ratio:.0%}")
        print(f"│ ⚠️ 风险等级: {risk_level}")
        print(f"└────────────────────────────────────────────────────────\n")

    # ====================== 量化交易静态接口 ======================
    @staticmethod
    def get_signal_for_quant() -> Optional[dict]:
        """【量化交易一站式调用】读取最新大盘环境信号
        用法:
            signal = MarketEnvironment.get_signal_for_quant()
            if signal and signal['market_safe']:
                weight = signal['position_ratio']
                atr_mult = signal['atr_multiplier']
            else:
                weight = 0.3
                atr_mult = 1.2
        """
        if not os.path.exists(MARKET_ENV_LATEST_FILE):
            return None
        try:
            with open(MARKET_ENV_LATEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def get_history(days: int = 30) -> List[dict]:
        if not os.path.exists(MARKET_ENV_HISTORY_FILE):
            return []
        try:
            with open(MARKET_ENV_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history[-days:] if len(history) > days else history
        except Exception:
            return []

    @staticmethod
    def get_env_change_signal() -> Optional[dict]:
        """【量化交易专用】获取大盘环境拐点信号"""
        history = MarketEnvironment.get_history(days=5)
        if len(history) < 2:
            return None
        latest = history[-1]
        prev = history[-2]
        current_status = latest.get('status', '')
        prev_status = prev.get('status', '')
        current_score = latest.get('total_score', 0)
        score_change = current_score - prev.get('total_score', 0)
        return {
            'status_changed': bool(current_status != prev_status),
            'turned_bull': bool(current_status in ('偏多', '强多头') and prev_status not in ('偏多', '强多头')),
            'turned_bear': bool(current_status in ('偏空', '强空头') and prev_status not in ('偏空', '强空头')),
            'score_accelerating': bool(abs(score_change) > 2.0),
            'current_status': current_status,
            'current_score': current_score,
        }


# ====================================================================
#  HTMLReporter - 手机响应式 + 大盘环境增强
# ====================================================================
class HTMLReporter:
    STYLE = {
        "极强波段多头": {"cls": "badge-bull-super", "icon": "🔥"},
        "波段多头": {"cls": "badge-bull-strong", "icon": "📈"},
        "偏多企稳": {"cls": "badge-bull-weak", "icon": "↗️"},
        "多空震荡": {"cls": "badge-neutral", "icon": "⚖️"},
        "偏空走弱": {"cls": "badge-bear-weak", "icon": "↘️"},
        "波段空头": {"cls": "badge-bear-strong", "icon": "📉"},
        "极弱波段空头": {"cls": "badge-bear-super", "icon": "❄️"},
    }
    MARKET_STATUS_STYLE = {
        "强多头": {"cls": "env-safe", "icon": "🚀"},
        "偏多": {"cls": "env-safe", "icon": "✅"},
        "震荡": {"cls": "env-neutral", "icon": "⚖️"},
        "偏空": {"cls": "env-danger", "icon": "⚠️"},
        "强空头": {"cls": "env-danger", "icon": "🚨"},
    }

    @classmethod
    def generate(cls, results: List[Dict], env_result: MarketEnvResult, filename="index.html"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        css, js = cls._get_assets()
        stats = cls._generate_stats(results)
        rows = cls._generate_rows(results)
        market_env_html = cls._generate_market_env_html(env_result)
        html = f"""<!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ETF实战波段雷达</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📡</text></svg>">
        <style>{css}</style>
    </head>
    <body>
    <div class="dashboard">
        <header class="hero">
            <div class="hero-inner">
                <div class="hero-badge">PRO 1.0</div>
                <h1>📡 ETF 实战波段交易雷达</h1>
                <p class="hero-sub">多周期加权评分框架 · 大盘四维风控 · Alpha-RPS 相对强度</p>
                <p class="hero-time">📅 数据更新: <strong>{timestamp}</strong></p>
            </div>
            <div class="hero-gold-line"></div>
        </header>
        {market_env_html}
        <section class="stats-grid">{stats}</section>
        <section class="table-section">
            <div class="table-header-bar">
                <h2>📊 标的评分明细</h2>
                <span class="table-hint">点击表头可排序</span>
            </div>
            <div class="table-card">
                <table id="radarTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">标的 / Alpha-RPS ⇅</th>
                            <th onclick="sortTable(1)">月线(±4) ⇅</th>
                            <th onclick="sortTable(2)">周线(±6) ⇅</th>
                            <th onclick="sortTable(3)">日线(±3) ⇅</th>
                            <th onclick="sortTable(4)" class="text-center">防守止损 ⇅</th>
                            <th onclick="sortTable(5)" class="text-center">加权总分 ⇅</th>
                            <th onclick="sortTable(6)" class="text-center">状态信号 ⇅</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </section>
        <footer class="footer">
            <div class="footer-accent"></div>
            <p>💡 <strong>交易铁律：</strong>顺大势、看节奏、抓时机 — 只做Alpha-RPS高且多周期共振的多头标的，破位必止损！</p>
            <p class="footer-disclaimer">⚠️ 本工具仅供学习研究，不构成投资建议。入市有风险，投资需谨慎。</p>
        </footer>
    </div>
    <script>{js}</script>
    </body></html>"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n🎉 交互看板已生成: {os.path.abspath(filename)}")

    @classmethod
    def _generate_market_env_html(cls, env: MarketEnvResult) -> str:
        style = cls.MARKET_STATUS_STYLE.get(env.status, cls.MARKET_STATUS_STYLE["震荡"])
        env_cls = style['cls']
        env_icon = style['icon']
        # 【修复】统一中国A股配色: 红涨绿跌
        score_color = "#dc2626" if env.total_score > 0 else "#16a34a" if env.total_score < 0 else "#64748b"
        score_str = f"+{env.total_score:.1f}" if env.total_score > 0 else f"{env.total_score:.1f}"
        change_color = "#dc2626" if env.score_change > 0 else "#16a34a" if env.score_change < 0 else "#64748b"
        change_arrow = "🔺" if env.score_change > 0 else "🔻" if env.score_change < 0 else "➡️"
        change_str = f"{change_arrow}{env.score_change:+.1f}" if env.score_change != 0 else "➡️0.0"

        # ---- 各维度评分条 ----
        def dimension_bar(label: str, score: float, max_score: float, reason: str) -> str:
            pct = min(abs(score) / max_score * 100, 100) if max_score > 0 else 0
            if score > 0:
                bar_bg = "linear-gradient(90deg, #fee2e2, #fca5a5)"
                text_color = "#b91c1c"
            elif score < 0:
                bar_bg = "linear-gradient(90deg, #dcfce7, #86efac)"
                text_color = "#15803d"
            else:
                bar_bg = "#cbd5e1"
                text_color = "#64748b"
            score_display = f"+{score:.1f}" if score > 0 else f"{score:.1f}"
            reason_html = f'<span class="dim-reason">{reason}</span>' if reason else ''
            return f'''
                <div class="dimension-row">
                    <span class="dim-label">{label}</span>
                    <div class="dim-bar-bg">
                        <div class="dim-bar-fill" style="width:{pct:.0f}%; background:{bar_bg}"></div>
                    </div>
                    <span class="dim-score" style="color:{text_color}">{score_display}</span>
                    {reason_html}
                </div>'''

        # 构造reason
        trend_reason = ""
        if env.trend_details.get('bullish_alignment'):
            trend_reason = "多头排列"
        elif env.trend_details.get('bearish_alignment'):
            trend_reason = "空头排列"
        momentum_reason = ""
        if env.momentum_details.get('golden_cross'):
            momentum_reason = "金叉"
        elif env.momentum_details.get('death_cross'):
            momentum_reason = "死叉"
        elif env.momentum_details.get('hist_expanding'):
            momentum_reason = "柱放大"
        elif env.momentum_details.get('hist_shrinking'):
            momentum_reason = "柱缩小"
        volume_reason = ""
        if env.volume_details.get('high_volume'):
            volume_reason = "放量"
        elif env.volume_details.get('low_volume'):
            volume_reason = "缩量"
        volatility_reason = ""
        if env.volatility_details.get('extreme_volatility'):
            volatility_reason = "极端波动"
        elif env.volatility_details.get('high_volatility'):
            volatility_reason = "波动偏高"
        elif env.volatility_details.get('very_low_volatility'):
            volatility_reason = "极低/蓄力"
        elif env.volatility_details.get('low_volatility'):
            volatility_reason = "低波动"
        elif env.volatility_details.get('normal_volatility'):
            volatility_reason = "正常"
        if env.volatility_details.get('percentile_high'):
            volatility_reason += " ↑历史偏高"
        elif env.volatility_details.get('percentile_low'):
            volatility_reason += " ↓历史偏低"

        # MA偏离度
        def ma_dev_tag(label: str, pct_val: float) -> str:
            color = "#dc2626" if pct_val > 0 else "#16a34a" if pct_val < 0 else "#64748b"
            icon = "▲" if pct_val > 0 else "▼" if pct_val < 0 else "—"
            return f'<span class="ma-tag" style="color:{color}; border-left:3px solid {color}">{icon} {label} {pct_val:+.2f}%</span>'

        # 安全描述
        if env.market_safe:
            safe_desc = (f"大盘环境安全，建议维持 <strong>{env.position_ratio:.0%}</strong> 仓位，"
                         f"止损倍数 <strong>{env.atr_multiplier}x ATR</strong>")
        else:
            safe_desc = (f"大盘环境防守！建议缩减至 <strong>{env.position_ratio:.0%}</strong> 仓位，"
                         f"收紧止损至 <strong>{env.atr_multiplier}x ATR</strong>")
        change_alert = ""
        if env.status_changed:
            change_alert = '<div class="status-change-alert">⚡ 状态切换！请关注仓位调整</div>'
        # ATR百分位
        pct_val = env.atr_percentile
        pct_color = "#16a34a" if pct_val < 40 else "#ca8a04" if pct_val < 70 else "#dc2626"
        pct_label = "偏低" if pct_val < 40 else "适中" if pct_val < 70 else "偏高"
        return f'''
        <div class="market-env {env_cls}">
            <div class="env-status-strip"></div>
            <div class="env-body">
                <div class="env-header">
                    <div class="env-title-group">
                        <h3>{env_icon} 大盘环境: <span class="env-status-text">{env.status}</span></h3>
                        <span class="env-index-name">{env.index_name} ({env.index_code})</span>
                    </div>
                    <div class="env-score-group">
                        <div class="env-score-big" style="color:{score_color}">{score_str}</div>
                        <div class="env-score-change" style="color:{change_color}">{change_str}</div>
                    </div>
                </div>
                {change_alert}
                <p class="env-desc">{safe_desc}</p>
                <div class="env-details-grid">
                    <div class="env-dimensions">
                        <div class="dim-section-title">四维评分</div>
                        {dimension_bar("趋势", env.trend_score, 4, trend_reason)}
                        {dimension_bar("动能", env.momentum_score, 3, momentum_reason)}
                        {dimension_bar("量价", env.volume_score, 2, volume_reason)}
                        {dimension_bar("波动", env.volatility_score, 2, volatility_reason)}
                    </div>
                    <div class="env-ma-devs">
                        <div class="dim-section-title">均线偏离度</div>
                        {ma_dev_tag("MA20", env.close_vs_ma20_pct)}
                        {ma_dev_tag("MA60", env.close_vs_ma60_pct)}
                        {ma_dev_tag("MA120", env.close_vs_ma120_pct)}
                        <div class="dim-section-title" style="margin-top:10px">辅助指标</div>
                        <span class="ma-tag">MACD柱 {env.macd_hist:.4f}</span>
                        <span class="ma-tag">量比 {env.vol_ratio:.2f}</span>
                        <span class="ma-tag">ATR% {env.atr_pct:.2f}%</span>
                        <div class="dim-section-title" style="margin-top:10px">ATR历史百分位</div>
                        <div class="pct-bar-container">
                            <div class="pct-bar-bg">
                                <div class="pct-bar-fill" style="width:{pct_val:.0f}%; background:{pct_color}"></div>
                            </div>
                            <span class="pct-label" style="color:{pct_color}">{pct_val:.0f}% ({pct_label})</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>'''

    @classmethod
    def _generate_rows(cls, results: List[Dict]) -> str:
        rows = []
        for r in sorted(results, key=lambda x: x['total_score'], reverse=True):
            style = cls.STYLE.get(r['status'], cls.STYLE["多空震荡"])
            # 行状态色条颜色
            if r['total_score'] >= 6:
                row_accent = "#dc2626"
            elif r['total_score'] >= 2:
                row_accent = "#f59e0b"
            elif r['total_score'] >= -2:
                row_accent = "#94a3b8"
            elif r['total_score'] >= -6:
                row_accent = "#16a34a"
            else:
                row_accent = "#166534"
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
            if tag_html:
                tag_html = f'<div class="tag-row">{tag_html}</div>'

            def sc_cls(s):
                return "score-pos" if s > 0 else ("score-neg" if s < 0 else "score-zero")

            def sc_str(s):
                return f"+{s:.1f}" if s > 0 else f"{s:.1f}"

            dist_color = "#16a34a" if r['stop_dist'] < 0 else ("#ca8a04" if r['stop_dist'] < 3 else "#dc2626")
            dist_icon = "🚨" if r['stop_dist'] < 0 else ("⚠️" if r['stop_dist'] < 3 else "🛡️")
            rps_color = "#dc2626" if r['rps'] >= 80 else "#d97706" if r['rps'] >= 50 else "#64748b"
            rps_bar = min(r['rps'], 100)
            rows.append(f"""
                <tr style="border-left:4px solid {row_accent}">
                    <td data-label="标的/Alpha-RPS" data-sort="{r['rps']}">
                        <div class="code-title">
                            <span class="code-name">{r['name']}</span>
                            <span class="code-num">{r['code']} · RPS <span class="rps-inline" style="color:{rps_color}">{r['rps']:.0f}</span>
                                <span class="rps-bar-bg"><span class="rps-bar-fill" style="width:{rps_bar:.0f}%; background:{rps_color}"></span></span>
                            </span>
                        </div>
                    </td>
                    <td data-label="月线" data-sort="{r['monthly_score']}"><div class="signal-box"><span class="score-pill {sc_cls(r['monthly_score'])}">{sc_str(r['monthly_score'])}</span><span class="signal-text">{r['monthly_reason']}</span></div></td>
                    <td data-label="周线" data-sort="{r['weekly_score']}"><div class="signal-box"><span class="score-pill {sc_cls(r['weekly_score'])}">{sc_str(r['weekly_score'])}</span><span class="signal-text">{r['weekly_reason']}</span></div></td>
                    <td data-label="日线" data-sort="{r['daily_score']}"><div class="signal-box"><span class="score-pill {sc_cls(r['daily_score'])}">{sc_str(r['daily_score'])}</span><span class="signal-text">{r['daily_reason']}</span></div></td>
                    <td data-label="防守" class="text-center" data-sort="{r['stop_dist']}">
                        <div class="stop-price">止损线 <strong>{r['stop_loss']:.3f}</strong></div>
                        <div class="stop-dist" style="color:{dist_color}">{dist_icon} {r['stop_dist']:.1f}%</div>
                    </td>
                    <td data-label="总分" class="text-center" data-sort="{r['total_score']}">
                        <span class="total-score" style="color:{'#dc2626' if r['total_score'] > 0 else '#16a34a' if r['total_score'] < 0 else '#475569'}">{sc_str(r['total_score'])}</span>
                    </td>
                    <td data-label="状态" class="text-center" data-sort="{r['total_score']}">
                        <span class="status-badge {style['cls']}">{style['icon']} {r['status']}</span>{tag_html}
                    </td>
                </tr>
            """)
        return "".join(rows)

    @classmethod
    def _generate_stats(cls, results: List[Dict]) -> str:
        if not results:
            return ""
        counts = {
            "total": len(results),
            "bull": sum(1 for r in results if '多' in r['status']),
            "king": sum(1 for r in results if r['rps'] >= 85),
            "new_bull": sum(1 for r in results if any("拐点" in t for t in r['tags']))
        }
        return f"""
            <div class="stat-card stat-blue"><div class="stat-icon">📋</div><div class="stat-info"><div class="stat-val">{counts['total']}</div><div class="stat-label">标的池总数</div></div></div>
            <div class="stat-card stat-red"><div class="stat-icon">📈</div><div class="stat-info"><div class="stat-val">{counts['bull']}</div><div class="stat-label">波段多头</div></div></div>
            <div class="stat-card stat-purple"><div class="stat-icon">👑</div><div class="stat-info"><div class="stat-val">{counts['king']}</div><div class="stat-label">领涨龙头</div></div></div>
            <div class="stat-card stat-orange"><div class="stat-icon">🔥</div><div class="stat-info"><div class="stat-val">{counts['new_bull']}</div><div class="stat-label">今日新拐点</div></div></div>
        """

    @staticmethod
    def _get_assets():
        css = """
            :root {
                --primary: #3b82f6; --bg: #f0f2f5; --card: #ffffff;
                --text: #1e293b; --text-muted: #64748b; --border: #e2e8f0;
                --red: #dc2626; --red-light: #fef2f2; --red-bg: #fee2e2;
                --green: #16a34a; --green-light: #f0fdf4; --green-bg: #dcfce7;
                --gold: #d97706; --gold-light: #fffbeb;
                --navy: #0f172a; --navy-mid: #1e293b; --indigo: #312e81;
            }
            * { margin:0; padding:0; box-sizing:border-box; }
            body {
                font-family: "PingFang SC","Microsoft YaHei","Hiragino Sans GB",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
                background: var(--bg); color: var(--text); line-height:1.5; padding:12px;
            }
            .dashboard { max-width:1480px; margin:0 auto; display:flex; flex-direction:column; gap:14px; }
            /* ====== Hero ====== */
            .hero {
                background: linear-gradient(135deg, var(--navy) 0%, #1a1a4e 50%, var(--indigo) 100%);
                color: #fff; border-radius:16px; overflow:hidden; position:relative;
            }
            .hero-inner { padding:28px 24px 18px; text-align:center; }
            .hero-badge {
                display:inline-block; background:linear-gradient(135deg,#d97706,#f59e0b);
                color:#fff; font-size:0.68rem; font-weight:800; padding:2px 10px;
                border-radius:99px; letter-spacing:1px; margin-bottom:8px;
            }
            .hero h1 { font-size:1.5rem; font-weight:900; letter-spacing:2px; }
            .hero-sub { color:#93c5fd; font-size:0.85rem; margin-top:6px; }
            .hero-time { color:#cbd5e1; font-size:0.8rem; margin-top:4px; }
            .hero-time strong { color:#fbbf24; }
            .hero-gold-line { height:3px; background:linear-gradient(90deg,transparent,#d97706 20%,#fbbf24 50%,#d97706 80%,transparent); }
            /* ====== 大盘环境 ====== */
            .market-env {
                border-radius:14px; overflow:hidden; display:flex;
                box-shadow:0 2px 12px rgba(0,0,0,0.07); background:var(--card);
            }
            .env-safe .env-status-strip { background:linear-gradient(180deg,#dc2626,#ef4444); }
            .env-neutral .env-status-strip { background:linear-gradient(180deg,#ca8a04,#eab308); }
            .env-danger .env-status-strip { background:linear-gradient(180deg,#16a34a,#22c55e); }
            .env-status-strip { width:6px; flex-shrink:0; }
            .env-body { flex:1; padding:16px 18px; }
            .env-safe .env-body { background:#fef2f2; }
            .env-neutral .env-body { background:#fefce8; }
            .env-danger .env-body { background:#f0fdf4; }
            .env-header { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
            .env-title-group h3 { font-size:1.05rem; margin:0; color:var(--navy-mid); }
            .env-status-text { font-weight:900; }
            .env-safe .env-status-text { color:#dc2626; }
            .env-neutral .env-status-text { color:#ca8a04; }
            .env-danger .env-status-text { color:#16a34a; }
            .env-index-name { font-size:0.78rem; color:var(--text-muted); }
            .env-score-group { display:flex; align-items:baseline; gap:8px; }
            .env-score-big { font-size:1.6rem; font-weight:900; font-family:monospace; line-height:1; }
            .env-score-change { font-size:0.82rem; font-weight:700; font-family:monospace; }
            .env-desc { font-size:0.85rem; margin-bottom:10px; line-height:1.6; }
            .status-change-alert {
                background:linear-gradient(90deg,#fef08a,#fef9c3); color:#854d0e;
                padding:6px 12px; border-radius:8px; font-weight:700; font-size:0.8rem;
                margin-bottom:8px; animation:pulse 2s infinite;
            }
            @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.65} }
            .env-details-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
            .env-dimensions { display:flex; flex-direction:column; gap:5px; }
            .dim-section-title { font-size:0.75rem; font-weight:800; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:2px; }
            .dimension-row { display:flex; align-items:center; gap:6px; }
            .dim-label { width:36px; font-weight:800; font-size:0.78rem; flex-shrink:0; color:var(--text); }
            .dim-bar-bg { flex:1; height:14px; background:#e2e8f0; border-radius:7px; overflow:hidden; min-width:50px; }
            .dim-bar-fill { height:100%; border-radius:7px; transition:width 0.4s ease; }
            .dim-score { width:40px; font-weight:900; font-size:0.82rem; font-family:monospace; text-align:right; flex-shrink:0; }
            .dim-reason { font-size:0.72rem; color:#475569; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:120px; }
            .env-ma-devs { display:flex; flex-direction:column; gap:3px; }
            .ma-tag {
                display:inline-block; font-size:0.75rem; padding:3px 8px; border-radius:5px;
                background:#f8fafc; margin-right:4px; margin-bottom:3px;
                font-family:monospace; font-weight:700; border-left:3px solid #94a3b8;
            }
            .pct-bar-container { display:flex; align-items:center; gap:8px; margin-top:4px; }
            .pct-bar-bg { flex:1; height:12px; background:#e2e8f0; border-radius:6px; overflow:hidden; }
            .pct-bar-fill { height:100%; border-radius:6px; transition:width 0.4s; }
            .pct-label { font-size:0.75rem; font-weight:800; font-family:monospace; white-space:nowrap; }
            /* ====== Stats ====== */
            .stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
            .stat-card {
                background:var(--card); border-radius:12px; padding:14px 12px;
                display:flex; align-items:center; gap:10px;
                box-shadow:0 1px 4px rgba(0,0,0,0.05); border:1px solid var(--border);
                transition:transform 0.2s,box-shadow 0.2s;
            }
            .stat-card:hover { transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,0.1); }
            .stat-icon { font-size:1.5rem; flex-shrink:0; }
            .stat-info { flex:1; }
            .stat-val { font-size:1.6rem; font-weight:900; line-height:1.1; }
            .stat-label { font-size:0.72rem; color:var(--text-muted); font-weight:700; margin-top:2px; }
            .stat-blue .stat-val { color:#3b82f6; }
            .stat-red .stat-val { color:#dc2626; }
            .stat-purple .stat-val { color:#7c3aed; }
            .stat-orange .stat-val { color:#ea580c; }
            /* ====== Table Section ====== */
            .table-section { background:var(--card); border-radius:14px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.06); border:1px solid var(--border); }
            .table-header-bar { display:flex; justify-content:space-between; align-items:center; padding:14px 18px; border-bottom:1px solid var(--border); }
            .table-header-bar h2 { font-size:1rem; font-weight:800; color:var(--navy-mid); }
            .table-hint { font-size:0.72rem; color:var(--text-muted); }
            .table-card { overflow-x:auto; }
            table { width:100%; border-collapse:collapse; text-align:left; }
            th {
                background:#f8fafc; padding:10px 10px; font-weight:800; color:#475569;
                font-size:0.78rem; border-bottom:2px solid #cbd5e1; cursor:pointer;
                white-space:nowrap; position:sticky; top:0; z-index:1;
            }
            th:hover { background:#e2e8f0; }
            td { padding:10px; border-bottom:1px solid #f1f5f9; font-size:0.85rem; vertical-align:middle; }
            tr { transition:background 0.15s; }
            tr:hover { background:#f8fafc; }
            .code-title { display:flex; flex-direction:column; }
            .code-name { font-weight:800; font-size:0.92rem; color:var(--text); }
            .code-num { font-size:0.72rem; color:var(--text-muted); font-family:monospace; display:flex; align-items:center; gap:4px; }
            .rps-bar-bg { display:inline-block; width:40px; height:5px; background:#e2e8f0; border-radius:3px; overflow:hidden; vertical-align:middle; }
            .rps-bar-fill { display:block; height:100%; border-radius:3px; }
            .rps-inline { font-weight:900; }
            .signal-box { display:flex; align-items:center; gap:5px; }
            .score-pill {
                min-width:38px; padding:2px 6px; border-radius:6px; text-align:center;
                font-weight:900; font-size:0.8rem; font-family:monospace;
            }
            .score-pos { background:#fee2e2; color:#b91c1c; }
            .score-neg { background:#dcfce7; color:#15803d; }
            .score-zero { background:#f1f5f9; color:#64748b; }
            .signal-text { font-size:0.75rem; color:#475569; }
            .text-center { text-align:center; }
            .total-score { font-size:1.5rem; font-weight:900; font-family:monospace; }
            .status-badge {
                display:inline-flex; align-items:center; gap:4px; padding:4px 12px;
                border-radius:99px; font-weight:800; font-size:0.68rem; white-space:nowrap;
            }
            .badge-bull-super { background:linear-gradient(135deg,#fee2e2,#fecaca); color:#991b1b; border:2px solid #f87171; }
            .badge-bull-strong { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }
            .badge-bull-weak { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }
            .badge-neutral { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }
            .badge-bear-weak { background:#f0fdfa; color:#0f766e; border:1px solid #99f6e4; }
            .badge-bear-strong { background:#dcfce7; color:#166534; border:1px solid #86efac; }
            .badge-bear-super { background:linear-gradient(135deg,#166534,#15803d); color:#f0fdf4; border:2px solid #86efac; }
            .tag-row { margin-top:5px; }
            .tag-mark { display:inline-block; font-size:0.68rem; padding:1px 7px; border-radius:4px; font-weight:800; margin:2px 3px 2px 0; }
            .tag-king { background:#ede9fe; color:#6d28d9; border:1px solid #c4b5fd; }
            .tag-pit { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
            .tag-new-bull { background:#ffedd5; color:#c2410c; border:1px solid #fdba74; }
            .tag-danger { background:#16a34a; color:#fff; border:1px solid #166534; }
            .tag-trap { background:#f59e0b; color:#fff; border:1px solid #d97706; }
            .tag-fire { background:#fffbeb; color:#3b82f6; border:1px solid #166534; }
            .stop-price { font-size:0.82rem; color:#4b5563; }
            .stop-price strong { font-family:monospace; }
            .stop-dist { font-size:0.82rem; font-weight:800; }
            /* ====== Footer ====== */
            .footer { background:var(--card); border-radius:14px; overflow:hidden; border:1px solid var(--border); text-align:center; }
            .footer-accent { height:3px; background:linear-gradient(90deg,transparent,#dc2626 20%,#f59e0b 50%,#dc2626 80%,transparent); }
            .footer p { padding:10px 14px; color:var(--text-muted); font-size:0.82rem; }
            .footer p:first-of-type { padding-bottom:2px; }
            .footer-disclaimer { font-size:0.72rem !important; color:#94a3b8 !important; padding-top:0 !important; }
            /* ====== 手机响应式 ====== */
            @media (max-width:768px) {
                body { padding:8px; }
                .dashboard { gap:10px; }
                .hero-inner { padding:18px 14px 12px; }
                .hero h1 { font-size:1.15rem; }
                .hero-sub { font-size:0.76rem; }
                .env-details-grid { grid-template-columns:1fr; gap:10px; }
                .dim-reason { max-width:160px; }
                .market-env { border-radius:10px; }
                .env-status-strip { width:5px; }
                .stats-grid { grid-template-columns:repeat(2,1fr); gap:8px; }
                .stat-card { padding:10px 8px; }
                .stat-val { font-size:1.3rem; }
                .stat-icon { font-size:1.2rem; }
                /* 表格→卡片 */
                .table-section { border-radius:10px; }
                .table-header-bar { padding:10px 12px; }
                .table-header-bar h2 { font-size:0.9rem; }
                .table-hint { display:none; }
                .table-card { border-radius:0; background:var(--bg); padding:0; }
                table thead { display:none; }
                table { display:block; }
                table tbody { display:block; }
                table tbody tr {
                    display:flex; flex-direction:column; background:var(--card);
                    border-radius:10px; margin:0 8px 10px; border:1px solid var(--border);
                    box-shadow:0 1px 3px rgba(0,0,0,0.04); overflow:hidden;
                }
                table tbody tr:hover { background:var(--card); }
                table tbody tr td {
                    display:flex; justify-content:space-between; align-items:center;
                    padding:7px 12px; border-bottom:1px solid #f1f5f9; font-size:0.82rem;
                }
                table tbody tr td:last-child { border-bottom:none; }
                table tbody tr td::before {
                    content:attr(data-label); font-weight:800; color:var(--text-muted);
                    font-size:0.72rem; flex-shrink:0; margin-right:8px;
                }
                /* 卡片头行 */
                table tbody tr td:first-child {
                    background:#f8fafc; border-bottom:2px solid var(--border);
                    flex-direction:column; align-items:flex-start; padding:10px 12px;
                }
                table tbody tr td:first-child::before { display:none; }
                /* 总分行 */
                table tbody tr td:nth-child(6) { justify-content:center; background:#f8fafc; }
                table tbody tr td:nth-child(6)::before { display:none; }
                /* 状态行 */
                table tbody tr td:last-child { justify-content:center; padding:10px; }
                table tbody tr td:last-child::before { display:none; }
                .signal-text { white-space:normal; font-size:0.72rem; }
                .total-score { font-size:1.3rem; }
                .tag-mark { font-size:0.65rem; }
                .status-badge { font-size:0.72rem; }
                .footer { border-radius:10px; }
            }
            @media (max-width:400px) {
                .env-header { flex-direction:column; align-items:flex-start; }
                .env-score-big { font-size:1.3rem; }
                .dim-label { width:28px; font-size:0.7rem; }
                .dim-reason { max-width:80px; font-size:0.65rem; }
                .ma-tag { font-size:0.65rem; padding:2px 5px; }
                .pct-bar-container { flex-direction:column; align-items:stretch; gap:2px; }
                .stat-card { gap:6px; }
            }
            """
        js = """
            let sortStates = [0,0,0,0,0,0,0];
            function sortTable(colIndex) {
                const table = document.getElementById("radarTable");
                const tbody = table.querySelector("tbody");
                const rows = Array.from(tbody.querySelectorAll("tr"));
                let isAsc = sortStates[colIndex] === 1;
                sortStates = [0,0,0,0,0,0,0];
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


# ====================== 辅助函数 ======================
def get_etf_name_map() -> dict:
    print("🔄 初始化名称映射...")
    try:
        spot = ak.fund_etf_spot_ths()
        return dict(zip(spot['基金代码'], spot['基金名称']))
    except Exception:
        return {}


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def save_history(results: List[Dict]):
    hist = {r['code']: {'total_score': r['total_score']} for r in results}
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def fetch_single_etf(code: str, name: str, market_safe: bool, atr_multiplier: float) -> Optional[MarketAnalyzer]:
    a = MarketAnalyzer(code, name, market_safe=market_safe, atr_multiplier=atr_multiplier)
    if a.fetch_data():
        return a
    return None


def calc_blended_return(df: pd.DataFrame) -> float:
    if len(df) < 21:
        return -999.0
    p_now = df['close'].iloc[-1]
    ret20 = (p_now - df['close'].iloc[-21]) / df['close'].iloc[-21]
    ret60 = (p_now - df['close'].iloc[-61]) / df['close'].iloc[-61] if len(df) >= 61 else ret20
    ret120 = (p_now - df['close'].iloc[-121]) / df['close'].iloc[-121] if len(df) >= 121 else ret60
    return 0.3 * ret20 + 0.3 * ret60 + 0.4 * ret120


def save_etf_signals(results: List[Dict], env_result: MarketEnvResult):
    """保存ETF信号到JSON(供量化交易读取)"""
    output = {
        "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_env": {
            "status": env_result.status,
            "market_safe": bool(env_result.market_safe),
            "total_score": float(env_result.total_score),
            "position_ratio": float(env_result.position_ratio),
            "atr_multiplier": float(env_result.atr_multiplier),
            "risk_level": env_result.risk_level,
        },
        "signals": []
    }
    for r in sorted(results, key=lambda x: x['total_score'], reverse=True):
        output["signals"].append({
            "code": r['code'],
            "name": r['name'],
            "total_score": float(r['total_score']),
            "status": r['status'],
            "rps": float(r.get('rps', 0)),
            "price": round(float(r['price']), 3) if pd.notna(r['price']) else 0.0,
            "stop_loss": round(float(r['stop_loss']), 3),
            "stop_dist_pct": round(float(r['stop_dist']), 1),
            "tags": r['tags'],
            "monthly_score": float(r['monthly_score']),
            "weekly_score": float(r['weekly_score']),
            "daily_score": float(r['daily_score']),
        })
    try:
        with open(ETF_SIGNALS_LATEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存ETF信号失败: {e}")


# ====================== 核心启动逻辑 ======================
def main():
    codes = [
        '159326', '512400', '512480', '512880', '159206', '159870', '515880', '159869', '516150',
        '159852', '515220', '159201', '515790', '512660', '159755', '515210', '159611', '512690',
        '512800', '159851', '560710', '159766', '512200', '518880', '513120', '513050', '513520',
        '159941', '159667', '159825', '560280'
    ]
    print(f"🚀 [ETF波段交易雷达] 启动! 标的池数量: {len(codes)}")
    name_map = get_etf_name_map()
    prev_history = load_history()
    # ======== 大盘环境评估 ========
    market_env = MarketEnvironment('510300', '沪深300ETF')
    env_result = market_env.evaluate()
    market_safe = env_result.market_safe
    atr_multiplier = env_result.atr_multiplier
    bm_analyzer = market_env.analyzer
    bm_blended_ret = calc_blended_return(
        bm_analyzer.df_daily) if bm_analyzer and not bm_analyzer.df_daily.empty else 0.0
    print(f"📈 沪深300基准混合收益率: {bm_blended_ret * 100:.2f}%")
    print(f"🛡️ 当前ATR止损倍数: {atr_multiplier}x | 建议仓位: {env_result.position_ratio:.0%}\n")
    analyzers = []
    print("⏳ 正在并发拉取 K线数据...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {
            executor.submit(fetch_single_etf, code, name_map.get(code, f"ETF_{code}"),
                            market_safe, atr_multiplier): code
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
        if res:
            results.append(res)
    if results:
        save_history(results)
        HTMLReporter.generate(results, env_result, "index.html")
        save_etf_signals(results, env_result)
        print(f"📊 量化交易数据已输出: {ETF_SIGNALS_LATEST_FILE}")
        print(f"📊 大盘环境数据已输出: {MARKET_ENV_LATEST_FILE}")
    else:
        print("\n❌ 未获取到有效数据。")


if __name__ == "__main__":
    main()
