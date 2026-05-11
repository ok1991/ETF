#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF实战波段交易雷达 - 行业ETF版
v3.2 增强版

v3.2 改进清单:
  [v3.2-1] 📊 评分变动方向箭头: sparkline 数据不足时显示 ▲/▼ 变动值
  [v3.2-2] 🔍 日线 MACD 顶/底背离检测: 价格新高但 MACD 未新高(反之亦然)

  [保留] v3.1-1~v3.1-4 全部改进
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, fields
from enum import Enum
import os
import json
import concurrent.futures
import time
import traceback
import warnings
import threading

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        Enum 状态类型                         ║
# ╚══════════════════════════════════════════════════════════════╝

class MarketStatus(Enum):
    """大盘环境状态"""
    STRONG_BULL = "强多头"
    BULL = "偏多"
    NEUTRAL = "震荡"
    BEAR = "偏空"
    STRONG_BEAR = "强空头"


class ETFStatus(Enum):
    """ETF波段状态"""
    EXTREME_BULL = "极强波段多头"
    BULL = "波段多头"
    WEAK_BULL = "偏多企稳"
    NEUTRAL = "多空震荡"
    WEAK_BEAR = "偏空走弱"
    BEAR = "波段空头"
    EXTREME_BEAR = "极弱波段空头"

    @property
    def is_bullish(self) -> bool:
        return self in (ETFStatus.EXTREME_BULL, ETFStatus.BULL, ETFStatus.WEAK_BULL)

    @property
    def is_bearish(self) -> bool:
        return self in (ETFStatus.EXTREME_BEAR, ETFStatus.BEAR, ETFStatus.WEAK_BEAR)


def _enum_value(status: Any) -> str:
    """统一获取 Enum 或 str 的字符串值（兼容旧 JSON）"""
    return status.value if isinstance(status, Enum) else str(status)


# ╔══════════════════════════════════════════════════════════════╗
# ║                          配置类                              ║
# ╚══════════════════════════════════════════════════════════════╝

class Config:
    """全局配置 — 路径、阈值、大盘权重"""

    HISTORY_FILE: str = "etf_history_state.json"
    MARKET_ENV_HISTORY_FILE: str = "market_env_history.json"
    MARKET_ENV_LATEST_FILE: str = "market_env_latest.json"
    ETF_SIGNALS_LATEST_FILE: str = "etf_signals_latest.json"
    LOG_FILE: str = "etf_radar.log"

    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    DATA_DIR: str = "etf_data"
    MIN_DATA_POINTS: int = 50

    DEFAULT_INDEX_CODE: str = '510300'
    DEFAULT_INDEX_NAME: str = '沪深300ETF'
    MAX_WORKERS: int = 10

    SCORE_THRESHOLDS: Dict[str, float] = {
        'strong_bull': 4.5,
        'weak_bull': 1.5,
        'neutral_low': -1.5,
        'weak_bear': -4.5,
    }

    ENV_WEIGHTS: Dict[str, float] = {
        'trend': 1.0,
        'momentum': 0.8,
        'volume': 0.6,
        'volatility': 0.4,
    }


class ETFScoringConfig:
    """ETF评分配置"""

    MULTI_PERIOD_WEIGHTS: Dict[str, float] = {
        'monthly': 1.3,
        'weekly': 1.6,
        'daily': 1.0,
    }

    RESONANCE_BONUS: float = 0.6
    DIVERGENCE_PENALTY: float = -1.8

    RSI_OVERBOUGHT: float = 75
    RSI_OVERSOLD: float = 25

    VOLUME_EXPANSION_THRESHOLD: float = 1.3
    VOLUME_CONTRACTION_THRESHOLD: float = 0.7

    SPARKLINE_DAYS: int = 14
    HISTORY_DAILY_SCORES_DAYS: int = 30


# ╔══════════════════════════════════════════════════════════════╗
# ║                    线程安全日志管理器                          ║
# ╚══════════════════════════════════════════════════════════════╝

class Logger:
    """线程安全日志 — 文件用覆盖模式"""

    _lock: threading.Lock = threading.Lock()

    @staticmethod
    def log(level: str, message: str, exception: Optional[Exception] = None) -> None:
        timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message: str = f"[{timestamp}] [{level.upper()}] {message}"
        if exception:
            log_message += f"\n异常详情: {str(exception)}\n{traceback.format_exc()}"
        print(log_message)
        try:
            with Logger._lock:
                with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n\n')
        except Exception:
            pass

    @staticmethod
    def info(message: str) -> None:
        Logger.log('INFO', message)

    @staticmethod
    def warning(message: str, exception: Optional[Exception] = None) -> None:
        Logger.log('WARNING', message, exception)

    @staticmethod
    def error(message: str, exception: Optional[Exception] = None) -> None:
        Logger.log('ERROR', message, exception)


def _init_log_file() -> None:
    try:
        with open(Config.LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== ETF波段交易雷达 日志 ===\n")
            f.write(f"=== 启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    except Exception:
        pass


# ╔══════════════════════════════════════════════════════════════╗
# ║                 MarketEnvResult — 大盘评估结果                ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class MarketEnvResult:
    """大盘环境结构化结果"""
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
    atr_percentile: float
    trend_score: float
    momentum_score: float
    volume_score: float
    volatility_score: float
    total_score: float
    trend_details: dict
    momentum_details: dict
    volume_details: dict
    volatility_details: dict
    status: MarketStatus
    market_safe: bool
    atr_multiplier: float
    position_ratio: float
    risk_level: str
    score_change: float
    status_changed: bool

    def _convert_value(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (np.bool_,)):
            return bool(value)
        elif isinstance(value, (np.integer,)):
            return int(value)
        elif isinstance(value, (np.floating,)):
            return float(value)
        elif isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._convert_value(item) for item in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def to_dict(self) -> dict:
        return {f.name: self._convert_value(getattr(self, f.name)) for f in fields(self)}


# ╔══════════════════════════════════════════════════════════════╗
# ║               DataNormalizer — 列名标准化                     ║
# ╚══════════════════════════════════════════════════════════════╝

class DataNormalizer:
    PRIORITY_MAP: Dict[str, List[str]] = {
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

        existing: Dict[str, str] = {str(c).lower(): c for c in df.columns}
        rename: Dict[str, str] = {}
        drop: set = set()

        for std, aliases in cls.PRIORITY_MAP.items():
            primary: Optional[str] = None
            for alias in aliases:
                if alias.lower() in existing:
                    orig: str = existing[alias.lower()]
                    if primary is None:
                        primary = orig
                        if orig != std:
                            rename[orig] = std
                    else:
                        drop.add(orig)

        df = df.drop(columns=list(drop)).rename(columns=rename)
        required: List[str] = ['date', 'open', 'high', 'low', 'close']
        missing: List[str] = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"缺少核心列: {missing}")
        df['date'] = pd.to_datetime(df['date'])
        if 'amount' in df.columns:
            df['volume'] = df['amount']
            df = df.drop(columns=['amount'])
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        return df


# ╔══════════════════════════════════════════════════════════════╗
# ║           TechnicalIndicators — 统一标准参数                  ║
# ╚══════════════════════════════════════════════════════════════╝

class TechnicalIndicators:
    """技术指标 — 全部使用业界标准默认参数"""

    @staticmethod
    def ma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        for p in periods:
            df[f'MA{p}'] = df['close'].rolling(window=p).mean()
        return df

    @staticmethod
    def ma_slope(df: pd.DataFrame, period: int = 20, lookback: int = 3) -> pd.DataFrame:
        if f'MA{period}' not in df.columns:
            return df
        ma_col: pd.Series = df[f'MA{period}']
        shifted: pd.Series = ma_col.shift(lookback)
        df[f'MA{period}_slope'] = np.where(
            shifted != 0, (ma_col - shifted) / shifted * 100, 0.0
        )
        return df

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        ema_fast: pd.Series = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow: pd.Series = df['close'].ewm(span=slow, adjust=False).mean()
        dif: pd.Series = ema_fast - ema_slow
        df['MACD_DIF'] = dif
        df['MACD_DEA'] = dif.ewm(span=signal, adjust=False).mean()
        df['MACD_hist'] = dif - df['MACD_DEA']
        return df

    @staticmethod
    def boll(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        df['BOLL_mid'] = df['close'].rolling(window=period).mean()
        std: pd.Series = df['close'].rolling(window=period).std()
        df['BOLL_upper'] = df['BOLL_mid'] + std_dev * std
        df['BOLL_lower'] = df['BOLL_mid'] - std_dev * std
        df['BOLL_bw'] = np.where(
            df['BOLL_mid'] > 0,
            (df['BOLL_upper'] - df['BOLL_lower']) / df['BOLL_mid'], 0.0
        )
        band_width: pd.Series = df['BOLL_upper'] - df['BOLL_lower']
        df['BOLL_pctB'] = np.where(
            band_width > 0,
            (df['close'] - df['BOLL_lower']) / band_width,
            0.5
        )
        return df

    @staticmethod
    def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df['VMA'] = df['volume'].rolling(window=period).mean()
        return df

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        h, l, pc = df['high'], df['low'], df['close'].shift()
        tr: pd.DataFrame = pd.concat([h - l, abs(h - pc), abs(l - pc)], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        return df

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """RSI — Wilder方法，标准参数(14) [v3.1-1] 修复 loss 条件"""
        delta: pd.Series = df['close'].diff()
        gain: pd.Series = delta.where(delta > 0, 0.0)
        loss: pd.Series = (-delta).where(delta < 0, 0.0)
        avg_gain: pd.Series = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss: pd.Series = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs: pd.Series = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df.loc[avg_loss == 0, 'RSI'] = 100.0
        return df

    @staticmethod
    def yesterday_macd_cross(df: pd.DataFrame) -> Optional[str]:
        if 'MACD_hist' not in df.columns or len(df) < 3:
            return None
        h_now: float = df['MACD_hist'].iloc[-1]
        h_prev: float = df['MACD_hist'].iloc[-2]
        if pd.isna(h_now) or pd.isna(h_prev):
            return None
        if h_prev <= 0 < h_now:
            return 'golden_cross'
        if h_prev >= 0 > h_now:
            return 'death_cross'
        return None


# ╔══════════════════════════════════════════════════════════════╗
# ║                 ETFAnalyzer — 核心分析引擎                    ║
# ╚══════════════════════════════════════════════════════════════╝

class ETFAnalyzer:
    """ETF分析器 — 统一标准参数 + 多周期评分 + 动态追踪止损"""

    def __init__(self, code: str, name: str,
                 force_download: bool = False,
                 market_safe: bool = True,
                 atr_multiplier: float = 2.0) -> None:
        self.code: str = code
        self.name: str = name
        self.force_download: bool = force_download
        self.data_dir: str = Config.DATA_DIR
        self.market_safe: bool = market_safe
        self.atr_multiplier: float = atr_multiplier
        self.df_daily: pd.DataFrame = pd.DataFrame()
        self.df_weekly: pd.DataFrame = pd.DataFrame()
        self.df_monthly: pd.DataFrame = pd.DataFrame()
        self.rps: float = 0.0
        self.stop_loss_price: float = 0.0
        self.data_loaded: bool = False
        self.last_error: Optional[Exception] = None
        self.prev_stop: float = 0.0

    # ────────────────── 数据获取 ──────────────────

    def _add_market_prefix(self, code: str) -> str:
        if code.startswith(('5', '6')):
            return f"sh{code}"
        if code.startswith(('1', '0', '3')):
            return f"sz{code}"
        return code

    def _validate_dataframe(self, df: pd.DataFrame, min_rows: int = 50) -> bool:
        if df.empty:
            return False
        if len(df) < min_rows:
            Logger.warning(f"{self.code} 数据量不足: {len(df)} < {min_rows}")
            return False
        if not all(c in df.columns for c in ['date', 'open', 'high', 'low', 'close']):
            Logger.error(f"{self.code} 缺少必要列")
            return False
        return True

    def fetch_data(self, max_retries: int = Config.MAX_RETRIES) -> bool:
        for attempt in range(max_retries):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
                today_str: str = datetime.now().strftime('%Y%m%d')
                df: pd.DataFrame = pd.DataFrame()

                existing: List[str] = sorted(
                    [f for f in os.listdir(self.data_dir)
                     if f.startswith(f"{self.code}_") and f.endswith('.csv')],
                    reverse=True
                )

                if not self.force_download and existing:
                    if today_str in existing[0]:
                        try:
                            df = pd.read_csv(os.path.join(self.data_dir, existing[0]),
                                             parse_dates=['date'])
                            if self._validate_dataframe(df):
                                self.df_daily = df.sort_values('date').reset_index(drop=True)
                                self._resample_data()
                                self.data_loaded = True
                                Logger.info(f"{self.code} 本地今日文件加载成功")
                                return True
                        except Exception as e:
                            Logger.warning(f"{self.code} 读取今日文件失败", e)

                    try:
                        df = pd.read_csv(os.path.join(self.data_dir, existing[0]),
                                         parse_dates=['date'])
                        if self._validate_dataframe(df):
                            self.df_daily = df.sort_values('date').reset_index(drop=True)
                            self._resample_data()
                            self.data_loaded = True
                            Logger.info(f"{self.code} 本地最近文件加载成功")
                            return True
                    except Exception as e:
                        Logger.warning(f"{self.code} 读取最近文件失败", e)

                Logger.info(f"{self.code} 网络获取 ({attempt + 1}/{max_retries})")
                df_net = ak.stock_zh_a_hist_tx(
                    symbol=self._add_market_prefix(self.code), adjust="qfq"
                )

                if df_net is not None and not df_net.empty:
                    df = DataNormalizer.normalize(df_net).sort_values('date').reset_index(drop=True)
                    if self._validate_dataframe(df, Config.MIN_DATA_POINTS):
                        new_file: str = f"{self.code}_{df['date'].iloc[-1].strftime('%Y%m%d')}.csv"
                        df.to_csv(os.path.join(self.data_dir, new_file),
                                  index=False, encoding='utf-8-sig')
                        self._cleanup_old_files(new_file, existing)
                        self.df_daily = df
                        self._resample_data()
                        self.data_loaded = True
                        Logger.info(f"{self.code} 网络获取成功")
                        return True

                if not self.force_download and existing:
                    try:
                        df = pd.read_csv(os.path.join(self.data_dir, existing[0]),
                                         parse_dates=['date'])
                        if self._validate_dataframe(df):
                            self.df_daily = df.sort_values('date').reset_index(drop=True)
                            self._resample_data()
                            self.data_loaded = True
                            Logger.warning(f"{self.code} 使用旧文件")
                            return True
                    except Exception:
                        pass

            except Exception as e:
                self.last_error = e
                Logger.error(f"{self.code} 尝试{attempt + 1}失败", e)
                if attempt < max_retries - 1:
                    time.sleep(Config.RETRY_DELAY * (attempt + 1))

        Logger.error(f"{self.code} 数据获取全部失败")
        return False

    _file_cleanup_lock: threading.Lock = threading.Lock()

    def _cleanup_old_files(self, current_file: str, existing_files: List[str]) -> None:
        with ETFAnalyzer._file_cleanup_lock:
            for f_name in existing_files:
                if f_name != current_file:
                    try:
                        os.remove(os.path.join(self.data_dir, f_name))
                    except (FileNotFoundError, Exception):
                        pass

    def _resample_data(self) -> None:
        if self.df_daily.empty:
            return
        self.df_weekly = self._resample('W-FRI')
        try:
            self.df_monthly = self._resample('ME')
        except (ValueError, KeyError):
            try:
                self.df_monthly = self._resample('M')
            except Exception:
                self.df_monthly = pd.DataFrame()

    def _resample(self, freq: str) -> pd.DataFrame:
        if self.df_daily.empty:
            return pd.DataFrame()
        df: pd.DataFrame = self.df_daily.set_index('date')
        agg: Dict[str, str] = {
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }
        return df.resample(freq).agg(agg).dropna(subset=['close']).reset_index()

    def get_data_date(self) -> Optional[str]:
        if self.df_daily.empty:
            return None
        return self.df_daily['date'].iloc[-1].strftime('%Y-%m-%d')

    def is_data_stale(self, max_age_days: int = 2) -> bool:
        if self.df_daily.empty:
            return True
        last = self.df_daily['date'].iloc[-1]
        return (datetime.now() - last).days > max_age_days

    # ────────────────── 指标计算 ──────────────────

    def calculate_indicators(self) -> None:
        if not self.data_loaded:
            return

        TechnicalIndicators.ma(self.df_monthly, [5, 10])
        TechnicalIndicators.ma_slope(self.df_monthly, period=5, lookback=3)
        TechnicalIndicators.macd(self.df_monthly)

        TechnicalIndicators.ma(self.df_weekly, [5, 10, 20])
        TechnicalIndicators.ma_slope(self.df_weekly, period=20, lookback=3)
        TechnicalIndicators.macd(self.df_weekly)

        TechnicalIndicators.boll(self.df_daily)
        TechnicalIndicators.macd(self.df_daily)
        TechnicalIndicators.volume_ma(self.df_daily)
        TechnicalIndicators.atr(self.df_daily)
        TechnicalIndicators.rsi(self.df_daily)

        self._calc_trailing_stop()

    def _calc_trailing_stop(self) -> None:
        if len(self.df_daily) < 20:
            self.stop_loss_price = 0.0
            return

        highest_20: float = self.df_daily['high'].rolling(20).max().iloc[-1]
        atr_val: float = self._get_value(self.df_daily, 'ATR')
        current_price: float = self._get_value(self.df_daily, 'close')

        if pd.isna(highest_20) or pd.isna(atr_val) or atr_val <= 0:
            self.stop_loss_price = 0.0
            return

        raw_stop: float = highest_20 - self.atr_multiplier * atr_val

        supports: List[float] = [raw_stop]
        ma20_val: float = self._get_value(self.df_daily, 'MA20')
        boll_lower: float = self._get_value(self.df_daily, 'BOLL_lower')

        if pd.notna(ma20_val) and ma20_val > 0:
            supports.append(ma20_val)
        if pd.notna(boll_lower) and boll_lower > 0:
            supports.append(boll_lower)

        if pd.notna(current_price) and current_price > 0:
            valid: List[float] = [s for s in supports if s < current_price]
            base_stop: float = max(valid) if valid else raw_stop
        else:
            base_stop = raw_stop

        if self.prev_stop > 0 and base_stop < self.prev_stop:
            if pd.notna(current_price) and current_price > self.prev_stop:
                self.stop_loss_price = self.prev_stop
                return

        self.stop_loss_price = base_stop

    # ────────────────── 工具方法 ──────────────────

    def _get_value(self, df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return np.nan
        v = df[col].iloc[-1]
        return np.nan if pd.isna(v) else float(v)

    @staticmethod
    def _s_format(value: float, precision: int = 3) -> str:
        if pd.isna(value):
            return 'N/A'
        try:
            return f'{value:.{precision}f}'
        except Exception:
            return str(value)

    def _log_details(self, ms: float, ws: float, ds: float, daily_reason: str = "") -> None:
        pm: float = self._get_value(self.df_monthly, 'close')
        hm: float = self._get_value(self.df_monthly, 'MACD_hist')
        Logger.info(
            f"   [月线] Score={ms:>5.1f} | Price={self._s_format(pm)} | "
            f"MA5/10={self._s_format(self._get_value(self.df_monthly, 'MA5'))}/"
            f"{self._s_format(self._get_value(self.df_monthly, 'MA10'))} | "
            f"Hist={self._s_format(hm, 3)}"
        )

        pw: float = self._get_value(self.df_weekly, 'close')
        hw: float = self._get_value(self.df_weekly, 'MACD_hist')
        sw: float = self._get_value(self.df_weekly, 'MA20_slope')
        Logger.info(
            f"   [周线] Score={ws:>5.1f} | Price={self._s_format(pw)} | "
            f"MA5/20={self._s_format(self._get_value(self.df_weekly, 'MA5'))}/"
            f"{self._s_format(self._get_value(self.df_weekly, 'MA20'))} | "
            f"Hist={self._s_format(hw, 3)} | Slope={self._s_format(sw, 2)}%"
        )

        pd_: float = self._get_value(self.df_daily, 'close')
        bu: float = self._get_value(self.df_daily, 'BOLL_upper')
        bl: float = self._get_value(self.df_daily, 'BOLL_lower')
        hd: float = self._get_value(self.df_daily, 'MACD_hist')
        pct_b: float = self._get_value(self.df_daily, 'BOLL_pctB')
        vol: float = self._get_value(self.df_daily, 'volume')
        vma: float = self._get_value(self.df_daily, 'VMA')
        vr: float = vol / vma if pd.notna(vma) and vma > 0 and pd.notna(vol) else 0.0
        Logger.info(
            f"   [日线] Score={ds:>5.1f} | Price={self._s_format(pd_)} | "
            f"BOLL={self._s_format(bl)}~{self._s_format(bu)} | "
            f"%B={self._s_format(pct_b, 2)} | "
            f"Hist={self._s_format(hd, 3)} | Vol={vr:.2f}x"
        )

    # ────────────────── MACD柱状图趋势 ──────────────────

    def _macd_hist_trend_adjustment(self) -> Tuple[float, str]:
        if 'MACD_hist' not in self.df_daily.columns or len(self.df_daily) < 5:
            return 0.0, ""
        recent: np.ndarray = self.df_daily['MACD_hist'].tail(4).values
        if any(pd.isna(v) for v in recent):
            return 0.0, ""

        expanding_up: bool = all(
            recent[i] > 0 and recent[i] > recent[i - 1]
            for i in range(1, len(recent))
        )
        expanding_down: bool = all(
            recent[i] < 0 and recent[i] < recent[i - 1]
            for i in range(1, len(recent))
        )
        contracting: bool = (
                len(recent) >= 2 and (
                (recent[-1] > 0 and recent[-1] < recent[-2]) or
                (recent[-1] < 0 and recent[-1] > recent[-2])
        )
        )

        if expanding_up:
            return 0.3, " 红柱放大"
        if expanding_down:
            return -0.3, " 绿柱放大"
        if contracting:
            if recent[-1] > 0:
                return -0.1, " 红柱缩小"
            return 0.1, " 绿柱缩小"
        return 0.0, ""

    # ────────────────── 量价背离检测 ──────────────────

    def _volume_price_divergence_adjustment(self) -> Tuple[float, str]:
        if len(self.df_daily) < 10:
            return 0.0, ""
        recent: pd.DataFrame = self.df_daily.tail(6)
        prices: np.ndarray = recent['close'].values
        volumes: np.ndarray = recent['volume'].values

        if any(pd.isna(v) for v in prices) or any(pd.isna(v) for v in volumes):
            return 0.0, ""
        if np.std(volumes) < 1e-10:
            return 0.0, ""

        price_slope: float = float(np.polyfit(range(len(prices)), prices, 1)[0])
        vol_slope: float = float(np.polyfit(range(len(volumes)), volumes, 1)[0])

        if price_slope > 0 and vol_slope < 0:
            return -0.4, " 量价背离"
        if price_slope < 0 and vol_slope > 0:
            return -0.3, " 放量下跌"
        return 0.0, ""

    # ────────────────── RSI调整 ──────────────────

    @staticmethod
    def _rsi_adjustment(rsi: float) -> Tuple[float, str]:
        if pd.isna(rsi):
            return 0.0, ""
        if rsi > ETFScoringConfig.RSI_OVERBOUGHT:
            return -0.5, " RSI超买"
        if rsi < ETFScoringConfig.RSI_OVERSOLD:
            return 0.4, " RSI超卖"
        return 0.0, ""

    # ────────────────── [v3.2-2] 日线 MACD 顶/底背离 ──────────────────

    def _macd_divergence_adjustment(self) -> Tuple[float, str]:
        """[v3.2-2] 检测日线级别 MACD 顶/底背离

        原理:
        - 顶背离: 价格创新高但对应位置的 MACD 柱未创新高 → 上涨动能衰竭
        - 底背离: 价格创新低但对应位置的 MACD 柱未创新低 → 下跌动能衰竭

        方法: 将最近 25 天分为前半段/后半段，比较两段各自价格极值
        对应位置的 MACD 柱值。
        """
        if 'MACD_hist' not in self.df_daily.columns or len(self.df_daily) < 25:
            return 0.0, ""

        tail: pd.DataFrame = self.df_daily.tail(25)
        close_arr: np.ndarray = tail['close'].astype(float).values
        hist_arr: np.ndarray = tail['MACD_hist'].astype(float).values

        if any(pd.isna(v) for v in hist_arr):
            return 0.0, ""

        n: int = len(close_arr)
        mid: int = n // 2

        # 前半段 / 后半段
        c1: np.ndarray = close_arr[:mid]
        c2: np.ndarray = close_arr[mid:]
        h1: np.ndarray = hist_arr[:mid]
        h2: np.ndarray = hist_arr[mid:]

        # ── 顶背离 ──
        # 后段最高价 > 前段最高价，但后段最高价处的 MACD < 前段最高价处的 MACD
        i1: int = int(np.argmax(c1))
        i2: int = int(np.argmax(c2))
        if c2[i2] > c1[i1] and h2[i2] < h1[i1] and h2[i2] > 0:
            # h2[i2] > 0: 还在零轴上方，背离才有意义
            return -0.5, " MACD顶背离"

        # ── 底背离 ──
        # 后段最低价 < 前段最低价，但后段最低价处的 MACD > 前段最低价处的 MACD
        j1: int = int(np.argmin(c1))
        j2: int = int(np.argmin(c2))
        if c2[j2] < c1[j1] and h2[j2] > h1[j1] and h2[j2] < 0:
            # h2[j2] < 0: 还在零轴下方，背离才有意义
            return 0.5, " MACD底背离"

        return 0.0, ""

    # ────────────────── 月线分析 ──────────────────

    def _analyze_monthly(self) -> Tuple[float, str]:
        ma5: float = self._get_value(self.df_monthly, 'MA5')
        ma10: float = self._get_value(self.df_monthly, 'MA10')
        price: float = self._get_value(self.df_monthly, 'close')
        hist: float = self._get_value(self.df_monthly, 'MACD_hist')
        slope: float = self._get_value(self.df_monthly, 'MA5_slope')

        if pd.isna(ma10) or pd.isna(hist):
            if pd.notna(slope):
                if slope > 3:
                    return 1.5, "数据有限但MA5上行"
                if slope < -3:
                    return -1.5, "数据有限但MA5下行"
            return 0.0, "数据不足"

        ma_gap: float = (ma5 - ma10) / ma10 * 100 if ma10 != 0 else 0
        price_dev: float = (price - ma5) / ma5 * 100 if pd.notna(ma5) and ma5 != 0 else 0

        hist_prev: float = np.nan
        if len(self.df_monthly) > 1:
            hist_prev = self._get_value(self.df_monthly.iloc[:-1], 'MACD_hist')
        hist_dir: int = 0
        if pd.notna(hist) and pd.notna(hist_prev):
            hist_dir = 1 if hist > hist_prev else (-1 if hist < hist_prev else 0)

        is_consolidation: bool = abs(ma_gap) < 1.5 and abs(price_dev) < 2.5

        if ma5 > ma10:
            if price > ma5 and hist > 0:
                s: float = min(5.0, 3.0 + abs(ma_gap) * 0.15 + abs(hist) * 0.3)
                if ma_gap > 4.0 and abs(hist) > 0.8 and price_dev > 2.0:
                    return s, f"极强多头(间距{ma_gap:.1f}%)"
                if hist_dir > 0:
                    return s * 0.85, f"强多头(间距{ma_gap:.1f}%)"
                return s * 0.7, f"多头趋势(间距{ma_gap:.1f}%)"
            if price < ma5 and hist > 0:
                if price > ma10:
                    return (2.0 if hist_dir >= 0 else 1.5), "多头回调"
                return 0.5, "深度回调(跌破MA10)"
            if price > ma5 and hist < 0:
                return 1.0 if hist_dir < 0 else 1.5, "动能衰减"
            if ma10 < price < ma5:
                g: float = min(2.5, 1.0 + abs(ma_gap) * 0.2) if hist > 0 else 0.3
                return g, ("中继" if hist > 0 else "挣扎")
            if is_consolidation:
                if hist > 0.2:
                    return 0.8, "偏多蓄力"
                if hist < -0.2:
                    return -0.8, "偏空蓄力"
                return 0.0, "横盘"

        if ma5 < ma10:
            if price < ma5 and hist < 0:
                s = max(-5.0, -3.0 - abs(ma_gap) * 0.15 - abs(hist) * 0.3)
                if ma_gap < -4.0 and abs(hist) > 0.8 and price_dev < -2.0:
                    return s, f"极强空头(间距{ma_gap:.1f}%)"
                if hist_dir < 0:
                    return s * 0.85, f"强空头(间距{ma_gap:.1f}%)"
                return s * 0.7, f"空头趋势(间距{ma_gap:.1f}%)"
            if price > ma5 and hist < 0:
                if price > ma10:
                    return 1.0, "强反弹(突破MA10)"
                return -1.5, "弱势反弹"
            if price < ma5 and hist > 0:
                if price > ma10:
                    return 1.0, "趋势转变"
                return -0.5, "底部信号"
            if is_consolidation:
                if hist > 0.2:
                    return 0.8, "偏多蓄力"
                if hist < -0.2:
                    return -0.8, "偏空蓄力"
                return 0.0, "横盘"

        if ma_gap > 1.5:
            a: float = min(3.0, 1.0 + abs(ma_gap) * 0.15 + abs(hist) * 0.2)
            if price_dev > 5:
                a *= 0.8
            return a, f"月线上行(间距{ma_gap:.1f}%)"
        if ma_gap < -1.5:
            return max(-3.0, -1.0 - abs(ma_gap) * 0.15 - abs(hist) * 0.2), f"月线下行(间距{ma_gap:.1f}%)"
        if hist > 0.2:
            return 0.5, "月线偏多"
        if hist < -0.2:
            return -0.5, "月线偏空"
        return 0.0, "月线震荡"

    # ────────────────── 周线分析 ──────────────────

    def _analyze_weekly(self) -> Tuple[float, str]:
        ma5: float = self._get_value(self.df_weekly, 'MA5')
        ma10: float = self._get_value(self.df_weekly, 'MA10')
        ma20: float = self._get_value(self.df_weekly, 'MA20')
        slope: float = self._get_value(self.df_weekly, 'MA20_slope')
        price: float = self._get_value(self.df_weekly, 'close')
        hist: float = self._get_value(self.df_weekly, 'MACD_hist')

        if pd.isna(ma20) or pd.isna(slope):
            return 0.0, "数据不足"

        if abs(slope) < 1.2 and pd.notna(price) and ma20 > 0:
            if abs((price - ma20) / ma20) < 0.05:
                return 0.0, "震荡：20周线走平+价格纠缠"

        strong: bool = pd.notna(price) and price > ma20 and slope > 0.3
        weak: bool = pd.notna(price) and price < ma20 and slope < -0.4
        bull_ma: bool = pd.notna(ma5) and pd.notna(ma10) and ma5 > ma10 > ma20 * 0.98
        bear_ma: bool = pd.notna(ma5) and pd.notna(ma10) and ma5 < ma10 < ma20 * 1.02

        if bull_ma and strong and hist > 0:
            return 6.0, "极强多：多头排列+斜率向上+MACD金叉"
        if strong and hist < 0:
            return 2.0, "多头回调：趋势向上但MACD死叉"
        if strong and hist > 0:
            return 4.5, "多头趋势：站稳20周线+MACD正"
        if bear_ma and weak and hist < 0 and slope < -0.7:
            return -6.0, "极弱空：空头排列+发散"
        if weak and hist > 0:
            return -2.5, "空头反弹(警惕诱多)"
        if weak and hist < 0:
            return -4.5, "空头趋势：跌破20周线"
        if pd.notna(price) and pd.notna(ma20):
            return (0.5 if price > ma20 else -0.5), "弱势震荡"
        return 0.0, "数据不足"

    # ────────────────── 日线分析 ──────────────────

    def _analyze_daily(self) -> Tuple[float, str]:
        """日线 — BOLL + MACD + RSI + %B + MACD趋势 + 量价背离 + MACD背离"""
        price: float = self._get_value(self.df_daily, 'close')
        mid: float = self._get_value(self.df_daily, 'BOLL_mid')
        upper: float = self._get_value(self.df_daily, 'BOLL_upper')
        lower: float = self._get_value(self.df_daily, 'BOLL_lower')
        pct_b: float = self._get_value(self.df_daily, 'BOLL_pctB')
        hist: float = self._get_value(self.df_daily, 'MACD_hist')
        vol: float = self._get_value(self.df_daily, 'volume')
        vma: float = self._get_value(self.df_daily, 'VMA')
        rsi: float = self._get_value(self.df_daily, 'RSI')

        if any(pd.isna(x) for x in [upper, lower, mid, price]):
            return 0.0, "数据不完整"
        if mid <= 0:
            return 0.0, "中轨异常"

        vr: float = vol / vma if pd.notna(vma) and vma > 0 and pd.notna(vol) else 0.0

        base_score: float = 0.0
        base_reason: str = ""

        bbw: float = (upper - lower) / mid

        if bbw < 0.06:
            bias: float = 0.3 if pd.notna(hist) and hist > 0 else (
                -0.3 if pd.notna(hist) and hist < 0 else 0
            )
            base_score = 0.5 + bias
            base_reason = "蓄力：布林收口"

        elif price >= upper * 0.99:
            if vr > ETFScoringConfig.VOLUME_EXPANSION_THRESHOLD:
                base_score, base_reason = 3.0, "真突破：放量上轨"
            else:
                base_score, base_reason = 1.0, "滞涨：缩量上轨"

        elif price <= lower * 1.015:
            if vr < ETFScoringConfig.VOLUME_CONTRACTION_THRESHOLD:
                base_score, base_reason = 2.8, "极佳洗盘：缩量下轨"
            else:
                base_score, base_reason = -3.0, "真破位：放量下轨"

        elif abs((price - mid) / mid) < 0.015:
            if hist > 0:
                base_score, base_reason = 1.3, "企稳中轨"
            elif hist < -0.1:
                base_score, base_reason = -1.6, "走弱中轨"
            else:
                base_score, base_reason = 0.0, "盘整中轨"

        elif pd.notna(pct_b):
            boll_pos: float = float(np.clip((pct_b - 0.5) * 2.0, -1.5, 1.5))
            hist_dir_adj: float = (
                0.3 if pd.notna(hist) and hist > 0
                else (-0.3 if pd.notna(hist) and hist < 0 else 0.0)
            )
            base_score = boll_pos + hist_dir_adj
            base_reason = f"{'多头' if pct_b > 0.5 else '空头'}通道(%B={pct_b:.2f})"

        else:
            if mid < price < upper:
                base_score, base_reason = 0.8, "多头通道"
            elif lower < price < mid:
                base_score, base_reason = -0.8, "空头通道"
            else:
                base_score, base_reason = 0.0, "震荡"

        rsi_adj: float
        rsi_tag: str
        rsi_adj, rsi_tag = self._rsi_adjustment(rsi)

        macd_adj: float
        macd_tag: str
        macd_adj, macd_tag = self._macd_hist_trend_adjustment()

        vp_adj: float
        vp_tag: str
        vp_adj, vp_tag = self._volume_price_divergence_adjustment()

        # [v3.2-2] 日线 MACD 顶/底背离
        div_adj: float
        div_tag: str
        div_adj, div_tag = self._macd_divergence_adjustment()

        total: float = base_score + rsi_adj + macd_adj + vp_adj + div_adj

        reason: str = base_reason
        for tag in [rsi_tag, macd_tag, vp_tag, div_tag]:
            if tag:
                reason += tag

        return total, reason

    # ────────────────── 共振/背离 ──────────────────

    def _apply_resonance_and_conflict(
            self, m: float, w: float, d: float, daily_reason: str
    ) -> Tuple[float, float, List[str]]:
        bonus: float = 0.0
        penalty: float = 0.0
        tags: List[str] = []
        if m > 0 and w > 0 and d > 0:
            bonus += min(m, w, d) * ETFScoringConfig.RESONANCE_BONUS
            tags.append("📈 三周期共振")
        if w >= 4.0 and d <= -1.0:
            penalty += ETFScoringConfig.DIVERGENCE_PENALTY
            tags.append("⚠️ 周日顶背离")
        if w <= -4.0 and d >= 1.0:
            penalty += ETFScoringConfig.DIVERGENCE_PENALTY * 0.8
            tags.append("⚠️ 周日底背离(诱多)")
        return bonus, penalty, tags

    def _generate_tags(
            self, m: float, w: float, d: float, total: float,
            prev_score: Optional[float], stop_dist: float, daily_reason: str
    ) -> Tuple[List[str], float]:
        tags: List[str] = []
        if total >= 15.0 and self.rps >= 80:
            tags.append("👑 领涨龙头")
        elif total >= 12.0:
            tags.append("🚀 主升浪")
        elif total <= -12.0:
            tags.append("❄️ 主跌崩盘")
        if "极佳洗盘" in daily_reason and w >= 4.0:
            tags.append("💎 黄金坑低吸")
        if stop_dist < 0:
            tags.append("🚨 破位止损离场")
        if prev_score is not None and prev_score <= 0 and total >= 10.0:
            tags.append("🔥 底部拐点")
        return tags, total

    # ────────────────── 主分析入口 ──────────────────

    def analyze(self, prev_score: Optional[float] = None,
                prev_stop: float = 0.0) -> Optional[Dict[str, Any]]:
        try:
            if not self.data_loaded:
                Logger.error(f"{self.code} 数据未加载")
                return None

            self.prev_stop = prev_stop
            self.calculate_indicators()

            ms, mr = self._analyze_monthly()
            ws, wr = self._analyze_weekly()
            ds, dr = self._analyze_daily()

            w: Dict[str, float] = ETFScoringConfig.MULTI_PERIOD_WEIGHTS
            raw: float = ms * w['monthly'] + ws * w['weekly'] + ds * w['daily']

            bonus, penalty, extra = self._apply_resonance_and_conflict(ms, ws, ds, dr)

            final: float = round(raw + bonus + penalty, 1)
            status: ETFStatus = self._determine_status(final)
            price: float = self._get_value(self.df_daily, 'close')

            stop_dist: float = 0.0
            if pd.notna(price) and price > 0 and self.stop_loss_price > 0:
                stop_dist = (price - self.stop_loss_price) / price * 100

            tags, final = self._generate_tags(ms, ws, ds, final, prev_score, stop_dist, dr)
            all_tags: List[str] = extra + tags

            Logger.info(f"▶️ {self.name} ({self.code})")
            self._log_details(ms, ws, ds, dr)
            tag_str: str = f" → [{', '.join(all_tags)}]" if all_tags else ""
            Logger.info(
                f"   └── 📊 RPS:{self.rps:>5.1f} | 总分:{final:>5.1f} | "
                f"止损距:{stop_dist:>.1f}% → {status.value}{tag_str}\n"
            )

            return {
                "code": self.code,
                "name": self.name,
                "monthly_score": ms,
                "weekly_score": ws,
                "daily_score": ds,
                "total_score": final,
                "status": status,
                "tags": all_tags,
                "monthly_reason": mr,
                "weekly_reason": wr,
                "daily_reason": dr,
                "price": price,
                "stop_loss": self.stop_loss_price,
                "rps": self.rps,
                "stop_dist": stop_dist,
                "data_date": self.get_data_date(),
                "is_stale": self.is_data_stale(),
            }
        except Exception as e:
            Logger.error(f"{self.code} 分析失败", e)
            return None

    @staticmethod
    def _determine_status(score: float) -> ETFStatus:
        if score >= 14.0:
            return ETFStatus.EXTREME_BULL
        if score >= 7.0:
            return ETFStatus.BULL
        if score >= 2.5:
            return ETFStatus.WEAK_BULL
        if score <= -14.0:
            return ETFStatus.EXTREME_BEAR
        if score <= -7.0:
            return ETFStatus.BEAR
        if score <= -2.5:
            return ETFStatus.WEAK_BEAR
        return ETFStatus.NEUTRAL


# ╔══════════════════════════════════════════════════════════════╗
# ║              MarketEnvironment — 大盘环境评估                 ║
# ╚══════════════════════════════════════════════════════════════╝

class MarketEnvironment:
    """大盘四维评估 — 趋势/动能/量价/波动"""

    def __init__(self, index_code: str = Config.DEFAULT_INDEX_CODE,
                 index_name: str = Config.DEFAULT_INDEX_NAME,
                 force_download: bool = False) -> None:
        self.index_code: str = index_code
        self.index_name: str = index_name
        self.force_download: bool = force_download
        self.analyzer: Optional[ETFAnalyzer] = None
        self.result: Optional[MarketEnvResult] = None

    def evaluate(self) -> MarketEnvResult:
        Logger.info(f"🌐 大盘风控 ({self.index_name} - {self.index_code})...")

        self.analyzer = ETFAnalyzer(
            self.index_code, self.index_name,
            force_download=self.force_download, market_safe=True
        )

        if not self.analyzer.fetch_data():
            Logger.warning("⚠️ 大盘数据失败，防守模式")
            self.result = self._default_danger("数据获取失败")
            self._save()
            return self.result

        TechnicalIndicators.ma(self.analyzer.df_daily, [20, 60, 120])
        TechnicalIndicators.macd(self.analyzer.df_daily)
        TechnicalIndicators.volume_ma(self.analyzer.df_daily)
        TechnicalIndicators.atr(self.analyzer.df_daily)

        df: pd.DataFrame = self.analyzer.df_daily
        if df.empty or len(df) < 120:
            Logger.warning("⚠️ 数据不足120日，防守模式")
            self.result = self._default_danger("数据不足")
            self._save()
            return self.result

        a: ETFAnalyzer = self.analyzer
        price: float = a._get_value(df, 'close')
        ma20: float = a._get_value(df, 'MA20')
        ma60: float = a._get_value(df, 'MA60')
        ma120: float = a._get_value(df, 'MA120')
        macd_hist: float = a._get_value(df, 'MACD_hist')
        vol: float = a._get_value(df, 'volume')
        vma: float = a._get_value(df, 'VMA')
        atr: float = a._get_value(df, 'ATR')

        vs_ma20: float = (price - ma20) / ma20 * 100 if pd.notna(ma20) and ma20 != 0 and pd.notna(price) else 0.0
        vs_ma60: float = (price - ma60) / ma60 * 100 if pd.notna(ma60) and ma60 != 0 and pd.notna(price) else 0.0
        vs_ma120: float = (price - ma120) / ma120 * 100 if pd.notna(ma120) and ma120 != 0 and pd.notna(price) else 0.0
        vr: float = vol / vma if pd.notna(vma) and vma > 0 and pd.notna(vol) else 1.0
        atr_pct: float = atr / price * 100 if pd.notna(atr) and pd.notna(price) and price > 0 else 0.0
        atr_pctl: float = self._atr_percentile(df)

        ts, tr, td = self._trend(price, ma20, ma60, ma120)
        ms, mr, md = self._momentum(df, macd_hist)
        vs, vrl, vd = self._volume(df, price, vr)
        vols, volrl, vold = self._volatility(atr_pct, atr_pctl)

        total: float = round(
            ts * Config.ENV_WEIGHTS['trend'] + ms * Config.ENV_WEIGHTS['momentum'] +
            vs * Config.ENV_WEIGHTS['volume'] + vols * Config.ENV_WEIGHTS['volatility'], 1
        )

        status: MarketStatus = self._status(total)
        safe: bool = total >= Config.SCORE_THRESHOLDS['weak_bull']
        atm: float = self._atr_mult(total)
        pos: float = self._pos_ratio(total)
        risk: str = self._risk_level(total)

        prev: Optional[dict] = self._last_record()
        sc: float = 0.0
        ch: bool = False
        if prev:
            sc = round(total - prev.get('total_score', 0), 1)
            ch = (status.value != prev.get('status', ''))

        self._log(ts, tr, ms, mr, vs, vrl, vols, volrl, total, status, safe, pos, atm, sc, ch)

        self.result = MarketEnvResult(
            date=datetime.now().strftime('%Y-%m-%d'),
            index_code=self.index_code, index_name=self.index_name,
            price=round(float(price), 3) if pd.notna(price) else 0.0,
            ma20=round(float(ma20), 3) if pd.notna(ma20) else 0.0,
            ma60=round(float(ma60), 3) if pd.notna(ma60) else 0.0,
            ma120=round(float(ma120), 3) if pd.notna(ma120) else 0.0,
            close_vs_ma20_pct=round(float(vs_ma20), 2),
            close_vs_ma60_pct=round(float(vs_ma60), 2),
            close_vs_ma120_pct=round(float(vs_ma120), 2),
            macd_hist=round(float(macd_hist), 4) if pd.notna(macd_hist) else 0.0,
            vol_ratio=round(float(vr), 2),
            atr_pct=round(float(atr_pct), 2),
            atr_percentile=round(float(atr_pctl), 1),
            trend_score=float(ts), momentum_score=float(ms),
            volume_score=float(vs), volatility_score=float(vols),
            total_score=float(total),
            trend_details=td, momentum_details=md,
            volume_details=vd, volatility_details=vold,
            status=status, market_safe=bool(safe),
            atr_multiplier=float(atm), position_ratio=float(pos),
            risk_level=risk,
            score_change=float(sc), status_changed=bool(ch),
        )
        self._save()
        return self.result

    @staticmethod
    def _atr_percentile(df: pd.DataFrame) -> float:
        if 'ATR' not in df.columns or len(df) < 30:
            return 50.0
        s: pd.Series = df['ATR'].dropna()
        if len(s) < 20:
            return 50.0
        lb: int = min(120, len(s))
        recent: pd.Series = s.iloc[-lb:]
        cur: float = recent.iloc[-1]
        if pd.isna(cur):
            return 50.0
        return float((recent < cur).sum() / len(recent) * 100)

    def _trend(self, price: float, ma20: float, ma60: float, ma120: float
               ) -> Tuple[float, str, dict]:
        h20: bool = pd.notna(ma20) and ma20 != 0
        h60: bool = pd.notna(ma60) and ma60 != 0
        h120: bool = pd.notna(ma120) and ma120 != 0
        d: dict = {
            'above_ma20': bool(price > ma20) if h20 else False,
            'above_ma60': bool(price > ma60) if h60 else False,
            'above_ma120': bool(price > ma120) if h120 else False,
            'ma20_above_ma60': bool(ma20 > ma60) if (h20 and h60) else False,
            'ma60_above_ma120': bool(ma60 > ma120) if (h60 and h120) else False,
            'bullish_alignment': False, 'bearish_alignment': False
        }
        if h20 and h60 and h120:
            if price > ma20 > ma60 > ma120:
                d['bullish_alignment'] = True
                return 4.0, "完美多头排列", d
            if price > ma60 > ma120 and price > ma20:
                d['bullish_alignment'] = True
                return 3.5, "强多头", d
            if price > ma60 > ma120:
                return 2.5, "中期多头(跌破MA20)", d
            if price > ma20 and price > ma60:
                return 2.0, "短期偏多", d
            if price > ma20:
                return (0.0 if abs(price - ma20) / ma20 * 100 < 1.5 else 1.0), "纠缠/偏多", d
            if price < ma20 < ma60 < ma120:
                d['bearish_alignment'] = True
                return -4.0, "完美空头排列", d
            if price < ma60 < ma120 and price < ma20:
                d['bearish_alignment'] = True
                return -3.5, "强空头", d
            if price < ma60 < ma120:
                return -2.5, "中期空头", d
            if price < ma20 and price < ma60:
                return -2.0, "短期偏空", d
            if price < ma20:
                return (0.0 if abs(price - ma20) / ma20 * 100 < 1.5 else -1.0), "纠缠/偏空", d
            return 0.0, "均线交叉区", d
        if h20:
            return (1.0 if price > ma20 else -1.0 if price < ma20 else 0.0), "MA20判断", d
        return 0.0, "趋势数据不足", d

    def _momentum(self, df: pd.DataFrame, hist: float) -> Tuple[float, str, dict]:
        d: dict = {
            'golden_cross': False, 'death_cross': False,
            'hist_positive': False, 'hist_expanding': False, 'hist_shrinking': False
        }
        if pd.isna(hist) or len(df) < 5:
            return 0.0, "动能不足", d
        d['hist_positive'] = bool(hist > 0)
        cross: Optional[str] = TechnicalIndicators.yesterday_macd_cross(df)
        if cross == 'golden_cross':
            d['golden_cross'] = True
        elif cross == 'death_cross':
            d['death_cross'] = True
        recent: np.ndarray = df['MACD_hist'].tail(4).values
        ep: bool = False
        en: bool = False
        if len(recent) >= 4 and all(pd.notna(v) for v in recent):
            if all(recent[i] > 0 and recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                ep = True
            if all(recent[i] < 0 and recent[i] <= recent[i - 1] for i in range(1, len(recent))):
                en = True
        d['hist_expanding'] = bool(ep or en)
        if len(recent) >= 3 and pd.notna(recent[-1]) and pd.notna(recent[-2]):
            if (recent[-1] > 0 and recent[-1] < recent[-2]) or (recent[-1] < 0 and recent[-1] > recent[-2]):
                d['hist_shrinking'] = True
        if d['golden_cross'] and ep:
            return 3.0, "金叉+红柱放大", d
        if d['golden_cross']:
            return 2.5, "刚金叉", d
        if hist > 0 and ep:
            return 2.0, "红柱放大", d
        if hist > 0:
            return (1.0 if d['hist_shrinking'] else 1.5), "红柱", d
        if d['death_cross'] and en:
            return -3.0, "死叉+绿柱放大", d
        if d['death_cross']:
            return -2.5, "刚死叉", d
        if hist < 0 and en:
            return -2.0, "绿柱放大", d
        if hist < 0:
            return (-1.0 if d['hist_shrinking'] else -1.5), "绿柱", d
        return 0.0, "动能中性", d

    def _volume(self, df: pd.DataFrame, price: float, vr: float) -> Tuple[float, str, dict]:
        d: dict = {
            'vol_ratio': round(float(vr), 2), 'price_up': False, 'price_down': False,
            'high_volume': bool(vr > 1.5), 'low_volume': bool(vr < 0.7), 'price_change_pct': 0.0
        }
        if len(df) < 2:
            return 0.0, "量价不足", d
        prev_price: float = df['close'].iloc[-2]
        if pd.notna(prev_price) and prev_price > 0 and pd.notna(price):
            pc: float = (price - prev_price) / prev_price * 100
            d['price_up'] = bool(pc > 0)
            d['price_down'] = bool(pc < 0)
            d['price_change_pct'] = round(float(pc), 2)
        if d['high_volume'] and d['price_up']:
            return 2.0, "放量上涨", d
        if vr > 1.0 and d['price_up']:
            return 1.0, "温和放量上涨", d
        if d['price_up'] and d['low_volume']:
            return 0.5, "缩量上涨", d
        if d['high_volume'] and d['price_down']:
            return -2.0, "放量下跌", d
        if vr > 1.0 and d['price_down']:
            return -1.0, "放量下跌", d
        if d['price_down'] and d['low_volume']:
            return -0.5, "缩量下跌", d
        return 0.0, "平量盘整", d

    def _volatility(self, atr_pct: float, atr_pctl: float) -> Tuple[float, str, dict]:
        d: dict = {
            'atr_pct': round(float(atr_pct), 2), 'atr_percentile': round(float(atr_pctl), 1),
            'very_low_volatility': bool(atr_pct < 0.5), 'low_volatility': bool(0.5 <= atr_pct < 0.8),
            'normal_volatility': bool(0.8 <= atr_pct <= 1.2), 'high_volatility': bool(1.2 < atr_pct <= 2.0),
            'extreme_volatility': bool(atr_pct > 2.0),
            'percentile_high': bool(atr_pctl > 75), 'percentile_low': bool(atr_pctl < 25)
        }
        b: float
        r: str
        if d['extreme_volatility']:
            b, r = -2.0, "极端波动"
        elif d['high_volatility']:
            b, r = -1.0, "波动偏高"
        elif d['normal_volatility']:
            b, r = 0.0, "正常"
        elif d['low_volatility']:
            b, r = 1.0, "低波动"
        else:
            b, r = 2.0, "极低/蓄力"
        if d['percentile_high'] and b >= 0:
            b -= 0.5
            r += " [历史偏高]"
        elif d['percentile_low'] and b < 0:
            b += 0.5
            r += " [历史偏低]"
        return max(-2.0, min(2.0, b)), r, d

    def _status(self, t: float) -> MarketStatus:
        th: Dict[str, float] = Config.SCORE_THRESHOLDS
        if t >= th['strong_bull']:
            return MarketStatus.STRONG_BULL
        if t >= th['weak_bull']:
            return MarketStatus.BULL
        if t >= th['neutral_low']:
            return MarketStatus.NEUTRAL
        if t >= th['weak_bear']:
            return MarketStatus.BEAR
        return MarketStatus.STRONG_BEAR

    def _atr_mult(self, s: float) -> float:
        c: float = max(-6.0, min(6.0, s))
        return round(1.2 + (c + 6.0) / 12.0 * 1.3, 1)

    def _pos_ratio(self, s: float) -> float:
        c: float = max(-6.0, min(6.0, s))
        r: float = 1.0 / (1.0 + np.exp(-3.0 * c / 6.0))
        return round(0.05 + r * 0.9, 2)

    def _risk_level(self, s: float) -> str:
        if s >= 3.0:
            return "低"
        if s >= -1.5:
            return "中"
        return "高"

    def _save(self) -> None:
        if self.result is None:
            return
        rd: dict = self.result.to_dict()
        try:
            with open(Config.MARKET_ENV_LATEST_FILE, 'w', encoding='utf-8') as f:
                json.dump(rd, f, ensure_ascii=False, indent=2)
        except Exception as e:
            Logger.error("保存最新环境失败", e)
        hist: list = []
        if os.path.exists(Config.MARKET_ENV_HISTORY_FILE):
            try:
                with open(Config.MARKET_ENV_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    hist = json.load(f)
            except Exception:
                hist = []
        today: str = self.result.date
        hist = [h for h in hist if h.get('date') != today]
        hist.append(rd)
        hist.sort(key=lambda x: x.get('date', ''))
        if len(hist) > 365:
            hist = hist[-365:]
        try:
            with open(Config.MARKET_ENV_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(hist, f, ensure_ascii=False, indent=2)
        except Exception as e:
            Logger.error("保存历史失败", e)

    @staticmethod
    def _last_record() -> Optional[dict]:
        if not os.path.exists(Config.MARKET_ENV_HISTORY_FILE):
            return None
        try:
            with open(Config.MARKET_ENV_HISTORY_FILE, 'r', encoding='utf-8') as f:
                hist: list = json.load(f)
            today: str = datetime.now().strftime('%Y-%m-%d')
            prev: list = [h for h in hist if h.get('date', '') != today]
            return prev[-1] if prev else None
        except Exception:
            return None

    def _default_danger(self, reason: str = "") -> MarketEnvResult:
        return MarketEnvResult(
            date=datetime.now().strftime('%Y-%m-%d'),
            index_code=self.index_code, index_name=self.index_name,
            price=0, ma20=0, ma60=0, ma120=0,
            close_vs_ma20_pct=0, close_vs_ma60_pct=0, close_vs_ma120_pct=0,
            macd_hist=0, vol_ratio=0, atr_pct=0, atr_percentile=50,
            trend_score=0, momentum_score=0, volume_score=0, volatility_score=0,
            total_score=-6.0,
            trend_details={}, momentum_details={}, volume_details={}, volatility_details={},
            status=MarketStatus.STRONG_BEAR, market_safe=False,
            atr_multiplier=1.2, position_ratio=0.05, risk_level="高",
            score_change=0, status_changed=False,
        )

    def _log(self, ts: float, tr: str, ms: float, mr: str, vs: float, vrl: str,
             vols: float, volr: str, total: float, status: MarketStatus,
             safe: bool, pos: float, atm: float, sc: float, ch: bool) -> None:
        icon: str = "✅" if safe else "🚨"
        ci: str = "📈" if sc > 0 else ("📉" if sc < 0 else "➡️")
        rl: str = self._risk_level(total)
        Logger.info(f"┌────────────────────────────────────────────")
        Logger.info(f"│ 🌐 {self.index_name} ({self.index_code})")
        Logger.info(f"├────────────────────────────────────────────")
        Logger.info(f"│ 【趋势】 {ts:>+5.1f} │ {tr}")
        Logger.info(f"│ 【动能】 {ms:>+5.1f} │ {mr}")
        Logger.info(f"│ 【量价】 {vs:>+5.1f} │ {vrl}")
        Logger.info(f"│ 【波动】 {vols:>+5.1f} │ {volr}")
        Logger.info(f"├────────────────────────────────────────────")
        Logger.info(f"│ 综合:{total:>+5.1f} | {status.value} | {ci}{sc:>+5.1f}")
        Logger.info(f"│ {icon} 安全:{'是' if safe else '否'} | 仓位:{pos:.0%} | ATR:{atm}x | 风险:{rl}")
        Logger.info(f"└────────────────────────────────────────────\n")

    @staticmethod
    def get_signal_for_quant() -> Optional[dict]:
        if not os.path.exists(Config.MARKET_ENV_LATEST_FILE):
            return None
        try:
            with open(Config.MARKET_ENV_LATEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def get_history(days: int = 30) -> list:
        if not os.path.exists(Config.MARKET_ENV_HISTORY_FILE):
            return []
        try:
            with open(Config.MARKET_ENV_HISTORY_FILE, 'r', encoding='utf-8') as f:
                h: list = json.load(f)
            return h[-days:]
        except Exception:
            return []

    @staticmethod
    def get_env_change_signal() -> Optional[dict]:
        hist: list = MarketEnvironment.get_history(days=5)
        if len(hist) < 2:
            return None
        l: dict = hist[-1]
        p: dict = hist[-2]
        cs: str = l.get('status', '')
        ps: str = p.get('status', '')
        return {
            'status_changed': bool(cs != ps),
            'turned_bull': bool(cs in ('偏多', '强多头') and ps not in ('偏多', '强多头')),
            'turned_bear': bool(cs in ('偏空', '强空头') and ps not in ('偏空', '强空头')),
            'score_accelerating': bool(abs(l.get('total_score', 0) - p.get('total_score', 0)) > 2.0),
            'current_status': cs,
            'current_score': l.get('total_score', 0),
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  HTMLReporter — [v3.2-1] sparkline 箭头 + [v3.2-2] 背离显示 ║
# ╚══════════════════════════════════════════════════════════════╝

class HTMLReporter:
    """HTML交互看板"""

    STYLE: Dict[ETFStatus, Dict[str, str]] = {
        ETFStatus.EXTREME_BULL: {"cls": "badge-bull-super", "icon": "🔥"},
        ETFStatus.BULL: {"cls": "badge-bull-strong", "icon": "📈"},
        ETFStatus.WEAK_BULL: {"cls": "badge-bull-weak", "icon": "↗️"},
        ETFStatus.NEUTRAL: {"cls": "badge-neutral", "icon": "⚖️"},
        ETFStatus.WEAK_BEAR: {"cls": "badge-bear-weak", "icon": "↘️"},
        ETFStatus.BEAR: {"cls": "badge-bear-strong", "icon": "📉"},
        ETFStatus.EXTREME_BEAR: {"cls": "badge-bear-super", "icon": "❄️"},
    }

    ENV_STYLE: Dict[MarketStatus, Dict[str, str]] = {
        MarketStatus.STRONG_BULL: {"cls": "env-danger", "icon": "🚀"},
        MarketStatus.BULL: {"cls": "env-danger", "icon": "✅"},
        MarketStatus.NEUTRAL: {"cls": "env-neutral", "icon": "⚖️"},
        MarketStatus.BEAR: {"cls": "env-safe", "icon": "⚠️"},
        MarketStatus.STRONG_BEAR: {"cls": "env-safe", "icon": "🚨"},
    }

    # ────────────────── [v3.2-1] Sparkline / 变动箭头 ──────────────────

    @staticmethod
    def _sparkline_svg(scores: List[Tuple[str, float]],
                       score_delta: Optional[float] = None) -> str:
        """[v3.2-1] SVG 迷你趋势图；数据不足时回退为评分变动箭头

        Args:
            scores: [(date_str, score), ...] 的历史评分列表
            score_delta: 与上次运行的总分差值（首次运行为 None）
        """
        # 数据不足：显示变动箭头或占位符
        if len(scores) < 2:
            if score_delta is not None:
                arrow: str = "▲" if score_delta > 0 else ("▼" if score_delta < 0 else "→")
                color: str = "#dc2626" if score_delta > 0 else ("#16a34a" if score_delta < 0 else "#94a3b8")
                sign: str = "+" if score_delta > 0 else ""
                return (f'<span class="sparkline-arrow" style="color:{color}" '
                        f'title="较上次 {score_delta:+.1f}">{arrow}{sign}{score_delta:.1f}</span>')
            return '<span class="sparkline-na">—</span>'

        values: List[float] = [s[1] for s in scores]
        dates: List[str] = [s[0] for s in scores]
        n: int = len(values)

        min_v: float = min(values)
        max_v: float = max(values)
        rng: float = max_v - min_v if max_v != min_v else 1.0

        w: int = 80
        h: int = 22
        pad: int = 3

        points: List[str] = []
        for i, v in enumerate(values):
            x: float = pad + i * (w - 2 * pad) / max(1, n - 1)
            y: float = h - pad - (v - min_v) / rng * (h - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")

        color: str = "#dc2626" if values[-1] > 0 else "#16a34a" if values[-1] < 0 else "#94a3b8"

        zero_line: str = ""
        if min_v < 0 < max_v:
            zy: float = h - pad - (0 - min_v) / rng * (h - 2 * pad)
            zero_line = f'<line x1="{pad}" y1="{zy:.1f}" x2="{w - pad}" y2="{zy:.1f}" stroke="#cbd5e1" stroke-width="0.6" stroke-dasharray="2,2"/>'

        # 末端圆点
        last_dot: str = ""
        if points:
            lx, ly = points[-1].split(',')
            last_dot = f'<circle cx="{lx}" cy="{ly}" r="2" fill="{color}"/>'

        # tooltip: 起止 + delta
        delta_str: str = f" ({score_delta:+.1f})" if score_delta is not None else ""
        title_text: str = f"{dates[0]}~{dates[-1]}: {values[0]:+.1f}→{values[-1]:+.1f}{delta_str}"

        return (
            f'<svg class="sparkline" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="none"><title>{title_text}</title>'
            f'{zero_line}'
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
            f'{last_dot}</svg>'
        )

    # ────────────────── [v3.1-4] 市场宽度计算 ──────────────────

    @staticmethod
    def _compute_breadth(results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {'total': 0, 'bull': 0, 'bear': 0, 'neutral': 0,
                    'bull_pct': 0, 'bear_pct': 0, 'ratio': 0, 'signal': '无数据'}
        total: int = len(results)
        bull: int = sum(1 for r in results if r['status'].is_bullish)
        bear: int = sum(1 for r in results if r['status'].is_bearish)
        neutral: int = total - bull - bear
        ratio: float = bull / max(1, bear)
        signal: str = (
            '极度乐观' if ratio > 3.0 else
            '偏多' if ratio > 1.5 else
            '中性' if ratio >= 0.67 else
            '偏空' if ratio >= 0.33 else
            '极度悲观'
        )
        return {
            'total': total, 'bull': bull, 'bear': bear, 'neutral': neutral,
            'bull_pct': round(bull / total * 100, 1),
            'bear_pct': round(bear / total * 100, 1),
            'neutral_pct': round(neutral / total * 100, 1),
            'ratio': round(ratio, 2),
            'signal': signal,
        }

    # ────────────────── HTML 生成 ──────────────────

    @classmethod
    def generate(cls, results: List[Dict], env_result: MarketEnvResult,
                 filename: str = "index.html") -> None:
        ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        css: str
        js: str
        css, js = cls._assets()

        breadth: Dict[str, Any] = cls._compute_breadth(results)
        stats: str = cls._stats(results, breadth)
        rows: str = cls._rows(results)
        env_h: str = cls._env_html(env_result, breadth)

        html: str = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF波段雷达</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📡</text></svg>">
<style>{css}</style>
</head>
<body>
<div class="dashboard">
<header class="hero">
<div class="hero-inner">
<div class="hero-badge">PRO 3.2</div>
<h1>📡 ETF波段交易雷达</h1>
<p class="hero-sub">标准指标(MACD/BOLL/RSI/%B) · 大盘四维风控 · Alpha-RPS · 追踪止损 · MACD背离</p>
<p class="hero-time">📅 更新: <strong>{ts}</strong></p>
</div><div class="hero-gold-line"></div>
</header>
{env_h}
<section class="stats-grid">{stats}</section>
<section class="table-section">
<div class="table-header-bar"><h2>📊 标的评分明细</h2><span class="table-hint">点击表头排序</span></div>
<div class="table-card"><table id="radarTable"><thead><tr>
<th onclick="sortTable(0)">标的 / RPS ⇅</th>
<th onclick="sortTable(1)">趋势 ⇅</th>
<th onclick="sortTable(2)">月线(±5) ⇅</th>
<th onclick="sortTable(3)">周线(±6) ⇅</th>
<th onclick="sortTable(4)">日线(±4) ⇅</th>
<th onclick="sortTable(5)" class="text-center">止损 ⇅</th>
<th onclick="sortTable(6)" class="text-center">总分 ⇅</th>
<th onclick="sortTable(7)" class="text-center">状态 ⇅</th>
</tr></thead><tbody>{rows}</tbody></table></div>
</section>
<footer class="footer"><div class="footer-accent"></div>
<p>💡 顺大势、看节奏、抓时机 — 只做RPS高且多周期共振标的，破位必止损！</p>
<p class="footer-disclaimer">⚠️ 仅供学习，不构成投资建议。</p></footer>
</div>
<script>{js}</script></body></html>"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
            Logger.info(f"\n🎉 看板: {os.path.abspath(filename)}")
        except Exception as e:
            Logger.error("生成HTML失败", e)

    # ─── 大盘环境 HTML (含市场宽度) ───

    @classmethod
    def _env_html(cls, env: MarketEnvResult, breadth: Dict[str, Any]) -> str:
        s: Dict[str, str] = cls.ENV_STYLE.get(env.status, cls.ENV_STYLE[MarketStatus.NEUTRAL])
        sc: str = "#dc2626" if env.total_score > 0 else "#16a34a" if env.total_score < 0 else "#64748b"
        ss: str = f"+{env.total_score:.1f}" if env.total_score > 0 else f"{env.total_score:.1f}"
        cc: str = "#dc2626" if env.score_change > 0 else "#16a34a" if env.score_change < 0 else "#64748b"
        ca: str = "🔺" if env.score_change > 0 else "🔻" if env.score_change < 0 else "➡️"
        cs: str = f"{ca}{env.score_change:+.1f}" if env.score_change != 0 else "➡️0.0"

        def bar(label: str, score: float, mx: float, reason: str) -> str:
            pct: float = min(abs(score) / mx * 100, 100) if mx > 0 else 0
            bg: str = ("linear-gradient(90deg,#fee2e2,#fca5a5)" if score > 0
                       else ("linear-gradient(90deg,#dcfce7,#86efac)" if score < 0 else "#cbd5e1"))
            tc: str = "#b91c1c" if score > 0 else "#15803d" if score < 0 else "#64748b"
            sd: str = f"+{score:.1f}" if score > 0 else f"{score:.1f}"
            rh: str = f'<span class="dim-reason">{reason}</span>' if reason else ''
            return (f'<div class="dimension-row"><span class="dim-label">{label}</span>'
                    f'<div class="dim-bar-bg"><div class="dim-bar-fill" style="width:{pct:.0f}%;background:{bg}"></div></div>'
                    f'<span class="dim-score" style="color:{tc}">{sd}</span>{rh}</div>')

        tr: str = ("多头排列" if env.trend_details.get('bullish_alignment')
                   else ("空头排列" if env.trend_details.get('bearish_alignment') else ""))
        mr: str = ""
        if env.momentum_details.get('golden_cross'):
            mr = "金叉"
        elif env.momentum_details.get('death_cross'):
            mr = "死叉"
        elif env.momentum_details.get('hist_expanding'):
            mr = "柱放大"
        elif env.momentum_details.get('hist_shrinking'):
            mr = "柱缩小"
        vr: str = ("放量" if env.volume_details.get('high_volume')
                   else ("缩量" if env.volume_details.get('low_volume') else ""))
        volr: str = ""
        if env.volatility_details.get('extreme_volatility'):
            volr = "极端"
        elif env.volatility_details.get('high_volatility'):
            volr = "偏高"
        elif env.volatility_details.get('very_low_volatility'):
            volr = "极低"
        elif env.volatility_details.get('low_volatility'):
            volr = "低"
        elif env.volatility_details.get('normal_volatility'):
            volr = "正常"
        if env.volatility_details.get('percentile_high'):
            volr += "↑"
        elif env.volatility_details.get('percentile_low'):
            volr += "↓"

        def mt(l: str, v: float) -> str:
            c: str = "#dc2626" if v > 0 else "#16a34a" if v < 0 else "#64748b"
            ic: str = "▲" if v > 0 else "▼" if v < 0 else "—"
            return f'<span class="ma-tag" style="color:{c};border-left:3px solid {c}">{ic} {l} {v:+.2f}%</span>'

        desc: str = (
            f"安全，建议 <strong>{env.position_ratio:.0%}</strong> 仓位，止损 <strong>{env.atr_multiplier}x ATR</strong>"
            if env.market_safe else
            f"防守！建议 <strong>{env.position_ratio:.0%}</strong> 仓位，止损 <strong>{env.atr_multiplier}x ATR</strong>"
        )
        chg: str = '<div class="status-change-alert">⚡ 状态切换！</div>' if env.status_changed else ''
        pv: float = env.atr_percentile
        pc: str = "#16a34a" if pv < 40 else "#ca8a04" if pv < 70 else "#dc2626"
        pl: str = "偏低" if pv < 40 else "适中" if pv < 70 else "偏高"

        b: Dict[str, Any] = breadth
        breadth_html: str = ""
        if b.get('total', 0) > 0:
            bc: str = (
                "#dc2626" if b['ratio'] > 1.5 else
                "#ca8a04" if b['ratio'] >= 0.67 else "#16a34a"
            )
            breadth_html = f'''
<div class="breadth-section">
<div class="dim-section-title">📊 市场宽度</div>
<div class="breadth-bar-wrap">
<div class="breadth-bar">
<div class="breadth-seg breadth-bull" style="width:{b['bull_pct']:.0f}%"></div>
<div class="breadth-seg breadth-neutral" style="width:{b['neutral_pct']:.0f}%"></div>
<div class="breadth-seg breadth-bear" style="width:{b['bear_pct']:.0f}%"></div>
</div>
<div class="breadth-labels">
<span class="bl-bull">多头 {b['bull']}</span>
<span class="bl-neutral">中性 {b['neutral']}</span>
<span class="bl-bear">空头 {b['bear']}</span>
</div>
</div>
<div class="breadth-ratio">
<span>多空比: <strong style="color:{bc}">{b['ratio']:.2f}</strong></span>
<span class="breadth-signal" style="color:{bc}">{b['signal']}</span>
</div>
</div>'''

        return f'''
<div class="market-env {s['cls']}"><div class="env-status-strip"></div><div class="env-body">
<div class="env-header">
<div class="env-title-group"><h3>{s['icon']} 大盘: <span class="env-status-text">{env.status.value}</span></h3>
<span class="env-index-name">{env.index_name} ({env.index_code})</span></div>
<div class="env-score-group"><div class="env-score-big" style="color:{sc}">{ss}</div>
<div class="env-score-change" style="color:{cc}">{cs}</div></div></div>{chg}
<p class="env-desc">{desc}</p>
<div class="env-details-grid"><div class="env-dimensions">
<div class="dim-section-title">四维评分</div>
{bar("趋势", env.trend_score, 4, tr)}{bar("动能", env.momentum_score, 3, mr)}{bar("量价", env.volume_score, 2, vr)}{bar("波动", env.volatility_score, 2, volr)}
</div><div class="env-ma-devs"><div class="dim-section-title">均线偏离</div>
{mt("MA20", env.close_vs_ma20_pct)}{mt("MA60", env.close_vs_ma60_pct)}{mt("MA120", env.close_vs_ma120_pct)}
<div class="dim-section-title" style="margin-top:10px">辅助</div>
<span class="ma-tag">MACD {env.macd_hist:.4f}</span><span class="ma-tag">量比 {env.vol_ratio:.2f}</span><span class="ma-tag">ATR% {env.atr_pct:.2f}%</span>
<div class="dim-section-title" style="margin-top:10px">ATR百分位</div>
<div class="pct-bar-container"><div class="pct-bar-bg"><div class="pct-bar-fill" style="width:{pv:.0f}%;background:{pc}"></div></div>
<span class="pct-label" style="color:{pc}">{pv:.0f}% ({pl})</span></div>
</div></div>
{breadth_html}
</div></div>'''

    # ─── 表格行 (含 sparkline / 变动箭头) ───

    @classmethod
    def _rows(cls, results: List[Dict]) -> str:
        rows: List[str] = []
        for r in sorted(results, key=lambda x: x['total_score'], reverse=True):
            st: Dict[str, str] = cls.STYLE.get(r['status'], cls.STYLE[ETFStatus.NEUTRAL])
            ra: str = ("#dc2626" if r['total_score'] >= 7
                       else "#f59e0b" if r['total_score'] >= 2.5
            else "#94a3b8" if r['total_score'] >= -2.5
            else "#16a34a" if r['total_score'] >= -7
            else "#166534")
            tags: str = ""
            for t in r['tags']:
                tc: str = "tag-mark "
                if "龙头" in t or "主升" in t:
                    tc += "tag-king"
                elif "黄金坑" in t:
                    tc += "tag-pit"
                elif "止损" in t or "破位" in t:
                    tc += "tag-danger"
                elif "诱多" in t or "背离" in t:
                    tc += "tag-trap"
                elif "拐点" in t:
                    tc += "tag-new-bull"
                else:
                    tc += "tag-fire"
                tags += f'<span class="{tc}">{t}</span> '
            if tags:
                tags = f'<div class="tag-row">{tags}</div>'

            def sc(s: float) -> str:
                return "score-pos" if s > 0 else ("score-neg" if s < 0 else "score-zero")

            def ss(s: float) -> str:
                return f"+{s:.1f}" if s > 0 else f"{s:.1f}"

            dc: str = "#dc2626" if r['stop_dist'] < 0 else ("#ca8a04" if r['stop_dist'] < 3 else "#16a34a")
            di: str = "🚨" if r['stop_dist'] < 0 else ("⚠️" if r['stop_dist'] < 3 else "🛡️")
            rc: str = "#dc2626" if r['rps'] >= 80 else "#d97706" if r['rps'] >= 50 else "#64748b"
            status_val: str = _enum_value(r['status'])

            # [v3.2-1] sparkline + score_delta
            sparkline_data: List[Tuple[str, float]] = r.get('sparkline_data', [])
            score_delta: Optional[float] = r.get('score_delta')
            sparkline_html: str = cls._sparkline_svg(sparkline_data, score_delta)
            spark_sort: float = sparkline_data[-1][1] if sparkline_data else r['total_score']

            rows.append(f"""
<tr style="border-left:4px solid {ra}">
<td data-label="标的/RPS" data-sort="{r['rps']}"><div class="code-title"><span class="code-name">{r['name']}</span>
<span class="code-num">{r['code']} · RPS <span class="rps-inline" style="color:{rc}">{r['rps']:.0f}</span>
<span class="rps-bar-bg"><span class="rps-bar-fill" style="width:{min(r['rps'], 100):.0f}%;background:{rc}"></span></span></span></div></td>
<td data-label="趋势" data-sort="{spark_sort}" class="col-spark">{sparkline_html}</td>
<td data-label="月线" data-sort="{r['monthly_score']}"><div class="signal-box"><span class="score-pill {sc(r['monthly_score'])}">{ss(r['monthly_score'])}</span><span class="signal-text">{r['monthly_reason']}</span></div></td>
<td data-label="周线" data-sort="{r['weekly_score']}"><div class="signal-box"><span class="score-pill {sc(r['weekly_score'])}">{ss(r['weekly_score'])}</span><span class="signal-text">{r['weekly_reason']}</span></div></td>
<td data-label="日线" data-sort="{r['daily_score']}"><div class="signal-box"><span class="score-pill {sc(r['daily_score'])}">{ss(r['daily_score'])}</span><span class="signal-text">{r['daily_reason']}</span></div></td>
<td data-label="止损" class="text-center" data-sort="{r['stop_dist']}"><div class="stop-price">止损 <strong>{r['stop_loss']:.3f}</strong></div>
<div class="stop-dist" style="color:{dc}">{di} {r['stop_dist']:.1f}%</div></td>
<td data-label="总分" class="text-center" data-sort="{r['total_score']}">
<span class="total-score" style="color:{'#dc2626' if r['total_score'] > 0 else '#16a34a' if r['total_score'] < 0 else '#475569'}">{ss(r['total_score'])}</span></td>
<td data-label="状态" class="text-center" data-sort="{r['total_score']}">
<span class="status-badge {st['cls']}">{st['icon']} {status_val}</span>{tags}</td>
</tr>""")
        return "".join(rows)

    @classmethod
    def _stats(cls, results: List[Dict], breadth: Dict[str, Any]) -> str:
        if not results:
            return ""
        c: Dict[str, int] = {
            't': len(results),
            'b': breadth.get('bull', sum(1 for r in results if r['status'].is_bullish)),
            'k': sum(1 for r in results if r['rps'] >= 85),
            's': sum(1 for r in results if '止损' in str(r.get('tags', []))),
        }
        return f"""
<div class="stat-card stat-blue"><div class="stat-icon">📋</div><div class="stat-info"><div class="stat-val">{c['t']}</div><div class="stat-label">标的池</div></div></div>
<div class="stat-card stat-red"><div class="stat-icon">📈</div><div class="stat-info"><div class="stat-val">{c['b']}</div><div class="stat-label">波段多头</div></div></div>
<div class="stat-card stat-purple"><div class="stat-icon">👑</div><div class="stat-info"><div class="stat-val">{c['k']}</div><div class="stat-label">领涨龙头</div></div></div>
<div class="stat-card stat-orange"><div class="stat-icon">🚨</div><div class="stat-info"><div class="stat-val">{c['s']}</div><div class="stat-label">破位止损</div></div></div>"""

    @staticmethod
    def _assets() -> Tuple[str, str]:
        css: str = """
:root{--primary:#3b82f6;--bg:#f0f2f5;--card:#fff;--text:#1e293b;--text-muted:#64748b;--border:#e2e8f0;--navy:#0f172a;--navy-mid:#1e293b;--indigo:#312e81}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"PingFang SC","Microsoft YaHei","Hiragino Sans GB",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);line-height:1.5;padding:12px}
.dashboard{max-width:1520px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.hero{background:linear-gradient(135deg,var(--navy) 0%,#1a1a4e 50%,var(--indigo) 100%);color:#fff;border-radius:16px;overflow:hidden}
.hero-inner{padding:28px 24px 18px;text-align:center}
.hero-badge{display:inline-block;background:linear-gradient(135deg,#d97706,#f59e0b);color:#fff;font-size:.68rem;font-weight:800;padding:2px 10px;border-radius:99px;letter-spacing:1px;margin-bottom:8px}
.hero h1{font-size:1.5rem;font-weight:900;letter-spacing:2px}
.hero-sub{color:#93c5fd;font-size:.85rem;margin-top:6px}
.hero-time{color:#cbd5e1;font-size:.8rem;margin-top:4px}
.hero-time strong{color:#fbbf24}
.hero-gold-line{height:3px;background:linear-gradient(90deg,transparent,#d97706 20%,#fbbf24 50%,#d97706 80%,transparent)}
.market-env{border-radius:14px;overflow:hidden;display:flex;box-shadow:0 2px 12px rgba(0,0,0,.07);background:var(--card)}
.env-danger .env-status-strip{background:linear-gradient(180deg,#dc2626,#ef4444)}
.env-neutral .env-status-strip{background:linear-gradient(180deg,#ca8a04,#eab308)}
.env-safe .env-status-strip{background:linear-gradient(180deg,#16a34a,#22c55e)}
.env-status-strip{width:6px;flex-shrink:0}
.env-body{flex:1;padding:16px 18px}
.env-danger .env-body{background:#fef2f2}
.env-neutral .env-body{background:#fefce8}
.env-safe .env-body{background:#f0fdf4}
.env-header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.env-title-group h3{font-size:1.05rem;margin:0;color:var(--navy-mid)}
.env-status-text{font-weight:900}
.env-danger .env-status-text{color:#dc2626}
.env-neutral .env-status-text{color:#ca8a04}
.env-safe .env-status-text{color:#16a34a}
.env-index-name{font-size:.78rem;color:var(--text-muted)}
.env-score-group{display:flex;align-items:baseline;gap:8px}
.env-score-big{font-size:1.6rem;font-weight:900;font-family:monospace;line-height:1}
.env-score-change{font-size:.82rem;font-weight:700;font-family:monospace}
.env-desc{font-size:.85rem;margin-bottom:10px;line-height:1.6}
.status-change-alert{background:linear-gradient(90deg,#fef08a,#fef9c3);color:#854d0e;padding:6px 12px;border-radius:8px;font-weight:700;font-size:.8rem;margin-bottom:8px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.65}}
.env-details-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.env-dimensions{display:flex;flex-direction:column;gap:5px}
.dim-section-title{font-size:.75rem;font-weight:800;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}
.dimension-row{display:flex;align-items:center;gap:6px}
.dim-label{width:36px;font-weight:800;font-size:.78rem;flex-shrink:0;color:var(--text)}
.dim-bar-bg{flex:1;height:14px;background:#e2e8f0;border-radius:7px;overflow:hidden;min-width:50px}
.dim-bar-fill{height:100%;border-radius:7px;transition:width .4s ease}
.dim-score{width:40px;font-weight:900;font-size:.82rem;font-family:monospace;text-align:right;flex-shrink:0}
.dim-reason{font-size:.72rem;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px}
.env-ma-devs{display:flex;flex-direction:column;gap:3px}
.ma-tag{display:inline-block;font-size:.75rem;padding:3px 8px;border-radius:5px;background:#f8fafc;margin-right:4px;margin-bottom:3px;font-family:monospace;font-weight:700;border-left:3px solid #94a3b8}
.pct-bar-container{display:flex;align-items:center;gap:8px;margin-top:4px}
.pct-bar-bg{flex:1;height:12px;background:#e2e8f0;border-radius:6px;overflow:hidden}
.pct-bar-fill{height:100%;border-radius:6px;transition:width .4s}
.pct-label{font-size:.75rem;font-weight:800;font-family:monospace;white-space:nowrap}
.breadth-section{margin-top:12px;padding-top:10px;border-top:1px solid #e2e8f0}
.breadth-bar-wrap{margin:6px 0}
.breadth-bar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:#e2e8f0}
.breadth-seg{transition:width .4s}
.breadth-bull{background:#dc2626}
.breadth-neutral{background:#94a3b8}
.breadth-bear{background:#16a34a}
.breadth-labels{display:flex;justify-content:space-between;margin-top:4px;font-size:.72rem;font-weight:700}
.bl-bull{color:#dc2626}
.bl-neutral{color:#94a3b8}
.bl-bear{color:#16a34a}
.breadth-ratio{display:flex;justify-content:space-between;align-items:center;font-size:.78rem;margin-top:6px}
.breadth-signal{font-weight:800;padding:2px 8px;border-radius:4px;background:#f1f5f9}
/* [v3.2-1] sparkline + 变动箭头 */
.sparkline{display:block}
.sparkline-na{font-size:.72rem;color:#94a3b8}
.sparkline-arrow{font-size:.75rem;font-weight:800;font-family:monospace;white-space:nowrap;display:inline-block;letter-spacing:-0.5px}
.col-spark{text-align:center;vertical-align:middle;padding:6px 8px!important}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.stat-card{background:var(--card);border-radius:12px;padding:14px 12px;display:flex;align-items:center;gap:10px;box-shadow:0 1px 4px rgba(0,0,0,.05);border:1px solid var(--border);transition:transform .2s,box-shadow .2s}
.stat-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.1)}
.stat-icon{font-size:1.5rem;flex-shrink:0}
.stat-info{flex:1}
.stat-val{font-size:1.6rem;font-weight:900;line-height:1.1}
.stat-label{font-size:.72rem;color:var(--text-muted);font-weight:700;margin-top:2px}
.stat-blue .stat-val{color:#3b82f6}
.stat-red .stat-val{color:#dc2626}
.stat-purple .stat-val{color:#7c3aed}
.stat-orange .stat-val{color:#ea580c}
.table-section{background:var(--card);border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid var(--border)}
.table-header-bar{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid var(--border)}
.table-header-bar h2{font-size:1rem;font-weight:800;color:var(--navy-mid)}
.table-hint{font-size:.72rem;color:var(--text-muted)}
.table-card{overflow-x:auto}
table{width:100%;border-collapse:collapse;text-align:left}
th{background:#f8fafc;padding:10px;font-weight:800;color:#475569;font-size:.78rem;border-bottom:2px solid #cbd5e1;cursor:pointer;white-space:nowrap;position:sticky;top:0;z-index:1}
th:hover{background:#e2e8f0}
td{padding:10px;border-bottom:1px solid #f1f5f9;font-size:.85rem;vertical-align:middle}
tr{transition:background .15s}
tr:hover{background:#f8fafc}
.code-title{display:flex;flex-direction:column}
.code-name{font-weight:800;font-size:.92rem;color:var(--text)}
.code-num{font-size:.72rem;color:var(--text-muted);font-family:monospace;display:flex;align-items:center;gap:4px}
.rps-bar-bg{display:inline-block;width:40px;height:5px;background:#e2e8f0;border-radius:3px;overflow:hidden;vertical-align:middle}
.rps-bar-fill{display:block;height:100%;border-radius:3px}
.rps-inline{font-weight:900}
.signal-box{display:flex;align-items:center;gap:5px}
.score-pill{min-width:38px;padding:2px 6px;border-radius:6px;text-align:center;font-weight:900;font-size:.8rem;font-family:monospace}
.score-pos{background:#fee2e2;color:#b91c1c}
.score-neg{background:#dcfce7;color:#15803d}
.score-zero{background:#f1f5f9;color:#64748b}
.signal-text{font-size:.75rem;color:#475569}
.text-center{text-align:center}
.total-score{font-size:1.5rem;font-weight:900;font-family:monospace}
.status-badge{display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:99px;font-weight:800;font-size:.68rem;white-space:nowrap}
.badge-bull-super{background:linear-gradient(135deg,#fee2e2,#fecaca);color:#991b1b;border:2px solid #f87171}
.badge-bull-strong{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5}
.badge-bull-weak{background:#fffbeb;color:#b45309;border:1px solid #fde68a}
.badge-neutral{background:#f1f5f9;color:#475569;border:1px solid #cbd5e1}
.badge-bear-weak{background:#f0fdfa;color:#0f766e;border:1px solid #99f6e4}
.badge-bear-strong{background:#dcfce7;color:#166534;border:1px solid #86efac}
.badge-bear-super{background:linear-gradient(135deg,#166534,#15803d);color:#f0fdf4;border:2px solid #86efac}
.tag-row{margin-top:5px}
.tag-mark{display:inline-block;font-size:.68rem;padding:1px 7px;border-radius:4px;font-weight:800;margin:2px 3px 2px 0}
.tag-king{background:#ede9fe;color:#6d28d9;border:1px solid #c4b5fd}
.tag-pit{background:#fef9c3;color:#854d0e;border:1px solid #fde047}
.tag-new-bull{background:#ffedd5;color:#c2410c;border:1px solid #fdba74}
.tag-danger{background:#16a34a;color:#fff;border:1px solid #166534}
.tag-trap{background:#f59e0b;color:#fff;border:1px solid #d97706}
.tag-fire{background:#fffbeb;color:#3b82f6;border:1px solid #bfdbfe}
.stop-price{font-size:.82rem;color:#4b5563}
.stop-price strong{font-family:monospace}
.stop-dist{font-size:.82rem;font-weight:800}
.footer{background:var(--card);border-radius:14px;overflow:hidden;border:1px solid var(--border);text-align:center}
.footer-accent{height:3px;background:linear-gradient(90deg,transparent,#dc2626 20%,#f59e0b 50%,#dc2626 80%,transparent)}
.footer p{padding:10px 14px;color:var(--text-muted);font-size:.82rem}
.footer p:first-of-type{padding-bottom:2px}
.footer-disclaimer{font-size:.72rem!important;color:#94a3b8!important;padding-top:0!important}
@media(max-width:768px){
body{padding:8px}.dashboard{gap:10px}.hero-inner{padding:18px 14px 12px}.hero h1{font-size:1.15rem}.hero-sub{font-size:.76rem}
.env-details-grid{grid-template-columns:1fr;gap:10px}.dim-reason{max-width:160px}.market-env{border-radius:10px}.env-status-strip{width:5px}
.stats-grid{grid-template-columns:repeat(2,1fr);gap:8px}.stat-card{padding:10px 8px}.stat-val{font-size:1.3rem}.stat-icon{font-size:1.2rem}
.table-section{border-radius:10px}.table-header-bar{padding:10px 12px}.table-header-bar h2{font-size:.9rem}.table-hint{display:none}
.col-spark{display:none}
.table-card{border-radius:0;background:var(--bg);padding:0}table thead{display:none}table{display:block}table tbody{display:block}
table tbody tr{display:flex;flex-direction:column;background:var(--card);border-radius:10px;margin:0 8px 10px;border:1px solid var(--border);box-shadow:0 1px 3px rgba(0,0,0,.04);overflow:hidden}
table tbody tr:hover{background:var(--card)}
table tbody tr td{display:flex;justify-content:space-between;align-items:center;padding:7px 12px;border-bottom:1px solid #f1f5f9;font-size:.82rem}
table tbody tr td:last-child{border-bottom:none}
table tbody tr td::before{content:attr(data-label);font-weight:800;color:var(--text-muted);font-size:.72rem;flex-shrink:0;margin-right:8px}
table tbody tr td:first-child{background:#f8fafc;border-bottom:2px solid var(--border);flex-direction:column;align-items:flex-start;padding:10px 12px}
table tbody tr td:first-child::before{display:none}
table tbody tr td[data-label="趋势"]{display:none}
table tbody tr td:nth-child(7){justify-content:center;background:#f8fafc}
table tbody tr td:nth-child(7)::before{display:none}
table tbody tr td:last-child{justify-content:center;padding:10px}
table tbody tr td:last-child::before{display:none}
.signal-text{white-space:normal;font-size:.72rem}.total-score{font-size:1.3rem}.tag-mark{font-size:.65rem}
.status-badge{font-size:.72rem}.footer{border-radius:10px}
}
@media(max-width:400px){
.env-header{flex-direction:column;align-items:flex-start}.env-score-big{font-size:1.3rem}
.dim-label{width:28px;font-size:.7rem}.dim-reason{max-width:80px;font-size:.65rem}
.ma-tag{font-size:.65rem;padding:2px 5px}.pct-bar-container{flex-direction:column;align-items:stretch;gap:2px}.stat-card{gap:6px}
}"""

        js: str = """
let sortStates=[0,0,0,0,0,0,0,0];
function sortTable(colIndex){
const table=document.getElementById("radarTable"),tbody=table.querySelector("tbody"),rows=Array.from(tbody.querySelectorAll("tr"));
let isAsc=sortStates[colIndex]===1;sortStates=[0,0,0,0,0,0,0,0];sortStates[colIndex]=isAsc?0:1;
rows.sort((a,b)=>{let va=parseFloat(a.cells[colIndex].getAttribute("data-sort"))||0,vb=parseFloat(b.cells[colIndex].getAttribute("data-sort"))||0;return isAsc?(va-vb):(vb-va)});
rows.forEach(row=>tbody.appendChild(row));
}"""
        return css, js


# ╔══════════════════════════════════════════════════════════════╗
# ║                        辅助函数                              ║
# ╚══════════════════════════════════════════════════════════════╝

def get_etf_name_map() -> Dict[str, str]:
    Logger.info("🔄 初始化名称映射...")
    try:
        spot = ak.fund_etf_spot_ths()
        return dict(zip(spot['基金代码'], spot['基金名称']))
    except Exception as e:
        Logger.warning("获取名称映射失败", e)
        return {}


def load_history() -> Dict[str, Dict]:
    if os.path.exists(Config.HISTORY_FILE):
        try:
            with open(Config.HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_history(results: List[Dict]) -> None:
    today: str = datetime.now().strftime('%Y-%m-%d')
    max_days: int = ETFScoringConfig.HISTORY_DAILY_SCORES_DAYS

    existing: Dict[str, Dict] = {}
    if os.path.exists(Config.HISTORY_FILE):
        try:
            with open(Config.HISTORY_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            pass

    data: Dict[str, Dict] = {}
    for r in results:
        code: str = r['code']
        old_scores: Dict[str, float] = existing.get(code, {}).get('daily_scores', {})
        old_scores[today] = round(r['total_score'], 1)
        sorted_scores: Dict[str, float] = dict(sorted(old_scores.items())[-max_days:])

        data[code] = {
            'total_score': r['total_score'],
            'stop_loss': r['stop_loss'],
            'stop_dist': r['stop_dist'],
            'status': _enum_value(r['status']),
            'daily_scores': sorted_scores,
        }
    try:
        with open(Config.HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        Logger.error("保存历史失败", e)


def detect_signal_changes(
        current_results: List[Dict],
        prev_history: Dict[str, Dict]
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []

    bull_values: set = {ETFStatus.BULL.value, ETFStatus.EXTREME_BULL.value, ETFStatus.WEAK_BULL.value}
    bear_values: set = {ETFStatus.BEAR.value, ETFStatus.EXTREME_BEAR.value, ETFStatus.WEAK_BEAR.value}

    for r in current_results:
        code: str = r['code']
        prev: Dict = prev_history.get(code, {})
        if not prev:
            continue

        if 'status' not in prev or 'stop_dist' not in prev:
            continue

        prev_status: str = prev.get('status', '')
        curr_status: str = _enum_value(r['status'])
        prev_stop_dist: float = prev.get('stop_dist', 999.0)
        curr_stop_dist: float = r['stop_dist']
        prev_score: float = prev.get('total_score', 0.0)

        if curr_status in bull_values and prev_status not in bull_values:
            changes.append({
                'code': code, 'name': r['name'], 'type': 'NEW_BUY',
                'from_status': prev_status, 'to_status': curr_status,
                'total_score': r['total_score'],
            })

        if curr_status in bear_values and prev_status not in bear_values:
            changes.append({
                'code': code, 'name': r['name'], 'type': 'NEW_SELL',
                'from_status': prev_status, 'to_status': curr_status,
                'total_score': r['total_score'],
            })

        if curr_stop_dist < 0 and prev_stop_dist >= 0:
            changes.append({
                'code': code, 'name': r['name'], 'type': 'STOP_LOSS',
                'stop_price': r['stop_loss'], 'price': r['price'],
            })

        if abs(r['total_score'] - prev_score) > 5.0:
            changes.append({
                'code': code, 'name': r['name'], 'type': 'SCORE_ACCEL',
                'from_score': prev_score, 'to_score': r['total_score'],
            })

    return changes


def fetch_single_etf(code: str, name: str, force_download: bool,
                     market_safe: bool, atr_multiplier: float,
                     prev_stop: float = 0.0) -> Optional[ETFAnalyzer]:
    try:
        a = ETFAnalyzer(code, name, force_download=force_download,
                        market_safe=market_safe, atr_multiplier=atr_multiplier)
        a.prev_stop = prev_stop
        if a.fetch_data():
            return a
    except Exception as e:
        Logger.error(f"获取{code}失败", e)
    return None


def calc_blended_return(df: pd.DataFrame) -> float:
    """0.3×20日 + 0.3×60日 + 0.4×120日"""
    try:
        if len(df) < 21:
            return -999.0
        p: float = df['close'].iloc[-1]
        r20: float = (p - df['close'].iloc[-21]) / df['close'].iloc[-21]
        r60: float = (p - df['close'].iloc[-61]) / df['close'].iloc[-61] if len(df) >= 61 else r20
        r120: float = (p - df['close'].iloc[-121]) / df['close'].iloc[-121] if len(df) >= 121 else r60
        return 0.3 * r20 + 0.3 * r60 + 0.4 * r120
    except Exception:
        return -999.0


def save_etf_signals(results: List[Dict], env_result: MarketEnvResult,
                     breadth: Dict[str, Any]) -> None:
    try:
        out: Dict[str, Any] = {
            "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "market_env": {
                "status": env_result.status.value,
                "market_safe": bool(env_result.market_safe),
                "total_score": float(env_result.total_score),
                "position_ratio": float(env_result.position_ratio),
                "atr_multiplier": float(env_result.atr_multiplier),
                "risk_level": env_result.risk_level,
            },
            "market_breadth": {k: v for k, v in breadth.items()},
            "signals": []
        }
        for r in sorted(results, key=lambda x: x['total_score'], reverse=True):
            out["signals"].append({
                "code": r['code'], "name": r['name'],
                "total_score": float(r['total_score']),
                "status": _enum_value(r['status']),
                "rps": float(r.get('rps', 0)),
                "price": round(float(r['price']), 3) if pd.notna(r['price']) else 0.0,
                "stop_loss": round(float(r['stop_loss']), 3),
                "stop_dist_pct": round(float(r['stop_dist']), 1),
                "tags": r['tags'],
                "monthly_score": float(r['monthly_score']),
                "weekly_score": float(r['weekly_score']),
                "daily_score": float(r['daily_score']),
                "data_date": r.get('data_date', ''),
                "is_stale": r.get('is_stale', False),
                "score_delta": r.get('score_delta'),
            })
        with open(Config.ETF_SIGNALS_LATEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        Logger.info(f"📊 信号已输出: {Config.ETF_SIGNALS_LATEST_FILE}")
    except Exception as e:
        Logger.error("保存信号失败", e)


def log_signal_changes(changes: List[Dict[str, Any]]) -> None:
    if not changes:
        Logger.info("📋 无新信号变化")
        return

    Logger.info(f"🔔 === 信号变化 ({len(changes)}项) ===")
    type_icons: Dict[str, str] = {
        'NEW_BUY': '🟢', 'NEW_SELL': '🔴',
        'STOP_LOSS': '🚨', 'SCORE_ACCEL': '⚡',
    }
    for ch in changes:
        icon: str = type_icons.get(ch['type'], '❓')
        t: str = ch['type']
        if t in ('NEW_BUY', 'NEW_SELL'):
            Logger.info(
                f"  {icon} {ch['name']}({ch['code']}) "
                f"{ch['from_status']} → {ch['to_status']} | 总分{ch['total_score']:+.1f}"
            )
        elif t == 'STOP_LOSS':
            Logger.info(
                f"  {icon} {ch['name']}({ch['code']}) 触发止损 | "
                f"止损价{ch['stop_price']:.3f} | 现价{ch['price']:.3f}"
            )
        elif t == 'SCORE_ACCEL':
            Logger.info(
                f"  {icon} {ch['name']}({ch['code']}) 评分大幅变化 | "
                f"{ch['from_score']:+.1f} → {ch['to_score']:+.1f}"
            )
    Logger.info("")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        主函数                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main() -> None:
    t_start: float = time.time()
    _init_log_file()

    FORCE_DOWNLOAD: bool = True
    if FORCE_DOWNLOAD:
        Logger.info("🚀 强制下载模式")
    else:
        Logger.info("📦 智能模式: 优先本地缓存")

    codes: List[str] = [
        '159326', '512400', '512480', '512880', '159206', '159870', '515880', '159869', '516150',
        '159852', '515220', '159201', '515790', '512660', '159755', '515210', '159611', '512690',
        '512800', '159851', '560710', '159766', '512200', '513120', '518880', '159667', '159825',
        '560280', '159732', '159259', '159996', '159698', '512220'
    ]

    Logger.info(f"🚀 [ETF波段雷达 v3.2] 启动! {len(codes)}个标的")
    Logger.info("📐 指标: MACD(12/26/9) BOLL(20/2.0+%B) RSI(14) ATR(14) VOL MA(20)")
    Logger.info("🆕 v3.2: 评分变动箭头 / 日线MACD顶底背离检测\n")

    name_map: Dict[str, str] = get_etf_name_map()
    prev_history: Dict[str, Dict] = load_history()

    # ═══ 大盘环境 ═══
    Logger.info("🌐 评估大盘环境...")
    market_env = MarketEnvironment(
        Config.DEFAULT_INDEX_CODE, Config.DEFAULT_INDEX_NAME,
        force_download=FORCE_DOWNLOAD
    )
    env_result: MarketEnvResult = market_env.evaluate()
    market_safe: bool = env_result.market_safe
    atr_multiplier: float = env_result.atr_multiplier

    bm_ret: float = 0.0
    if market_env.analyzer and not market_env.analyzer.df_daily.empty:
        bm_ret = calc_blended_return(market_env.analyzer.df_daily)

    Logger.info(f"📈 基准收益: {bm_ret * 100:.2f}%")
    Logger.info(f"🛡️ ATR止损: {atr_multiplier}x | 仓位: {env_result.position_ratio:.0%}\n")
    t_env: float = time.time()

    # ═══ 并发获取数据 ═══
    analyzers: List[ETFAnalyzer] = []
    Logger.info("⏳ 并发拉取K线...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
        futures: Dict[concurrent.futures.Future, str] = {}
        for code in codes:
            name: str = name_map.get(code, f"ETF_{code}")
            prev_stop: float = 0.0
            if code in prev_history:
                prev_stop = float(prev_history[code].get('stop_loss', 0.0))
            futures[ex.submit(
                fetch_single_etf, code, name,
                FORCE_DOWNLOAD, market_safe, atr_multiplier, prev_stop
            )] = code

        done: int = 0
        for f in concurrent.futures.as_completed(futures):
            done += 1
            try:
                a: Optional[ETFAnalyzer] = f.result()
                if a:
                    analyzers.append(a)
            except Exception as e:
                Logger.error("拉取失败", e)
            if done % 5 == 0 or done == len(codes):
                Logger.info(f"进度: {done}/{len(codes)}")

    Logger.info(f"✅ 获取 {len(analyzers)}/{len(codes)}\n")
    t_fetch: float = time.time()

    if not analyzers:
        Logger.error("❌ 无有效数据")
        return

    # ═══ Alpha-RPS ═══
    Logger.info("🧮 计算 Alpha-RPS...")
    alphas: Dict[str, float] = {}
    for a in analyzers:
        try:
            ret: float = calc_blended_return(a.df_daily)
            alphas[a.code] = ret - bm_ret if ret != -999.0 else -999.0
        except Exception:
            alphas[a.code] = -999.0

    rps_map: Dict[str, float] = {}
    valid: Dict[str, float] = {k: v for k, v in alphas.items() if v != -999.0}
    if valid:
        sorted_c: List[str] = sorted(valid.keys(), key=lambda k: valid[k])
        n: int = len(sorted_c)
        for i, c in enumerate(sorted_c):
            rps_map[c] = (i / max(1, n - 1)) * 100.0

    # ═══ 多周期评分 ═══
    Logger.info("🧠 多周期评分...\n")
    results: List[Dict] = []
    for a in analyzers:
        try:
            a.rps = rps_map.get(a.code, 0.0)
            prev: Optional[float] = prev_history.get(a.code, {}).get('total_score')
            prev_s: float = float(prev_history.get(a.code, {}).get('stop_loss', 0.0))
            res: Optional[Dict] = a.analyze(prev, prev_stop=prev_s)
            if res:
                results.append(res)
        except Exception as e:
            Logger.error(f"{a.code} 分析失败", e)

    t_score: float = time.time()

    # ═══ [v3.2-1] 附加 sparkline 数据 + score_delta 到结果 ═══
    today_str: str = datetime.now().strftime('%Y-%m-%d')
    for r in results:
        code: str = r['code']
        scores: Dict[str, float] = prev_history.get(code, {}).get('daily_scores', {})
        scores[today_str] = r['total_score']
        r['sparkline_data'] = sorted(scores.items())[-ETFScoringConfig.SPARKLINE_DAYS:]

        # 评分变动: 与上次运行的 total_score 比较
        prev_total: Optional[float] = prev_history.get(code, {}).get('total_score')
        if prev_total is not None:
            r['score_delta'] = round(r['total_score'] - prev_total, 1)
        else:
            r['score_delta'] = None  # 首次运行，无对比基准

    # ═══ 输出 ═══
    if results:
        try:
            save_history(results)

            breadth: Dict[str, Any] = HTMLReporter._compute_breadth(results)

            HTMLReporter.generate(results, env_result, "index.html")
            save_etf_signals(results, env_result, breadth)

            signal_changes: List[Dict[str, Any]] = detect_signal_changes(results, prev_history)
            log_signal_changes(signal_changes)

            Logger.info(f"📊 环境: {Config.MARKET_ENV_LATEST_FILE}")

            Logger.info(f"📊 市场宽度: 多头{breadth['bull']} 空头{breadth['bear']} "
                        f"中性{breadth['neutral']} | 多空比{breadth['ratio']:.2f} → {breadth['signal']}")

            Logger.info(f"🎉 完成! {len(results)}个ETF\n")

            Logger.info("📊 === 多头 TOP10 ===")
            for i, r in enumerate(sorted(results, key=lambda x: x['total_score'], reverse=True)[:10]):
                delta_str: str = ""
                if r.get('score_delta') is not None:
                    d: float = r['score_delta']
                    arrow: str = "▲" if d > 0 else ("▼" if d < 0 else "→")
                    delta_str = f" | {arrow}{d:+.1f}"
                Logger.info(
                    f"  {i + 1:>2}. {r['name']:<16s} {r['code']} | "
                    f"总分{r['total_score']:>+6.1f} | RPS{r['rps']:>5.0f} | "
                    f"{_enum_value(r['status'])}{delta_str}"
                )

            Logger.info("\n📊 === 空头 BOTTOM5 ===")
            for r in sorted(results, key=lambda x: x['total_score'])[:5]:
                delta_str = ""
                if r.get('score_delta') is not None:
                    d = r['score_delta']
                    arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
                    delta_str = f" | {arrow}{d:+.1f}"
                Logger.info(
                    f"     {r['name']:<16s} {r['code']} | "
                    f"总分{r['total_score']:>+6.1f} | RPS{r['rps']:>5.0f} | "
                    f"{_enum_value(r['status'])}{delta_str}"
                )

        except Exception as e:
            Logger.error("输出失败", e)
    else:
        Logger.error("❌ 无有效结果")

    Logger.info(f"\n⏱️ 耗时: 大盘{t_env - t_start:.1f}s | 数据{t_fetch - t_env:.1f}s | "
                f"评分{t_score - t_fetch:.1f}s | 输出{time.time() - t_score:.1f}s | "
                f"总计{time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
