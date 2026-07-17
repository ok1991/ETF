#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF V4 趋势波段信号生成器核心实现。

整体优先逻辑（从高到低）：
1. 大盘体制（Regime） — 最高优先
   - MarketEnvironment 先执行，决定 market_safe、atr_multiplier 与风险等级。
   - 强空头环境下应显著压制信号（当前仅扣1.5分，建议后续改成渐进惩罚）。
2. 数据新鲜度与合约完整性
   - validate_signal_contract 必须在输出后立即执行。
3. 相对强度 (RPS Tier + 自适应窗口)
   - 市场大反转时缩短至60天。
   - RPS ≥ 75 应显著提升优先级。
4. 多周期共振 + 形态质量（Core Technical Edge）
   - 周线权重最高（1.6）合理。
   - 共振加分 > 背离/冲突惩罚。
   - 无危险tag（顶背离、诱多、破位止损）是硬门槛。
5. 风险可执行性（Risk Quality）
   - 止损距离甜蜜区（3~7.5%最佳）、动态追踪止损、ATR健康度。
   - score_delta 正向强烈加分，负向大幅扣分。
6. 特殊标签（Context Bonus）
   - “黄金坑”“领涨龙头”“底部拐点”标签应额外加分。
新增 composite_priority（0~100）正是为了把以上雷达维度显式量化，仅用于雷达排序和看板解释。
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

from .signals.contract import (
    CONFIDENCE_LEVELS,
    DATA_QUALITY_STATES,
    ENTRY_SETUPS,
    ENTRY_STATES,
    TREND_STATES,
    align_price_bases,
    align_return_series,
    confirmed_resample,
    fingerprint_price_directory,
    V4CalibrationModel,
    V4_SCHEMA_VERSION,
    build_v4_signal,
    v4_calibration_features,
)
from .signals.factors import (
    compute_asset_factors,
    final_priority as v4_final_priority,
    market_policy as build_v4_market_policy,
    normalised_atr_percentile,
    rank_relative_strength,
    weekly_trend_factor,
)

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
    V4_CALIBRATION_FILE: str = "v4_calibration.json"
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
    ENV_HYSTERESIS_FAST_CHANGE: float = 2.0
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


# V4 公共信号合约
SIGNAL_SCHEMA: Dict[str, Any] = {
    "schema_version": V4_SCHEMA_VERSION,
    "required": (
        "schema_version", "signal_id", "code", "name", "data_date", "price",
        "data_quality", "trend", "entry", "relative_strength", "risk",
        "market_policy", "calibration",
    ),
}
# ╔══════════════════════════════════════════════════════════════╗
# ║ [OPT-#8] count_trading_days — 交易日天数 (排除周末)          ║
# ╚══════════════════════════════════════════════════════════════╝

def count_trading_days(from_date_str: str, to_date: Optional[datetime] = None) -> int:
    """计算交易日天数（排除周六日，不含法定假日）。

    用于替代自然日 stale_days，避免周末虚高导致信号被误拒。

    Args:
        from_date_str: 起始日期字符串 (YYYY-MM-DD)
        to_date: 截止日期，默认今天

    Returns:
        交易日天数，解析失败返回 999
    """
    if not from_date_str:
        return 999
    try:
        from_dt = datetime.strptime(from_date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 999
    to_dt = (to_date or datetime.now()).date()
    if from_dt >= to_dt:
        return 0
    trading_days = 0
    current = from_dt + timedelta(days=1)
    while current <= to_dt:
        if current.weekday() < 5:  # 0-4 = 周一至周五
            trading_days += 1
        current += timedelta(days=1)
    return trading_days


# ╔══════════════════════════════════════════════════════════════╗
# ║ [OPT-#9] validate_signal_contract — 信号合约校验             ║
# ╚══════════════════════════════════════════════════════════════╝

def validate_signal_contract(sig_data: Dict[str, Any]) -> List[str]:
    """验证输出是否为完整且唯一的 V4 信号合约。"""
    errors: List[str] = []
    signals = sig_data.get("signals", [])
    if not isinstance(signals, list) or not signals:
        return ["signals 列表为空"]

    for index, signal in enumerate(signals):
        code = str(signal.get("code", f"index_{index}"))
        for field in SIGNAL_SCHEMA["required"]:
            if field not in signal:
                errors.append(f"{code}: 缺少必需字段 '{field}'")
        if signal.get("schema_version") != V4_SCHEMA_VERSION:
            errors.append(f"{code}.schema_version: 必须为 {V4_SCHEMA_VERSION}")

        for field in (
            "data_quality", "trend", "entry", "relative_strength",
            "risk", "market_policy", "calibration",
        ):
            if not isinstance(signal.get(field), dict):
                errors.append(f"{code}.{field}: 必须为 dict")

        quality = signal.get("data_quality") or {}
        trend = signal.get("trend") or {}
        entry = signal.get("entry") or {}
        calibration = signal.get("calibration") or {}
        risk = signal.get("risk") or {}

        if quality.get("status") not in DATA_QUALITY_STATES:
            errors.append(f"{code}.data_quality.status: 非法枚举 {quality.get('status')!r}")
        if trend.get("state") not in TREND_STATES:
            errors.append(f"{code}.trend.state: 非法枚举 {trend.get('state')!r}")
        if entry.get("state") not in ENTRY_STATES:
            errors.append(f"{code}.entry.state: 非法枚举 {entry.get('state')!r}")
        if entry.get("setup") not in ENTRY_SETUPS:
            errors.append(f"{code}.entry.setup: 非法枚举 {entry.get('setup')!r}")
        if calibration.get("confidence") not in CONFIDENCE_LEVELS:
            errors.append(
                f"{code}.calibration.confidence: 非法枚举 {calibration.get('confidence')!r}"
            )
        if not isinstance(entry.get("setup_score"), (int, float)):
            errors.append(f"{code}.entry.setup_score: 必须为数值")
        if not isinstance(entry.get("priority"), (int, float)):
            errors.append(f"{code}.entry.priority: 必须为数值")
        if not isinstance(risk.get("executable"), bool):
            errors.append(f"{code}.risk.executable: 必须为 bool")
        if not isinstance(calibration.get("approved"), bool):
            errors.append(f"{code}.calibration.approved: 必须为 bool")
    return errors
# ╔══════════════════════════════════════════════════════════════╗
# ║ [OPT-#3] atomic_json_save — 原子写入 JSON 文件               ║
# ╚══════════════════════════════════════════════════════════════╝

def atomic_json_save(data: Any, path: str) -> bool:
    """原子写入 JSON 文件 (先写 .tmp 再 rename)。"""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        Logger.error(f"原子写入失败: {path}", e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


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
        try:
            print(log_message)
        except UnicodeEncodeError:
            print(log_message.encode("gbk", errors="replace").decode("gbk", errors="replace"))
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
    risk_level: str
    score_change: float
    status_changed: bool
    schema_version: int = V4_SCHEMA_VERSION
    regime_level: str = "NEUTRAL"
    entry_permission: str = "OBSERVE_ONLY"
    max_exposure_ratio: float = 0.3

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
        """RSI — Wilder方法，标准参数(14)"""
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

    _calendar_cache: Optional[pd.DatetimeIndex] = None
    _calendar_lock: threading.Lock = threading.Lock()

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
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_weekly: pd.DataFrame = pd.DataFrame()
        self.df_monthly: pd.DataFrame = pd.DataFrame()
        self.df_weekly_preview: pd.DataFrame = pd.DataFrame()
        self.df_monthly_preview: pd.DataFrame = pd.DataFrame()
        self.trading_calendar: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.data_quality: Dict[str, Any] = {
            "status": "BLOCKED",
            "price_basis": "RAW",
            "analysis_basis": "QFQ",
            "adjustment_factor": 0.0,
            "adjustment_changed": False,
            "reasons": ["PRICE_DATA_NOT_LOADED"],
        }
        self.executable_price: float = 0.0
        self.rps: float = 0.0
        self.alpha_score: float = 0.0
        self.vol_adj_alpha: float = 0.0
        self.beta_to_benchmark: float = 1.0
        self.stop_loss_price: float = 0.0
        self.data_loaded: bool = False
        self.last_error: Optional[Exception] = None
        self.prev_stop: float = 0.0

    def set_price_frames(
            self,
            qfq_df: pd.DataFrame,
            raw_df: Optional[pd.DataFrame] = None,
            trading_calendar: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        """Install aligned analysis/execution price frames and confirmed periods."""
        qfq = qfq_df.sort_values("date").reset_index(drop=True).copy()
        if raw_df is None or raw_df.empty:
            raw = qfq.copy()
            aligned, quality = align_price_bases(raw, qfq)
            quality["status"] = "DEGRADED"
            quality["reasons"] = list(dict.fromkeys(
                list(quality.get("reasons", [])) + ["RAW_PRICE_FALLBACK_TO_QFQ"]
            ))
        else:
            raw = raw_df.sort_values("date").reset_index(drop=True).copy()
            aligned, quality = align_price_bases(raw, qfq)
        if aligned.empty:
            raise ValueError("raw and qfq price frames have no aligned dates")

        valid_dates = set(aligned["date"])
        self.df_daily = qfq[qfq["date"].isin(valid_dates)].reset_index(drop=True)
        self.df_raw = raw[raw["date"].isin(valid_dates)].reset_index(drop=True)
        self.data_quality = quality
        self.executable_price = float(self.df_raw["close"].iloc[-1])
        if trading_calendar is None or len(trading_calendar) == 0:
            start = self.df_daily["date"].iloc[0]
            end = self.df_daily["date"].iloc[-1] + pd.Timedelta(days=40)
            self.trading_calendar = pd.bdate_range(start, end)
            if self.data_quality["status"] == "VALID":
                self.data_quality["status"] = "DEGRADED"
            self.data_quality["reasons"] = list(dict.fromkeys(
                list(self.data_quality.get("reasons", [])) + ["FALLBACK_BUSINESS_CALENDAR"]
            ))
        else:
            self.trading_calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar))
        self.data_loaded = True
        self._resample_data()

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

    @classmethod
    def _load_trading_calendar(cls, start: Any, end: Any) -> pd.DatetimeIndex:
        with cls._calendar_lock:
            if cls._calendar_cache is None:
                try:
                    calendar_df = ak.tool_trade_date_hist_sina()
                    date_column = "trade_date" if "trade_date" in calendar_df.columns else calendar_df.columns[0]
                    cls._calendar_cache = pd.DatetimeIndex(
                        pd.to_datetime(calendar_df[date_column], errors="coerce").dropna()
                    )
                except Exception:
                    cls._calendar_cache = pd.DatetimeIndex([])
        if cls._calendar_cache is not None and len(cls._calendar_cache) > 0:
            return cls._calendar_cache
        return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=40))

    def _load_cached_raw(self, data_date: Any) -> Optional[pd.DataFrame]:
        stamp = pd.Timestamp(data_date).strftime("%Y%m%d")
        path = os.path.join(self.data_dir, f"{self.code}_raw_{stamp}.csv")
        if not os.path.exists(path):
            return None
        try:
            raw = pd.read_csv(path, parse_dates=["date"])
            return raw if self._validate_dataframe(raw) else None
        except Exception:
            return None

    def fetch_data(self, max_retries: int = Config.MAX_RETRIES) -> bool:
        for attempt in range(max_retries):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
                today_str: str = datetime.now().strftime('%Y%m%d')
                df: pd.DataFrame = pd.DataFrame()

                existing: List[str] = sorted(
                    [f for f in os.listdir(self.data_dir)
                     if f.startswith(f"{self.code}_")
                     and "_raw_" not in f
                     and f.endswith('.csv')],
                    reverse=True
                )

                if not self.force_download and existing:
                    if today_str in existing[0]:
                        try:
                            df = pd.read_csv(os.path.join(self.data_dir, existing[0]),
                                             parse_dates=['date'])
                            if self._validate_dataframe(df):
                                raw = self._load_cached_raw(df["date"].iloc[-1])
                                calendar = self._load_trading_calendar(df["date"].iloc[0], df["date"].iloc[-1])
                                self.set_price_frames(df, raw, trading_calendar=calendar)
                                Logger.info(f"{self.code} 本地今日文件加载成功")
                                return True
                        except Exception as e:
                            Logger.warning(f"{self.code} 读取今日文件失败", e)

                    try:
                        df = pd.read_csv(os.path.join(self.data_dir, existing[0]),
                                         parse_dates=['date'])
                        if self._validate_dataframe(df):
                            raw = self._load_cached_raw(df["date"].iloc[-1])
                            calendar = self._load_trading_calendar(df["date"].iloc[0], df["date"].iloc[-1])
                            self.set_price_frames(df, raw, trading_calendar=calendar)
                            Logger.info(f"{self.code} 本地最近文件加载成功")
                            return True
                    except Exception as e:
                        Logger.warning(f"{self.code} 读取最近文件失败", e)

                Logger.info(f"{self.code} 网络获取 ({attempt + 1}/{max_retries})")
                df_net = ak.stock_zh_a_hist_tx(
                    symbol=self._add_market_prefix(self.code), adjust="qfq"
                )
                raw_net = ak.stock_zh_a_hist_tx(
                    symbol=self._add_market_prefix(self.code), adjust=""
                )

                if df_net is not None and not df_net.empty and raw_net is not None and not raw_net.empty:
                    df = DataNormalizer.normalize(df_net).sort_values('date').reset_index(drop=True)
                    raw = DataNormalizer.normalize(raw_net).sort_values('date').reset_index(drop=True)
                    if (
                            self._validate_dataframe(df, Config.MIN_DATA_POINTS)
                            and self._validate_dataframe(raw, Config.MIN_DATA_POINTS)
                    ):
                        new_file: str = f"{self.code}_{df['date'].iloc[-1].strftime('%Y%m%d')}.csv"
                        raw_file: str = f"{self.code}_raw_{raw['date'].iloc[-1].strftime('%Y%m%d')}.csv"
                        df.to_csv(os.path.join(self.data_dir, new_file),
                                  index=False, encoding='utf-8-sig')
                        raw.to_csv(os.path.join(self.data_dir, raw_file),
                                   index=False, encoding='utf-8-sig')
                        self._cleanup_old_files(new_file, existing)
                        calendar = self._load_trading_calendar(df["date"].iloc[0], df["date"].iloc[-1])
                        self.set_price_frames(df, raw, trading_calendar=calendar)
                        Logger.info(f"{self.code} 网络获取成功")
                        return True

                if not self.force_download and existing:
                    try:
                        df = pd.read_csv(os.path.join(self.data_dir, existing[0]),
                                         parse_dates=['date'])
                        if self._validate_dataframe(df):
                            raw = self._load_cached_raw(df["date"].iloc[-1])
                            calendar = self._load_trading_calendar(df["date"].iloc[0], df["date"].iloc[-1])
                            self.set_price_frames(df, raw, trading_calendar=calendar)
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
                    except Exception:
                        pass

    def _resample_data(self) -> None:
        if self.df_daily.empty:
            return
        self.df_weekly_preview = self._resample('W-FRI')
        try:
            self.df_monthly_preview = self._resample('ME')
        except (ValueError, KeyError):
            self.df_monthly_preview = self._resample('M')

        calendar = self.trading_calendar
        if len(calendar) == 0:
            start = self.df_daily["date"].iloc[0]
            end = self.df_daily["date"].iloc[-1] + pd.Timedelta(days=40)
            calendar = pd.bdate_range(start, end)
        as_of = self.df_daily["date"].iloc[-1]
        self.df_weekly = confirmed_resample(self.df_daily, "W-FRI", as_of, calendar)
        self.df_monthly = confirmed_resample(self.df_daily, "ME", as_of, calendar)

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
        """判断数据是否过期，使用交易日计算，避免周末误判。"""
        data_date = self.get_data_date()
        if not data_date:
            return True
        return count_trading_days(data_date) > max_age_days

    # ────────────────── 指标计算 ──────────────────

    def calculate_indicators(self) -> None:
        if not self.data_loaded:
            return

        TechnicalIndicators.ma(self.df_monthly, [5, 10, 20])
        TechnicalIndicators.ma_slope(self.df_monthly, period=5, lookback=3)
        TechnicalIndicators.ma_slope(self.df_monthly, period=10, lookback=3)
        TechnicalIndicators.ma_slope(self.df_monthly, period=20, lookback=3)
        TechnicalIndicators.macd(self.df_monthly)

        TechnicalIndicators.ma(self.df_weekly, [5, 10, 20])
        TechnicalIndicators.ma_slope(self.df_weekly, period=5, lookback=3)
        TechnicalIndicators.ma_slope(self.df_weekly, period=10, lookback=3)
        TechnicalIndicators.ma_slope(self.df_weekly, period=20, lookback=3)
        TechnicalIndicators.volume_ma(self.df_weekly)
        TechnicalIndicators.macd(self.df_weekly)
        TechnicalIndicators.ma(self.df_daily, [5, 10, 20])
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

        chandelier_stop: float = highest_20 - self.atr_multiplier * atr_val

        supports: List[float] = [chandelier_stop]
        ma20_val: float = self._get_value(self.df_daily, 'MA20')
        boll_lower: float = self._get_value(self.df_daily, 'BOLL_lower')

        if pd.notna(ma20_val) and ma20_val > 0:
            supports.append(ma20_val - 0.5 * atr_val)
        if pd.notna(boll_lower) and boll_lower > 0:
            supports.append(boll_lower - 0.5 * atr_val)

        if pd.notna(current_price) and current_price > 0:
            valid: List[float] = [s for s in supports if s < current_price]
            base_stop: float = max(valid) if valid else min(chandelier_stop, current_price * 0.985)
        else:
            base_stop = chandelier_stop

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

    # ────────────────── [OPT-#4] 日线 MACD 顶/底背离（增加最小间距） ──────────────────

    def _macd_divergence_adjustment(self) -> Tuple[float, str]:
        """检测日线级别 MACD 顶/底背离。

        原理:
        - 顶背离: 价格创新高但对应位置的 MACD 柱未创新高 → 上涨动能衰竭
        - 底背离: 价格创新低但对应位置的 MACD 柱未创新低 → 下跌动能衰竭

        [OPT] 改进: 增加最小间距约束 (i2 > 2 / j2 > 2)，
        避免极值落在段边界导致误触发。
        """
        if 'MACD_hist' not in self.df_daily.columns or len(self.df_daily) < 25:
            return 0.0, ""

        tail: pd.DataFrame = self.df_daily.tail(30)  # [OPT] 取30天，给间距留余量
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
        i1: int = int(np.argmax(c1))
        i2: int = int(np.argmax(c2))
        # [OPT-#4] i2 > 2: 后段峰值不在开头，才是有意义的"新高"
        if c2[i2] > c1[i1] and h2[i2] < h1[i1] and h2[i2] > 0 and i2 > 2:
            return -0.5, " MACD顶背离"

        # ── 底背离 ──
        j1: int = int(np.argmin(c1))
        j2: int = int(np.argmin(c2))
        # [OPT-#4] j2 > 2: 后段谷值不在开头
        if c2[j2] < c1[j1] and h2[j2] > h1[j1] and h2[j2] < 0 and j2 > 2:
            return 0.5, " MACD底背离"

        return 0.0, ""

    # ────────────────── 月线分析 ──────────────────

    def _analyze_monthly(self) -> Tuple[float, str]:
        ma5: float = self._get_value(self.df_monthly, 'MA5')
        ma10: float = self._get_value(self.df_monthly, 'MA10')
        ma20: float = self._get_value(self.df_monthly, 'MA20')
        price: float = self._get_value(self.df_monthly, 'close')
        hist: float = self._get_value(self.df_monthly, 'MACD_hist')
        slope5: float = self._get_value(self.df_monthly, 'MA5_slope')
        slope10: float = self._get_value(self.df_monthly, 'MA10_slope')
        slope20: float = self._get_value(self.df_monthly, 'MA20_slope')

        if pd.isna(ma10) or pd.isna(hist):
            if pd.notna(slope5):
                if slope5 > 3:
                    return 1.5, "数据有限但MA5上行"
                if slope5 < -3:
                    return -1.5, "数据有限但MA5下行"
            return 0.0, "数据不足"

        ma_gap: float = (ma5 - ma10) / ma10 * 100 if ma10 != 0 else 0
        price_dev: float = (price - ma5) / ma5 * 100 if pd.notna(ma5) and ma5 != 0 else 0
        price_dev20: float = (
            (price - ma20) / ma20 * 100
            if pd.notna(price) and pd.notna(ma20) and ma20 > 0 else 0.0
        )

        hist_prev: float = np.nan
        if len(self.df_monthly) > 1:
            hist_prev = self._get_value(self.df_monthly.iloc[:-1], 'MACD_hist')
        hist_dir: int = 0
        if pd.notna(hist) and pd.notna(hist_prev):
            hist_dir = 1 if hist > hist_prev else (-1 if hist < hist_prev else 0)

        ret3: float = 0.0
        ret6: float = 0.0
        if pd.notna(price) and price > 0 and len(self.df_monthly) > 3:
            p3 = float(self.df_monthly['close'].iloc[-4])
            if p3 > 0:
                ret3 = (price / p3 - 1.0) * 100
        if pd.notna(price) and price > 0 and len(self.df_monthly) > 6:
            p6 = float(self.df_monthly['close'].iloc[-7])
            if p6 > 0:
                ret6 = (price / p6 - 1.0) * 100

        is_consolidation: bool = abs(ma_gap) < 1.5 and abs(price_dev) < 2.5
        long_ma_ok: bool = pd.notna(ma20) and ma20 > 0
        above_long: bool = not long_ma_ok or price > ma20
        below_long: bool = long_ma_ok and price < ma20
        long_slope_up: bool = pd.notna(slope20) and slope20 > 0.4
        long_slope_down: bool = pd.notna(slope20) and slope20 < -0.4
        long_bull: bool = above_long and ma5 > ma10 and (not long_ma_ok or ma10 >= ma20 * 0.98)
        long_bear: bool = below_long and ma5 < ma10 and (not long_ma_ok or ma10 <= ma20 * 1.02)

        score_adj: float = 0.0
        if ret3 > 3 and ret6 > 5:
            score_adj += 0.5
        elif ret3 < -3 and ret6 < -5:
            score_adj -= 0.5
        elif ret3 > 2 and ret6 < -2:
            score_adj += 0.3
        elif ret3 < -2 and ret6 > 3:
            score_adj -= 0.3

        overheat_penalty: float = 0.0
        if price_dev20 > 22:
            overheat_penalty = 1.0
        elif price_dev20 > 16:
            overheat_penalty = 0.6
        elif price_dev > 10:
            overheat_penalty = 0.4

        def finalize(score: float, reason: str) -> Tuple[float, str]:
            adjusted = round(max(-5.0, min(5.0, score + score_adj - overheat_penalty)), 1)
            tags: List[str] = []
            if abs(ret3) >= 2 or abs(ret6) >= 4:
                tags.append(f"3/6月{ret3:+.1f}%/{ret6:+.1f}%")
            if overheat_penalty > 0:
                tags.append(f"月线乖离偏高{price_dev20:.1f}%")
            return adjusted, reason + (f"({';'.join(tags)})" if tags else "")

        if long_bull and hist > 0:
            if price > ma5 and long_slope_up and hist_dir >= 0:
                s = 3.8 + min(0.8, max(0.0, ma_gap) * 0.12) + min(0.4, max(0.0, ret3) / 20)
                return finalize(s, f"月线主升结构(间距{ma_gap:.1f}%)")
            if price < ma5 and price > ma20:
                return finalize(2.2 if hist_dir >= 0 else 1.7, "月线主升回调")
            return finalize(2.6 if hist_dir >= 0 else 2.0, f"月线多头结构(间距{ma_gap:.1f}%)")

        if long_bear and hist < 0:
            if price < ma5 and long_slope_down and hist_dir <= 0:
                s = -3.8 - min(0.8, abs(ma_gap) * 0.12) - min(0.4, abs(min(ret3, 0.0)) / 20)
                return finalize(s, f"月线下跌延续(间距{ma_gap:.1f}%)")
            if price > ma5 and price < ma20:
                return finalize(-1.8, "月线空头反弹")
            return finalize(-2.8 if hist_dir <= 0 else -2.2, f"月线空头结构(间距{ma_gap:.1f}%)")

        if below_long and hist_dir > 0 and ret3 > 0:
            return finalize(1.0 if price > ma10 else 0.4, "月线筑底修复")
        if above_long and hist_dir < 0 and ret3 < 0:
            return finalize(1.2 if price > ma10 else 0.2, "月线趋势老化")

        if ma5 > ma10:
            if price > ma5 and hist > 0:
                s: float = min(5.0, 3.0 + abs(ma_gap) * 0.15 + abs(hist) * 0.3)
                if ma_gap > 4.0 and abs(hist) > 0.8 and price_dev > 2.0:
                    return finalize(s, f"极强多头(间距{ma_gap:.1f}%)")
                if hist_dir > 0:
                    return finalize(s * 0.85, f"强多头(间距{ma_gap:.1f}%)")
                return finalize(s * 0.7, f"多头趋势(间距{ma_gap:.1f}%)")
            if price < ma5 and hist > 0:
                if price > ma10:
                    return finalize((2.0 if hist_dir >= 0 else 1.5), "多头回调")
                return finalize(0.5, "深度回调(跌破MA10)")
            if price > ma5 and hist < 0:
                return finalize(1.0 if hist_dir < 0 else 1.5, "动能衰减")
            if ma10 < price < ma5:
                g: float = min(2.5, 1.0 + abs(ma_gap) * 0.2) if hist > 0 else 0.3
                return finalize(g, ("中继" if hist > 0 else "挣扎"))
            if is_consolidation:
                if hist > 0.2:
                    return finalize(0.8, "偏多蓄力")
                if hist < -0.2:
                    return finalize(-0.8, "偏空蓄力")
                return finalize(0.0, "横盘")

        if ma5 < ma10:
            if price < ma5 and hist < 0:
                s = max(-5.0, -3.0 - abs(ma_gap) * 0.15 - abs(hist) * 0.3)
                if ma_gap < -4.0 and abs(hist) > 0.8 and price_dev < -2.0:
                    return finalize(s, f"极强空头(间距{ma_gap:.1f}%)")
                if hist_dir < 0:
                    return finalize(s * 0.85, f"强空头(间距{ma_gap:.1f}%)")
                return finalize(s * 0.7, f"空头趋势(间距{ma_gap:.1f}%)")
            if price > ma5 and hist < 0:
                if price > ma10:
                    return finalize(1.0, "强反弹(突破MA10)")
                return finalize(-1.5, "弱势反弹")
            if price < ma5 and hist > 0:
                if price > ma10:
                    return finalize(1.0, "趋势转变")
                return finalize(-0.5, "底部信号")
            if is_consolidation:
                if hist > 0.2:
                    return finalize(0.8, "偏多蓄力")
                if hist < -0.2:
                    return finalize(-0.8, "偏空蓄力")
                return finalize(0.0, "横盘")

        if ma_gap > 1.5:
            a: float = min(3.0, 1.0 + abs(ma_gap) * 0.15 + abs(hist) * 0.2)
            if price_dev > 5:
                a *= 0.8
            return finalize(a, f"月线上行(间距{ma_gap:.1f}%)")
        if ma_gap < -1.5:
            return finalize(max(-3.0, -1.0 - abs(ma_gap) * 0.15 - abs(hist) * 0.2), f"月线下行(间距{ma_gap:.1f}%)")
        if hist > 0.2:
            return finalize(0.5, "月线偏多")
        if hist < -0.2:
            return finalize(-0.5, "月线偏空")
        return finalize(0.0, "月线震荡")

    # ────────────────── 周线分析 ──────────────────

    def _analyze_weekly(self) -> Tuple[float, str]:
        ma5: float = self._get_value(self.df_weekly, 'MA5')
        ma10: float = self._get_value(self.df_weekly, 'MA10')
        ma20: float = self._get_value(self.df_weekly, 'MA20')
        slope5: float = self._get_value(self.df_weekly, 'MA5_slope')
        slope10: float = self._get_value(self.df_weekly, 'MA10_slope')
        slope20: float = self._get_value(self.df_weekly, 'MA20_slope')
        price: float = self._get_value(self.df_weekly, 'close')
        hist: float = self._get_value(self.df_weekly, 'MACD_hist')
        vol: float = self._get_value(self.df_weekly, 'volume')
        vma: float = self._get_value(self.df_weekly, 'VMA')

        if pd.isna(ma20) or pd.isna(slope20):
            return 0.0, "数据不足"

        dist20: float = (price - ma20) / ma20 * 100 if pd.notna(price) and ma20 > 0 else 0.0
        vr: float = vol / vma if pd.notna(vol) and pd.notna(vma) and vma > 0 else 1.0
        recent = self.df_weekly.tail(12)
        prev = self.df_weekly.iloc[:-1].tail(12)
        range_pct: float = 0.0
        prev_high: float = np.nan
        prev_low: float = np.nan
        if not recent.empty:
            low_v = float(recent['low'].min())
            high_v = float(recent['high'].max())
            if low_v > 0:
                range_pct = (high_v - low_v) / low_v * 100
        if not prev.empty:
            prev_high = float(prev['high'].max())
            prev_low = float(prev['low'].min())

        last_open = self._get_value(self.df_weekly, 'open')
        last_high = self._get_value(self.df_weekly, 'high')
        upper_wick_pct: float = 0.0
        if pd.notna(last_high) and pd.notna(last_open) and pd.notna(price) and last_high > 0:
            body_top = max(last_open, price)
            upper_wick_pct = max(0.0, (last_high - body_top) / last_high * 100)

        strong: bool = pd.notna(price) and price > ma20 and slope20 > 0.3
        weak: bool = pd.notna(price) and price < ma20 and slope20 < -0.4
        bull_ma: bool = pd.notna(ma5) and pd.notna(ma10) and ma5 > ma10 > ma20 * 0.98
        bear_ma: bool = pd.notna(ma5) and pd.notna(ma10) and ma5 < ma10 < ma20 * 1.02
        short_slope_up: bool = pd.notna(slope5) and pd.notna(slope10) and slope5 > 0.4 and slope10 > 0.2
        short_slope_down: bool = pd.notna(slope5) and pd.notna(slope10) and slope5 < -0.4 and slope10 < -0.2
        platform: bool = range_pct > 0 and range_pct <= 13.0 and abs(slope20) < 1.2
        breakout: bool = (
            pd.notna(prev_high) and prev_high > 0 and price >= prev_high * 0.995
            and vr >= 1.08 and hist > 0
        )
        failed_breakout: bool = (
            pd.notna(prev_high) and prev_high > 0 and last_high >= prev_high * 0.995
            and price < prev_high * 0.985 and (upper_wick_pct >= 1.2 or vr >= 1.25)
        )

        def volume_note() -> str:
            if vr >= 1.25:
                return f"+放量{vr:.1f}x"
            if vr <= 0.75:
                return f"+缩量{vr:.1f}x"
            return f"+量能{vr:.1f}x"

        if failed_breakout and price > ma20:
            return 1.0, f"突破失败：上影回落{volume_note()}"
        if failed_breakout:
            return -1.5, f"突破失败：未收回平台{volume_note()}"

        if breakout and bull_ma and short_slope_up and slope20 > 0.5:
            return 6.0, f"主升加速：平台放量突破{volume_note()}"
        if breakout:
            return 4.8, f"平台突破：收上前高{volume_note()}"

        if bull_ma and strong and hist > 0 and short_slope_up:
            if dist20 <= 12.0:
                return 5.6, f"主升加速：多头排列+斜率共振({dist20:.1f}%)"
            return 4.6, f"主升延伸：离20周线偏远({dist20:.1f}%)"

        if strong and price <= ma10 * 1.02 and price >= ma20 * 0.98:
            if hist >= -0.05 and vr <= 1.15:
                return 4.2, f"健康回踩：靠近10/20周线{volume_note()}"
            return 2.4, f"回踩待确认：动能转弱{volume_note()}"

        if platform and price >= ma20 * 0.98 and hist >= -0.05:
            return 2.8 if price >= ma10 else 1.6, f"平台蓄势：12周振幅{range_pct:.1f}%"

        if strong and hist < 0:
            return 2.0, "多头回调：趋势向上但MACD死叉"
        if strong and hist > 0:
            return 4.3, "多头趋势：站稳20周线+MACD正"
        if bear_ma and weak and hist < 0 and slope20 < -0.7 and short_slope_down:
            return -6.0, "极弱空：空头排列+发散"
        if weak and hist > 0:
            return -2.5, "空头反弹(警惕诱多)"
        if weak and hist < 0:
            return -4.8, "下跌延续：跌破20周线+MACD负"
        if pd.notna(prev_low) and prev_low > 0 and price <= prev_low * 1.01 and weak:
            return -5.2, "下跌延续：接近12周低位"
        if abs(slope20) < 1.2 and pd.notna(price) and ma20 > 0:
            if abs(dist20) < 5.0:
                return 0.0, "震荡：20周线走平+价格纠缠"
        if pd.notna(price) and pd.notna(ma20):
            return (0.5 if price > ma20 else -0.5), "弱势震荡"
        return 0.0, "数据不足"

    # ────────────────── 日线分析 ──────────────────

    def _analyze_daily(self) -> Tuple[float, str]:
        """日线 — 只负责买点和风险时机，不覆盖月周线趋势判断。"""
        price: float = self._get_value(self.df_daily, 'close')
        open_p: float = self._get_value(self.df_daily, 'open')
        high: float = self._get_value(self.df_daily, 'high')
        low: float = self._get_value(self.df_daily, 'low')
        ma5: float = self._get_value(self.df_daily, 'MA5')
        ma10: float = self._get_value(self.df_daily, 'MA10')
        ma20: float = self._get_value(self.df_daily, 'MA20')
        mid: float = self._get_value(self.df_daily, 'BOLL_mid')
        upper: float = self._get_value(self.df_daily, 'BOLL_upper')
        lower: float = self._get_value(self.df_daily, 'BOLL_lower')
        pct_b: float = self._get_value(self.df_daily, 'BOLL_pctB')
        hist: float = self._get_value(self.df_daily, 'MACD_hist')
        vol: float = self._get_value(self.df_daily, 'volume')
        vma: float = self._get_value(self.df_daily, 'VMA')
        rsi: float = self._get_value(self.df_daily, 'RSI')
        atr: float = self._get_value(self.df_daily, 'ATR')

        if any(pd.isna(x) for x in [upper, lower, mid, price]):
            return 0.0, "数据不完整"
        if mid <= 0:
            return 0.0, "中轨异常"

        vr: float = vol / vma if pd.notna(vma) and vma > 0 and pd.notna(vol) else 0.0
        atr_pct: float = atr / price * 100 if pd.notna(atr) and pd.notna(price) and price > 0 else 0.0
        prev_close: float = np.nan
        prev_high20: float = np.nan
        prev_low20: float = np.nan
        if len(self.df_daily) > 1:
            prev_close = float(self.df_daily['close'].iloc[-2])
        if len(self.df_daily) > 20:
            prev20 = self.df_daily.iloc[:-1].tail(20)
            prev_high20 = float(prev20['high'].max())
            prev_low20 = float(prev20['low'].min())
        upper_wick_ratio: float = 0.0
        if pd.notna(high) and pd.notna(low) and high > low and pd.notna(open_p):
            body_top = max(open_p, price)
            upper_wick_ratio = max(0.0, (high - body_top) / (high - low))

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
                base_score, base_reason = 0.5, "滞涨：缩量上轨"

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

        setup_adj: float = 0.0
        setup_tags: List[str] = []
        ma_bull: bool = pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and ma5 > ma10 > ma20 * 0.995
        ma_bear: bool = pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and ma5 < ma10 < ma20 * 1.005

        if ma_bull and price > ma5:
            setup_adj += 0.4
            setup_tags.append("短均多头")
        elif ma_bull and price >= ma10 * 0.99:
            setup_adj += 0.2
            setup_tags.append("回踩短均")
        elif ma_bear and price < ma10:
            setup_adj -= 0.5
            setup_tags.append("短均空头")

        if pd.notna(prev_high20) and prev_high20 > 0 and price >= prev_high20 * 0.998:
            if vr >= ETFScoringConfig.VOLUME_EXPANSION_THRESHOLD and upper_wick_ratio < 0.35:
                setup_adj += 0.6
                setup_tags.append("20日新高有效")
            else:
                setup_adj -= 0.5
                setup_tags.append("20日新高未确认")
        elif pd.notna(prev_low20) and prev_low20 > 0 and price <= prev_low20 * 1.002:
            if vr > 1.1:
                setup_adj -= 0.7
                setup_tags.append("20日新低放量")
            else:
                setup_adj -= 0.2
                setup_tags.append("20日新低缩量")

        if pd.notna(prev_high20) and prev_high20 > 0 and pd.notna(high):
            if high >= prev_high20 * 0.998 and price < prev_high20 * 0.99 and upper_wick_ratio >= 0.35:
                setup_adj -= 0.8
                setup_tags.append("放量冲高回落" if vr >= 1.15 else "上影线回落")

        if pd.notna(prev_close) and prev_close >= mid and price < mid:
            setup_adj -= 0.6
            setup_tags.append("跌破中轨")
        elif pd.notna(prev_close) and prev_close <= mid and price > mid and hist > 0:
            setup_adj += 0.3
            setup_tags.append("收复中轨")

        if atr_pct > 4.5 and price > upper * 0.98:
            setup_adj -= 0.4
            setup_tags.append(f"ATR偏热{atr_pct:.1f}%")
        elif 1.0 <= atr_pct <= 3.5 and base_score > 0:
            setup_adj += 0.2
            setup_tags.append(f"ATR健康{atr_pct:.1f}%")

        rsi_adj: float
        rsi_tag: str
        rsi_adj, rsi_tag = self._rsi_adjustment(rsi)

        macd_adj: float
        macd_tag: str
        macd_adj, macd_tag = self._macd_hist_trend_adjustment()

        vp_adj: float
        vp_tag: str
        vp_adj, vp_tag = self._volume_price_divergence_adjustment()

        div_adj: float
        div_tag: str
        div_adj, div_tag = self._macd_divergence_adjustment()

        total: float = base_score + setup_adj + rsi_adj + macd_adj + vp_adj + div_adj
        total = round(max(-5.0, min(5.0, total)), 1)

        reason: str = base_reason
        if setup_tags:
            reason += " " + " ".join(setup_tags)
        for tag in [rsi_tag, macd_tag, vp_tag, div_tag]:
            if tag:
                reason += tag

        return total, reason

    # ────────────────── 共振/背离 ──────────────────

    @staticmethod
    def _resonance_strength(m: float, w: float, d: float) -> str:
        if not (m > 0 and w > 0 and d > 0):
            return ""
        if m >= 2.5 and w >= 4.0 and d >= 1.5:
            return "strong"
        if min(m, w, d) < 1.0:
            return "weak"
        return "medium"

    @staticmethod
    def _ramp(value: float, start: float, full: float) -> float:
        if full <= start:
            return 1.0
        return max(0.0, min(1.0, (value - start) / (full - start)))

    def _apply_resonance_and_conflict(
            self, m: float, w: float, d: float, daily_reason: str
    ) -> Tuple[float, float, List[str]]:
        bonus: float = 0.0
        penalty: float = 0.0
        tags: List[str] = []

        resonance_strength = self._resonance_strength(m, w, d)
        if resonance_strength:
            weakest = min(m, w, d)
            base_bonus = weakest * ETFScoringConfig.RESONANCE_BONUS
            if resonance_strength == "strong":
                bonus += min(2.4, base_bonus * 1.15)
                tags.append("📈 强三周期共振")
            elif resonance_strength == "medium":
                bonus += min(1.5, base_bonus)
                tags.append("📈 三周期共振")
            else:
                bonus += min(0.6, base_bonus * 0.65)
                tags.append("📈 弱三周期共振")

        if w > 2.5 and d < 0:
            weekly_force = self._ramp(w, 2.5, 5.0)
            daily_damage = self._ramp(abs(d), 0.0, 3.0)
            severity = max(0.2, min(1.0, weekly_force * 0.55 + daily_damage * 0.45))
            conflict_penalty = ETFScoringConfig.DIVERGENCE_PENALTY * severity
            # 如果日线已经因为这些原因扣分，则降低额外惩罚，避免重复重罚
            if any(key in daily_reason for key in ["MACD顶背离", "量价背离", "RSI超买", "放量下跌", "真破位"]):
                conflict_penalty *= 0.5
            penalty += conflict_penalty
            if any(key in daily_reason for key in ["真破位", "跌破中轨", "20日新低", "冲高回落"]):
                tags.append("⚠️ 周强日破")
            else:
                tags.append("⚠️ 周强日弱")

        if w < -2.5 and d > 0:
            weekly_damage = self._ramp(abs(w), 2.5, 5.0)
            daily_repair = self._ramp(d, 0.0, 3.0)
            severity = max(0.2, min(1.0, weekly_damage * 0.55 + daily_repair * 0.45))
            conflict_penalty = ETFScoringConfig.DIVERGENCE_PENALTY * 0.8 * severity
            if any(key in daily_reason for key in ["MACD底背离", "RSI超卖", "极佳洗盘"]):
                conflict_penalty *= 0.5
            penalty += conflict_penalty
            tags.append("⚠️ 周弱日强")

        if m < -0.5 and w > 2.0:
            monthly_drag = self._ramp(abs(m), 0.5, 3.0)
            weekly_force = self._ramp(w, 2.0, 5.0)
            severity = max(0.2, min(1.0, monthly_drag * 0.45 + weekly_force * 0.55))
            penalty += ETFScoringConfig.DIVERGENCE_PENALTY * 0.45 * severity
            tags.append("⚠️ 月弱周强")

        if m > 1.5 and w < 1.5:
            monthly_force = self._ramp(m, 1.5, 4.0)
            weekly_drag = self._ramp(1.5 - w, 0.0, 3.0)
            severity = max(0.2, min(1.0, monthly_force * 0.45 + weekly_drag * 0.55))
            penalty += ETFScoringConfig.DIVERGENCE_PENALTY * 0.35 * severity
            tags.append("⚠️ 月强周弱")
        return bonus, penalty, tags

    @staticmethod
    def _phase_from_reason(reason: str) -> str:
        if not reason:
            return ""
        head = reason.split("(", 1)[0].strip()
        head = head.split("：", 1)[0].strip()
        return head

    def _cycle_conflict_label(self, m: float, w: float, d: float, daily_reason: str) -> str:
        if m > 0 and w > 0 and d > 0:
            return "三周期共振"
        if w > 2.5 and d < 0:
            if any(key in daily_reason for key in ["真破位", "跌破中轨", "20日新低", "冲高回落"]):
                return "周强日破"
            return "周强日弱"
        if w < -2.5 and d > 0:
            return "周弱日强"
        if m < -0.5 and w > 2.0:
            return "月弱周强"
        if m > 1.5 and w < 1.5:
            return "月强周弱"
        return ""

    def _generate_tags(
            self,
            weekly_score: float,
            raw_total_score: float,
            prev_score: Optional[float],
            stop_dist: float,
            daily_reason: str
    ) -> List[str]:
        tags: List[str] = []
        danger_text = daily_reason
        has_danger = stop_dist < 0 or any(
            kw in danger_text for kw in ["止损", "诱多", "顶背离", "周强日破", "周强日弱", "破位"]
        )
        if (
                raw_total_score >= 15.0
                and self.rps >= 80
                and weekly_score >= 4.0
                and 2.8 <= stop_dist <= 8.0
                and not has_danger
        ):
            tags.append("👑 领涨龙头")
        elif raw_total_score >= 12.0:
            tags.append("🚀 主升浪")
        elif raw_total_score <= -12.0:
            tags.append("❄️ 主跌崩盘")
        if "极佳洗盘" in daily_reason and weekly_score >= 4.0:
            tags.append("💎 黄金坑低吸")
        if stop_dist < 0:
            tags.append("🚨 破位止损离场")
        if prev_score is not None and prev_score <= 0 and raw_total_score >= 10.0:
            tags.append("🔥 底部拐点")
        return tags

    # ────────────────── 主分析入口 ──────────────────

    def analyze(
            self,
            prev_score: Optional[float] = None,
            prev_stop: float = 0.0,
            prev_raw_score: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not self.data_loaded:
                Logger.error(f"{self.code} 数据未加载")
                return None
            adjustment_factor = float(self.data_quality.get("adjustment_factor", 1.0) or 1.0)
            self.prev_stop = prev_stop * adjustment_factor if prev_stop > 0 else 0.0
            self.calculate_indicators()
            ms, mr = self._analyze_monthly()
            ws, wr = self._analyze_weekly()
            ds, dr = self._analyze_daily()
            w: Dict[str, float] = ETFScoringConfig.MULTI_PERIOD_WEIGHTS
            raw: float = ms * w['monthly'] + ws * w['weekly'] + ds * w['daily']
            bonus, penalty, extra = self._apply_resonance_and_conflict(ms, ws, ds, dr)
            # === 第1点修复：分离原始强弱分和风险调整分 ===
            raw_total_score: float = round(raw + bonus + penalty, 1)
            total_score: float = raw_total_score
            # raw_status 只反映 ETF 自身技术状态，不包含大盘扣分。
            # test.py 买入资格优先看 raw_status，避免大盘惩罚通过 status 重复传导。
            raw_status: ETFStatus = self._determine_status(raw_total_score)
            # V4 中大盘只控制权限和总暴露，不再污染 ETF 自身技术分。
            market_tags: List[str] = ["🚨 大盘防守"] if not self.market_safe else []
            status: ETFStatus = self._determine_status(total_score)
            analysis_price: float = self._get_value(self.df_daily, 'close')
            price: float = self.executable_price if self.executable_price > 0 else analysis_price
            v4_factors = compute_asset_factors(
                self.df_daily,
                self.df_weekly,
                self.df_monthly,
                executable_price=price,
                adjustment_factor=adjustment_factor,
            )
            v4_stop = float((v4_factors.get("risk") or {}).get("stop_loss", 0.0) or 0.0)
            output_stop_loss: float = v4_stop or (
                self.stop_loss_price / adjustment_factor
                if adjustment_factor > 0 else self.stop_loss_price
            )
            stop_dist: float = 0.0
            if pd.notna(price) and price > 0 and output_stop_loss > 0:
                stop_dist = (price - output_stop_loss) / price * 100
            conflict_text = self._cycle_conflict_label(ms, ws, ds, dr)
            monthly_phase = self._phase_from_reason(mr)
            weekly_phase = self._phase_from_reason(wr)
            daily_setup = self._phase_from_reason(dr)
            tags = self._generate_tags(ws, raw_total_score, prev_score, stop_dist, f"{dr} {' '.join(extra)}")
            all_tags: List[str] = market_tags + extra + tags
            # ==================== 【新增】复合优先分计算 — 精确放在这里 ====================
            # 这就是之前说的“放在 ETFAnalyzer.analyze() 末尾”的位置
            # （Logger.info 打印之后、return 字典之前）
            rps_norm = max(0.0, min(1.0, self.rps / 100.0))
            # 修复：
            # composite_priority 的动量项应优先使用：
            # 本期 raw_total_score - 上期 raw_total_score。
            # 如果旧历史文件里还没有 raw_total_score，则回退使用 prev_score，兼容旧数据。
            if prev_raw_score is not None:
                score_delta_for_priority = raw_total_score - float(prev_raw_score)
            elif prev_score is not None:
                score_delta_for_priority = raw_total_score - float(prev_score)
            else:
                score_delta_for_priority = 0.0
            delta_norm = max(0.0, min(1.0, (score_delta_for_priority + 4.0) / 8.0))
            trend_strength = max(0.0, min(1.0, (raw_total_score - 2.5) / 14.5))
            resonance_strength = self._resonance_strength(ms, ws, ds)
            resonance_factor = {
                "strong": 1.0,
                "medium": 0.75,
                "weak": 0.5,
            }.get(resonance_strength, 0.25)
            risk_text = " ".join([dr, conflict_text, " ".join(all_tags)])
            clean_factor = 0.0 if any(
                bad in risk_text
                for bad in ["顶背离", "诱多", "止损", "破位", "真破位", "冲高回落"]
            ) else 1.0
            if 2.8 <= stop_dist <= 8.0:
                setup_quality = 1.0
            elif 1.5 <= stop_dist < 2.8:
                setup_quality = max(0.0, stop_dist / 2.8) * 0.7
            elif 8.0 < stop_dist <= 12.0:
                setup_quality = max(0.35, 1.0 - (stop_dist - 8.0) / 4.0 * 0.65)
            else:
                setup_quality = 0.35
            risk_quality = max(0.0, min(1.0, setup_quality * clean_factor))
            market_regime_factor = 1.0 if self.market_safe else 0.35
            signal_quality = max(0.0, min(1.0, (
                trend_strength * 0.35
                + rps_norm * 0.25
                + delta_norm * 0.20
                + resonance_factor * 0.10
                + risk_quality * 0.10
            )))
            composite_priority = 100.0 * (
                trend_strength * 0.30
                + rps_norm * 0.25
                + delta_norm * 0.18
                + resonance_factor * 0.12
                + risk_quality * 0.10
                + market_regime_factor * 0.05
            )
            composite_priority = round(max(0.0, min(100.0, composite_priority)), 1)
            conviction_tier = (
                "S" if composite_priority >= 78 else
                "A" if composite_priority >= 65 else
                "B" if composite_priority >= 48 else "C"
            )
            is_preferred = composite_priority >= 68 and clean_factor > 0.9
            # =================================================================================
            Logger.info(f"▶️ {self.name} ({self.code})")
            self._log_details(ms, ws, ds, dr)
            tag_str: str = f" → [{', '.join(all_tags)}]" if all_tags else ""
            Logger.info(
                f"   └── 📊 风险RPS:{self.rps:>5.1f} | 原始:{raw_total_score:>5.1f} | "
                f"展示:{total_score:>5.1f} | 止损距:{stop_dist:>.1f}% → {status.value}"
                f" | 优先级:{composite_priority:.1f}({conviction_tier}){' ⭐' if is_preferred else ''}{tag_str}\n"
            )
            return {
                "code": self.code,
                "name": self.name,
                "monthly_score": ms,
                "weekly_score": ws,
                "daily_score": ds,
                "raw_total_score": raw_total_score,
                "total_score": total_score,
                "raw_status": raw_status,
                "status": status,
                "tags": all_tags,
                "monthly_reason": mr,
                "weekly_reason": wr,
                "daily_reason": dr,
                "monthly_phase": monthly_phase,
                "weekly_phase": weekly_phase,
                "daily_setup": daily_setup,
                "cycle_conflict": conflict_text,
                "price": price,
                "stop_loss": output_stop_loss,
                "rps": self.rps,
                "alpha_score": self.alpha_score,
                "vol_adj_alpha": self.vol_adj_alpha,
                "beta_to_benchmark": self.beta_to_benchmark,
                "signal_quality": round(signal_quality, 2),
                "risk_quality": round(risk_quality, 2),
                "stop_dist": stop_dist,
                "data_date": self.get_data_date(),
                "is_stale": self.is_data_stale(),
                "data_quality": dict(self.data_quality),
                "v4_factors": v4_factors,
                "relative_strength": {},
                "v4_market": {},
                "v4_priority": 0.0,
                "weekly_confirmed": True,
                "monthly_confirmed": True,
                "score_delta": None,       # 后续主流程填充：展示分变化
                "raw_score_delta": None,   # 后续主流程填充：原始技术分变化
                # === 雷达排序字段：供信号 JSON 和看板解释使用 ===
                "composite_priority": composite_priority,
                "conviction_tier": conviction_tier,
                "is_preferred": is_preferred,
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

        prev: Optional[dict] = self._last_record()
        status: MarketStatus = self._status(total)
        sc: float = 0.0
        if prev:
            sc = round(total - prev.get('total_score', 0), 1)
        status = self._apply_status_hysteresis(status, total, prev, sc)
        safe: bool = status in (MarketStatus.STRONG_BULL, MarketStatus.BULL)
        atm: float = self._atr_mult(total)
        risk: str = self._risk_level(total)
        ch: bool = bool(prev and status.value != prev.get('status', ''))
        self._log(ts, tr, ms, mr, vs, vrl, vols, volrl, total, status, safe, atm, sc, ch)

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
            atr_multiplier=float(atm),
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

    @staticmethod
    def _status_from_value(value: Any, default: MarketStatus) -> MarketStatus:
        for status in MarketStatus:
            if value == status.value or value == status.name:
                return status
        return default

    def _apply_status_hysteresis(
            self,
            candidate: MarketStatus,
            total: float,
            prev: Optional[dict],
            score_change: float,
    ) -> MarketStatus:
        """普通状态切换需要连续确认，强变化允许立即切换。"""
        if not prev:
            return candidate

        prev_status = self._status_from_value(prev.get('status'), candidate)
        if candidate == prev_status:
            return candidate

        if abs(score_change) >= Config.ENV_HYSTERESIS_FAST_CHANGE:
            return candidate

        prev_candidate = self._status(float(prev.get('total_score', 0.0)))
        if prev_candidate == candidate:
            return candidate

        Logger.info(
            f"🧭 大盘状态待确认: {prev_status.value} → {candidate.value}，"
            "需连续2次或强变化触发"
        )
        return prev_status

    def _atr_mult(self, s: float) -> float:
        c: float = max(-6.0, min(6.0, s))
        return round(1.2 + (c + 6.0) / 12.0 * 1.3, 1)

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
        # [OPT-#3] 原子写入最新环境文件
        atomic_json_save(rd, Config.MARKET_ENV_LATEST_FILE)
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
        # [OPT-#3] 原子写入历史文件
        atomic_json_save(hist, Config.MARKET_ENV_HISTORY_FILE)

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
        if reason:
            Logger.warning(f"进入大盘防守默认模式: {reason}")
        return MarketEnvResult(
            date=datetime.now().strftime('%Y-%m-%d'),
            index_code=self.index_code,
            index_name=self.index_name,
            price=0,
            ma20=0,
            ma60=0,
            ma120=0,
            close_vs_ma20_pct=0,
            close_vs_ma60_pct=0,
            close_vs_ma120_pct=0,
            macd_hist=0,
            vol_ratio=0,
            atr_pct=0,
            atr_percentile=50,
            trend_score=0,
            momentum_score=0,
            volume_score=0,
            volatility_score=0,
            total_score=-6.0,
            trend_details={},
            momentum_details={},
            volume_details={},
            volatility_details={},
            status=MarketStatus.STRONG_BEAR,
            market_safe=False,
            atr_multiplier=1.2,
            risk_level="高",
            score_change=0,
            status_changed=False,
        )

    def _log(self, ts: float, tr: str, ms: float, mr: str, vs: float, vrl: str,
             vols: float, volr: str, total: float, status: MarketStatus,
             safe: bool, atm: float, sc: float, ch: bool) -> None:
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
        Logger.info(f"│ {icon} 安全:{'是' if safe else '否'} | ATR:{atm}x | 风险:{rl}")
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
# ║              HTMLReporter — sparkline + 背离显示             ║
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

    # ────────────────── Sparkline / 变动箭头 ──────────────────

    @staticmethod
    def _sparkline_svg(scores: List[Tuple[str, float]],
                       score_delta: Optional[float] = None) -> str:
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

        last_dot: str = ""
        if points:
            lx, ly = points[-1].split(',')
            last_dot = f'<circle cx="{lx}" cy="{ly}" r="2" fill="{color}"/>'

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

    # ────────────────── 市场宽度计算 ──────────────────

    @staticmethod
    def _compute_breadth(results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {'total': 0, 'bull': 0, 'bear': 0, 'neutral': 0,
                    'bull_pct': 0, 'bear_pct': 0, 'ratio': 0, 'signal': '无数据'}
        total: int = len(results)
        bull_values = {ETFStatus.EXTREME_BULL.value, ETFStatus.BULL.value, ETFStatus.WEAK_BULL.value}
        bear_values = {ETFStatus.EXTREME_BEAR.value, ETFStatus.BEAR.value, ETFStatus.WEAK_BEAR.value}

        def status_value(r: Dict[str, Any]) -> str:
            return _enum_value(r.get('raw_status', r.get('status', '')))

        bull: int = sum(1 for r in results if status_value(r) in bull_values)
        bear: int = sum(1 for r in results if status_value(r) in bear_values)
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
            'basis': 'raw_status',
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
<div class="hero-badge">PRO 3.3</div>
<h1>📡 ETF波段交易雷达</h1>
<p class="hero-sub">标准指标(MACD/BOLL/RSI/%B) · 大盘四维风控 · Alpha-RPS · 追踪止损 · MACD背离</p>
<p class="hero-time">📅 更新: <strong>{ts}</strong></p>
</div><div class="hero-gold-line"></div>
</header>
{env_h}
<section class="stats-grid">{stats}</section>
<section class="table-section">
<div class="table-header-bar"><h2>📊 标的评分明细</h2><span class="table-hint">点击表头排序 · 雷达优先级仅用于机会排序</span></div>
<div class="table-card"><table id="radarTable"><thead><tr>
<th onclick="sortTable(0)">标的 / 雷达优先级 ⇅</th>
<th onclick="sortTable(1)">趋势 ⇅</th>
<th onclick="sortTable(2)">月线(±5) ⇅</th>
<th onclick="sortTable(3)">周线(±6) ⇅</th>
<th onclick="sortTable(4)">日线(±4) ⇅</th>
<th onclick="sortTable(5)" class="text-center">参考止损 ⇅</th>
<th onclick="sortTable(6)" class="text-center">总分 ⇅</th>
<th onclick="sortTable(7)" class="text-center">状态 ⇅</th>
</tr></thead><tbody>{rows}</tbody></table></div>
</section>
<footer class="footer"><div class="footer-accent"></div>
<p>💡 顺大势、看节奏、抓时机 — 只做风险调整RPS高且多周期共振标的，破位必止损！</p>
<p class="footer-disclaimer">⚠️ 仅供学习，不构成投资建议。</p></footer>
</div>
<script>{js}</script></body></html>"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
            Logger.info(f"\n🎉 看板: {os.path.abspath(filename)}")
        except Exception as e:
            Logger.error("生成HTML失败", e)

    # ─── 大盘环境 HTML ───

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
            f"环境偏安全，雷达仅提示机会强弱；账户仓位由交易引擎独立裁决，止损参考 <strong>{env.atr_multiplier}x ATR</strong>"
            if env.market_safe else
            f"环境偏防守，雷达仅提示风险与机会排序；账户仓位由交易引擎独立裁决，止损参考 <strong>{env.atr_multiplier}x ATR</strong>"
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

    # ─── 表格行 ───

    @classmethod
    def _rows(cls, results: List[Dict]) -> str:
        rows: List[str] = []
        for r in sorted(results, key=lambda x: x.get('composite_priority', x['total_score']), reverse=True):
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

            sparkline_data: List[Tuple[str, float]] = r.get('sparkline_data', [])
            score_delta: Optional[float] = r.get('score_delta')
            sparkline_html: str = cls._sparkline_svg(sparkline_data, score_delta)
            spark_sort: float = sparkline_data[-1][1] if sparkline_data else r['total_score']

            priority = float(r.get('composite_priority', r['total_score']))
            tier = str(r.get('conviction_tier', 'C'))

            rows.append(f"""
<tr style="border-left:4px solid {ra}">
<td data-label="标的/雷达优先级" data-sort="{priority}"><div class="code-title"><span class="code-name">{r['name']}</span>
<span class="code-num">{r['code']} · 风险RPS <span class="rps-inline" style="color:{rc}">{r['rps']:.0f}</span>
<span class="rps-bar-bg"><span class="rps-bar-fill" style="width:{min(r['rps'], 100):.0f}%;background:{rc}"></span></span></span>
<span class="code-num">雷达优先级 {priority:.1f} · {tier}</span></div></td>
<td data-label="趋势" data-sort="{spark_sort}" class="col-spark">{sparkline_html}</td>
<td data-label="月线" data-sort="{r['monthly_score']}"><div class="signal-box"><span class="score-pill {sc(r['monthly_score'])}">{ss(r['monthly_score'])}</span><span class="signal-text">{r['monthly_reason']}</span></div></td>
<td data-label="周线" data-sort="{r['weekly_score']}"><div class="signal-box"><span class="score-pill {sc(r['weekly_score'])}">{ss(r['weekly_score'])}</span><span class="signal-text">{r['weekly_reason']}</span></div></td>
<td data-label="日线" data-sort="{r['daily_score']}"><div class="signal-box"><span class="score-pill {sc(r['daily_score'])}">{ss(r['daily_score'])}</span><span class="signal-text">{r['daily_reason']}</span></div></td>
<td data-label="参考止损" class="text-center" data-sort="{r['stop_dist']}"><div class="stop-price">参考止损 <strong>{r['stop_loss']:.3f}</strong></div>
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
            'k': sum(1 for r in results if "领涨龙头" in str(r.get('tags', []))),
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
    /* ═══════════════════════════════════════════════════════════
       ETF波段雷达 V4 - 专业级样式系统
       ═══════════════════════════════════════════════════════════ */
    /* ────────────────── CSS变量系统 ────────────────── */
    :root {
      /* 主色调 */
      --primary: #3b82f6;
      --primary-dark: #2563eb;
      --primary-light: #60a5fa;

      /* 语义色 */
      --success: #10b981;
      --success-light: #d1fae5;
      --danger: #ef4444;
      --danger-light: #fee2e2;
      --warning: #f59e0b;
      --warning-light: #fef3c7;
      --info: #06b6d4;

      /* 中性色 */
      --gray-50: #f9fafb;
      --gray-100: #f3f4f6;
      --gray-200: #e5e7eb;
      --gray-300: #d1d5db;
      --gray-400: #9ca3af;
      --gray-500: #6b7280;
      --gray-600: #4b5563;
      --gray-700: #374151;
      --gray-800: #1f2937;
      --gray-900: #111827;

      /* 背景色 */
      --bg-primary: #ffffff;
      --bg-secondary: #f8fafc;
      --bg-tertiary: #f1f5f9;

      /* 文字色 */
      --text-primary: #0f172a;
      --text-secondary: #475569;
      --text-tertiary: #94a3b8;

      /* 边框 */
      --border-color: #e2e8f0;
      --border-radius-sm: 8px;
      --border-radius-md: 12px;
      --border-radius-lg: 16px;
      --border-radius-xl: 20px;

      /* 阴影 */
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
      --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);

      /* 动画 */
      --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
      --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
      --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);

      /* 间距 */
      --spacing-xs: 4px;
      --spacing-sm: 8px;
      --spacing-md: 16px;
      --spacing-lg: 24px;
      --spacing-xl: 32px;
    }
    /* ────────────────── 全局重置 ────────────────── */
    *, *::before, *::after {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", 
                   "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      background-attachment: fixed;
      color: var(--text-primary);
      line-height: 1.6;
      padding: var(--spacing-md);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    /* ────────────────── 主容器 ────────────────── */
    .dashboard {
      max-width: 1600px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: var(--spacing-lg);
    }
    /* ────────────────── Hero区域 ────────────────── */
    .hero {
      background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
      border-radius: var(--border-radius-xl);
      overflow: hidden;
      box-shadow: var(--shadow-xl);
      position: relative;
    }
    .hero::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
      opacity: 0.4;
    }
    .hero-inner {
      padding: var(--spacing-xl) var(--spacing-lg);
      text-align: center;
      position: relative;
      z-index: 1;
    }
    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #fff;
      font-size: 0.75rem;
      font-weight: 800;
      padding: 6px 16px;
      border-radius: 999px;
      letter-spacing: 1px;
      margin-bottom: var(--spacing-md);
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
      animation: pulse-badge 2s ease-in-out infinite;
    }
    @keyframes pulse-badge {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.05); }
    }
    .hero h1 {
      font-size: 2.5rem;
      font-weight: 900;
      letter-spacing: -0.5px;
      color: #fff;
      margin-bottom: var(--spacing-sm);
      text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .hero-sub {
      color: #cbd5e1;
      font-size: 1rem;
      margin-bottom: var(--spacing-sm);
      font-weight: 500;
    }
    .hero-time {
      color: #e2e8f0;
      font-size: 0.875rem;
      font-weight: 600;
    }
    .hero-time strong {
      color: #fbbf24;
      font-weight: 700;
    }
    .hero-gold-line {
      height: 4px;
      background: linear-gradient(90deg, 
        transparent 0%, 
        #f59e0b 20%, 
        #fbbf24 50%, 
        #f59e0b 80%, 
        transparent 100%);
      box-shadow: 0 2px 8px rgba(251, 191, 36, 0.5);
    }
    /* ────────────────── 大盘环境卡片 ────────────────── */
    .market-env {
      background: var(--bg-primary);
      border-radius: var(--border-radius-lg);
      overflow: hidden;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border-color);
      display: grid;
      grid-template-columns: 6px 1fr;
      transition: transform var(--transition-base), box-shadow var(--transition-base);
    }
    .market-env:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-xl);
    }
    .env-status-strip {
      background: linear-gradient(180deg, var(--primary), var(--primary-dark));
    }
    .env-danger .env-status-strip {
      background: linear-gradient(180deg, #dc2626, #b91c1c);
    }
    .env-neutral .env-status-strip {
      background: linear-gradient(180deg, #f59e0b, #d97706);
    }
    .env-safe .env-status-strip {
      background: linear-gradient(180deg, #10b981, #059669);
    }
    .env-body {
      padding: var(--spacing-lg);
    }
    .env-danger .env-body {
      background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    .env-neutral .env-body {
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    .env-safe .env-body {
      background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    .env-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: var(--spacing-md);
      gap: var(--spacing-md);
      flex-wrap: wrap;
    }
    .env-title-group h3 {
      font-size: 1.5rem;
      font-weight: 800;
      color: var(--text-primary);
      margin-bottom: var(--spacing-xs);
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
    }
    .env-status-text {
      font-weight: 900;
    }
    .env-danger .env-status-text {
      color: #dc2626;
    }
    .env-neutral .env-status-text {
      color: #f59e0b;
    }
    .env-safe .env-status-text {
      color: #10b981;
    }
    .env-index-name {
      font-size: 0.875rem;
      color: var(--text-secondary);
      font-weight: 600;
    }
    .env-score-group {
      display: flex;
      align-items: baseline;
      gap: var(--spacing-sm);
      background: rgba(255, 255, 255, 0.6);
      padding: var(--spacing-sm) var(--spacing-md);
      border-radius: var(--border-radius-md);
      backdrop-filter: blur(10px);
    }
    .env-score-big {
      font-size: 2.5rem;
      font-weight: 900;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      line-height: 1;
    }
    .env-score-change {
      font-size: 1rem;
      font-weight: 700;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }
    .env-desc {
      font-size: 1rem;
      line-height: 1.7;
      margin-bottom: var(--spacing-md);
      padding: var(--spacing-md);
      background: rgba(255, 255, 255, 0.5);
      border-radius: var(--border-radius-sm);
      border-left: 4px solid var(--primary);
    }
    .status-change-alert {
      background: linear-gradient(135deg, #fef08a, #fde047);
      color: #854d0e;
      padding: var(--spacing-sm) var(--spacing-md);
      border-radius: var(--border-radius-sm);
      font-weight: 700;
      font-size: 0.875rem;
      margin-bottom: var(--spacing-md);
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      animation: pulse-alert 2s ease-in-out infinite;
      box-shadow: 0 4px 12px rgba(234, 179, 8, 0.3);
    }
    @keyframes pulse-alert {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.9; transform: scale(0.98); }
    }
    .env-details-grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: var(--spacing-lg);
      margin-bottom: var(--spacing-md);
    }
    /* 维度评分 */
    .env-dimensions {
      display: flex;
      flex-direction: column;
      gap: var(--spacing-md);
    }
    .dim-section-title {
      font-size: 0.75rem;
      font-weight: 800;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 1.5px;
      margin-bottom: var(--spacing-xs);
      display: flex;
      align-items: center;
      gap: var(--spacing-xs);
    }
    .dim-section-title::before {
      content: '';
      width: 3px;
      height: 14px;
      background: var(--primary);
      border-radius: 2px;
    }
    .dimension-row {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-sm);
      background: rgba(255, 255, 255, 0.6);
      border-radius: var(--border-radius-sm);
      transition: all var(--transition-fast);
    }
    .dimension-row:hover {
      background: rgba(255, 255, 255, 0.9);
      transform: translateX(4px);
    }
    .dim-label {
      min-width: 48px;
      font-weight: 800;
      font-size: 0.875rem;
      color: var(--text-primary);
    }
    .dim-bar-bg {
      flex: 1;
      height: 20px;
      background: var(--gray-200);
      border-radius: 10px;
      overflow: hidden;
      position: relative;
    }
    .dim-bar-fill {
      height: 100%;
      border-radius: 10px;
      transition: width var(--transition-slow);
      position: relative;
      overflow: hidden;
    }
    .dim-bar-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, 
        rgba(255, 255, 255, 0) 0%, 
        rgba(255, 255, 255, 0.3) 50%, 
        rgba(255, 255, 255, 0) 100%);
      animation: shimmer 2s infinite;
    }
    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
    .dim-score {
      min-width: 50px;
      font-weight: 900;
      font-size: 0.875rem;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      text-align: right;
    }
    .dim-reason {
      font-size: 0.75rem;
      color: var(--text-secondary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 140px;
      font-weight: 600;
    }
    /* 均线偏离 */
    .env-ma-devs {
      display: flex;
      flex-direction: column;
      gap: var(--spacing-sm);
    }
    .ma-tag {
      display: inline-flex;
      align-items: center;
      gap: var(--spacing-xs);
      font-size: 0.8rem;
      padding: var(--spacing-xs) var(--spacing-sm);
      border-radius: var(--border-radius-sm);
      background: rgba(255, 255, 255, 0.7);
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      font-weight: 700;
      border-left: 3px solid var(--gray-400);
      transition: all var(--transition-fast);
    }
    .ma-tag:hover {
      background: rgba(255, 255, 255, 0.95);
      transform: translateX(4px);
    }
    .pct-bar-container {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-sm);
      background: rgba(255, 255, 255, 0.6);
      border-radius: var(--border-radius-sm);
    }
    .pct-bar-bg {
      flex: 1;
      height: 16px;
      background: var(--gray-200);
      border-radius: 8px;
      overflow: hidden;
    }
    .pct-bar-fill {
      height: 100%;
      border-radius: 8px;
      transition: width var(--transition-slow);
    }
    .pct-label {
      font-size: 0.8rem;
      font-weight: 800;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      white-space: nowrap;
    }
    /* 市场宽度 */
    .breadth-section {
      margin-top: var(--spacing-md);
      padding-top: var(--spacing-md);
      border-top: 2px solid rgba(255, 255, 255, 0.5);
    }
    .breadth-bar-wrap {
      margin: var(--spacing-sm) 0;
    }
    .breadth-bar {
      display: flex;
      height: 16px;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .breadth-seg {
      transition: width var(--transition-slow);
    }
    .breadth-bull {
      background: linear-gradient(135deg, #dc2626, #ef4444);
    }
    .breadth-neutral {
      background: linear-gradient(135deg, #94a3b8, #cbd5e1);
    }
    .breadth-bear {
      background: linear-gradient(135deg, #10b981, #34d399);
    }
    .breadth-labels {
      display: flex;
      justify-content: space-between;
      margin-top: var(--spacing-xs);
      font-size: 0.75rem;
      font-weight: 700;
    }
    .bl-bull { color: #dc2626; }
    .bl-neutral { color: #64748b; }
    .bl-bear { color: #10b981; }
    .breadth-ratio {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.875rem;
      margin-top: var(--spacing-sm);
      padding: var(--spacing-sm);
      background: rgba(255, 255, 255, 0.6);
      border-radius: var(--border-radius-sm);
    }
    .breadth-signal {
      font-weight: 800;
      padding: 4px var(--spacing-sm);
      border-radius: var(--border-radius-sm);
      background: rgba(255, 255, 255, 0.9);
    }
    /* ────────────────── 统计卡片网格 ────────────────── */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: var(--spacing-md);
    }
    .stat-card {
      background: var(--bg-primary);
      border-radius: var(--border-radius-lg);
      padding: var(--spacing-lg);
      display: flex;
      align-items: center;
      gap: var(--spacing-md);
      box-shadow: var(--shadow-md);
      border: 1px solid var(--border-color);
      transition: all var(--transition-base);
      position: relative;
      overflow: hidden;
    }
    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: var(--primary);
      transform: scaleY(0);
      transition: transform var(--transition-base);
    }
    .stat-card:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-xl);
    }
    .stat-card:hover::before {
      transform: scaleY(1);
    }
    .stat-icon {
      font-size: 2.5rem;
      flex-shrink: 0;
      filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
    }
    .stat-info {
      flex: 1;
    }
    .stat-val {
      font-size: 2rem;
      font-weight: 900;
      line-height: 1;
      margin-bottom: var(--spacing-xs);
    }
    .stat-label {
      font-size: 0.875rem;
      color: var(--text-secondary);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .stat-blue .stat-val { color: #3b82f6; }
    .stat-red .stat-val { color: #dc2626; }
    .stat-purple .stat-val { color: #7c3aed; }
    .stat-orange .stat-val { color: #ea580c; }
    /* ────────────────── 表格区域 ────────────────── */
    .table-section {
      background: var(--bg-primary);
      border-radius: var(--border-radius-lg);
      overflow: hidden;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border-color);
    }
    .table-header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: var(--spacing-lg);
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      border-bottom: 2px solid var(--border-color);
    }
    .table-header-bar h2 {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
    }
    .table-hint {
      font-size: 0.8rem;
      color: var(--text-secondary);
      font-weight: 600;
    }
    .table-card {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    thead {
      background: var(--gray-100);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    th {
      padding: var(--spacing-md);
      font-weight: 800;
      color: var(--text-secondary);
      font-size: 0.875rem;
      text-align: left;
      border-bottom: 2px solid var(--border-color);
      cursor: pointer;
      white-space: nowrap;
      transition: all var(--transition-fast);
      user-select: none;
    }
    th:hover {
      background: var(--gray-200);
      color: var(--text-primary);
    }
    th:active {
      transform: scale(0.98);
    }
    td {
      padding: var(--spacing-md);
      border-bottom: 1px solid var(--gray-100);
      font-size: 0.9rem;
      vertical-align: middle;
    }
    tbody tr {
      transition: all var(--transition-fast);
      position: relative;
    }
    tbody tr:hover {
      background: var(--gray-50);
    }
    tbody tr:hover::before {
      background: var(--primary);
    }
    /* 代码标题列 */
    .code-title {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .code-name {
      font-weight: 800;
      font-size: 1rem;
      color: var(--text-primary);
    }
    .code-num {
      font-size: 0.75rem;
      color: var(--text-tertiary);
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      display: flex;
      align-items: center;
      gap: var(--spacing-xs);
      flex-wrap: wrap;
    }
    .rps-inline {
      font-weight: 900;
    }
    .rps-bar-bg {
      display: inline-block;
      width: 50px;
      height: 6px;
      background: var(--gray-200);
      border-radius: 3px;
      overflow: hidden;
      vertical-align: middle;
    }
    .rps-bar-fill {
      display: block;
      height: 100%;
      border-radius: 3px;
      transition: width var(--transition-slow);
    }
    /* Sparkline */
    .col-spark {
      text-align: center;
      vertical-align: middle;
      padding: var(--spacing-sm) !important;
    }
    .sparkline {
      display: block;
      filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
    }
    .sparkline-na {
      font-size: 0.75rem;
      color: var(--text-tertiary);
    }
    .sparkline-arrow {
      font-size: 0.8rem;
      font-weight: 800;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      gap: 2px;
      padding: 2px 6px;
      border-radius: var(--border-radius-sm);
      background: rgba(0, 0, 0, 0.05);
    }
    /* 信号盒子 */
    .signal-box {
      display: flex;
      align-items: center;
      gap: var(--spacing-xs);
      flex-wrap: wrap;
    }
    .score-pill {
      min-width: 44px;
      padding: 4px var(--spacing-sm);
      border-radius: var(--border-radius-sm);
      text-align: center;
      font-weight: 900;
      font-size: 0.875rem;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      box-shadow: var(--shadow-sm);
      transition: all var(--transition-fast);
    }
    .score-pill:hover {
      transform: scale(1.05);
    }
    .score-pos {
      background: linear-gradient(135deg, #fee2e2, #fecaca);
      color: #b91c1c;
      border: 1px solid #fca5a5;
    }
    .score-neg {
      background: linear-gradient(135deg, #dcfce7, #bbf7d0);
      color: #15803d;
      border: 1px solid #86efac;
    }
    .score-zero {
      background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
      color: #64748b;
      border: 1px solid #cbd5e1;
    }
    .signal-text {
      font-size: 0.8rem;
      color: var(--text-secondary);
      font-weight: 600;
    }
    /* 止损信息 */
    .text-center {
      text-align: center;
    }
    .stop-price {
      font-size: 0.875rem;
      color: var(--text-secondary);
      margin-bottom: 4px;
    }
    .stop-price strong {
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      color: var(--text-primary);
    }
    .stop-dist {
      font-size: 0.875rem;
      font-weight: 800;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }
    /* 总分 */
    .total-score {
      font-size: 1.75rem;
      font-weight: 900;
      font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    /* 状态徽章 */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px var(--spacing-md);
      border-radius: 999px;
      font-weight: 800;
      font-size: 0.75rem;
      white-space: nowrap;
      box-shadow: var(--shadow-sm);
      transition: all var(--transition-fast);
    }
    .status-badge:hover {
      transform: scale(1.05);
    }
    .badge-bull-super {
      background: linear-gradient(135deg, #fee2e2, #fecaca);
      color: #991b1b;
      border: 2px solid #f87171;
      box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .badge-bull-strong {
      background: linear-gradient(135deg, #fee2e2, #fef2f2);
      color: #b91c1c;
      border: 1px solid #fca5a5;
    }
    .badge-bull-weak {
      background: linear-gradient(135deg, #fffbeb, #fef3c7);
      color: #b45309;
      border: 1px solid #fde68a;
    }
    .badge-neutral {
      background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
      color: #475569;
      border: 1px solid #cbd5e1;
    }
    .badge-bear-weak {
      background: linear-gradient(135deg, #f0fdfa, #ccfbf1);
      color: #0f766e;
      border: 1px solid #99f6e4;
    }
    .badge-bear-strong {
      background: linear-gradient(135deg, #dcfce7, #f0fdf4);
      color: #166534;
      border: 1px solid #86efac;
    }
    .badge-bear-super {
      background: linear-gradient(135deg, #166534, #15803d);
      color: #f0fdf4;
      border: 2px solid #86efac;
      box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    /* 标签行 */
    .tag-row {
      margin-top: var(--spacing-sm);
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .tag-mark {
      display: inline-flex;
      align-items: center;
      font-size: 0.7rem;
      padding: 3px var(--spacing-sm);
      border-radius: var(--border-radius-sm);
      font-weight: 800;
      transition: all var(--transition-fast);
    }
    .tag-mark:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }
    .tag-king {
      background: linear-gradient(135deg, #ede9fe, #ddd6fe);
      color: #6d28d9;
      border: 1px solid #c4b5fd;
    }
    .tag-pit {
      background: linear-gradient(135deg, #fef9c3, #fef08a);
      color: #854d0e;
      border: 1px solid #fde047;
    }
    .tag-new-bull {
      background: linear-gradient(135deg, #ffedd5, #fed7aa);
      color: #c2410c;
      border: 1px solid #fdba74;
    }
    .tag-danger {
      background: linear-gradient(135deg, #16a34a, #22c55e);
      color: #fff;
      border: 1px solid #166534;
    }
    .tag-trap {
      background: linear-gradient(135deg, #f59e0b, #fbbf24);
      color: #fff;
      border: 1px solid #d97706;
    }
    .tag-fire {
      background: linear-gradient(135deg, #dbeafe, #bfdbfe);
      color: #1e40af;
      border: 1px solid #93c5fd;
    }
    /* ────────────────── 页脚 ────────────────── */
    .footer {
      background: var(--bg-primary);
      border-radius: var(--border-radius-lg);
      overflow: hidden;
      border: 1px solid var(--border-color);
      text-align: center;
      box-shadow: var(--shadow-md);
    }
    .footer-accent {
      height: 4px;
      background: linear-gradient(90deg, 
        transparent 0%, 
        #dc2626 20%, 
        #f59e0b 50%, 
        #dc2626 80%, 
        transparent 100%);
    }
    .footer p {
      padding: var(--spacing-md);
      color: var(--text-secondary);
      font-size: 0.9rem;
      line-height: 1.6;
    }
    .footer p:first-of-type {
      padding-bottom: var(--spacing-xs);
      font-weight: 600;
    }
    .footer-disclaimer {
      font-size: 0.75rem !important;
      color: var(--text-tertiary) !important;
      padding-top: 0 !important;
      font-weight: 500 !important;
    }
    /* ────────────────── 响应式设计 ────────────────── */
    @media (max-width: 1200px) {
      .env-details-grid {
        grid-template-columns: 1fr;
      }

    }
    @media (max-width: 768px) {
      body {
        padding: var(--spacing-sm);
      }

      .dashboard {
        gap: var(--spacing-md);
      }

      /* Hero */
      .hero-inner {
        padding: var(--spacing-lg) var(--spacing-md);
      }

      .hero h1 {
        font-size: 1.75rem;
      }

      .hero-sub {
        font-size: 0.875rem;
      }

      /* 大盘环境 */
      .env-header {
        flex-direction: column;
        align-items: flex-start;
      }

      .env-score-group {
        width: 100%;
        justify-content: space-between;
      }

      .env-details-grid {
        grid-template-columns: 1fr;
        gap: var(--spacing-md);
      }

      .dim-reason {
        max-width: 100px;
      }

      /* 统计卡片 */
      .stats-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: var(--spacing-sm);
      }

      .stat-card {
        padding: var(--spacing-md);
      }

      .stat-icon {
        font-size: 2rem;
      }

      .stat-val {
        font-size: 1.5rem;
      }

      /* 表格 */
      .table-header-bar {
        padding: var(--spacing-md);
      }

      .table-header-bar h2 {
        font-size: 1rem;
      }

      .table-hint {
        display: none;
      }

      .col-spark {
        display: none;
      }

      /* 移动端卡片式表格 */
      .table-card {
        background: transparent;
        padding: 0;
      }

      table thead {
        display: none;
      }

      table, table tbody {
        display: block;
      }

      table tbody tr {
        display: flex;
        flex-direction: column;
        background: var(--bg-primary);
        border-radius: var(--border-radius-md);
        margin: 0 var(--spacing-sm) var(--spacing-md);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        overflow: hidden;
      }

      table tbody tr:hover {
        background: var(--bg-primary);
        box-shadow: var(--shadow-md);
      }

      table tbody tr td {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--spacing-sm) var(--spacing-md);
        border-bottom: 1px solid var(--gray-100);
        font-size: 0.875rem;
      }

      table tbody tr td:last-child {
        border-bottom: none;
      }

      table tbody tr td::before {
        content: attr(data-label);
        font-weight: 800;
        color: var(--text-secondary);
        font-size: 0.75rem;
        flex-shrink: 0;
        margin-right: var(--spacing-sm);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      table tbody tr td:first-child {
        background: var(--gray-50);
        border-bottom: 2px solid var(--border-color);
        flex-direction: column;
        align-items: flex-start;
        padding: var(--spacing-md);
      }

      table tbody tr td:first-child::before {
        display: none;
      }

      table tbody tr td[data-label="趋势"] {
        display: none;
      }

      table tbody tr td:nth-child(7) {
        justify-content: center;
        background: var(--gray-50);
      }

      table tbody tr td:nth-child(7)::before {
        display: none;
      }

      table tbody tr td:last-child {
        justify-content: center;
        padding: var(--spacing-md);
      }

      table tbody tr td:last-child::before {
        display: none;
      }

      .signal-text {
        white-space: normal;
        font-size: 0.75rem;
      }

      .total-score {
        font-size: 1.5rem;
      }

      .tag-mark {
        font-size: 0.65rem;
      }

      .status-badge {
        font-size: 0.75rem;
      }

    }
    @media (max-width: 480px) {
      .hero h1 {
        font-size: 1.5rem;
      }

      .env-score-big {
        font-size: 2rem;
      }

      .dim-label {
        min-width: 36px;
        font-size: 0.75rem;
      }

      .dim-reason {
        max-width: 80px;
        font-size: 0.7rem;
      }

      .stats-grid {
        grid-template-columns: 1fr;
      }

      .stat-card {
        gap: var(--spacing-sm);
      }
    }
    /* ────────────────── 打印样式 ────────────────── */
    @media print {
      body {
        background: white;
        padding: 0;
      }

      .dashboard {
        max-width: 100%;
      }

      .hero {
        background: #1e3a8a;
        page-break-after: avoid;
      }

      .market-env,
      .table-section {
        page-break-inside: avoid;
        box-shadow: none;
        border: 1px solid #000;
      }

      .stat-card:hover,
      tbody tr:hover {
        transform: none;
        box-shadow: none;
      }

      .col-spark {
        display: none;
      }
    }
    /* ────────────────── 辅助动画 ────────────────── */
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    .dashboard > * {
      animation: fadeIn 0.5s ease-out backwards;
    }
    .dashboard > *:nth-child(1) { animation-delay: 0.05s; }
    .dashboard > *:nth-child(2) { animation-delay: 0.1s; }
    .dashboard > *:nth-child(3) { animation-delay: 0.15s; }
    .dashboard > *:nth-child(4) { animation-delay: 0.2s; }
    .dashboard > *:nth-child(5) { animation-delay: 0.25s; }
    .dashboard > *:nth-child(6) { animation-delay: 0.3s; }
    /* ────────────────── 滚动条美化 ────────────────── */
    ::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    ::-webkit-scrollbar-track {
      background: var(--gray-100);
      border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
      background: var(--gray-400);
      border-radius: 5px;
      transition: background var(--transition-fast);
    }
    ::-webkit-scrollbar-thumb:hover {
      background: var(--gray-500);
    }
    /* ────────────────── 选中文本样式 ────────────────── */
    ::selection {
      background: rgba(59, 130, 246, 0.2);
      color: var(--text-primary);
    }
    ::-moz-selection {
      background: rgba(59, 130, 246, 0.2);
      color: var(--text-primary);
    }
    """
        js: str = """
    /* ═══════════════════════════════════════════════════════════
       ETF波段雷达 V4 - 交互脚本
       ═══════════════════════════════════════════════════════════ */
    // 排序状态数组
    let sortStates = [0, 0, 0, 0, 0, 0, 0, 0];
    // 表格排序函数
    function sortTable(colIndex) {
      const table = document.getElementById("radarTable");
      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));

      // 切换排序方向
      let isAsc = sortStates[colIndex] === 1;
      sortStates = [0, 0, 0, 0, 0, 0, 0, 0];
      sortStates[colIndex] = isAsc ? 0 : 1;

      // 排序
      rows.sort((a, b) => {
        let valA = parseFloat(a.cells[colIndex].getAttribute("data-sort")) || 0;
        let valB = parseFloat(b.cells[colIndex].getAttribute("data-sort")) || 0;
        return isAsc ? (valA - valB) : (valB - valA);
      });

      // 重新插入
      rows.forEach(row => tbody.appendChild(row));

      // 更新表头视觉反馈
      const headers = table.querySelectorAll("th");
      headers.forEach((th, idx) => {
        th.style.background = idx === colIndex ? "#e2e8f0" : "";
        th.style.color = idx === colIndex ? "#0f172a" : "";
      });
    }
    // 页面加载完成后执行
    document.addEventListener('DOMContentLoaded', function() {
      // 添加加载动画
      document.body.style.opacity = '0';
      setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-in';
        document.body.style.opacity = '1';
      }, 100);

      // 为所有卡片添加入场动画
      const cards = document.querySelectorAll('.stat-card');
      cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
          card.style.transition = 'all 0.5s ease-out';
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, 100 + index * 50);
      });

      // Sparkline tooltip (简化版)
      const sparklines = document.querySelectorAll('.sparkline');
      sparklines.forEach(svg => {
        svg.style.cursor = 'pointer';
        svg.addEventListener('mouseenter', function() {
          this.style.filter = 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2))';
        });
        svg.addEventListener('mouseleave', function() {
          this.style.filter = 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1))';
        });
      });

      // 表格行点击高亮
      const tableRows = document.querySelectorAll('tbody tr');
      tableRows.forEach(row => {
        row.addEventListener('click', function() {
          tableRows.forEach(r => r.style.background = '');
          this.style.background = '#f0f9ff';
          setTimeout(() => {
            this.style.background = '';
          }, 2000);
        });
      });

      // 添加返回顶部按钮
      const backToTop = document.createElement('button');
      backToTop.innerHTML = '↑';
      backToTop.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        opacity: 0;
        transition: all 0.3s ease;
        z-index: 1000;
      `;
      document.body.appendChild(backToTop);

      // 滚动显示返回顶部按钮
      window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
          backToTop.style.opacity = '1';
          backToTop.style.transform = 'scale(1)';
        } else {
          backToTop.style.opacity = '0';
          backToTop.style.transform = 'scale(0.8)';
        }
      });

      backToTop.addEventListener('click', () => {
        window.scrollTo({
          top: 0,
          behavior: 'smooth'
        });
      });

      backToTop.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.1)';
      });

      backToTop.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
      });

      // 性能优化：懒加载图片（如果有）
      if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              const img = entry.target;
              img.src = img.dataset.src;
              img.classList.remove('lazy');
              imageObserver.unobserve(img);
            }
          });
        });

        const lazyImages = document.querySelectorAll('img.lazy');
        lazyImages.forEach(img => imageObserver.observe(img));
      }

      console.log('%c📡 ETF波段雷达 V4', 'color: #3b82f6; font-size: 20px; font-weight: bold;');
      console.log('%c系统已就绪 | 数据实时更新', 'color: #10b981; font-size: 14px;');
    });
    """
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
    """[OPT-#3] 保存历史记录（原子写入）。"""
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
            'raw_total_score': r.get('raw_total_score', r['total_score']),
            'stop_loss': r['stop_loss'],
            'stop_dist': r['stop_dist'],
            'status': _enum_value(r['status']),
            'daily_scores': sorted_scores,
        }
    # [OPT-#3] 使用原子写入
    if not atomic_json_save(data, Config.HISTORY_FILE):
        Logger.error("保存历史失败（原子写入回退）")


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


def fetch_single_etf(
        code: str,
        name: str,
        force_download: bool,
        market_safe: bool,
        atr_multiplier: float
) -> Optional[ETFAnalyzer]:
    try:
        a = ETFAnalyzer(
            code,
            name,
            force_download=force_download,
            market_safe=market_safe,
            atr_multiplier=atr_multiplier,
        )
        if a.fetch_data():
            return a
    except Exception as e:
        Logger.error(f"获取{code}失败", e)
    return None


def calc_blended_return(df: pd.DataFrame, window: int = 120) -> float:
    """混合收益率：20日、60日、长周期收益加权。"""
    try:
        if len(df) < 21:
            return -999.0
        close = df['close']
        p: float = close.iloc[-1]
        r20: float = (p - close.iloc[-21]) / close.iloc[-21]
        w60: int = min(61, len(df))
        r60: float = (p - close.iloc[-w60]) / close.iloc[-w60]
        w_long: int = min(window + 1, len(df))
        r_long: float = (p - close.iloc[-w_long]) / close.iloc[-w_long]
        return 0.4 * r20 + 0.3 * r60 + 0.3 * r_long
    except Exception:
        return -999.0


def calc_risk_adjusted_alpha(
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
        window: int = 120,
) -> Dict[str, float]:
    """相对基准的风险调整 Alpha，用作 RPS 排名底层分。"""
    empty = {
        "alpha_score": -999.0,
        "vol_adj_alpha": -999.0,
        "beta_to_benchmark": 1.0,
        "ret_blend": -999.0,
    }
    try:
        ret = calc_blended_return(df, window=window)
        if ret == -999.0 or df.empty or "close" not in df.columns:
            return empty

        bm_ret = calc_blended_return(benchmark_df, window=window) if benchmark_df is not None else 0.0
        if bm_ret == -999.0:
            bm_ret = 0.0

        etf_rets = df["close"].pct_change().dropna()
        beta = 1.0
        if benchmark_df is not None and not benchmark_df.empty and "close" in benchmark_df.columns:
            aligned_returns = align_return_series(df, benchmark_df, window=max(20, window))
            n = len(aligned_returns)
            if n >= 20:
                e = aligned_returns["etf_return"].to_numpy(dtype=float)
                b = aligned_returns["benchmark_return"].to_numpy(dtype=float)
                bm_var = float(np.var(b))
                if bm_var > 1e-8:
                    beta = float(np.cov(e, b)[0, 1] / bm_var)
                    beta = max(-0.5, min(2.5, beta))

        alpha = ret - beta * bm_ret
        vol20 = float(etf_rets.tail(20).std()) if len(etf_rets) >= 20 else 0.0
        vol_adj_alpha = alpha / max(vol20, 0.005)
        return {
            "alpha_score": round(float(alpha), 6),
            "vol_adj_alpha": round(float(vol_adj_alpha), 6),
            "beta_to_benchmark": round(float(beta), 3),
            "ret_blend": round(float(ret), 6),
        }
    except Exception:
        return empty


_V4_CALIBRATION_CACHE: Optional[V4CalibrationModel] = None
_V4_CALIBRATION_LOADED: bool = False
_V4_CALIBRATION_STATUS_REASON: str = "CALIBRATION_NOT_LOADED"


def v4_calibration_status_reason() -> str:
    return _V4_CALIBRATION_STATUS_REASON


def load_v4_calibration(path: Optional[str] = None) -> Optional[V4CalibrationModel]:
    global _V4_CALIBRATION_CACHE, _V4_CALIBRATION_LOADED, _V4_CALIBRATION_STATUS_REASON
    path = path or Config.V4_CALIBRATION_FILE
    if _V4_CALIBRATION_LOADED:
        return _V4_CALIBRATION_CACHE
    _V4_CALIBRATION_LOADED = True
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        model = V4CalibrationModel.from_dict(value)
        trained_until = pd.to_datetime(model.trained_until, errors="coerce")
        if pd.isna(trained_until) or (pd.Timestamp.now().normalize() - trained_until).days > 180:
            _V4_CALIBRATION_STATUS_REASON = "CALIBRATION_STALE"
            return None
        current_fingerprint = fingerprint_price_directory(Config.DATA_DIR, model.trained_until)
        if not current_fingerprint or current_fingerprint != model.data_fingerprint:
            _V4_CALIBRATION_STATUS_REASON = "CALIBRATION_FINGERPRINT_MISMATCH"
            return None
        if not bool((model.thresholds or {}).get("approved", False)):
            _V4_CALIBRATION_STATUS_REASON = "CALIBRATION_NOT_APPROVED"
            return None
        _V4_CALIBRATION_CACHE = model
        _V4_CALIBRATION_STATUS_REASON = "APPROVED"
        return model
    except Exception as error:
        _V4_CALIBRATION_STATUS_REASON = "CALIBRATION_UNAVAILABLE"
        Logger.warning("v4校准产物不可用，新仓保持阻断", error)
        return None


def enrich_v4_signal(
        result: Dict[str, Any],
        calibration_model: Optional[V4CalibrationModel] = None,
) -> Dict[str, Any]:
    model = calibration_model if calibration_model is not None else load_v4_calibration()
    if model is None:
        calibration: Dict[str, Any] = {
            "early_stop_probability_3d": 1.0,
            "win_probability_10d": 0.0,
            "expected_excess_return_10d": 0.0,
            "sample_count": 0,
            "confidence": "LOW",
            "version": "v4-unapproved",
            "approved": False,
            "status_reason": v4_calibration_status_reason(),
        }
    else:
        calibration = model.predict(v4_calibration_features(result))
        calibration["approved"] = True
        calibration["status_reason"] = "APPROVED"
    return build_v4_signal(result, calibration=calibration)


def save_etf_signals(
    results: List[Dict[str, Any]],
    env_result: MarketEnvResult,
    breadth: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """保存唯一的 V4 公共信号合约。"""
    try:
        signals = [enrich_v4_signal(dict(result)) for result in results]
        signals.sort(
            key=lambda signal: float((signal.get("entry") or {}).get("priority", 0.0)),
            reverse=True,
        )
        market_policy = dict((signals[0].get("market_policy") or {})) if signals else {}
        out: Dict[str, Any] = {
            "schema_version": V4_SCHEMA_VERSION,
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_policy": market_policy,
            "market_breadth": dict(breadth),
            "signals": [],
        }
        for signal in signals:
            out["signals"].append(
                {
                    "schema_version": V4_SCHEMA_VERSION,
                    "signal_id": str(signal.get("signal_id", "")),
                    "code": str(signal.get("code", "")),
                    "name": str(signal.get("name", "")),
                    "data_date": str(signal.get("data_date", "")),
                    "price": round(float(signal.get("price", 0.0) or 0.0), 3),
                    "data_quality": dict(signal.get("data_quality") or {}),
                    "trend": dict(signal.get("trend") or {}),
                    "entry": dict(signal.get("entry") or {}),
                    "relative_strength": dict(signal.get("relative_strength") or {}),
                    "risk": dict(signal.get("risk") or {}),
                    "market_policy": dict(signal.get("market_policy") or {}),
                    "calibration": dict(signal.get("calibration") or {}),
                }
            )
        atomic_json_save(out, Config.ETF_SIGNALS_LATEST_FILE)
        Logger.info(f"📊 V4 信号已输出: {Config.ETF_SIGNALS_LATEST_FILE}")
        return out
    except Exception as error:
        Logger.error("保存 V4 信号失败", error)
        return None
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

    FORCE_DOWNLOAD: bool = os.environ.get("FORCE_DOWNLOAD", "false").lower() == "true"
    if FORCE_DOWNLOAD:
        Logger.info("🚀 强制下载模式")
    else:
        Logger.info("📦 智能模式: 优先本地缓存")

    codes: List[str] = [
        '159326', '588170', '513090', '159206', '515880', '159869', '516150', '562950', '562500',
        '515220', '515790', '512660', '159566', '515210', '159611', '512690', '159930', '560280', 
        '512800', '159851', '513120', '513050', '159667', '159259', '159996', '518880'
    ]

    Logger.info(f"🚀 [ETF波段雷达 V4] 启动! {len(codes)}个标的")
    Logger.info("📐 指标: MACD(12/26/9) BOLL(20/2.0+%B) RSI(14) ATR(14) VOL MA(20)")
    Logger.info("🆕 V4: 趋势效率·顺势形态·beta调整相对强度·结构止损·校准门控\n")

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
    Logger.info(f"🛡️ ATR止损参考: {atr_multiplier}x | 风险等级: {env_result.risk_level}\n")
    t_env: float = time.time()

    # ═══ 并发获取数据 ═══
    analyzers: List[ETFAnalyzer] = []
    Logger.info("⏳ 并发拉取K线...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
        futures: Dict[concurrent.futures.Future, str] = {}
        for code in codes:
            name: str = name_map.get(code, f"ETF_{code}")
            futures[ex.submit(
                fetch_single_etf,
                code,
                name,
                FORCE_DOWNLOAD,
                market_safe,
                atr_multiplier,
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

    # ═══ [OPT-#5] 自适应RPS窗口 ═══
    is_major_reversal: bool = False
    if len(market_env.get_history(days=5)) >= 2:
        env_signal = market_env.get_env_change_signal()
        if env_signal and env_signal.get('score_accelerating') and env_signal.get('status_changed'):
            is_major_reversal = True
            Logger.info("⚡ 检测到市场大反转，RPS窗口缩短至60天")
    rps_window = 60 if is_major_reversal else 120

    Logger.info("🧮 计算 V4 20/60/120日 beta 调整 RPS...")
    alphas: Dict[str, float] = {}
    alpha_profiles: Dict[str, Dict[str, float]] = {}
    benchmark_df = market_env.analyzer.df_daily if market_env.analyzer is not None else None
    for a in analyzers:
        try:
            profile = calc_risk_adjusted_alpha(a.df_daily, benchmark_df, window=rps_window)
            alpha_profiles[a.code] = profile
            alphas[a.code] = profile.get("vol_adj_alpha", -999.0)
        except Exception:
            alphas[a.code] = -999.0

    v4_relative_strength: Dict[str, Dict[str, Any]] = {}
    if benchmark_df is not None and not benchmark_df.empty:
        v4_relative_strength = rank_relative_strength(
            {item.code: item.df_daily for item in analyzers},
            benchmark_df,
        )
    rps_map: Dict[str, float] = {
        code: float(profile.get("score", 0.0) or 0.0)
        for code, profile in v4_relative_strength.items()
    }
    if rps_map:
        n: int = len(rps_map)
        rps85_count: int = sum(1 for v in rps_map.values() if v >= 85.0)
        Logger.info(
            f"📌 RPS有效样本:{n} | RPS>=85约等于池内前{rps85_count}名，"
            "采用20/60/120日跟踪误差标准化后的固定池分位"
        )

    # ═══ 多周期评分 ═══
    Logger.info("🧠 多周期评分...\n")
    results: List[Dict] = []
    for a in analyzers:
        try:
            a.rps = rps_map.get(a.code, 0.0)
            profile = alpha_profiles.get(a.code, {})
            a.alpha_score = float(profile.get("alpha_score", 0.0) or 0.0)
            a.vol_adj_alpha = float(profile.get("vol_adj_alpha", 0.0) or 0.0)
            a.beta_to_benchmark = float(profile.get("beta_to_benchmark", 1.0) or 1.0)
            prev_item = prev_history.get(a.code, {})
            prev: Optional[float] = prev_item.get('total_score')
            prev_raw: Optional[float] = prev_item.get('raw_total_score')
            prev_s: float = float(prev_item.get('stop_loss', 0.0))
            res: Optional[Dict] = a.analyze(
                prev_score=prev,
                prev_stop=prev_s,
                prev_raw_score=prev_raw,
            )
            if res:
                relative_strength = dict(v4_relative_strength.get(a.code, {}))
                res["relative_strength"] = relative_strength
                res["v4_priority"] = v4_final_priority(res.get("v4_factors", {}), relative_strength)
                res["rps"] = float(relative_strength.get("score", 0.0) or 0.0)
                results.append(res)
        except Exception as e:
            Logger.error(f"{a.code} 分析失败", e)

    benchmark_weekly_score = 0.0
    benchmark_natr_percentile = 50.0
    if market_env.analyzer is not None:
        benchmark_weekly_score = float(
            weekly_trend_factor(market_env.analyzer.df_weekly).get("score", 0.0) or 0.0
        )
        benchmark_natr_percentile = normalised_atr_percentile(market_env.analyzer.df_daily)
    v4_market = build_v4_market_policy(
        results,
        benchmark_weekly_score=benchmark_weekly_score,
        benchmark_natr_percentile=benchmark_natr_percentile,
    )
    for result in results:
        result["v4_market"] = dict(v4_market)
    env_result.entry_permission = str(v4_market.get("entry_permission", "BLOCKED"))
    env_result.max_exposure_ratio = float(v4_market.get("max_exposure_ratio", 0.0) or 0.0)
    env_result.regime_level = str(v4_market.get("state", "RISK_OFF"))
    env_result.atr_percentile = float(benchmark_natr_percentile)

    t_score: float = time.time()

    # ═══ 附加 sparkline 数据 + score_delta 到结果 ═══
    today_str: str = datetime.now().strftime('%Y-%m-%d')
    for r in results:
        code: str = r['code']
        scores: Dict[str, float] = prev_history.get(code, {}).get('daily_scores', {})
        scores[today_str] = r['total_score']
        r['sparkline_data'] = sorted(scores.items())[-ETFScoringConfig.SPARKLINE_DAYS:]
        prev_total: Optional[float] = prev_history.get(code, {}).get('total_score')
        prev_raw_total: Optional[float] = prev_history.get(code, {}).get('raw_total_score')
        if prev_total is not None:
            r['score_delta'] = round(r['total_score'] - prev_total, 1)
        else:
            r['score_delta'] = None
        # raw_score_delta 只反映 ETF 自身技术分变化，不包含大盘防守扣分。
        # test.py 的 EV、评分骤降、恢复模式优先使用这个字段。
        if prev_raw_total is not None:
            r['raw_score_delta'] = round(
                r.get('raw_total_score', r['total_score']) - prev_raw_total,
                1,
            )
        else:
            r['raw_score_delta'] = None

    # ═══ 输出 ═══
    if results:
        try:
            save_history(results)
            breadth: Dict[str, Any] = HTMLReporter._compute_breadth(results)
            HTMLReporter.generate(results, env_result, "index.html")
            sig_output = save_etf_signals(results, env_result, breadth)
            signal_changes: List[Dict[str, Any]] = detect_signal_changes(results, prev_history)
            log_signal_changes(signal_changes)
            # 合约校验（直接用内存对象，不重复读文件）
            if sig_output:
                contract_errors = validate_signal_contract(sig_output)
                if contract_errors:
                    Logger.warning(f"⚠️ 信号合约校验失败 ({len(contract_errors)}项):")
                    for err in contract_errors[:5]:
                        Logger.warning(f"  → {err}")
                else:
                    Logger.info("✅ 信号合约校验通过")
            else:
                Logger.warning("⚠️ 信号输出失败，跳过合约校验")
            Logger.info(f"📊 环境: {Config.MARKET_ENV_LATEST_FILE}")
            Logger.info(f"📊 市场宽度: 多头{breadth['bull']} 空头{breadth['bear']} "
                        f"中性{breadth['neutral']} | 多空比{breadth['ratio']:.2f} → {breadth['signal']}")
            Logger.info(f"🎉 完成! {len(results)}个ETF\n")
            Logger.info("📊 === 雷达优先 TOP10 ===")
            top10 = sorted(results, key=lambda x: x.get('composite_priority', x['total_score']), reverse=True)[:10]
            for i, r in enumerate(top10):
                delta_str: str = ""
                if r.get('score_delta') is not None:
                    d: float = r['score_delta']
                    arrow: str = "▲" if d > 0 else ("▼" if d < 0 else "→")
                    delta_str = f" | {arrow}{d:+.1f}"
                priority = float(r.get('composite_priority', r['total_score']))
                tier = str(r.get('conviction_tier', 'C'))
                Logger.info(
                    f"  {i + 1:>2}. {r['name']:<16s} {r['code']} | "
                    f"优先{priority:>5.1f}({tier}) | 总分{r['total_score']:>+6.1f} | "
                    f"RPS{r['rps']:>5.0f} | {_enum_value(r['status'])}{delta_str}"
                )
            Logger.info("\n📊 === 空头 BOTTOM5 ===")
            bottom5 = sorted(results, key=lambda x: x['total_score'])[:5]
            for r in bottom5:
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
