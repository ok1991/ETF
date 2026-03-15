# -*- coding: utf-8 -*-
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import akshare as ak
import yaml
import lightgbm as lgb
class ETFScreener:
    def __init__(self, etf_codes: List[str], data_dir: str = "etf_data",
                 output_file: str = "index.html", db_path: str = "etf_history.db",
                 config_path: str = "config.yaml"):
        self.etf_codes = etf_codes
        self.data_dir = os.path.abspath(data_dir)
        self.output_file = os.path.abspath(output_file)
        self.db_path = os.path.abspath(db_path)
        self.latest_trade_date = None
        self.debug_mode = True
        self.raw_results_for_db = []
        self.benchmark_df_analyzed = None
        self.tier_results: Dict[str, List[Dict]] = {'S': [], 'A': [], 'B': [], 'F': []}
        self.market_is_bullish = False
        self.market_status_text = "评估中..."
        self.market_phase = "RANGING"
        self.total_valid_etfs = 0
        self._ensure_dir(self.data_dir)
        self._init_db()
        self._load_config(config_path)
        self.name_map = self._get_etf_name_map()
        self.model_path = "ml_model.txt"
        self.model = self._load_ml_model()
        # V22.2 颜色映射安全初始化
        if not hasattr(self, 'profile_to_color_map') or not isinstance(self.profile_to_color_map, dict):
            self.profile_to_color_map = {
                "全能冠军": "tag-gold", "强力突破者": "tag-red", "动能猛兽": "tag-red",
                "稳健爬升者": "tag-green", "高位旗形整理者": "tag-blue", "主线洗盘中": "tag-blue",
                "逆势孤狼": "tag-purple", "潜力观察股": "tag-purple", "筑顶高危": "tag-orange",
                "动能衰竭预警": "tag-orange", "假强势预警": "tag-orange", "假突破警报": "tag-black",
                "冰点反转": "tag-ice-blue", "弱市调整中": "tag-grey", "主线共振突破": "tag-red",
            }
        # 新增：默认启用评分趋势箭头（上下分数），便于用户直观看到分数变化
        self.show_score_trend = True
    # ==================== V22.2 混合评分（已锁定14特征，与你的模型完全匹配） ====================
    def _load_ml_model(self):
        try:
            model = lgb.Booster(model_file=self.model_path)
            print(f"[V22.2 ML] LightGBM模型加载成功（混合评分已启用）")
            return model
        except Exception:
            print("[V22.2 ML警告] 未找到 ml_model.txt，使用纯规则评分。请先运行 train_ml_model.py")
            return None
    def _extract_ml_features(self, metrics: dict) -> pd.DataFrame:
        features = {
            'ma20_slope': metrics.get('ma20_slope', 0),
            'rs_slope': metrics.get('rs_slope', 0),
            'rsi': metrics.get('rsi', 50),
            'macd_hist_slope': metrics.get('macd_hist_slope', 0),
            'rank': metrics.get('rank', 999),
            'ret20': metrics.get('ret20', 0),
            'ret60': metrics.get('ret60', 0),
            'atr_pct': metrics.get('atr_pct', 0),
            'bias20': metrics.get('bias20', 0),
            'volume_ratio': metrics.get('volume_ratio', 1),
            'mom_score': metrics.get('mom_score', 0),
            'days_below_ma20': metrics.get('days_below_ma20', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'is_bullish_alignment': 1 if metrics.get('is_bullish_alignment', False) else 0,
        }
        return pd.DataFrame([features])
    def get_hybrid_score(self, metrics: dict, df: pd.DataFrame) -> Tuple[int, List[str]]:
        rule_score, rule_msgs = self._calculate_score(metrics)
        if self.model is None:
            return rule_score, rule_msgs + ["[ML未加载] 使用纯规则评分"]
        features_df = self._extract_ml_features(metrics)
        try:
            ml_prob = self.model.predict(features_df)[0] * 100
            hybrid_score = int(self.hybrid_rule_weight * rule_score + self.hybrid_ml_weight * ml_prob)
            return max(0, hybrid_score), rule_msgs + [f"ML预测上涨概率: {ml_prob:.1f}% → 混合分: {hybrid_score}"]
        except Exception as e:
            print(f"[ML预测异常] {e}，回退纯规则")
            return rule_score, rule_msgs + ["[ML预测失败] 使用纯规则"]
    def _load_config(self, config_path: str):
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        required = ['ma_mid', 'ma_long', 's_top_pct', 'a_top_pct']
        for k in required:
            if k not in config:
                raise ValueError(f"config.yaml 缺失关键参数: {k}")
        for key, value in config.items():
            setattr(self, key, value)
        self.hybrid_rule_weight = config.get('hybrid_rule_weight', 0.7)
        self.hybrid_ml_weight = config.get('hybrid_ml_weight', 0.3)
        print(f"[配置成功] V22.4 已加载 config.yaml（ML混合权重 {self.hybrid_rule_weight:.0%}+{self.hybrid_ml_weight:.0%}）")
    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS daily_profile (
                           date TEXT NOT NULL, 
                           code TEXT NOT NULL, 
                           profile_name TEXT NOT NULL, 
                           score INTEGER NOT NULL, 
                           ma20_slope REAL DEFAULT 0.0,
                           rs_ma_slope REAL DEFAULT 0.0,
                           momentum_score REAL DEFAULT 0.0,
                           rank_position INTEGER DEFAULT 999,
                           PRIMARY KEY (date, code)
                       )""")
                cursor.execute("PRAGMA table_info(daily_profile)")
                columns = [info[1] for info in cursor.fetchall()]
                if 'ma20_slope' not in columns: cursor.execute(
                    "ALTER TABLE daily_profile ADD COLUMN ma20_slope REAL DEFAULT 0.0")
                if 'rs_ma_slope' not in columns: cursor.execute(
                    "ALTER TABLE daily_profile ADD COLUMN rs_ma_slope REAL DEFAULT 0.0")
                if 'momentum_score' not in columns: cursor.execute(
                    "ALTER TABLE daily_profile ADD COLUMN momentum_score REAL DEFAULT 0.0")
                if 'rank_position' not in columns: cursor.execute(
                    "ALTER TABLE daily_profile ADD COLUMN rank_position INTEGER DEFAULT 999")
        except sqlite3.Error as e:
            print(f"[数据库严重警告] 数据库初始化失败: {e}")
    def _get_yesterday_data(self, code: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT profile_name, rank_position FROM daily_profile WHERE code= ? AND date < ? ORDER BY date DESC LIMIT 1"
                today_q_date = self.latest_trade_date if self.latest_trade_date else datetime.now().strftime('%Y-%m-%d')
                cursor.execute(query, (code, today_q_date))
                result = cursor.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            print(f"[数据库警告] 查询昨日数据失败 for {code}: {e}")
            return None
    def _get_historical_profiles(self, code: str, num_days: int) -> List[Dict]:
        history = []
        today_str = self.latest_trade_date or datetime.now().strftime('%Y-%m-%d')
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = """
                            SELECT profile_name, date, score 
                            FROM daily_profile 
                            WHERE code = ? AND date < ?
                            ORDER BY date DESC 
                            LIMIT ?
                        """
                cursor.execute(query, (code, today_str, num_days))
                results = cursor.fetchall()
                for result in reversed(results):
                    p_name = (result['profile_name'] or "").split(' ')[0]
                    history.append({
                        "date": result['date'],
                        "name": p_name,
                        "color_class": self.profile_to_color_map.get(p_name, "tag-grey"),
                        "score": int(result['score']) if result['score'] is not None else 0
                    })
            return history
        except Exception:
            return []
    @staticmethod
    def _ensure_dir(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
    @staticmethod
    def _add_market_prefix(code: str) -> str:
        if code.startswith(('5', '6')): return f"sh{code}"
        if code.startswith(('1', '0', '3')): return f"sz{code}"
        return code
    def _get_etf_name_map(self) -> Dict:
        try:
            spot_df = ak.fund_etf_spot_ths()
            name_dict = dict(zip(spot_df['基金代码'], spot_df['基金名称']))
            name_dict[self.benchmark_code] = "沪深300ETF"
            return name_dict
        except Exception:
            return {self.benchmark_code: "沪深300ETF"}
    def get_etf_data(self, code: str) -> pd.DataFrame:
        date_str = self.latest_trade_date.replace('-', '') if self.latest_trade_date else datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(self.data_dir, f"{code}_{date_str}.csv")
        df = pd.DataFrame()
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['日期'], encoding="utf-8-sig")
            except Exception:
                pass
        required_length = self.ma_long + self.stat_period
        if df.empty or len(df) < required_length:
            try:
                df_ak = ak.stock_zh_a_hist_tx(symbol=self._add_market_prefix(code), adjust="qfq")
                if not df_ak.empty:
                    df_ak['日期'] = pd.to_datetime(df_ak['date'])
                    df_ak.sort_values('日期', inplace=True, ignore_index=True)
                    df = df_ak[['日期', 'open', 'close', 'high', 'low', 'amount']].rename(
                        columns={'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低', 'amount': '成交量'})
                    latest_date_in_data = df['日期'].iloc[-1].strftime('%Y%m%d')
                    actual_file_path = os.path.join(self.data_dir, f"{code}_{latest_date_in_data}.csv")
                    df.to_csv(actual_file_path, index=False, encoding="utf-8-sig")
            except Exception:
                return pd.DataFrame()
        df.rename(columns={'成交额': '成交量', 'volume': '成交量'}, inplace=True, errors='ignore')
        return df
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high_low = df['最高'] - df['最低']
        high_close = np.abs(df['最高'] - df['收盘'].shift())
        low_close = np.abs(df['最低'] - df['收盘'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=self.atr_period).mean()
        df['ATR_Pct'] = df['ATR'] / df['收盘']
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - 100 / (1 + rs)
        ema_fast = df['收盘'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['收盘'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        hist_recent = df['MACD_Hist'].iloc[-self.slope_period:].dropna()
        if len(hist_recent) >= 2:
            slope = np.polyfit(range(len(hist_recent)), hist_recent.values, 1)[0]
            df.loc[df.index[-1], 'MACD_Hist_Slope'] = slope / hist_recent.mean() if hist_recent.mean() != 0 else 0
        else:
            df['MACD_Hist_Slope'] = np.nan
        return df
    def _calculate_etf_indicators(self, etf_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(etf_df, benchmark_df, on='日期', suffixes=('_etf', '_benchmark'), how='inner')
        df['收盘'] = pd.to_numeric(df['收盘_etf'], errors='coerce')
        df['最高'] = pd.to_numeric(df['最高_etf'], errors='coerce')
        df['最低'] = pd.to_numeric(df['最低_etf'], errors='coerce')
        df['涨跌幅'] = df['收盘'].pct_change()
        for p in [self.ma_very_short, self.ma_mid, self.ma_long]:
            df[f'MA{p}'] = df['收盘'].rolling(p, min_periods=1).mean()
        df[f'BIAS{self.ma_mid}'] = (df['收盘'] - df[f'MA{self.ma_mid}']) / df[f'MA{self.ma_mid}'] * 100
        df['成交量'] = pd.to_numeric(df['成交量_etf'], errors='coerce')
        if not df['成交量'].isnull().all():
            avg_vol = df['成交量'].rolling(self.ma_mid, min_periods=1).mean()
            df['成交量比'] = np.where(avg_vol > 0, df['成交量'] / avg_vol, np.nan)
        else:
            df['成交量比'] = np.nan
        df['RS_Ratio'] = df['收盘_etf'] / df['收盘_benchmark']
        df[f'RS_MA{self.rs_ma_period}'] = df['RS_Ratio'].rolling(self.rs_ma_period, min_periods=1).mean()
        df['涨幅_20日'] = df['收盘'].pct_change(periods=20)
        df['涨幅_60日'] = df['收盘'].pct_change(periods=60)
        df['机构动量得分'] = df['涨幅_20日'] * 1.0 + df['涨幅_60日'] * 1.5
        df = df.dropna(subset=[f'MA{self.ma_long}', f'RS_MA{self.rs_ma_period}', f'BIAS{self.ma_mid}']).reset_index(drop=True)
        df = self._add_advanced_indicators(df)
        return df
    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df_calc = df.copy()
        df_calc['收盘'] = pd.to_numeric(df_calc['收盘'], errors='coerce')
        for p in [self.ma_mid, self.ma_long]:
            df_calc[f'MA{p}'] = df_calc['收盘'].rolling(p, min_periods=1).mean()
        return df_calc
    def _analyze_market_environment(self):
        df = self.benchmark_df_analyzed
        if df is None or df.empty or len(df) < self.ma_long:
            self.market_is_bullish, self.market_status_text = False, "大盘数据不足，无法评估"
            self.market_phase = "RANGING"
            return
        today = df.iloc[-1]
        close = today.get('收盘', 0)
        ma20 = today.get(f'MA{self.ma_mid}', 0)
        ma60 = today.get(f'MA{self.ma_long}', 0)
        ma20_slope = np.nan
        ma60_slope = np.nan
        ym20 = df[f'MA{self.ma_mid}'].iloc[-self.slope_period:].dropna()
        if len(ym20) >= 2:
            ma20_slope = (np.polyfit(range(len(ym20)), ym20.values, 1)[0] / ym20.mean()) * 100
        ym60 = df[f'MA{self.ma_long}'].iloc[-self.slope_period:].dropna()
        if len(ym60) >= 2:
            ma60_slope = (np.polyfit(range(len(ym60)), ym60.values, 1)[0] / ym60.mean()) * 100
        SLOPE_THRESHOLD_POSITIVE = 0.05
        SLOPE_THRESHOLD_NEGATIVE = -0.05
        if close > ma20 and ma20 > ma60 and ma20_slope > SLOPE_THRESHOLD_POSITIVE and ma60_slope > 0:
            self.market_phase = "STRONG_BULL"
            self.market_is_bullish = True
            self.market_status_text = "强多头市场 - 牛市主升浪，环境极佳"
        elif close < ma20 and ma20 < ma60 and ma20_slope < SLOPE_THRESHOLD_NEGATIVE and ma60_slope < 0:
            self.market_phase = "STRONG_BEAR"
            self.market_is_bullish = False
            self.market_status_text = "强空头市场 - 熊市主跌浪，环境恶劣"
        elif close > ma20 and ma20_slope > SLOPE_THRESHOLD_POSITIVE:
            self.market_phase = "WEAK_BULL"
            self.market_is_bullish = True
            self.market_status_text = "弱多头市场 - 熊市反弹或牛初，谨慎乐观" if ma20 < ma60 else "弱多头市场 - 牛市回调结束初期，趋势待确认"
        elif close < ma20 and ma20_slope < SLOPE_THRESHOLD_NEGATIVE:
            self.market_phase = "WEAK_BEAR"
            self.market_is_bullish = False
            self.market_status_text = "弱空头市场 - 牛市回调或熊初，提高警惕" if ma20 > ma60 else "弱空头市场 - 熊市反弹结束，风险加剧"
        else:
            self.market_phase = "RANGING"
            self.market_is_bullish = False
            self.market_status_text = "震荡市场 - 方向不明，谨慎行事"
    def _get_dynamic_thresholds(self, scores: List[float]) -> Tuple[int, int]:
        if not scores:
            return 75, 45
        sorted_scores = sorted([s for s in scores if s >= 0], reverse=True)
        n = len(sorted_scores)
        if n == 0:
            return 75, 45
        is_weak_bear = self.market_phase in ("WEAK_BEAR", "STRONG_BEAR")
        s_pct = self.weak_bear_s_top_pct if is_weak_bear else self.s_top_pct
        s_idx = max(0, int(n * s_pct / 100) - 1)
        a_idx = max(0, int(n * self.a_top_pct / 100) - 1)
        return int(sorted_scores[s_idx]), int(sorted_scores[a_idx])
    def run(self):
        print(f"======== V22.3 形态演化轴防溢出版 启动 ========")
        benchmark_raw_df = self.get_etf_data(self.benchmark_code)
        if benchmark_raw_df.empty:
            print("[核心错误] 无法获取大盘数据，中止。")
            return
        self.benchmark_df_analyzed = self._calculate_base_indicators(benchmark_raw_df)
        if not self.benchmark_df_analyzed.empty:
            self.latest_trade_date = self.benchmark_df_analyzed['日期'].iloc[-1].strftime('%Y-%m-%d')
        else:
            self.latest_trade_date = datetime.now().strftime('%Y-%m-%d')
        self._clear_old_data_files()
        self._analyze_market_environment()
        print(f"[战情评估] {self.market_status_text} (Phase: {self.market_phase})\n")
        print("[Pass 1] 扫描全市场标的，计算机构动量截面排名...")
        etf_cache = {}
        for code in self.etf_codes:
            if code == self.benchmark_code: continue
            df_raw = self.get_etf_data(code)
            if df_raw.empty: continue
            df_analyzed = self._calculate_etf_indicators(df_raw, self.benchmark_df_analyzed)
            if df_analyzed.empty: continue
            today = df_analyzed.iloc[-1]
            mom_score = today.get('机构动量得分', -999)
            if pd.isna(mom_score): mom_score = -999
            etf_cache[code] = {'df': df_analyzed, 'mom_score': mom_score,
                               'ret20': today.get('涨幅_20日', 0), 'ret60': today.get('涨幅_60日', 0)}
        sorted_codes = sorted([k for k, v in etf_cache.items() if v['mom_score'] > -900],
                              key=lambda k: etf_cache[k]['mom_score'], reverse=True)
        total_valid = len(sorted_codes)
        self.total_valid_etfs = total_valid
        for rank, code in enumerate(sorted_codes, 1):
            etf_cache[code]['rank'] = rank
            etf_cache[code]['is_top_third'] = rank <= max(1, total_valid // 3)
            etf_cache[code]['is_bottom_third'] = rank > (total_valid * 2 // 3)
        print("[Pass 1.5] 预计算所有ETF原始分数 → 生成每日动态百分位门槛...")
        preliminary_scores = []
        for code, cache_data in etf_cache.items():
            df_analyzed = cache_data['df']
            is_match, msg, *a_flags = self._analyze_stock_conditions(df_analyzed)
            if is_match:
                (d_b_20, net_v, max_d, ma20_s, v_rat, rs_s, b20, is_bull, vol20, l20, h20,
                 close_p, daily_chg, atr_pct, rsi_val, macd_hist_slope) = a_flags
                metrics = {
                    'days_below_ma20': d_b_20, 'net_volatility_days': net_v, 'max_drawdown': max_d,
                    'ma20_slope': ma20_s, 'volume_ratio': v_rat, 'rs_slope': rs_s, 'bias20': b20,
                    'is_bullish_alignment': is_bull, 'volatility_20d': vol20, 'daily_change': daily_chg,
                    'mom_score': cache_data['mom_score'], 'rank': cache_data.get('rank', 999),
                    'ret20': cache_data['ret20'], 'ret60': cache_data['ret60'],
                    'is_top_third': cache_data.get('is_top_third', False),
                    'is_bottom_third': cache_data.get('is_bottom_third', False),
                    'atr_pct': atr_pct, 'rsi': rsi_val, 'macd_hist_slope': macd_hist_slope
                }
                score, _ = self.get_hybrid_score(metrics, df_analyzed)
                preliminary_scores.append(score)
        self.s_min_score, self.a_min_score = self._get_dynamic_thresholds(preliminary_scores)
        weak_str = "（弱市前8%）" if self.market_phase in ("WEAK_BEAR", "STRONG_BEAR") else ""
        print(f"[科学动态门槛] S级 >= {self.s_min_score}分 (前{self.s_top_pct if not weak_str else self.weak_bear_s_top_pct}%{weak_str})")
        print(f"[科学动态门槛] A级 >= {self.a_min_score}分 (前{self.a_top_pct}%)\n")
        print("[Pass 2] 融合截面排名，执行微观战术研判...")
        for code, cache_data in etf_cache.items():
            df_analyzed = cache_data['df']
            is_match, msg, *a_flags = self._analyze_stock_conditions(df_analyzed)
            if is_match:
                (d_b_20, net_v, max_d, ma20_s, v_rat, rs_s, b20, is_bull, vol20, l20, h20,
                 close_p, daily_chg, atr_pct, rsi_val, macd_hist_slope) = a_flags
                metrics = {
                    'days_below_ma20': d_b_20, 'net_volatility_days': net_v, 'max_drawdown': max_d,
                    'ma20_slope': ma20_s, 'volume_ratio': v_rat, 'rs_slope': rs_s, 'bias20': b20,
                    'is_bullish_alignment': is_bull, 'volatility_20d': vol20, 'daily_change': daily_chg,
                    'mom_score': cache_data['mom_score'], 'rank': cache_data.get('rank', 999),
                    'ret20': cache_data['ret20'], 'ret60': cache_data['ret60'],
                    'is_top_third': cache_data.get('is_top_third', False),
                    'is_bottom_third': cache_data.get('is_bottom_third', False),
                    'atr_pct': atr_pct, 'rsi': rsi_val, 'macd_hist_slope': macd_hist_slope
                }
                score, score_msgs = self.get_hybrid_score(metrics, df_analyzed)
                metrics['score'] = score
                yesterday_data = self._get_yesterday_data(code)
                yesterday_profile_name = "N/A"
                yesterday_rank = 999
                if yesterday_data:
                    yesterday_profile_name = yesterday_data.get('profile_name', 'N/A').split(' ')[0]
                    yesterday_rank = yesterday_data.get('rank_position', 999)
                is_strong_breakthrough = (
                        yesterday_profile_name in {"潜力观察股", "高位旗形整理者", "稳健爬升者"} and daily_chg > 0.01 and v_rat > 1.2 and rs_s > 0)
                metrics['is_strong_breakthrough'] = is_strong_breakthrough
                is_ice_point = self._detect_ice_point_reversal(metrics, df_analyzed)
                profile, p_desc, tag_class, sig_text, sig_desc, tier = self._get_final_assessment(
                    metrics, yesterday_profile_name, yesterday_rank, is_ice_point, self.market_is_bullish)
                if self.debug_mode:
                    print(f"DEBUG {code}: Score={score}, MomRank={metrics['rank']}, RSI={metrics.get('rsi', 0):.1f}, "
                          f"MACD_Slope={metrics.get('macd_hist_slope', 0):.4f}, FinalProfile={profile}, Tier={tier}")
                sig_level_map = {'S': 'buy-strong', 'A': 'posture-follow', 'B': 'signal-reversal', 'F': 'posture-avoid'}
                if profile in ["筑顶高危", "假强势预警", "动能衰竭预警"]:
                    sig_level_map[tier] = 'risk-high'
                elif profile in ["高位旗形整理者", "潜力观察股", "主线洗盘中", "逆势孤狼"]:
                    sig_level_map[tier] = 'risk-medium'
                pos_pct = 50.0 if not (h20 > l20 and pd.notna(close_p)) else max(0.0, min(100.0, (
                        (close_p - l20) / (h20 - l20) * 100.0)))
                vp_label, vp_class, vp_tooltip = self._get_volume_price_profile(pos_pct, v_rat, daily_chg)
                vp_html = f'<div class="has-tooltip"><span class="vp-tag {vp_class}">{vp_label}</span><span class="tooltip">{vp_tooltip}</span></div>'
                signal_html = f'<div class="has-tooltip"><span class="signal-cell signal-{sig_level_map.get(tier, "posture-wait")}">{sig_text}</span><span class="tooltip">{sig_desc}</span></div>'
                rank = metrics.get('rank', 999)
                medal = " " if rank == 1 else " " if rank == 2 else " " if rank == 3 else ""
                rank_change_html = ""
                if yesterday_rank != 999 and rank != 999:
                    change = yesterday_rank - rank
                    if change > 0:
                        rank_change_html = f'<span class="rank-change rank-up">↑{change}</span>'
                    elif change < 0:
                        rank_change_html = f'<span class="rank-change rank-down">↓{-change}</span>'
                mom_html = f'<div class="has-tooltip mom-container"><div class="mom-rank-line"><span class="rank-main">{medal}#{rank}</span>{rank_change_html}</div><div class="rank-score">评分: {metrics["mom_score"]:.2f}</div><span class="tooltip">【动量指标】&#10;20日涨幅: {metrics["ret20"]:.2%}&#10;60日涨幅: {metrics["ret60"]:.2%}</span></div>'
                history_list = self._get_historical_profiles(code, self.history_days)
                evo_html, tt_text = '<div class="profile-evolution-cell">', "【5天形态演化轴】<br>"
                for i, item in enumerate(history_list):
                    evo_html += f'<div class="spark-box {item["color_class"]}"></div>'
                    tt_text += f"T-{len(history_list) - i}: {item['name']}（{item['score']}分）<br>"
                current_score = metrics.get('score', 0)
                yesterday_score = history_list[-2]['score'] if len(history_list) >= 2 else current_score
                delta = current_score - yesterday_score
                arrow = ""
                if self.show_score_trend and delta != 0:
                    arrow_color = "color:#dc3545" if delta > 0 else "color:#198754"
                    arrow = f'<span style="margin-left:4px;font-size:0.3em;font-weight:700;{arrow_color}">{"↑" if delta > 0 else "↓"}{abs(delta)}</span>'
                # === 修改点1：形态演化轴不再附加上下分数箭头（更简洁）===
                evo_html += f'<span class="tag {self.profile_to_color_map.get(profile, "tag-grey")}" style="margin-left: 5px;">{profile}</span></div>'
                tt_text += f"👉 今: {profile}（{current_score}分）{p_desc}"
                combined_profile_html = f'<div class="has-tooltip" style="justify-content: flex-start;">{evo_html}<span class="tooltip">{tt_text}</span></div>'
                p_color = "#198754" if pos_pct < 30 else ("#fd7e14" if pos_pct < 70 else "#dc3545")
                pos_bar_html = f'<div class="has-tooltip"><div class="pos-bar-wrapper"><div class="pos-center-line"></div><div class="pos-bar-marker" style="left: {pos_pct:.1f}%; background-color: {p_color};"></div></div><span class="tooltip">20日区间: {pos_pct:.0f}%<br>高:{h20:.3f}, 低:{l20:.3f}</span></div>'
                self.raw_results_for_db.append({
                    'code': code, 'score': score, 'profile_name': profile,
                    'ma20_slope': ma20_s if pd.notna(ma20_s) else 0.0,
                    'rs_slope': rs_s if pd.notna(rs_s) else 0.0,
                    'momentum_score': metrics['mom_score'],
                    'rank_position': metrics['rank']
                })
                row_data = {
                    'Tier': tier, 'raw_score': score,
                    '代码': f'<span style="font-size:1.1em;font-weight:500;color:#212529;font-family:monospace;">{code}</span>',
                    'ETF名称': f"<strong style='letter-spacing:0.5px;'>{self.name_map.get(code, code)}</strong>",
                    # === 修改点2：评分列现在包含上下分数箭头 ===
                    '评分': f'<div class="has-tooltip" style="font-weight:700;font-size:1.1em;color:{"#dc3545" if score > 60 else ("#6c757d" if score < 45 else "#495057")}">{score}{arrow}<span class="tooltip">{"&#10;".join(score_msgs)}</span></div>',
                    '机构动量(排位)': mom_html,
                    '战术指令': signal_html,
                    '形态演化轴': combined_profile_html,
                    '量价特征': vp_html,
                    'MA20趋势': f"{ma20_s:.2%}" if pd.notna(ma20_s) else "-",
                    # === 修改点3：20日位置挪到表格最右（最后一列）===
                    '20日位置': pos_bar_html,
                }
                self.tier_results[tier].append(row_data)
        for t in self.tier_results.keys():
            if self.tier_results[t]:
                self.tier_results[t].sort(key=lambda x: x['raw_score'], reverse=True)
        self._generate_html_report()
        try:
            today_str = self.latest_trade_date
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                data_to_insert = [(today_str, i['code'], i['profile_name'], i['score'],
                                   i['ma20_slope'], i['rs_slope'], i['momentum_score'], i['rank_position'])
                                  for i in self.raw_results_for_db]
                cursor.executemany(
                    "INSERT OR REPLACE INTO daily_profile (date, code, profile_name, score, ma20_slope, rs_ma_slope, momentum_score, rank_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    data_to_insert)
                print(f"[数据库] 成功存入 {len(data_to_insert)} 条数据。")
        except sqlite3.Error as e:
            print(f"[数据库错误] 无法保存数据: {e}")
    def _get_final_assessment(self, metrics: dict, yesterday_profile: str, yesterday_rank: int,
                              is_ice_point: bool, market_is_bullish: bool) -> Tuple[
        str, str, str, str, str, str]:
        score = metrics.get('score', 0)
        rank = metrics.get('rank', 999)
        total = max(1, self.total_valid_etfs)
        rank_pct = rank / total
        is_top15 = rank_pct <= self.top15_pct
        is_top20 = rank_pct <= self.top20_pct
        is_top40 = rank_pct <= self.top40_pct
        ma20_slope = metrics.get('ma20_slope', 0.0)
        rsi = metrics.get('rsi', 50.0)
        macd_slope = metrics.get('macd_hist_slope', 0.0)
        daily_chg = metrics.get('daily_change', 0.0)
        ret60 = metrics.get('ret60', 0.0)
        ret20 = metrics.get('ret20', 0.0)
        vol_ratio = metrics.get('volume_ratio', 0.0)
        s_min_score = self.s_min_score
        a_min_score = self.a_min_score
        if rank <= self.top_rank_for_decay and score < self.risk_top_rank_decay_score and (
                ret20 < -0.02 or macd_slope < 0 or rsi > self.rsi_decay):
            return "动能衰竭预警", "龙头历史动量强但当前破位", "tag-orange", "动能衰竭", "Rank#{}历史动量强，但Score仅{}（20日{:.1%}）+MACD/RSI确认衰竭，短期风险巨大，可关注企稳博弈。".format(rank, score, ret20), "B"
        if (is_top20 and score < self.top20_low_score) or (
                rank <= self.top_rank_for_decay and score < self.risk_top_rank_decay_score):
            if rsi > self.rsi_decay or macd_slope < self.macd_decay:
                return "动能衰竭预警", "龙头动能确认衰竭", "tag-orange", "动能衰竭", "高机构动量(Rank {})但 RSI({:.1f})/MACD({:.4f}) 双衰竭，短期风险巨大，可关注企稳博弈。".format(rank, rsi, macd_slope), "B"
        if yesterday_profile == "强力突破者" and daily_chg < self.breakthrough_daily_chg_min:
            return "假突破警报", "假突破引发强烈看空", "tag-black", "假突破陷阱", "多头陷阱确认，立即清仓避险。", "F"
        if ret60 > self.fake_strong_ret60_min and ret20 < self.fake_strong_ret20_max and score > self.top20_low_score:
            return "假强势预警", "中期向好但短期转负", "tag-orange", "强弩之末", "60日强势({:.1%})但20日回落({:.1%})，靠惯性死撑，短期已不赚钱。".format(ret60, ret20), "B"
        if is_ice_point:
            return "冰点反转", "卖盘枯竭引发极寒反转", "tag-ice-blue", "冰点反转", "高赔率左侧机会，严格设止损。", "B"
        if score < a_min_score:
            return "弱市调整中", "技术分过低", "tag-grey", "空仓规避", "切勿操作，坚决不要抄底。", "F"
        if score >= s_min_score and is_top15:
            return "全能冠军", "技术资金双击", "tag-gold", "全能冠军", "技术满分且机构重仓，持有的核心理由。", "S"
        if metrics.get('is_strong_breakthrough', False) and score >= self.strong_breakthrough_min_score:
            return "强力突破者", "脱离震荡平台", "tag-red", "强势破局", "量价配合良好，果断跟随。", "S"
        if score >= self.beast_min_score and is_top40:
            return "动能猛兽", "短期暴拉", "tag-red", "动能加速", "进入主升加速期，注意风险。", "S"
        if (score >= self.steady_climb_min_score and
                ma20_slope > self.steady_ma_slope_min and
                rsi >= self.steady_rsi_min and
                metrics.get('volatility_20d', 999) < self.steady_vol_max):
            p_desc = "主线稳健派" if is_top40 else "独立稳健派"
            sig_text = "顺势做多" if is_top40 else "持有观察"
            sig_desc = (
                "形态稳健且处于市场主流，核心关注对象。" if is_top40 else "自身形态良好，但暂未获市场共识，注意仓位。")
            return "稳健爬升者", p_desc, "tag-green", sig_text, sig_desc, "A"
        if score >= self.flag_min_score and ma20_slope > 0:
            return "高位旗形整理者", "整理末端", "tag-blue", "精准狙击", "重点观察，等待放量突破信号。", "A"
        if is_top20 and self.wash_min_score <= score < self.beast_min_score:
            return "主线洗盘中", "主线强势震荡", "tag-blue", "空中加油", "机构动量排位极高，短期回撤洗盘，缩量企稳可低吸。", "A"
        if score >= self.wolf_min_score and rank_pct > self.wolf_rank_pct_threshold:
            return "逆势孤狼", "形态好但无资金", "tag-purple", "警惕骗炮", "自娱自乐品种，缺乏主线资金共识，随时可能补跌。", "A"
        if rank > yesterday_rank and yesterday_rank <= max(1, total // 3) and score < self.top_warning_max_score:
            return "筑顶高危", "排名连续下滑", "tag-orange", "逢高减仓", "趋势破位前夜，主力资金可能正在撤出。", "A"
        if score >= a_min_score:
            return "潜力观察股", "特征不显", "tag-purple", "边缘试探", "多看少动，等待明确信号。", "A"
        return "弱市调整中", "技术分过低", "tag-grey", "空仓规避", "切勿操作，坚决不要抄底。", "F"
    def _detect_ice_point_reversal(self, metrics: dict, df: pd.DataFrame) -> bool:
        if df.empty or len(df) < 2: return False
        today, yesterday = df.iloc[-1], df.iloc[-2]
        score, daily_change, vol_ratio = metrics.get('score', 100), metrics.get('daily_change', 0), metrics.get('volume_ratio', 0)
        is_deeply_oversold = (score < self.ice_point_oversold_score and
                              today['收盘'] < today[f'MA{self.ma_mid}'] and
                              today['收盘'] < today[f'MA{self.ma_long}'] and
                              metrics.get('bias20', 0) < self.ice_point_bias_max)
        return (is_deeply_oversold and
                daily_change > self.ice_point_daily_chg_min and
                vol_ratio > self.ice_point_vol_ratio_min and
                today['收盘'] > yesterday['最高'])
    def _analyze_stock_conditions(self, df: pd.DataFrame) -> tuple:
        if df.empty or len(df) < self.ma_mid:
            return False, "数据不足", *([np.nan] * 16)
        today, r20 = df.iloc[-1], df.iloc[-self.ma_mid:]
        d_below = r20[r20['收盘'] < r20[f'MA{self.ma_mid}']].shape[0]
        net_v = r20[r20['涨跌幅'] > self.price_change_up_threshold].shape[0] - \
                r20[r20['涨跌幅'] < self.price_change_down_threshold].shape[0]
        rmax, r20_close = r20['收盘'].cummax(), r20['收盘']
        mdd = (r20_close - rmax).div(rmax).min() if not rmax.empty else 0
        l20, h20 = r20['最低'].min(), r20['最高'].max()
        ms, rs, v20 = np.nan, np.nan, np.nan
        ym = df[f'MA{self.ma_mid}'].iloc[-self.slope_period:].dropna()
        yr = df[f'RS_MA{self.rs_ma_period}'].iloc[-self.slope_period:].dropna()
        if len(ym) >= 2:
            ms = (np.polyfit(range(len(ym)), ym.values, 1)[0] / ym.mean())
        if len(yr) >= 2:
            rs = (np.polyfit(range(len(yr)), yr.values, 1)[0] / yr.mean())
        isa = (today.get(f'MA{self.ma_very_short}', 0) > today.get(f'MA{self.ma_mid}', 0) > today.get(
            f'MA{self.ma_long}', 0))
        if not r20['涨跌幅'].isnull().all():
            v20 = r20['涨跌幅'].std()
        atr_pct = today.get('ATR_Pct', np.nan)
        rsi_val = today.get('RSI', np.nan)
        macd_hist_slope = today.get('MACD_Hist_Slope', np.nan)
        return (True, "✅", d_below, net_v, mdd, ms, today.get('成交量比'), rs,
                today.get(f'BIAS{self.ma_mid}'), isa, v20, l20, h20, today['收盘'],
                today.get('涨跌幅'), atr_pct, rsi_val, macd_hist_slope)
    def _calculate_score(self, m: dict) -> Tuple[int, List[str]]:
        sd = {}
        phase = getattr(self, 'market_phase', "RANGING")
        ms = m.get('ma20_slope', np.nan)
        rs = m.get('rs_slope', np.nan)
        md = m.get('max_drawdown', -1)
        bs = m.get('bias20', 0)
        b20 = m.get('days_below_ma20', 0)
        nv = m.get('net_volatility_days', 0)
        atr_pct = m.get('atr_pct', np.nan)
        vol_ratio = m.get('volume_ratio', np.nan)
        rsi = m.get('rsi', np.nan)
        macd_hist_slope = m.get('macd_hist_slope', 0)
        rank = m.get('rank', 999)
        rs_weight = 15
        drawdown_weight = 15
        if phase == "STRONG_BULL":
            rs_weight += 5
        elif phase == "WEAK_BEAR":
            drawdown_weight += 8
            rs_weight -= 3
        sd['MA斜率'] = 0 if pd.isna(ms) or ms <= 0 else 15 if ms > self.ma_slope_strong else 10 if ms > self.ma_slope_mid else 5
        sd['RS斜率'] = 0 if pd.isna(rs) or rs <= 0 else rs_weight if rs > self.rs_slope_strong else 10 if rs > self.rs_slope_mid else 5
        sd['多头排列'] = 15 if m.get('is_bullish_alignment', False) else 0
        sd['最大回撤'] = drawdown_weight if md > self.drawdown_strong else 10 if md > self.drawdown_mid else 5 if md > self.drawdown_weak else 0
        sd['平滑度(ATR%)'] = 0 if pd.isna(atr_pct) or atr_pct > self.atr_punish_threshold else 10 if atr_pct < self.atr_strong else 7 if atr_pct < self.atr_mid else 3
        sd['健康乖离(BIAS)'] = (10 if self.bias_strong_low <= bs <= self.bias_strong_high else
                                 7 if self.bias_good_low < bs < self.bias_good_high else
                                 3 if self.bias_mid_low < bs < self.bias_mid_high else 0)
        sd['支撑度(低于MA20)'] = 15 if b20 <= self.below_ma_strong else 10 if b20 <= self.below_ma_mid else 5 if b20 <= self.below_ma_weak else 0
        sd['攻击性(高波)'] = 5 if nv >= self.net_vol_strong else 3 if nv >= self.net_vol_mid else 0
        divergence_penalty = -10 if vol_ratio > self.vol_ratio_divergence and rs < 0 else 0
        sd['量价背离'] = divergence_penalty
        rsi_penalty = 0
        if phase == "WEAK_BEAR":
            if rsi > self.rsi_weakbear_high:
                rsi_penalty = -12
            elif rsi > self.rsi_weakbear_mid:
                rsi_penalty = -8
        sd['RSI超买惩罚'] = rsi_penalty
        macd_penalty = 0
        if macd_hist_slope < self.macd_penalty_strong:
            macd_penalty = -12
        elif macd_hist_slope < self.macd_penalty_mid:
            macd_penalty = -8
        sd['MACD动能衰竭'] = macd_penalty
        sc = sum(sd.values())
        return max(0, int(sc)), [f"总分: {sc} "] + [f"{k}: {v}" for k, v in sorted(sd.items(), key=lambda x: x[1], reverse=True)]
    def _get_volume_price_profile(self, pos_pct: float, vol_ratio: float, daily_change: float) -> Tuple[str, str, str]:
        if pd.isna(vol_ratio) or pd.isna(daily_change) or pd.isna(pos_pct):
            return "数据不足", "vp-na", "缺少所需数据"
        if pos_pct > self.vp_high_pos:
            if vol_ratio > self.vp_high_vol and daily_change < -self.vp_big_change:
                return "高位放量杀跌", "vp-danger", "价格在高位，成交量巨大但收盘大跌，出货信号，风险极高！"
            if vol_ratio > self.vp_high_vol and abs(daily_change) < self.vp_small_change:
                return "高位天量滞涨", "vp-danger", "天量但价格涨不动，买盘衰竭迹象，警惕见顶反转。"
        if pos_pct < self.vp_low_pos:
            if vol_ratio < self.vp_low_vol and abs(daily_change) < self.vp_small_change:
                return "地量地价", "vp-success", "成交量极度萎缩，卖盘枯竭，见底可靠信号之一。"
            if vol_ratio > self.vp_high_vol and daily_change > self.vp_big_change:
                return "底部放量突破", "vp-success", "长期低位突然放量大涨，新一轮行情启动信号。"
        if vol_ratio > self.vol_ratio_divergence and abs(daily_change) < self.vp_small_change:
            return "多空拉锯", "vp-warn", "成交量显著放大但价格原地踏步，多空博弈激烈。"
        if vol_ratio > 1.2 and daily_change > 0.01:
            return "价涨量增", "vp-neutral-good", "健康的上涨形态。"
        return "常规波动", "vp-na", "量价关系正常，无明显异动。"
    def _render_tier_table(self, data_list: list) -> str:
        if not data_list:
            return '<div class="no-data-msg">该战术区域暂无符合条件的标的。</div>'
        df = pd.DataFrame(data_list)
        for col in ['Tier', 'raw_score']:
            if col in df.columns:
                df = df.drop(columns=[col])
        table_html = df.to_html(index=False, classes="styled-table", escape=False)
        return f'<div class="table-wrapper">{table_html}</div>'
    def _generate_html_report(self):
        banner_class = "banner-bull" if self.market_is_bullish else "banner-bear"
        market_banner_html = f'<div class="market-banner {banner_class}">{self.market_status_text}</div>'
        s_table = self._render_tier_table(self.tier_results['S'])
        a_table = self._render_tier_table(self.tier_results['A'])
        b_table = self._render_tier_table(self.tier_results['B'])
        f_table = self._render_tier_table(self.tier_results['F'])
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_title = f'ETF基金收市分析报告'
        html_template = f"""
                <!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
                <!-- 【新增】移动端总体缩放关键：自动适配 + 允许轻微放大 -->
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.8">
                <title>ETF基金收市分析报告</title><style>
                body {{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",Roboto, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #212529;}}
                .container {{max-width: 1600px; margin: 0 auto;}}h1 {{text-align: center; color: #343a40; font-weight: 600; letter-spacing:1px; margin-bottom: 5px;}}
                .info {{text-align: center; color: #6c757d; font-size: .9em; margin-bottom: 25px;}}
                .market-banner {{padding: 12px 20px;margin-bottom: 25px; border-radius: 8px; font-size: 1.1em;font-weight: 600; text-align: center;}}
                .banner-bull {{background-color: #d1e7dd; color: #0a3622; border: 1px solid #a3cfbb;}}
                .banner-bear {{background-color: #f8d7da; color: #58151c; border: 1px solid #f1aeb5;}}
                .tier-panel{{margin-bottom: 30px; background: #fff; border-radius: 12px; overflow: hidden; border: 1px solid #dee2e6; box-shadow: 0 4px 12px rgba(0,0,0,.05);}}
                .tier-header{{padding: 14px 24px; font-size: 1.15em;font-weight: 600; letter-spacing: 1px; border-bottom: 1px solid #dee2e6;}}
                .header-s {{background-color: #f0fff4; color: #2f6f4f;}} .header-a{{background-color: #fff8e1; color: #8d6e63;}} .header-b {{background-color: #e0f7fa; color: #006064;}} .header-f {{background-color: #f5f5f5; color: #757575;}}
                .styled-table {{width: 100%; border-collapse: collapse; table-layout: fixed;}}
                .styled-table th, .styled-table td {{padding: 10px 8px; border-bottom: 1px solid #e9ecef; vertical-align: middle; text-align: center; word-wrap: break-word;}}
                .styled-table th {{background-color: #f8f9fa; color: #495057;font-weight: 600; font-size: .9em; border-bottom: 2px solid #dee2e6; position: sticky; top: 0; z-index: 10;}}
                .styled-table tbody tr:hover {{background-color: #f1f3f5;}}
                .no-data-msg {{padding: 30px; text-align: center; color: #6c757d; font-style: italic;}}
                .styled-table th:nth-child(1), .styled-table td:nth-child(1) {{ width: 5%; text-align: left; padding-left: 15px;}}
                .styled-table th:nth-child(2), .styled-table td:nth-child(2) {{ width: 16%; text-align: left;}}
                .styled-table th:nth-child(3) {{ width: 5%; }}
                .styled-table th:nth-child(4) {{ width: 10%; }}
                .styled-table th:nth-child(5) {{ width: 9%; }}
                .styled-table th:nth-child(6), .styled-table td:nth-child(6) {{ 
                    width: 13% !important; 
                    min-width: 235px; 
                    max-width: 270px;
                    overflow: visible !important;
                    position: relative;
                }}
                /* 新增：20日位置现在是第9列，固定合适宽度 */
                .styled-table th:nth-child(9), .styled-table td:nth-child(9) {{ width: 9%; min-width: 95px; }}
                .profile-evolution-cell{{display:inline-flex;align-items:center;gap:3px;background-color:#f8f9fa;padding:4px 6px;border-radius:18px;border:1px solid #e9ecef;max-width:100%;flex-wrap:wrap;overflow:hidden;font-size:0.82em;line-height:1.05;}}
                .spark-box{{margin:0 1px;width:12px;height:12px;border-radius:50%}}
                .tag{{padding:4px 12px;border-radius:14px;font-weight:600;font-size:0.8em;white-space:nowrap}}
                /* 颜色全部恢复（与原版完全一致） */
                .tag-gold{{background:#ffc107;color:#343a40}}.tag-red{{background:#ef4444;color:#fff}}.tag-green{{background:#22c55e;color:#fff}}
                .tag-orange{{background:#f97316;color:#fff}}.tag-blue{{background:#3b82f6;color:#fff}}.tag-purple{{background:#8b5cf6;color:#fff}}
                .tag-grey{{background:#6b7280;color:#fff}}.tag-ice-blue{{background:#06b6d4;color:#fff}}.tag-black{{background:#111827;color:#fff}}
                /* 工具提示永不遮挡 */
                .has-tooltip{{position:relative;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;width:100%}}
                .tooltip{{visibility:hidden;opacity:0;position:absolute;bottom:calc(100% + 12px);left:50%;transform:translateX(-50%);background-color:#343a40;color:#f8f9fa;padding:10px 15px;border-radius:6px;font-size:.85em;line-height:1.6;white-space:pre-wrap;text-align:left;width:max-content;max-width:350px;box-shadow:0 5px 15px rgba(0,0,0,.3);transition:opacity .2s,visibility .2s;z-index:1000 !important;pointer-events:none}}
                .has-tooltip:hover .tooltip{{visibility:visible;opacity:1}}
                .mom-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; line-height: 1.2; }}
                .mom-rank-line {{ display: flex; align-items: baseline; gap: 5px; }}
                .rank-main {{ font-size: 1.1em; font-weight: 700; color: #212529; font-family: 'Segoe UI',Roboto,Arial,sans-serif; }}
                .rank-change {{ font-size: 0.3em; font-weight: 700; }}
                .rank-up {{ color: #dc3545; }}
                .rank-down {{ color: #198754; }}
                .rank-score {{ font-size: 0.8em; color: #6c757d; }}
                .signal-cell{{font-weight:600;padding:5px 12px;border-radius:16px;display:inline-block;border:1px solid transparent}}.signal-buy-strong,.signal-reversal{{background-color:#cce9e0;border-color:#b8ddd1;color:#05513e}}.signal-reversal{{background-color:#cfe2ff;border-color:#9ec5fe;color:#0a58ca}}.signal-risk-high{{background-color:#f8d7da;border-color:#f1aeb5;color:#58151c}}.signal-risk-medium{{background-color:#fff3cd;border-color:#ffecb5;color:#664d03}}
                .signal-posture-hold-strong{{color:#0d6efd;background-color:#e7f1ff}}.signal-posture-follow{{color:#0dcaf0}}.signal-posture-wait{{color:#fd7e14}}.signal-posture-avoid{{color:#fff;background:#343a40}}
                .pos-bar-wrapper{{width:100px;height:8px;background:#e9ecef;border-radius:4px;display:inline-block;position:relative;vertical-align:middle}}.pos-bar-marker{{height:12px;width:4px;border-radius:2px;position:absolute;top:-2px;transform:translateX(-50%)}}.pos-center-line{{height:8px;width:1px;background:#ced4da;position:absolute;left:50%;top:0}}
                .vp-tag{{padding:4px 10px;border-radius:4px;font-size:.85em;font-weight:600;border:1px solid}}.vp-danger{{background-color:#f8d7da;border-color:#f1aeb5;color:#b02a37}}.vp-success{{background-color:#d1e7dd;border-color:#a3cfbb;color:#146c43}}.vp-warn{{background-color:#fff3cd;border-color:#ffecb5;color:#664d03}}.vp-buy{{background-color:#cfe2ff;border-color:#9ec5fe;color:#0a58ca}}.vp-na{{color:#6c757d;font-size:.9em}}.vp-neutral-good{{color:#0a3622}}
                footer {{ text-align: center; padding: 20px; margin-top: 40px; font-size: 0.85em; color: #6c757d; border-top: 1px solid #dee2e6; }}
                /* ==================== 【新增】移动端友好层（不影响桌面） ==================== */
                .table-wrapper {{
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                }}
                .table-wrapper::-webkit-scrollbar {{ height: 6px; }}
                .table-wrapper::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 3px; }}
                @media (max-width: 768px) {{
                    body {{ padding: 12px 8px; }}
                    .container {{ max-width: 100%; }}
                    h1 {{ font-size: 1.35em; }}
                    .styled-table th, .styled-table td {{ padding: 8px 5px !important; font-size: 0.84em; }}
                    /* 重点优化形态演化轴（最容易溢出的列） */
                    .styled-table th:nth-child(6), .styled-table td:nth-child(6) {{
                        min-width: 165px !important; max-width: 195px !important;
                    }}
                    /* 新增：移动端第9列（20日位置）紧凑处理 */
                    .styled-table th:nth-child(9), .styled-table td:nth-child(9) {{ min-width: 75px !important; }}
                    .profile-evolution-cell {{ font-size: 0.78em; padding: 3px 5px; gap: 2px; }}
                    .spark-box {{ width: 9px; height: 9px; }}
                    .tag {{ font-size: 0.73em; padding: 2px 7px; }}
                    .pos-bar-wrapper {{ width: 68px; }}
                    .mom-container {{ font-size: 0.9em; }}
                    .tier-header {{ font-size: 1.05em; padding: 12px 16px; }}
                }}
                @media (max-width: 480px) {{
                    .styled-table th, .styled-table td {{ font-size: 0.81em; }}
                    .styled-table th:nth-child(6), .styled-table td:nth-child(6) {{ min-width: 150px !important; }}
                }}
                </style></head><body>
                    <div class="container">
                        <h1>{report_title}</h1>
                        <div class="info">生成时间: {report_time}</div>
                        {market_banner_html}
                        <div class="tier-panel"><div class="tier-header header-s">S级：主线共振突破与核心多头 (顺势做多)</div>{s_table}</div>
                        <div class="tier-panel"><div class="tier-header header-a">A级：蓄势观察与筑顶防守 (重点监控)</div>{a_table}</div>
                        <div class="tier-panel"><div class="tier-header header-b">B级：高风险博弈区 (冰点反转/动能衰竭)</div>{b_table}</div>
                        <div class="tier-panel"><div class="tier-header header-f">F级：绝对规避与陷阱区 (拒绝抄底)</div>{f_table}</div>
                        <footer>
                            数据来源于腾讯历史行情数据, 本报告仅供参考，非投资建议。
                        </footer>
                    </div>
                </body></html>
                """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
            print(f"\n[系统] 分析报告已生成: {self.output_file}")
        except IOError as e:
            print(f"[文件错误] 无法写入HTML报告: {e}")
    def _clear_old_data_files(self):
        today_str = self.latest_trade_date.replace('-', '') if self.latest_trade_date else datetime.now().strftime('%Y%m%d')
        try:
            for f_name in os.listdir(self.data_dir):
                if f_name.endswith('.csv') and today_str not in f_name:
                    try:
                        os.remove(os.path.join(self.data_dir, f_name))
                    except OSError:
                        pass
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    ETF_WATCHLIST = ['159326', '512400', '159516', '512880', '159206', '159870', '515880', '159869', '516150',
                     '159852', '515220', '159201', '515790', '512660', '159755', '515210', '159611', '512690',
                     '512800', '159851', '561360', '560710', '159766', '512200', '518880', '562500', '513120',
                     '513050', '513520', '159941', '159667', '159825', '560280']
    screener = ETFScreener(etf_codes=ETF_WATCHLIST)
    screener.run()
