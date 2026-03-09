# train_ml_model.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from test import ETFScreener
print("🚀 V22.1 ML训练启动（预计 30~90 秒）...")
ETF_WATCHLIST = [
                '159326', '512400', '159516', '512880', '159206', '159870',
                '515880', '159869', '516150', '159852', '515220', '159201',
                '515790', '512660', '159755', '515210', '159611', '512690',
                '512170', '512800', '159851', '561360', '560710', '159766',
                '512200', '159865', '518880', '513050', '513520', '159941'
                ]
screener = ETFScreener(ETF_WATCHLIST)
X, y = [], []
print(f"正在收集历史样本（{len(ETF_WATCHLIST)} 只 ETF）...")
for code in ETF_WATCHLIST:
    try:
        df_full = screener.get_etf_data(code)
        if len(df_full) < 100: continue
        for i in range(60, len(df_full) - 10, 4):  # 更宽松采样，速度更快
            slice_df = df_full.iloc[:i+1].copy()
            analyzed = screener._calculate_etf_indicators(slice_df, slice_df)
            if analyzed.empty or len(analyzed) < screener.ma_mid: continue
            is_match, msg, *a_flags = screener._analyze_stock_conditions(analyzed)
            if not is_match: continue
            (d_b_20, net_v, max_d, ma20_s, v_rat, rs_s, b20, is_bull, vol20, l20, h20,
             close_p, daily_chg, atr_pct, rsi_val, macd_hist_slope) = a_flags
            metrics = {
                'days_below_ma20': d_b_20, 'net_volatility_days': net_v, 'max_drawdown': max_d,
                'ma20_slope': ma20_s, 'volume_ratio': v_rat, 'rs_slope': rs_s, 'bias20': b20,
                'is_bullish_alignment': is_bull, 'volatility_20d': vol20, 'daily_change': daily_chg,
                'mom_score': 0, 'rank': 999, 'ret20': 0, 'ret60': 0,
                'atr_pct': atr_pct, 'rsi': rsi_val, 'macd_hist_slope': macd_hist_slope
            }
            features_df = screener._extract_ml_features(metrics)  # ← 14特征一致
            X.append(features_df.iloc[0])
            future_ret = (df_full.iloc[i+10]['收盘'] / df_full.iloc[i]['收盘'] - 1)
            y.append(1 if future_ret > 0 else 0)
    except Exception as e:
        print(f"  └─ {code} 跳过（{type(e).__name__}）")
X = pd.DataFrame(X)
y = np.array(y)
if len(X) == 0:
    print("❌ 没有收集到样本，请检查 akshare")
else:
    print(f"✅ 样本收集完成！共 {len(X)} 个训练样本")
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                               num_leaves=31, random_state=42, verbosity=-1)
    model.fit(X, y)
    model.booster_.save_model("ml_model.txt")
    print("🎉 模型已保存！现在运行 test.py 即可看到混合评分")
print("\n💡 提示：以后每天先跑这个脚本，再跑主程序。")