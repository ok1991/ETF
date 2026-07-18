# ETF V4 行业 ETF 自进化信号系统

ETF-main 下载中国行业 ETF 行情，生成 schema V4 信号，并通过严格的样本外门槛决定是否允许新仓。系统保持 fail-closed：数据、因子注册表、校准、市场权限或风险条件不满足时仍生成报告，但不输出可执行新仓。

> “跑赢沪深300”是 Walk-Forward 验收目标，不是收益承诺。注册表和校准只有在样本外超额收益、信息比率、稳定折数及回撤门槛同时通过时才会启用。

## 主要模块

- `etf_radar/signals/factors.py`：原有 V4 月/周趋势、入场形态、相对强度和结构止损。
- `etf_radar/factor_evolution.py`：监控因子 IC、IC-IR、5/10/20 日衰减和换手；自动退休失效因子；用符号遗传编程生成候选并用 Ridge 集成。
- `etf_radar/universe.py`：行业 ETF 的宽行业分组，因子先组内去均值再做截面标准化。
- `etf_radar/trading.py`：ETF 佣金万 1.5、无最低佣金，并计入交易所费用、买卖价差、滑点、成交额参与率冲击和 100 份整手约束。
- `etf_radar/validation.py`：按日历滚动的 expanding Walk-Forward，含 20 个交易日 purge 和 5 个交易日 embargo。
- `etf_radar/rotation.py`：两个错开5个交易日的行业轮动袖套，每个袖套持有10日，按相对强度、趋势效率、量能确认和V4优先级选择Top3。
- `etf_radar/calibration/pipeline.py`：历史截面生成、每折无泄漏进化、组合回测、沪深300比较和最终验收。

## 本地运行

Windows 新终端：

```powershell
python -m pip install -r requirements.txt
python main.py
```

如果当前终端尚未刷新 PATH，可使用：

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 main.py
```

主要输出：

- `public/index.html`
- `public/etf_signals_latest.json`
- `public/market_env_latest.json`
- `artifacts/calibration/v4_calibration.json`
- `artifacts/calibration/v4_acceptance_report.json`
- `artifacts/calibration/adaptive_factor_registry.json`
- `artifacts/calibration/rotation_model.json`
- `public/etf_rotation_latest.json`
- `.runtime/data/`、`.runtime/state/`、`.runtime/logs/`

## 运行闭环进化

先刷新行情，再运行带 Walk-Forward 的进化和校准，最后重新生成信号：

```powershell
$env:FORCE_DOWNLOAD="true"
python main.py
python calibrate_v4.py --sample-step 5
$env:FORCE_DOWNLOAD="false"
python main.py
```

`adaptive_factor_registry.json` 会记录每个新因子的表达式、简短经济逻辑、训练/验证 IC、IR、衰减、换手、退休原因和替换关系。未通过样本外门槛时，日常策略自动回退到原 V4 优先级，不使用新因子。

完整历史行会缓存到 `.runtime/state/v4_calibration_rows.json`。仅当行情指纹与 `sample-step` 一致时才可复用：

```powershell
python calibrate_v4.py --sample-step 5 --reuse-rows-cache
```

轮动模型与事件信号分别审批。事件入场不通过时仍保持阻断；只有 `rotation_model.json` 的总超额、IR、回撤和年度稳定性全部通过时，才生成非空的 `public/etf_rotation_latest.json` 目标权重。

## 测试与迭代

```powershell
python -m unittest discover -s tests -v
python -m py_compile etf_radar\factor_evolution.py etf_radar\calibration\pipeline.py
```

建议迭代流程：

1. 每日刷新行情并运行 `main.py`；双周运行 `calibrate_v4.py`，避免过度频繁挖掘。
2. 查看注册表中的 `retired_factors`、`new_replacements` 与 `ensemble_validation_metrics`。
3. 只有在多个 Walk-Forward 折中保持正超额收益、正信息比率且成本后回撤可接受时才提升风险预算。
4. 调整 GP 种群或代数后必须重新跑全量 Walk-Forward，不要依据全样本结果直接上线。

GitHub Actions 已配置日常分析与双周校准；双周任务会提交校准模型、验收报告、因子注册表及公共分析产物。
