# ETF 波段信号

`main.py` 继续输出兼容 v2 的字段，并新增 schema v3：

- `data_quality`：RAW 交易价、QFQ 分析价及复权事件；
- `trend`：仅使用已确认周线/月线的趋势状态；
- `entry`：`READY / WATCH / BLOCKED` 入场状态；
- `calibration`：3 日早停概率、10 日胜率和预期超额收益。

`trade_gate`、`gate_reasons` 与 `entry_channel` 是给交易端使用的兼容建议字段；
最终新仓裁决由 Swing-trading 根据实时价格、统一止损边界和账户状态完成。
v3 的 `calibration.approved/status_reason` 会明确说明人工校准审批状态。

生成历史校准与样本外验收报告：

```powershell
python calibrate_v3.py --sample-step 5
```

输出文件：

- `v3_calibration.json`
- `v3_acceptance_report.json`

只有 `v3_calibration.json` 中 `thresholds.approved=true` 时，生产信号才可能输出
`entry.state=READY`。校准缺失、过期或未通过验收时统一输出 `WATCH`。

运行测试：

```powershell
python -m unittest discover -s tests -v
```
