# 研究：buffer=3 为什么被踢（2026-07-27）

## 结论（先看这个）

**主因归类：数据指纹变化 + 规则硬门槛贴线，不是因子权重/防御规格被改掉。**

| 类别 | 是否主因 | 说明 |
|------|----------|------|
| 数据 | **是** | `data_fingerprint` 从 `887dfb45...` 变为 `3d94592e...`（`trained_until` 同为 2026-06-23） |
| 规则 | **是（触发器）** | 开发窗硬门槛 `max_drawdown <= 0.25` 把 buffer 1/2/3 全部剔除 |
| 信号/规格 | **否（非主因）** | `factor_weights`、`weekly_trend_min`、`exposure_authority`、成本模型两边一致 |
| 随机/贴线边界 | **是（形态）** | buffer3 开发窗回撤 0.2377 → 0.2526，只越线 **0.26 个百分点** |

一句话：历史联合指纹变了后，buffer 1/2/3 的开发窗回撤一起抬到约 25.26%，刚好压过 25% 门槛；选择器退化到只剩 buffer=0，而 0 在全样本滚动 12 个月正超额上不过关。

## 选择机制（代码）

位置：`etf_radar/calibration/pipeline.py`

1. `select_rotation_rank_buffer` 在 holdout 前的开发窗，对 buffer ∈ {0,1,2,3} 各自回测  
2. `choose_rotation_rank_buffer` 入选条件：
   - `excess_return > 0`
   - `information_ratio > 0`
   - `max_drawdown <= 0.25`
3. 排序键：`IR` ↓，`excess` ↓，`turnover` ↑，`buffer` ↑（并列时偏好更小 buffer）
4. 若无人合格：退化为在全部候选中仍按上面排序（**这就是 7/27 只剩 0 仍被选中的路径**）

## 关键对比

### 数据身份

| 项 | 冻结包 15be8c14 | 研究包 3a9086f4 |
|----|-----------------|-----------------|
| generated_at | 2026-07-26 16:17:37 | 2026-07-27 20:00:21 |
| trained_until | 2026-06-23 | 2026-06-23 |
| data_fingerprint | `887dfb45...` | `3d94592e...` |
| 政策 | `qfq-raw-joint-v2` | `qfq-raw-joint-v2` |
| factor_weights | 相同 | 相同 |
| weekly_trend_min | -0.1 | -0.1 |
| selected buffer | **3** | **0** |
| rotation_approved | true | false |

> `strategy_specification_fingerprint` 也不同，但规格哈希**包含 `rank_buffer`**。因此它主要是“选中 buffer 变化后的结果指纹”，不是防御阶梯被重写的证据。

### 开发窗：buffer3 被踢的直接数字

| buffer | 冻结 dd | 研究 dd | Δdd | 研究是否合格 |
|--------|---------|---------|-----|--------------|
| 0 | 0.237651 | 0.226198 | -0.0115 | 是 |
| 1 | 0.237651 | 0.252709 | +0.0151 | **否** |
| 2 | 0.237651 | 0.252626 | +0.0150 | **否** |
| 3 | 0.237651 | 0.252626 | +0.0150 | **否** |

- 冻结日：4 个 buffer 全合格，buffer3 以最高 IR=0.309 胜出  
- 研究日：仅 buffer0 合格；1/2/3 全部因 dd>0.25 出局  
- buffer3 越线幅度：`0.252626 - 0.25 = 0.002626`（贴线）

### 全样本：退化选择的后果

| 包 | 所选 buffer | IR | 滚动12月正超额 | 批准 |
|----|-------------|----|----------------|------|
| 冻结 | 3 | 0.527 | 0.600 | 是 |
| 研究 | 0 | 0.413 | **0.569** | 否（卡 `rolling_12m_positive_excess_ratio_min_0_60`） |

## 根因链条

```text
行情联合指纹变化（同 cutoff）
    → 开发窗净值路径重算
    → buffer1/2/3 回撤同步抬升至 ~25.26%
    → 硬门槛 max_dd<=0.25 剔除 1/2/3
    → 选择器只剩 buffer0
    → 全样本滚动12月正超额 0.569 < 0.60
    → 未批准；若无生产保护会覆盖冻结包
```

## 不是什么

1. **不是**因子权重被重搜改掉（两边权重逐项相同）  
2. **不是** acceptance policy 版本切换（都是 `rolling-excess-stability-v1`）  
3. **不是**“buffer3 策略逻辑突然失效很多”的大幅崩坏——它只是刚过线  
4. **不能**据此认为 buffer0 更优：它只是唯一幸存者，全样本稳定性更差

## 研究含义 / 后续动作

1. **生产继续钉死 15be8c14**，buffer3 冻结规格保留优先权  
2. 对“开发窗仅 1 个 buffer 合格且贴线”打 `HIGH_RISK_NEAR_THRESHOLD`，禁止自动挑战冻结包  
3. 下周可做：在同一研究指纹上强制重放 buffer=3 全样本（对照实验包 B），验证“若不允许退化到 0，全样本是否仍可过关”  
4. 中期：选择目标函数不应只看开发窗 dd 硬切，可加入换手、近端稳定性、贴线惩罚

## 证据路径

- 冻结生产包：`artifacts/calibration/`
- 研究包：`artifacts/calibration_research_20260727_buffer0/`
- 机器可读对照：`artifacts/calibration_research_20260727_buffer0/buffer_stability_comparison.json`
- 选择代码：`etf_radar/calibration/pipeline.py`（`choose_rotation_rank_buffer` / `select_rotation_rank_buffer`）
- 事故总档：`docs/incident-20260726-20260727.md`
