# 贴线警告规则（研究草稿）

## 目的

把“刚好过线/刚好被踢”从“稳健通过”里拆出来，避免自动晋升与错误自信。

## 标签

| 标签 | 条件 | 处置 |
|------|------|------|
| `NEAR_PASS_DRAWDOWN` | `max_drawdown ∈ (0.24, 0.25]` | 可过关但高风险，不得单独作为挑战冻结包的理由 |
| `NEAR_FAIL_DRAWDOWN` | `max_drawdown ∈ (0.25, 0.255]` | 虽不合格，视为贴线噪声区，优先复查数据指纹而非大改策略 |
| `NEAR_PASS_ROLL12` | 滚动12月正超额 ∈ [0.60, 0.62) | 贴线通过，挑战冻结包时一票否决“稳过” |
| `SELECTION_COLLAPSE` | 开发窗合格 buffer 数 = 1 | 禁止自动晋升；必须人工复核 |
| `DEGENERATE_BUFFER0_ONLY` | 仅 buffer0 合格且 1/2/3 均因 dd 出局 | 默认保留冻结规格 |

## 与 2026-07-27 的映射

- buffer1/2/3：`NEAR_FAIL_DRAWDOWN` + 共同抬升  
- 选择结果：`SELECTION_COLLAPSE` + `DEGENERATE_BUFFER0_ONLY`  
- 全样本 roll12=0.5688：明确失败（不是贴线通过）

## 落地建议

1. 文档层：已用于本周研究结论与挑战规则  
2. 代码层（后续）：在 `choose_rotation_rank_buffer` 输出中附加警告列表；`run_cycle` 对带 `SELECTION_COLLAPSE` 的包强制 `CALIBRATION_STAGED_NOT_PROMOTED`
