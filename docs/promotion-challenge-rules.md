# 新包挑战冻结包规则（可执行草稿）

## 目标

在研究继续的同时，保证生产不被“刚好重搜出来但更弱/未过关”的包打翻。

## 状态机（简版）

```text
研究中 → 隔离候选 → 人工确认挑战 → 已批准 → 已上线 →（可）已撤销
```

- **研究完成 ≠ 可上线**
- **隔离候选 ≠ 已批准**
- **已批准** 之后才允许替换冻结包

## 自动路径（机器）

1. 校准只写 `.runtime/calibration-*` 或研究目录
2. `validate_staged_bundle` 通过只代表“包完整且可审计”
3. **自动晋升额外要求**：
   - `rotation_approved == true`
   - 全部 `approval_gates == true`
   - 若存在 active 冻结钉，则 `challenge_open == true` 且候选通过挑战清单
4. 任一不满足 → `CALIBRATION_STAGED_NOT_PROMOTED`，生产不动

## 挑战清单（人工确认前必须全满足）

新包相对冻结包 `15be8c14...` 须同时满足：

1. **全部门槛通过**（当前政策 `rolling-excess-stability-v1`）
2. **不是贴线通过**
   - 滚动12个月正超额不得刚好贴 0.60（建议 ≥ 0.62 才算稳过）
   - 开发窗/全样本最大回撤不得刚好贴 0.25（建议 ≤ 0.24）
3. **关键稳定性不弱于冻结包**
   - 信息比率 ≥ 冻结包
   - 滚动12个月正超额 ≥ 冻结包
   - 最大回撤不劣于冻结包（数值不更高）
4. **选择逻辑可解释**
   - 记录 `rank_buffer` 及开发窗候选表
   - 说明为何不是“只剩 0 可选”的退化选择
5. **数据指纹与防御/恢复规格可审计**
   - 写明 fingerprint 是否变化
   - 防御阶梯 / 恢复路径是否故意变更

## 贴线警告标签

出现任一情况，标记 `HIGH_RISK_NEAR_THRESHOLD`，不得视为稳过：

- `rolling_12m_positive_excess_ratio` ∈ [0.60, 0.62)
- `max_drawdown` ∈ (0.24, 0.25]
- 开发窗仅 1 个 buffer 合格且其回撤贴线

## 人工确认动作

1. 对照表签字（见事故文档模板）
2. 将研究包路径与 bundle id 写入挑战记录
3. 打开 pin：`challenge_open=true`（仅在挑战窗口）
4. `validate_staged_bundle` + `promote_staged_bundle`
5. 更新 `production_promotion_receipt.json` 与 pin（新冻结 id，重新 `challenge_open=false`）

## 默认决策

> 若上一份已批准冻结包仍在有效期，且新搜索结果未全面优于它，**默认沿用旧规格**。
