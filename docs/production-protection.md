# 生产保护规则

## 一句话

**线上只认冻结包 `15be8c14f7d55db947b7fac5`。未过关包不得覆盖生产。**

## 现役冻结包

| 字段 | 值 |
|------|----|
| `artifact_bundle_id` | `15be8c14f7d55db947b7fac5` |
| 冻结源 | `.runtime/calibration-defense-recovery-freeze-20260726/` |
| 生产目录 | `artifacts/calibration/` |
| 保护钉 | `artifacts/calibration/frozen_production_pin.json` |
| `rank_buffer` | 3 |
| rotation | approved |
| 关键验收 | 开发窗回撤 0.237651；信息比率 0.527056；滚动12个月正超额 0.60；21/21 门槛通过 |

## 硬规则

1. **生产权威唯一**：`artifacts/calibration/` 是唯一线上校准权威目录（`PATHS.calibration`）。
2. **未过关不得覆盖**：`rotation_approved != true` 的候选包，禁止写入/晋升生产目录。
3. **冻结钉优先**：`frozen_production_pin.json` 中 `active=true` 且 `challenge_open=false` 时，禁止用其他 bundle 替换现役冻结包。
4. **研究可继续**：新校准必须先进入隔离区 / 研究目录，不得默认 `force-calibration` 直推生产。
5. **正式替换路径**：仅 `validate_staged_bundle` → 生产保护门禁 → 人工确认 → `promote_staged_bundle`。
6. **失败回滚**：晋升中异常必须按备份回滚；回滚后仍应保留冻结包。

## 代码门禁（2026-07-27 起）

`etf_radar/cycle.py` 在自动闭环中：

- 校验通过但 **rotation 未批准** → 状态 `CALIBRATION_STAGED_NOT_PROMOTED`，staging 保留，**不**调用 `promote_staged_bundle`
- 现役存在 **已批准生产包** 时，拒绝未批准候选覆盖
- 存在 **active 冻结钉** 且未 `challenge_open` 时，拒绝替换为其他 bundle

## 明确禁止

- 禁止在保护规则落地前对生产目录直接 `python calibrate_v4.py`（默认会写生产路径）
- 禁止 `run_cycle.py --force-calibration` 把未过关结果当成可交易包提交
- 禁止手改 `artifacts/calibration/` 成员 hash 以绕过 bundle 校验
- 禁止把研究包目录软链/复制覆盖生产目录

## 允许的研究动作

- 在 `.runtime/calibration-*` staging 重跑
- 在 `artifacts/calibration_*` 或 `artifacts/calibration_research_*` 落研究包
- 写对照表、稳定性扫描、挑战赛草稿
- 影子成本校准（`shadow_only=true`, `promotion_allowed=false`）
