# 对照实验：冻结包 A vs 重搜包 B vs 强制 buffer3

日期：2026-07-28  
隔离产物：`artifacts/calibration_experiment_dual_pack_20260728/`

## 一句话

**同研究指纹上强制 buffer=3 也不能过关；生产应继续只用冻结包 A。**

## 三包结果

| 包 | 设定 | 关键结果 | 可上线 |
|----|------|----------|--------|
| A | 冻结规格 buffer=3 | 21/21 过关；IR 0.527；roll12 0.60；dd 0.238 | 是 |
| B | 重搜自动选 buffer=0 | roll12 0.569 失败 | 否 |
| B3 | 重搜指纹强制 buffer=3 | 开发窗 dd 0.2526⇒全样本 dd≥0.2526，绝对回撤门失败 | 否 |

## 为什么 B3 的下界成立

1. 选型开发窗 = 全样本在 2024-01-01 前的数据前缀  
2. 同一 `rank_buffer` 下，全样本模拟在前缀期路径与开发窗模拟一致  
3. 因此全样本最大回撤不会低于开发窗最大回撤  
4. 研究指纹 buffer3 开发窗 dd=0.252626 > 0.25 → 全样本绝对回撤门必败  

> 本地没有 7/27 研究指纹的行情缓存，故 B3 用官方 development_candidates 做下界证明，而不是假装点位重放。

## 决策

- 不打开冻结钉挑战  
- 不把 B/B3 写入 `artifacts/calibration/`  
- 后续若要真点位重放 B3，需保存研究指纹对应的 predictions/rows 或当日数据快照  

相关：

- `docs/research-buffer3-ejection-20260727.md`
- `docs/buffer-stability-20260727.md`
- `docs/promotion-challenge-rules.md`
