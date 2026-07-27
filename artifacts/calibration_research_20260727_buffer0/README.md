# 研究包：2026-07-27 buffer=0 未过关重校准

- status: research_only_not_promoted
- artifact_bundle_id: 3a9086f4d3fb3bff20023036
- rank_buffer: 0
- rotation_approved: false
- 失败门槛: rolling_12m_positive_excess_ratio_min_0_60
- 全样本滚动12个月正超额: 0.568831 (< 0.60)
- 用途: 只对比、不自动上线
- 对照冻结包: 15be8c14f7d55db947b7fac5 (buffer=3, approved)

本目录不是生产权威。生产只认 `artifacts/calibration/` 且受 `frozen_production_pin.json` 保护。

## ??????

- `buffer_stability_comparison.json`?0/1/2/3 ????????????????
- ???`docs/research-buffer3-ejection-20260727.md`?`docs/buffer-stability-20260727.md`
