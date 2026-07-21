# 近五年官方口径隔离验收包（可促进，未上线）

生成时间：2026-07-21 13:20:33

## 状态
- **样本口径**：近5年滚动（entry_date ≥ 2021-06-02，至 2026-06-02）
- **轮动验收**：**21/21**
- **rotation_approved / strategy_approved**：**true**
- **factor_registry_approved**：**false**（与当前生产一致；不启用自适应因子覆盖，不影响轮动批准）
- **包 ID**：2a5c9780b2b333dd6bbd8cbf
- **生产目录**：未切换，仍为 1920278640edbabd7946738c（16/21，Fail-Closed）

## 主指标（全窗）
- 超额约 **+150%**，IR 约 **1.29**
- 最大回撤约 **15.9%**（≤25%）
- 相对回撤约 **12.7%**（≤30%）
- 滚动12月正超额约 **98.9%**，最差约 **-2.9%**
- 近三年留出超额约 **+88%**，相对回撤约 **12.6%**
- rank_buffer = **3**

## 分年超额（约）
2021 +10.2pp · 2022 +15.3pp · 2023 +22.5pp · 2024 +2.1pp · 2025 +59.4pp · 2026 +11.0pp

## 目录
D:\Lam\V4\ETF-main\artifacts\calibration_pulse_v9_5y

含：
- v4_acceptance_report.json
- rotation_model.json（approved=true）
- v4_calibration.json
- adaptive_factor_registry.json
- llm_factor_proposals.json
- calibration_bundle.json（PROMOTION_READY_ISOLATED_5Y）

## 如何促进到生产（需人工确认）
1. 确认接受「近5年」为官方验收样本口径
2. 用 promote_staged_bundle 将本目录提升到 rtifacts/calibration/
3. 发布说明更新；确认实盘仍受其它现金闸门（成本/执行证据等）约束
4. **未执行促进前，生产继续 Fail-Closed（现金）**

## 未自动做的事
- 未覆盖 rtifacts/calibration/
- 未改 public/release_note_latest.md
- 未放开实盘
