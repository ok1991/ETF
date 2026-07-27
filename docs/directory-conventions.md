# 目录约定与命名规则

## 总原则

| 类型 | 可交易？ | 自动上线？ | 典型路径 |
|------|----------|------------|----------|
| 生产包 | 是（须 rotation approved） | 仅经保护门禁 + 人工确认 | `artifacts/calibration/` |
| 研究包 | 否 | 否 | `artifacts/calibration_research_*` / `artifacts/calibration_*` |
| 运行时隔离 | 否 | 否（通过后才可能晋升） | `.runtime/calibration-*` |
| 冻结证据 | 否（源）/ 是（已晋升副本） | 否 | `.runtime/*-freeze-YYYYMMDD/` |

## 生产包

- **唯一权威**：`artifacts/calibration/`
- **成员**：`v4_calibration.json`、`v4_acceptance_report.json`、`adaptive_factor_registry.json`、`llm_factor_proposals.json`、`rotation_model.json`、`calibration_bundle.json`
- **治理附件**：
  - `production_promotion_receipt.json`：晋升回执
  - `frozen_production_pin.json`：冻结保护钉
  - `.bundle_backups/<bundle_id>/`：晋升前旧包备份（目录名=新 bundle id）

## 研究包

命名：

```text
artifacts/calibration_research_<YYYYMMDD>_<tag>/
artifacts/calibration_<experiment_name>/
```

示例：

- `artifacts/calibration_research_20260727_buffer0/` — 未过关对照
- `artifacts/calibration_pulse_v9_5y/` — 历史隔离候选
- `artifacts/calibration_pulse_v9_5y_research/` — 研究说明

规则：

1. 研究包 **不得** 使用裸名 `artifacts/calibration/`
2. README 必须写：`research_only` / `promotion_allowed` / 对照的生产 bundle id
3. 研究完成 ≠ 可上线

## 运行时隔离

| 模式 | 含义 |
|------|------|
| `.runtime/calibration-<random>` | 事务 staging（`run_cycle` 自动创建） |
| `.runtime/calibration-*-freeze-YYYYMMDD` | 人工冻结证据包 |
| `.runtime/*-shadow` / `cost-shadow/` | 影子实验，不可晋升 |
| `.runtime/incident-YYYYMMDD/` | 事故取证缓存（可选） |

## 公共产物

- `public/`：可发布分析与审计；**不是**校准权威
- 日常分析可更新 `public/`，但不得借此改写生产校准语义

## 命名速记

- 生产：无后缀 `calibration`
- 研究：`calibration_research_*` 或带实验名后缀
- 冻结：路径含 `freeze-YYYYMMDD`
- 影子：路径含 `shadow`
