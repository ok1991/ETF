# ETF V4 行业 ETF 自进化信号系统

ETF-main 下载中国行业 ETF 行情，生成 **schema V4 事件信号** 与 **rotation V2 风险预算目标**，并与同级 `Swing-trading` 执行端形成闭环。

系统默认 **fail-closed**：数据、因子注册表、校准、市场权限或风险条件任一不满足时，发布现金目标，不保留失效 Alpha 的旧仓授权。

> “跑赢沪深300”是 Walk-Forward 验收目标，不是收益承诺。注册表和校准只有在样本外超额收益、信息比率、稳定折数及回撤门槛同时通过时才会启用。

## 目录

1. [系统概览](#系统概览)
2. [快速开始](#快速开始)
3. [日常运行](#日常运行)
4. [主要输出](#主要输出)
5. [LLM 因子提案（可选）](#llm-因子提案可选)
6. [测试与 CI](#测试与-ci)
7. [建议迭代节奏](#建议迭代节奏)
8. [附录：架构与治理细则](#附录架构与治理细则)

---

## 系统概览

| 层次 | 作用 |
|------|------|
| 行情与数据质量 | 腾讯历史 QFQ/RAW 主源 + 缓存指纹完整性校验 |
| V4 事件信号 | 月/周趋势、入场形态、相对强度、结构止损等 |
| 自适应因子 | GP / 透明因子 / LLM 候选 → 三段审批 → 注册表 |
| Rotation V2 | 双袖套行业轮动 + 点时风险预算 + 容量约束 |
| 生产闭环 | `run_cycle.py` 事务化校准、整包晋升、失败回滚 |
| 执行闭环 | 读取 Swing 反馈与实盘绩效，异常时撤权发现金 |

**入口脚本：**

| 脚本 | 用途 |
|------|------|
| `main.py` | 日常分析：刷新行情、生成信号 / rotation / 健康产物 |
| `run_cycle.py` | 生产调度：检查是否需要校准，需要则事务化 Walk-Forward 并晋升 |
| `calibrate_v4.py` | 显式全量校准 / 影子成本校准 |
| `joint_health.py` | ETF 生产端 + Swing 执行端联合健康门禁 |

**包内关键模块：**

| 路径 | 作用 |
|------|------|
| `etf_radar/pipeline.py` | 日常分析主链路 |
| `etf_radar/cycle.py` | 生产闭环与产物晋升 |
| `etf_radar/calibration/pipeline.py` | Walk-Forward 校准 |
| `etf_radar/signals/factors.py` | V4 基础因子 |
| `etf_radar/factor_evolution.py` | 因子监控、GP、组合与退休 |
| `etf_radar/llm_factor_proposals.py` | 受约束 LLM 候选 |
| `etf_radar/rotation.py` | 行业轮动与风险预算 |
| `etf_radar/trading.py` | 成本、冲击、整手约束 |
| `etf_radar/validation.py` | Expanding Walk-Forward（purge / embargo） |
| `etf_radar/universe.py` | 行业分组与截面标准化 |
| `contracts/*.schema.json` | 对外 JSON 契约 |

---

## 快速开始

日常分析与双周校准默认在 **GitHub Actions** 跑：

- 日常分析：`.github/workflows/etf-daily-analysis.yml`
- 双周校准：`.github/workflows/calibrate-v4.yml`

本地只在需要调试时使用：

```powershell
cd ETF-main
python -m pip install -r requirements.txt
python main.py
```

可选：从模板配置环境变量（密钥不要提交仓库）：

```powershell
Copy-Item .env.example .env
# 编辑 .env 后，在当前会话导出所需变量，例如：
# $env:OPENAI_API_KEY="..."
```

`.env.example` 只是配置样板；程序读取的是进程环境变量，不是 `.env.example` 本身。

---

## 日常运行

### 1. 仅刷新信号（最常用）

```powershell
python main.py
```

强制重新下载行情：

```powershell
$env:FORCE_DOWNLOAD="true"
python main.py
$env:FORCE_DOWNLOAD="false"
```

### 2. 生产闭环（推荐调度入口）

```powershell
python run_cycle.py
```

常用参数：

```powershell
python run_cycle.py --force-calibration
python run_cycle.py --sample-step 5 --workers 6
python run_cycle.py --check-last-status
```

`run_cycle.py` 会：

1. 刷新并验证行情
2. 检查模型 `generated_at`、标签训练滞后、QFQ+RAW 联合指纹
3. 在需要时（约 14 天预防性重校准、训练滞后约 53 天、指纹变化或核心产物缺失）于 `.runtime` 隔离目录跑完整 Walk-Forward
4. 仅当 V4、rotation、因子注册表、LLM 审计与验收报告共享同一 `artifact_bundle_id` 且整包验收通过后才晋升
5. 晋升失败则回滚已替换文件，保留安全生产产物

### 3. 显式校准

先强制刷新行情，再校准，再重生信号：

```powershell
$env:FORCE_DOWNLOAD="true"
python main.py
python calibrate_v4.py --sample-step 5
$env:FORCE_DOWNLOAD="false"
python main.py
```

复用历史截面缓存（仅当行情指纹与 `sample-step` 一致）：

```powershell
python calibrate_v4.py --sample-step 5 --reuse-rows-cache
```

缓存路径：`.runtime/state/v4_calibration_rows.json`。

### 4. 联合健康检查

```powershell
python joint_health.py
python joint_health.py --require-remote-distribution
```

状态为 `BLOCKED` 时进程退出码为 `2`。

### 5. 影子成本校准（不可晋升）

当 `public/execution_cost_recalibration_latest.json` 达到  
`READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION` 时：

```powershell
python calibrate_v4.py `
  --cost-model-candidate public/execution_cost_recalibration_latest.json `
  --shadow-output-dir .runtime/cost-shadow/<candidate-id> `
  --reuse-rows-cache
```

产物只写入隔离目录，清单固定含 `shadow_only=true` 与 `promotion_allowed=false`，不会覆盖生产校准目录或 rotation 授权。

---

## 主要输出

### 公共产物 `public/`

| 文件 | 含义 |
|------|------|
| `index.html` | 本地看板 |
| `etf_signals_latest.json` | V4 事件信号 |
| `etf_rotation_latest.json` | Rotation V2 目标与风险预算 |
| `data_manifest_latest.json` | 行情验收清单与权限结论 |
| `market_env_latest.json` | 市场环境 |
| `factor_health_latest.json` | 线上因子健康 |
| `factor_promotion_readiness_latest.json` | 因子推广就绪度 |
| `distribution_audit_latest.json` | 分发审计 |
| `joint_health_latest.json` | ETF + Swing 联合健康 |
| `execution_feedback_audit_latest.json` | 执行反馈审计 |
| `live_performance_audit_latest.json` | 实盘相对绩效审计 |
| `execution_cost_recalibration_latest.json` | 真实成本重估候选 |
| `cycle_status_latest.json` | 最近生产闭环状态 |
| `release_note_latest.md` | 最近发布说明 |

### 校准产物 `artifacts/calibration/`

| 文件 | 含义 |
|------|------|
| `v4_calibration.json` | V4 校准模型 |
| `v4_acceptance_report.json` | Walk-Forward 验收报告 |
| `adaptive_factor_registry.json` | 自适应因子注册表 |
| `llm_factor_proposals.json` | LLM 候选审计 |
| `rotation_model.json` | 轮动模型 |
| `calibration_bundle.json` | 整包身份与晋升元数据 |
| `production_promotion_receipt.json` | ?????? |
| `frozen_production_pin.json` | ?????????????????? |

????? [`docs/`](./docs/)????????????????????? [2026-07-26/27 ????](./docs/incident-20260726-20260727.md)?

**????????? `15be8c14f7d55db947b7fac5`??????????????**

### 运行时 `.runtime/`

- `.runtime/data/`：行情缓存
- `.runtime/state/`：状态与截面缓存
- `.runtime/logs/`：日志
- `.runtime/` 下隔离目录：事务化校准 staging、LLM shadow、成本影子校准等

### 契约 `contracts/`

- `etf_signal_v4.schema.json`
- `etf_rotation_v2.schema.json`（及历史 v1）
- `execution_feedback_*.schema.json`
- `live_performance_v1.schema.json`
- `execution_cost_recalibration_v1.schema.json`

---

## LLM 因子提案（可选）

LLM **只生成研究候选**（表达式、经济假设、失效条件），**不能**直接改权重或获得交易权限。候选与 GP / 种子 / 透明因子一起走三段审批。

### 配置

仓库提供 `.env.example`。推荐用环境变量注入，勿提交密钥。

**PowerShell 示例：**

```powershell
$env:OPENAI_API_KEY="..."
$env:LLM_FACTOR_PROPOSALS_ENABLED="auto"
$env:OPENAI_MODEL="gpt-5.6"
$env:LLM_FACTOR_PROPOSAL_COUNT="6"
python calibrate_v4.py --sample-step 5 --reuse-rows-cache
```

**云端 Responses API：**

```text
LLM_FACTOR_PROPOSALS_ENABLED=true
LLM_FACTOR_PROVIDER=openai_responses
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6
```

**本地 / OpenAI 兼容 Chat API：**

```text
LLM_FACTOR_PROPOSALS_ENABLED=true
LLM_FACTOR_PROVIDER=local
LLM_LOCAL_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
LLM_LOCAL_MODEL=<本地模型名>
# 远端兼容端点必须设置：
# LLM_LOCAL_API_KEY=...
```

### 行为要点

| 项 | 说明 |
|----|------|
| `auto` | 有 `OPENAI_API_KEY` 则调用，否则记 `MISSING_API_KEY` 并继续 GP/ML |
| `false` | 明确禁用 LLM |
| 缓存 | 合法提案约 45 天；绑定提供方 / 模型 / 端点指纹 |
| 失败 | 网络或额度问题只关闭本轮 LLM，不降低审批门槛 |
| 离线复用 | 无活动提供方但缓存有效时，只读 `CACHED_OFFLINE`，不覆盖成 `MISSING_API_KEY` |
| 生产播种 | `LLM_FACTOR_CACHE_SOURCE=/path/llm_factor_proposals.json` |
| 闭环刷新 | 默认在隔离目录刷新候选；`LLM_CYCLE_PROVIDER_REFRESH=false` 可关闭联网刷新 |

兼容端点可只写到 `/v1`，程序会规范为 `/v1/chat/completions`。无密钥仅允许 loopback；表达式经白名单 AST 解析，不做 `eval`。

---

## 测试与 CI

```powershell
python -m unittest discover -s tests -v
python -m py_compile etf_radar\factor_evolution.py etf_radar\calibration\pipeline.py
```

GitHub Actions：

- 日常分析：`.github/workflows/etf-daily-analysis.yml`
- 双周校准：`.github/workflows/calibrate-v4.yml`（提交校准模型、验收报告、注册表与公共分析产物）

CI 中设置 `REQUIRE_FRESH_MARKET_DATA=true` 时，行情不新鲜会使作业失败。

---

## 建议迭代节奏

1. **每日**：`python main.py` 或调度 `python run_cycle.py`
2. **约双周**：完整校准（`calibrate_v4.py` 或由 `run_cycle.py` 触发），避免过度频繁挖掘
3. 查看注册表中的 `retired_factors`、`new_replacements`、`ensemble_validation_metrics`
4. 只有在多个 Walk-Forward 折上保持正超额、正 IR、成本后回撤可接受时，才考虑提高风险预算
5. 调整 GP 种群或代数后必须重跑全量 Walk-Forward，不要凭全样本结果直接上线

轮动、事件信号、自适应因子 **分别审批**：

- 轮动不依赖因子注册表授权
- 事件或自适应不通过时各自阻断
- 轮动失去审批时发布 rotation V2 现金目标，通知执行端撤旧仓风险

---

## 附录：架构与治理细则

以下内容供深入运维与审计使用；日常使用通常只需上面的快速开始与日常运行。

### A. 行情来源与失败关闭

- 历史前复权与原始价格 **只用腾讯**（`TENCENT_PRIMARY` / `CACHE_TENCENT_PRIMARY`）。
- **禁止** 将东方财富加入生产、备用或降级链路。
- 新浪 **价格交叉验证已关闭**（`SINA_CROSSCHECK_DISABLED`）；新浪仅用于交易日历（`tool_trade_date_hist_sina`）。旧标签 `TENCENT_SINA_VALIDATED` / `CACHE_TENCENT_SINA_VALIDATED` 仍可被清单识别，但新下载不再做新浪收盘价核验。
- 数据质量政策：`tencent-primary-cache-integrity-v3`。缓存绑定 QFQ/RAW 全帧 SHA-256、行数、数据日期与验证政策版本。旧元数据、CSV 篡改或任一指纹不一致 → 废止缓存并尝试腾讯联网重认证，失败则清单 fail-closed。
- 每次实时启动用当前认证目录重算 `qfq-raw-joint-v2`；历史被修订或本地文件变化后，旧模型即使未到期也失去权威。V4 事件校准关闭；rotation 发布带 `ROTATION_DATA_FINGERPRINT_MISMATCH` 的现金目标。
- 必需标的须同时满足：QFQ/RAW 日期一致、达到上海时区最新已完成交易日、主源/缓存审计通过、全标的数据日期一致。
- 验收写入 `public/data_manifest_latest.json`；缺失、滞后、日期混合、来源未批准或缓存完整性失败时市场权限归零并发现金。

### B. Rotation V2 与容量

- 两个错开 **5** 个交易日的行业袖套，各持有 **10** 日；按相对强度、趋势效率、量能确认与 V4 优先级选 Top3。
- 排名缓冲：Top3 仅当跌出 `Top3 + rank_buffer` 才替换（`rank_buffer` 在开发窗样本外选择，生产包可能为 0–3）。参数只在 **2024 年以前** 样本外选择，**2024–2026** 留作近年验收。
- 只使用 V4 市场策略的点时风险预算；目标 ETF 权重之和 = 唯一权威 `max_exposure_ratio`，其余为现金；成本后最大回撤不超过 **25%**。
- 使用新浪交易日历写入明确 `execution_date`；收盘后数据只能在下一交易日用当日实时价执行。
- 发布截至 `data_date` 的 20 日 ADV 与按 **10% ADV** 计算的 `max_new_risk_amount`。
- `max_participation_rate=10%` 是新增风险单日硬上限；减仓可超上限但按上限估冲击并记容量告警。
- 全期与近三年留出期 `capacity_fill_ratio` 均不得低于 **90%**。
- 执行契约：`single-exposure-authority-v4`，只接受 `exposure_authority=v4_market_policy`。
- 点时流动性筛选：候选 10% ADV 须能承载与 Swing 生产账户一致的 **1 万元** 参考组合完整目标权重；两袖套共享同一标的当日 10% ADV，禁止重复占用。资金规模变化须重校准参考规模。
- 发布前校验 signal V4、rotation V2 schema、风险预算、目标权重与完整实盘成本契约。
- 双袖套与排名缓冲状态按 rotation schema 2 持久化。经济字段未变时可复用发布时间与完整 SHA，避免无意义使 shadow / release / Swing 缓存失效。

### C. 模型时间治理

| 字段 | 规则 |
|------|------|
| `generated_at` | 产物生成时间；默认超过 **21** 天立即失效；缺失、格式错误或位于未来同样 fail-closed |
| `trained_until` | 前瞻标签可用的训练截止；独立允许最多 **60** 天滞后（避免把约 20 交易日前瞻标签的天然延迟当老化） |

- V4、rotation、自适应注册表分别执行门槛。
- rotation 失效 → 全现金目标；注册表失效 → 仅停自适应叠加；V4 事件层失效 → 事件入场关闭。
- 日常分析仍可安全发布被阻断结果；双周校准负责更新。双周任务连续失败时，21 天硬门槛仍会撤旧模型权限。

### D. 因子审批与自适应组合

- 审批三段：**训练 / 选择 / 独立批准**；边界 purge 约 **20** 交易日；Walk-Forward 另含 **5** 交易日 embargo。
- 复杂 GP 必须与透明原始因子同池竞争。退役表达式约 **183** 天冷却，禁止静默复活。
- Purged 选择集或独立批准集不足 → 拒绝该折并记 `FACTOR_PURGE_INSUFFICIENT`，禁止回退无 purge 数据。
- 独立批准集不参与搜索；未通过者只能 `RESEARCH`，不得写入 `new_replacements`、退休线上因子或获交易权限。
- 名义多因子至少两个归一化权重 ≥ **5%** 的有效因子。
- 组合：最高相关不得达 **0.95**，最小残余方差 ≥ **10%**，选择期 IC 比最佳单因子至少 +**0.0005**。
- 选择期拆三连子阶段：至少两段 IC 非负，最差 ≥ **-0.02**。
- ML 权重只在训练集拟合、选择前冻结。
- 因子搜索政策变更后，即使历史留出曾通过，也须累计至少 **13** 个真正新增校准日期才可生产授权。
- 熟化进度绑定候选表达式集合 SHA-256；GP/LLM/原始集合变化则重置 anchor。
- 选择期对唯一表达式做 Benjamini–Hochberg；`q>10%` 不得进组合搜索。政策版本：`complementary-stability-fdr-seasoning-v7`。
- LLM 被拒表达式族 **90** 天冷却，记 `LLM_REJECTED_EXPRESSION_COOLDOWN`。
- `factor_health_latest.json`：训练截止后非重叠 10 日样本、T+1 入场、真实成本；至少 **8** 个独立观察前仅 `WARMUP`。证据错配记 `LIVE_HEALTH_REGISTRY_MISMATCH` 并拒绝授权。有效因子不足或成熟线上显著负 IC 时，仅暂停自适应叠加。

### E. Rotation 稳定超额验收

除累计超额、IR、年度胜率、绝对回撤、真实成本与容量外，还须：

| 区间 | 门槛（摘要） |
|------|----------------|
| 全期 | ≥104 个滚动 12 月观察；滚动超额为正 ≥60%；最差滚动超额 ≥ -25%；相对沪深300最大回撤 ≤30%；最长相对水下 ≤260 轮动周期 |
| 近三年留出 | ≥26 个滚动观察；滚动胜率 ≥60%；最差滚动超额 ≥ -20%；相对回撤 ≤25%；最长水下 ≤104 周期 |

任一稳定性失败 → 撤 alpha 权限并发布全现金 rotation V2。公共目标须带  
`acceptance_policy_version=rolling-excess-stability-v1`。

### F. 真实执行证据

- 含真实订单但仅有 `MODEL_ESTIMATE_ONLY` 的计划进入待确认账本；**7** 个自然日内可补交完整/部分/未成交证据；逾期撤 rotation 并发现金。
- 预期执行场次 = 上一份已批准 rotation 的 `model_version + execution_date + 策略指纹 + 目标权重`。
- 仅执行日当天、可交易且允许写入的 Swing 反馈可核销；`NO_ORDERS` 也可证明任务已跑。执行日结束仍无匹配反馈 → `EXECUTION_SESSION_MISSED`，下周期撤权发现金。
- 同日开盘前新目标原子替换旧场次，旧指纹进 `superseded_execution_keys`，避免双重计划或次日误报漏跑。
- 与 Swing-trading **同级部署** 时，若本地已有 `execution_feedback_history.json` / `live_performance_latest.json` 且未显式设远程源，优先读本地；独立部署用批准的远程源；**环境变量显式配置始终优先**。

### G. 实盘相对绩效

- 只使用当前 rotation 模型连续记录。
- 少于 20 交易日：`WARMUP`。
- 20 日相对收益低于 -5% 或 60 日低于 -8% → 完整事务化重校准。
- 模型相对最大回撤达 -10% 或策略最大回撤达 -15% → 立即撤 rotation 发现金。
- 证据指纹 / 日期 / 基准 / 历史顺序 / 净值不一致，或已发布证据超过 **7** 天未更新 → fail-closed。
- 执行日结束后无 `live_performance_latest.json` → `LIVE_PERFORMANCE_SESSION_MISSED`；净值仍归属旧模型 → `LIVE_PERFORMANCE_MODEL_SESSION_MISMATCH`。

### H. 交易成本约定（摘要）

- ETF 佣金万 **1.5**、无最低佣金；计入交易所费用；ETF 零过户费 / 印花税。
- 买卖价差、基础滑点、成交额参与率冲击、**100** 份整手。
- 发布与批准回测的成本契约必须逐项一致。

### I. 常用环境变量（摘录）

| 变量 | 作用 |
|------|------|
| `FORCE_DOWNLOAD` | 强制重新下载行情 |
| `REQUIRE_FRESH_MARKET_DATA` | CI/门禁：行情不新鲜则失败 |
| `LLM_FACTOR_PROPOSALS_ENABLED` | `auto` / `true` / `false` |
| `LLM_FACTOR_PROVIDER` | `auto` / `openai_responses` / `local` 等 |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | 云端提供方 |
| `LLM_LOCAL_ENDPOINT` / `LLM_LOCAL_MODEL` / `LLM_LOCAL_API_KEY` | 本地或兼容端点 |
| `LLM_FACTOR_PROPOSAL_COUNT` | 候选数量（闭环默认约 6，硬上限 8） |
| `LLM_FACTOR_CACHE_SOURCE` | 只读缓存播种路径 |
| `LLM_CYCLE_PROVIDER_REFRESH` | 是否在闭环中联网刷新提供方 |

更完整的默认值见 `.env.example` 与 `etf_radar/llm_factor_proposals.py`。
