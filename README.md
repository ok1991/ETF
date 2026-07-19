# ETF V4 行业 ETF 自进化信号系统

ETF-main 下载中国行业 ETF 行情，生成 schema V4 事件信号与 rotation V2 风险预算目标，并通过严格的样本外门槛决定是否允许风险暴露。系统保持 fail-closed：数据、因子注册表、校准、市场权限或风险条件不满足时发布现金目标，不保留失效 Alpha 的旧仓授权。

> “跑赢沪深300”是 Walk-Forward 验收目标，不是收益承诺。注册表和校准只有在样本外超额收益、信息比率、稳定折数及回撤门槛同时通过时才会启用。

## 主要模块

- `etf_radar/signals/factors.py`：原有 V4 月/周趋势、入场形态、相对强度和结构止损。
- `etf_radar/factor_evolution.py`：监控因子 IC、IC-IR、5/10/20 日衰减和换手；自动退休失效因子；用符号遗传编程生成候选，并在训练集拟合后于选择集搜索低相关、真正贡献权重的互补组合，再用 Ridge 集成。
- 因子审批严格分为训练/选择/独立批准三段，并在边界purge约20个交易日；复杂GP必须与透明原始因子同池竞争。退役表达式进入约183天冷却期，不能在下一轮静默复活。
- Purged 选择集或独立批准集不足时直接拒绝该折并记录 `FACTOR_PURGE_INSUFFICIENT`，禁止回退到无 purge 数据；每个有效折会在验收报告中记录实际因子 purge 方法和截点。
- 独立批准集不参与因子或组合搜索。未通过独立批准的候选只能标记为 `RESEARCH`，不得写入 `new_replacements`、退休线上因子或获得交易权限；名义多因子组合还必须至少有两个归一化权重不低于 5% 的有效因子。
- `etf_radar/universe.py`：行业 ETF 的宽行业分组，因子先组内去均值再做截面标准化。
- `etf_radar/trading.py`：ETF 佣金万 1.5、无最低佣金，并计入交易所费用、买卖价差、滑点、成交额参与率冲击和 100 份整手约束。
- `etf_radar/validation.py`：按日历滚动的 expanding Walk-Forward，含 20 个交易日 purge 和 5 个交易日 embargo。
- `etf_radar/rotation.py`：两个错开5个交易日的行业轮动袖套，每个袖套持有10日，按相对强度、趋势效率、量能确认和V4优先级选择Top3。
- rotation V2 将轮动专用市场风险预算写入跨项目契约；目标 ETF 权重之和等于 `max_exposure_ratio`，剩余部分为现金。当前主线档位为 `RISK_OFF=50%`、`DEFENSIVE/NORMAL=100%`，并要求成本后最大回撤不超过 25%。
- 轮动换手控制采用开发期选择的排名缓冲：Top3 持仓只有跌出 Top5 才被替换。参数只在 2024 年以前的样本外区间选择，2024–2026 留作独立近年验收。
- `etf_radar/calibration/pipeline.py`：历史截面生成、每折无泄漏进化、组合回测、沪深300比较和最终验收。

## 行情来源与失败关闭

- 历史前复权与原始价格只使用腾讯行情；新浪仅用于独立核对最新交易日和最新收盘价。
- 明确禁止将东方财富数据源加入生产、备用或降级链路。
- 腾讯前复权与原始行情必须通过新浪最近5个共同交易日收盘价交叉验证；缓存同时绑定QFQ/RAW全帧SHA-256、行数、数据日期和验证政策版本。旧元数据、CSV篡改、单日价格偶然吻合或任一指纹不一致都会废止缓存认证并尝试联网重新认证，失败则行情清单 fail-closed。
- V4校准模型与rotation模型在每次实时启动时都会用当前认证目录重新计算 `qfq-raw-joint-v2` 指纹；历史数据被供应商修订或本地文件变化后，旧模型即使仍在有效期内也会失去权威。V4事件校准关闭，rotation则发布带 `ROTATION_DATA_FINGERPRINT_MISMATCH` 原因的现金目标。
- 每个必需标的必须同时满足：前复权/原始价格日期一致、达到上海时区最新已完成交易日、腾讯与新浪交叉校验通过、全标的数据日期一致。
- rotation V2 使用新浪交易日历写入明确的 `execution_date`；收盘后生成的数据只能在指定的下一交易日用当日实时价格执行。
- rotation V2 同时发布全标的、截至 `data_date` 的20日平均成交额，以及按10% ADV计算的 `max_new_risk_amount`；Swing 用这些字段计算与批准回测一致的参与率、市场冲击和实时容量余量，新增风险目标缺失、错日或额度不匹配时拒绝执行。
- `max_participation_rate=10%` 是新增风险的单日硬容量上限，不只是冲击公式的截断值；轮动回测和 Swing 都按 ADV 将新增买量截断到可成交整手。风险减仓允许超过该上限，但按上限冲击估算并显式记录容量告警。
- 批准回测正式披露容量截断次数、请求/执行/未成交金额、容量成交率和现金限制金额；全期与最近三年留出期的 `capacity_fill_ratio` 均不得低于90%。执行契约版本为 `adv-capacity-audit-authority-v3`，旧模型一律 fail-closed；模型版本、执行政策或策略指纹变化时，同周旧袖套状态也必须失效重建。
- 轮动选择在打分后先做点时可得流动性筛选：候选的10% ADV必须能承载与 Swing 生产账户一致的1万元参考组合完整目标权重；同一交易日两个袖套共享同一标的的10% ADV额度，禁止每个袖套各自重复使用容量。资金规模变化后必须重新校准参考规模，不得静默外推容量结论。
- 验收结果写入 `public/data_manifest_latest.json`。任一标的缺失、滞后、日期混合或未通过独立校验时，市场权限归零并发布现金目标；CI 中 `REQUIRE_FRESH_MARKET_DATA=true` 会进一步让作业失败。
- 流水线会在发布前同时验证 signal V4、rotation V2 JSON Schema、风险预算、目标权重和完整实盘成本契约；佣金、交易所费用、ETF 零过户费/印花税、价差、基础滑点、市场冲击、成交额参与率及整手必须与批准回测逐项一致。
- 双袖套与排名缓冲状态使用 rotation schema 2 持久化并可恢复，避免日常运行误重置持仓连续性。
- rotation 的模型、目标、风险预算、流动性和全部非时间字段未变化时复用原发布时间与完整 SHA，避免重复运行无意义地使 pre-trade shadow、分发 release 和 Swing 缓存失效；任一经济字段变化仍生成新发布身份并强制重新核验。
- `public/factor_health_latest.json` 使用注册表训练截止后的非重叠10日样本，按 T+1 入场并计入真实成本监控线上 IC；至少积累8个独立观察前只标记 WARMUP，不因少量重叠样本过度退役。
- 实时健康证据绑定注册表的表达式、Ridge系数、有效权重、训练截止和Bundle身份；旧健康文件与新因子组合错配时，推广链路直接标记 `LIVE_HEALTH_REGISTRY_MISMATCH` 并拒绝授权。
- 名义多因子组合必须至少有两个权重不低于5%的有效因子。有效因子不足或成熟线上证据出现显著负 IC 时，仅暂停自适应优先级叠加，基础V4和独立验收的轮动模型继续运行。

## 本地运行

Windows 新终端：

```powershell
python -m pip install -r requirements.txt
python main.py
```

如果当前终端尚未刷新 PATH，可使用：

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 main.py
```

主要输出：

- `public/index.html`
- `public/etf_signals_latest.json`
- `public/data_manifest_latest.json`
- `public/distribution_audit_latest.json`
- `public/joint_health_latest.json`
- `public/factor_health_latest.json`
- `public/market_env_latest.json`
- `artifacts/calibration/v4_calibration.json`
- `artifacts/calibration/v4_acceptance_report.json`
- `artifacts/calibration/adaptive_factor_registry.json`
- `artifacts/calibration/llm_factor_proposals.json`
- `artifacts/calibration/rotation_model.json`
- `public/etf_rotation_latest.json`
- `.runtime/data/`、`.runtime/state/`、`.runtime/logs/`

## 运行闭环进化

先刷新行情，再运行带 Walk-Forward 的进化和校准，最后重新生成信号：

```powershell
$env:FORCE_DOWNLOAD="true"
python main.py
python calibrate_v4.py --sample-step 5
$env:FORCE_DOWNLOAD="false"
python main.py
```

`adaptive_factor_registry.json` 会记录每个新因子的表达式、简短经济逻辑、训练/验证 IC、IR、衰减、换手、退休原因和替换关系。未通过样本外门槛时，日常策略自动回退到原 V4 优先级，不使用新因子。

### 受约束 LLM 因子提案

LLM 只生成候选表达式、经济假设和失效条件，不能直接改变权重或获得交易权限。表达式必须通过特征/运算符白名单、深度和复杂度校验，并与 GP、种子因子和透明原始因子一起接受三段式独立审批。

PowerShell 配置示例（不要把密钥提交到仓库）：

```powershell
$env:OPENAI_API_KEY="..."
$env:LLM_FACTOR_PROPOSALS_ENABLED="auto"
$env:OPENAI_MODEL="gpt-5.6"       # 可覆盖
$env:LLM_FACTOR_PROPOSAL_COUNT="6"
python calibrate_v4.py --sample-step 5 --reuse-rows-cache
```

- `auto`：存在 `OPENAI_API_KEY` 时调用，否则写入 `MISSING_API_KEY` 审计并继续 GP/ML 校准。
- `false`：明确禁用 LLM 提案。
- 合法提案缓存 45 天；篡改、越权特征、未知运算符和过度复杂表达式不会被复用。
- 网络/API失败只关闭本轮 LLM 挑战者，不会绕过或降低原有审批门槛。

完整历史行会缓存到 `.runtime/state/v4_calibration_rows.json`。仅当行情指纹与 `sample-step` 一致时才可复用：

```powershell
python calibrate_v4.py --sample-step 5 --reuse-rows-cache
```

轮动模型、事件信号与自适应因子叠加分别审批。轮动不依赖自适应因子注册表获得授权，只按自身成本后总超额、IR、回撤、年度稳定性及独立近年留出验收；事件入场或自适应叠加不通过时各自保持阻断。轮动失去自身审批时会生成 rotation V2 现金目标，通知执行端撤销旧仓风险。

## 测试与迭代

```powershell
python -m unittest discover -s tests -v
python -m py_compile etf_radar\factor_evolution.py etf_radar\calibration\pipeline.py
```

建议迭代流程：

1. 每日刷新行情并运行 `main.py`；双周运行 `calibrate_v4.py`，避免过度频繁挖掘。
2. 查看注册表中的 `retired_factors`、`new_replacements` 与 `ensemble_validation_metrics`。
3. 只有在多个 Walk-Forward 折中保持正超额收益、正信息比率且成本后回撤可接受时才提升风险预算。
4. 调整 GP 种群或代数后必须重新跑全量 Walk-Forward，不要依据全样本结果直接上线。

GitHub Actions 已配置日常分析与双周校准；双周任务会提交校准模型、验收报告、因子注册表及公共分析产物。

生产调度统一使用：

```powershell
python run_cycle.py
```

该入口先刷新并验证行情，再检查模型生成时间、标签训练滞后和 QFQ+RAW 联合指纹。达到 14 天预防性重校准门槛、训练滞后达到 53 天、行情指纹变化或核心产物缺失时，会在 `.runtime` 隔离目录执行完整 Walk-Forward；只有 V4、rotation、因子注册表、LLM审计与验收报告共享同一 `artifact_bundle_id` 并通过整包验收后才晋升。晋升失败会回滚全部已替换文件并保留安全生产产物。

本机闭环每次真正触发事务化校准时，会先在 `.runtime/llm-shadow/provider-health` 隔离刷新 Gemini 研究候选，默认请求6个且硬上限8个。只有主提供方、模型身份、候选结构和凭据泄露检查全部通过，候选才复制到 staging；网关失败时保留已验证缓存并继续 GP/ML 研究，绝不覆盖生产 Bundle。显式 `LLM_FACTOR_CACHE_SOURCE` 固定缓存或 `LLM_CYCLE_PROVIDER_REFRESH=false` 时不会联网刷新。

### 模型时间治理

- `generated_at` 表示产物实际生成时间，默认超过 21 天立即失效；缺失、格式错误或明显位于未来同样 fail-closed。
- `trained_until` 表示前瞻标签可用的训练截止日，独立允许最多 60 天滞后，避免把 20 个交易日前瞻标签的天然延迟误判为模型老化。
- V4 校准、rotation 模型和自适应因子注册表分别执行上述门槛。rotation 失效时发布全现金目标；因子注册表失效时只停用自适应叠加；V4 事件层失效时保持事件入场关闭。
- 日常分析继续安全发布被阻断的结果；双周校准任务负责更新模型。即使双周任务连续失败，21 天硬门槛也会撤销旧模型权限。

### 自适应因子组合治理

- 候选先通过训练期与严格 purge 后选择期的 IC、IR、近期稳定性、衰减和换手门槛。
- 组合允许存在可解释的相关性，但最高相关性不得达到 0.95、最小残余方差不得低于 10%，且组合选择期 IC 必须比最佳单因子至少增加 0.0005。
- 选择期拆成三个连续子阶段，至少两个阶段 IC 非负，最差阶段 IC 不得低于 -0.02。
- 获胜组合的 ML 权重只用训练集拟合并在选择前冻结，选择期和独立审批期都不参与权重重拟合。
- 因子搜索政策发生变化后，即使历史审批留出通过，也必须累计至少 13 个真正新增的校准日期才能获得生产授权，避免研究人员查看留出结果后立即上线。
- 熟化进度绑定候选表达式集合的 SHA-256 规格指纹；GP、LLM 或原始因子组合发生变化时自动重置 anchor，不能让新候选借用旧候选已累计的熟化日期。
- 每轮演化都会从 Registry 内实际表达式重新计算上一候选指纹；存储指纹缺失时兼容旧 Registry，存储值与表达式不一致时 fail-closed 并重置熟化。
- 已批准 Registry 在加载和应用前再次核对候选指纹；旧文件只在指纹缺失时以内存计算值兼容，显式不匹配则暂停因子叠加层。
- GP、透明原始因子、ML组合与LLM候选共同构成一次候选发现族。选择期先计算跨日期IC的保守显著性（双侧正态近似与方向符号检验取更差者），再对全部唯一候选表达式执行Benjamini–Hochberg校正，`q>10%`的候选不得进入组合搜索。ML组合搜索对所有实际评估组合单独执行同样的FDR校正；只有校正后组合才可进入独立审批留出。该政策版本为 `complementary-stability-fdr-seasoning-v7`，变更后重新触发13个新增日期的政策熟化期。

### Rotation 稳定超额验收

除累计超额、IR、年度胜率、绝对回撤、真实成本和容量外，rotation 还必须通过：

- 全期至少 104 个滚动12个月观察，滚动超额为正比例不低于 60%。
- 全期最差滚动12个月超额不低于 -25%，相对沪深300最大回撤不超过 30%，最长相对水下期不超过 260 个轮动周期。
- 最近三年独立留出至少 26 个滚动观察，滚动胜率不低于 60%，最差滚动超额不低于 -20%，相对回撤不超过 25%，最长水下期不超过 104 个周期。
- 任一稳定性门槛失败时撤销 alpha 权限并发布 rotation V2 全现金风险控制目标。
- 公共 rotation V2 目标必须携带 `acceptance_policy_version=rolling-excess-stability-v1`；该字段与模型版本、执行政策和策略规格指纹共同构成下游执行权威。
## LLM因子候选提供方

LLM只负责提出结构化研究候选，不能批准因子或直接改变交易。所有候选仍必须经过与GP候选相同的Purged Walk-Forward、选择期、独立审批期、成本、换手、容量和稳定性门槛。

云端Responses API：

```text
LLM_FACTOR_PROPOSALS_ENABLED=true
LLM_FACTOR_PROVIDER=openai_responses
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6
```

本地或OpenAI兼容Chat API：

```text
LLM_FACTOR_PROPOSALS_ENABLED=true
LLM_FACTOR_PROVIDER=local
LLM_LOCAL_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
LLM_LOCAL_MODEL=<已安装的本地模型名>
```

兼容端点也可只配置到 `/v1`，程序会规范化为 `/v1/chat/completions`。无密钥调用只允许 `localhost`、`127.0.0.1` 或 `::1`；远端兼容端点必须设置 `LLM_LOCAL_API_KEY`。若兼容网关忽略JSON Schema，程序最多追加一次严格JSON修复请求；供应商返回的函数表达式文本只通过无 `eval` 的白名单语法解析器转换为AST，未知符号、额外字段、错误元数或超复杂表达式仍拒绝。名称、经济逻辑、假设、预测周期和失败模式会再次执行本地类型与长度校验，其中 `failure_modes` 必须是真实的1至5项字符串数组，不能把单个字符串逐字符当作列表；离线缓存也会重新核验这些字段、提示版本、表达式签名和复杂度。缓存同时绑定提供方、模型和规范化端点指纹，切换任一项都会重新生成。没有可用提供方或服务返回限流/额度错误时保持fail-closed，不生成伪LLM候选。

如果当前进程没有配置活动提供方，但现有成功缓存仍在45天有效期内、提示版本一致、候选表达式和候选内提供方身份全部通过校验，程序会以 `CACHED_OFFLINE` 只读复用且绝不把成功缓存覆盖成 `MISSING_API_KEY`。事务化生产校准可通过 `LLM_FACTOR_CACHE_SOURCE=/path/llm_factor_proposals.json` 显式把只读研究缓存播种到临时staging；生产bundle仅在完整Purged Walk-Forward和发布前校验通过后才会替换。

## 真实执行证据治理

含真实订单但只有 `MODEL_ESTIMATE_ONLY` 的计划会进入待确认账本。7个自然日内允许补交券商完整成交、部分成交或未成交证据；逾期仍未确认时撤销rotation权限并发布现金目标。反馈指纹、成本权威或历史账本结构异常同样立即fail-closed，不能把被拒绝证据当成成本模型有效。

ETF端还会把上一份已批准rotation的 `model_version + execution_date + 策略指纹 + 目标权重` 登记为预期执行场次。只有执行日当天、实时行情可交易且组合状态允许写入的Swing反馈才能核销该场次，`NO_ORDERS` 也可以证明任务确实运行；若执行日结束后仍完全没有匹配反馈，则判定 `EXECUTION_SESSION_MISSED`，下一周期撤销rotation权限并发布现金目标。已核销场次保留指纹，后续重复审计不会重新登记造成误报。

同一执行日若在开盘前发布了新的已批准rotation或现金风险控制目标，新场次会按成本权威和执行日原子替换旧场次，旧指纹进入 `superseded_execution_keys` 而不是继续等待，避免模型换代后出现双重计划或次日误报漏跑。

在ETF-main与Swing-trading同级部署时，如果Swing本地 `public/execution_feedback_history.json` 或 `live_performance_latest.json` 已存在且未显式设置远程来源，ETF端优先读取本地证据，避免发布服务临时故障切断闭环；独立部署仍使用批准的远程来源，环境变量显式配置始终优先。

## 真实成本候选影子校准

当 `public/execution_cost_recalibration_latest.json` 达到 `READY_FOR_PURGED_WALK_FORWARD_RECALIBRATION` 后，可运行完整但不可晋升的影子校准：

```text
python calibrate_v4.py \
  --cost-model-candidate public/execution_cost_recalibration_latest.json \
  --shadow-output-dir .runtime/cost-shadow/<candidate-id> \
  --reuse-rows-cache
```

候选成本会用于重新生成T+1真实成本标签、Walk-Forward预测、阈值组合、rotation缓冲选择及组合模拟。所有产物强制写入指定隔离目录，生成的 `shadow_cost_validation_manifest.json` 固定包含 `shadow_only=true` 和 `promotion_allowed=false`。生产校准目录与rotation授权不会被覆盖；即使影子指标全部通过，也仍需正式变更执行政策并再次通过生产bundle验收。

## 实盘相对绩效治理

ETF端读取Swing发布的 `live_performance_latest.json`，只使用当前rotation模型连续产生的记录。少于20个交易日时保持WARMUP；20日相对收益低于-5%或60日低于-8%时触发完整事务化重校准。当前模型相对最大回撤达到-10%或策略最大回撤达到-15%时立即撤销rotation授权并发布现金目标。证据指纹、日期、基准、历史顺序或净值字段不一致，以及已发布证据超过7天未更新时，均fail-closed。

上一份已批准rotation的 `execution_date` 同时构成预期实盘绩效观察。执行日之前允许尚无净值证据；执行日结束后仍完全没有 `live_performance_latest.json`，则触发 `LIVE_PERFORMANCE_SESSION_MISSED`。若执行日净值存在但仍归属旧模型，则触发 `LIVE_PERFORMANCE_MODEL_SESSION_MISMATCH`。两种情况都会立即撤销rotation权限，防止执行反馈正常而回撤监控长期空转。
