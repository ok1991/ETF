# 灾难恢复手册

## 使用说明

本手册覆盖 ETF 生产闭环（`etf_radar`）与 Swing-trading 执行端在常见故障场景下的检测方法与恢复步骤。
配合 `python selfcheck.py` 自检脚本使用：先跑自检定位问题类别，再对照本手册对应章节处理。

## 场景一：远程分发数据被篡改或过期

- **症状**：`distribution_audit_latest.json` 校验失败，或 `joint_health` 报告 `remote_only_execution_allowed=false`。
- **检测**：运行 `python joint_health_cli.py`，查看 `status` 与 `reasons`。
- **恢复**：
  1. 确认 `artifacts/calibration/frozen_production_pin.json` 中的冻结包哈希未变。
  2. 重新执行 `etf_radar.pipeline.run()` 生成新的 `distribution_audit_latest.json` 与发布快照。
  3. 若远程仓库内容被覆盖，先回滚远程到最近一次已知良好的发布提交，再重新发布。

## 场景二：本地缓存缺失或损坏

- **症状**：管道启动时报 `FileNotFoundError` 或 `data_manifest` 校验失败。
- **检测**：查看 `.runtime/data/` 或 `PATHS.data` 指向目录是否存在必要的行情缓存文件。
- **恢复**：
  1. 从最近一次成功产物的 `data_manifest.json` 核对期望的文件清单。
  2. 重新执行数据采集步骤，补齐缺失文件后重跑管道。
  3. 若历史数据不可恢复，标记 `data_manifest.approved=false`，管道应自动进入安全兜底状态，禁止交易信号发布。

## 场景三：行情 API 故障或延迟报价

- **症状**：Swing 端 `source_diagnostics` 标记价格来源异常，或 `VALUATION_BLOCKED` 频繁出现。
- **检测**：查看 `decision_diagnostics.reasons` 中是否出现 `VALUATION_BLOCKED`。
- **恢复**：
  1. 优先使用备用行情源或延迟容忍窗口内的最近有效价。
  2. 若价格长时间不可用，`_rebalance()` 应保持只读观望，不生成新订单（现有 `VALUATION_BLOCKED` 分支已覆盖此逻辑）。
  3. 恢复后核对最新持仓市值与 `peak_capital`/`max_drawdown` 是否需要人工修正。

## 场景四：券商成交回报文件延迟

- **症状**：`pending_broker_confirmation_plan_id` 长时间未被清除，`execution_status` 持续为 `AWAITING_BROKER_CONFIRMATION`。
- **检测**：读取 `state.json` 中 `pending_broker_confirmation_plan_id` 字段，对比其写入日期。
- **恢复**：
  1. 先联系券商核实实际成交结果，不要臆断。
  2. 成交确认后，人工写入 `last_execution_satisfied_plan_id` 与实际持仓，再清空 `pending_broker_confirmation_plan_id`。
  3. 若长期无法核实，保持冻结状态，禁止新一轮再平衡覆盖未确认订单。

## 场景五：错过既定执行时段

- **症状**：当日 `run_date` 与 `execution_date` 不一致，触发 `EXECUTION_DATE_MISMATCH`。
- **检测**：查看 `decision_diagnostics.reasons` 中的 `EXECUTION_DATE_MISMATCH`。
- **恢复**：
  1. 不要手动修改系统日期强行匹配；等待下一个正确的 `execution_date` 窗口。
  2. 若确需补跑，先由人工确认轮动模型未过期，再显式指定 `run_date` 参数重跑。
  3. 补跑前先跑自检脚本，确认没有其他阻塞状态叠加。

## 场景六：状态文件损坏

- **症状**：`state.json` 或 `exposure_ratchet_history.json` 等 JSON 文件解析失败。
- **检测**：`python -c "import json; json.load(open(path))"` 抛出异常。
- **恢复**：
  1. 立即停止写入该文件的所有进程，避免二次损坏。
  2. 从最近一次 `atomic_json_save`/`_atomic_json_write` 生成的备份或历史提交中恢复。
  3. 若无法恢复，重建为最小合法结构（空持仓、`last_plan_id` 置空），并在恢复日志中记录本次重建。

## 场景七：Git 发布失败

- **症状**：`_publish_contract_assets()` 或发布脚本执行时 `git push` 报错。
- **检测**：查看发布脚本的退出码与 stderr 输出。
- **恢复**：
  1. 本地产物已生成时，`git push` 失败不影响本地状态；先排查网络/凭证问题。
  2. 重试推送；若远程有冲突，先 `git fetch` 确认远程未被意外修改，再决定 rebase 或强制发布。
  3. 若确认远程被篡改，参照场景一处理，不要用本地未审查的内容覆盖远程。

## 场景八：并发重复运行

- **症状**：同一交易日内管道或执行脚本被意外触发两次，可能导致重复下单或数据竞争。
- **检测**：`swing_trading` 已有文件锁机制（见 `test_locking.py`），检查锁文件是否存在及其持有者。
- **恢复**：
  1. 确认锁机制正常拒绝了并发进程（预期行为，无需干预）。
  2. 若锁未生效导致重复执行，立即核对 `buy_orders`/`sell_orders` 与实际持仓，人工去重。
  3. 排查触发重复运行的调度配置（如 Windows 计划任务是否重复注册）。

## 通用原则

- 任何恢复操作前先运行自检脚本，明确当前处于哪种 `overall_severity`（P0-P4）。
- 不确定时优先选择“安全空仓/保持现状”，不要凭猜测下单或覆盖生产数据。
- 所有人工修正都应记录（时间、操作人、修改内容），便于事后复盘。
- P0/P1 级别问题处理完成后，重新运行自检脚本确认已恢复至 P4，再解除人工冻结。

