# 操作纪律

## 临时纪律（保护规则落地后立即生效）

1. **暂停无保护重校准**
   - 在冻结钉 `active=true` 且未正式开挑战前，不对生产目录直接 force 晋升
   - 禁止默认 `python calibrate_v4.py`（会直写生产路径）
2. **需要重跑时**
   - 使用隔离输出目录
   - 或允许 `run_cycle --force-calibration` 仅生成 staging，但确认状态为 `CALIBRATION_STAGED_NOT_PROMOTED` 且生产 hash 未变
3. **生产只监控，不重钉**
   - 冻结包 `15be8c14...` 负责可交易授权
   - 研究包只对比

## 日常命令边界

| 命令 | 是否可碰生产 | 说明 |
|------|--------------|------|
| `python run_cycle.py` | 读取生产；可能触发校准 | 受保护门禁约束 |
| `python run_cycle.py --force-calibration` | 仅当候选过关且挑战打开才晋升 | 未过关应停在隔离 |
| `python calibrate_v4.py` | **默认危险** | 必须显式改输出到研究目录 |
| 影子成本校准 | 否 | `promotion_allowed=false` |

## 推荐研究命令形态

```bash
# 显式把输出指到研究隔离目录（示例）
python calibrate_v4.py \
  --calibration-out artifacts/calibration_research_YYYYMMDD/v4_calibration.json
```

（具体 CLI 参数以 `calibrate_v4.py -h` 为准；若缺少隔离参数，优先用 `run_cycle` staging 或复制脚本输出，禁止手改生产。）

## 提交纪律

- 研究包、事故档案、治理文档可以进 main
- **不得**把未过关 `artifacts/calibration/*` 提交为生产权威
- 若 CI 产生未过关校准，应保留为研究产物或失败状态，而不是覆盖冻结包
