# ETF V4 信号生产端

ETF-main 下载行情、计算 schema V4 信号并生成 GitHub Pages 产物。交易规则保持
fail-closed：数据、校准、市场权限或风险条件不满足时仍生成报告，但不会产生可执行新仓。

## 本地运行

```bash
python -m pip install -r requirements.txt
python main.py
```

输出目录：

- `public/index.html`
- `public/etf_signals_latest.json`
- `public/market_env_latest.json`
- `public/schema/etf-signal-v4.json`
- `.runtime/data/`、`.runtime/state/`、`.runtime/logs/`

校准：

```bash
python calibrate_v4.py --sample-step 5
python main.py
```

校准产物保存在 `artifacts/calibration/`。测试命令：

```bash
python -m unittest discover -s tests -v
```

GitHub Pages 必须选择 **GitHub Actions** 作为发布源；工作流会部署 `public/` 并通过
`CNAME` 绑定 `etf.imlam.com`。
