# Research Backtests

For a strategy that combines several compatible factor scores, use the shared
factor-portfolio entry point rather than creating another execution engine:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/research/run_factor_portfolio.py \
  --config departments/research/strategies/examples/cn_futures_value_momentum.yaml \
  --data-file runtime/data/futures_cn/daily/YOUR_LONG_FORMAT_FILE.parquet \
  --build-only
```

The command composes factor scores first and then reuses the same execution
mode and `AlphaEvaluator` used by the single-factor backtest path.

This directory contains executable factor, sleeve and router tests. It is not a paper or
notebook project area.

- Canonical factor logic stays under `departments/research/factors/`.
- Reusable engine infrastructure stays under `src/oqp/research/backtesting/`.
- Each runner here loads a fixed YAML specification and writes generated outputs under
  `runtime/artifacts/research/`.
- Dashboard labels may be descriptive, while artifact strategy ids remain stable for
  backward compatibility.

Run the Chinese-futures product-state and EMA comparison with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/research/backtests/run_cn_futures_product_state_ema.py
```
