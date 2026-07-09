# Options Risk Department

This department owns the workflow, policy, and review artifacts for listed-options risk. Executable option logic should live in `src/oqp/options` so every dashboard and service imports the same implementation.

Current shared modules:

- `src/oqp/options/analytics.py`: scanner, volatility, Black-Scholes, simulation helpers.
- `src/oqp/options/book.py`: live option-book Greeks, moneyness, IV/HV, intrinsic/extrinsic, and model-quality diagnostics.
- `src/oqp/options/contracts.py`: vendor-neutral option contract, quote, right, exercise, and settlement contracts.
- `src/oqp/options/chain_loader.py`: Massive/Yahoo/CN static-parquet chain normalization and lookup.
- `src/oqp/options/lifecycle.py`: DTE, expiry, intrinsic value, and settlement helpers.
- `src/oqp/options/liquidity.py`: bid/ask-aware fills, spread filters, and settlement-proxy labels.
- `src/oqp/options/margin.py`: premium cashflow and conservative option margin helpers.
- `src/oqp/options/greeks.py`: BSM Greek filling and scaled exposure helpers.
- `src/oqp/options/backtesting/`: daily event-driven option backtesting engine, ledger, models, and research adapters.
- `src/oqp/options/pricing.py`: option marks, Massive-primary pricing, Yahoo fallback.
- `src/oqp/options/payoff.py`: payoff curves, illustrative surfaces, Greeks display frames, risk checkpoints.
- `src/oqp/options/spread_recognition.py`: portfolio option leg parsing, spread recognition, underlying exposure.

Dashboard consumers:

- Ops Live Portfolio: option book, payoff hub, Greeks, TP/SL scaffolding.
- Ops Risk Control Room: aggregate option exposure and stress views.
- Ops Paper Trading: approved option proposals and paper execution monitoring.
- Discretionary Workbench: option scanner and idea evaluation.
- Research Dashboard: option strategy research, event-driven backtests, and candidate promotion.

Rule of thumb: if a function calculates, prices, parses, or simulates options, put it in `src/oqp/options`; if it documents how Oscar reviews or governs option risk, keep it here.

Phase 2 promotion map:

- `phase2_readaptation.md`: how the stochastic-modelling notebooks should be promoted into production modules and dashboard tabs.
