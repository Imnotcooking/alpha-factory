# Listed Options Risk Policy

This folder owns workflow, model-risk standards, and review policy for listed
options. Executable option logic lives in `src/oqp/options` so dashboards,
research, backtests, and services import one implementation. Portfolio limits
and breach authority remain in the Risk root catalog and appetite policy.

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
- Ops Paper Trading: approved option proposals and paper execution monitoring.
- Discretionary Workbench: option scanner and idea evaluation.
- Research Dashboard: option strategy research, event-driven backtests, and candidate promotion.

Rule of thumb: if a function calculates, prices, parses, or simulates options,
put it in `src/oqp/options`; if it governs model use, limit interpretation, or
risk review, keep it here.

Current limitations that must remain visible:

- margin helpers are conservative research approximations, not broker margin;
- BSM fallback is a vanilla baseline and does not model early exercise;
- the event-driven backtester does not yet provide full assignment/exercise,
  intraday NBBO, or broker-grade liquidation behavior;
- portfolio VaR/scenarios do not yet reprice the complete option book;
- missing spot, IV, expiry, multiplier, or quote context is a model gap, not zero
  risk.

The standalone Risk Control Room remains retired. Operational option risk belongs
inside the unified Live Portfolio workflow backed by package-owned calculations.

Advanced model promotion map:

- `model_promotion.md`: how stochastic-modelling notebooks should be promoted
  into reviewed production modules and dashboard views.
