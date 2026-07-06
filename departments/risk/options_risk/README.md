# Options Risk Department

This department owns the workflow, policy, and review artifacts for listed-options risk. Executable option logic should live in `src/oqp/options` so every dashboard and service imports the same implementation.

Current shared modules:

- `src/oqp/options/analytics.py`: scanner, volatility, Black-Scholes, simulation helpers.
- `src/oqp/options/book.py`: live option-book Greeks, moneyness, IV/HV, intrinsic/extrinsic, and model-quality diagnostics.
- `src/oqp/options/pricing.py`: option marks, Massive-primary pricing, Yahoo fallback.
- `src/oqp/options/payoff.py`: payoff curves, illustrative surfaces, Greeks display frames, risk checkpoints.
- `src/oqp/options/spread_recognition.py`: portfolio option leg parsing, spread recognition, underlying exposure.

Dashboard consumers:

- Ops Live Portfolio: option book, payoff hub, Greeks, TP/SL scaffolding.
- Ops Risk Control Room: aggregate option exposure and stress views.
- Ops Paper Trading: approved option proposals and paper execution monitoring.
- Discretionary Workbench: option scanner and idea evaluation.
- Research Dashboard: future option strategy research and candidate promotion.

Rule of thumb: if a function calculates, prices, parses, or simulates options, put it in `src/oqp/options`; if it documents how Oscar reviews or governs option risk, keep it here.

Phase 2 promotion map:

- `phase2_readaptation.md`: how the stochastic-modelling notebooks should be promoted into production modules and dashboard tabs.
