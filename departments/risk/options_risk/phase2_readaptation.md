# Phase 2 Options Readaptation Plan

This note maps the Phase 2 stochastic-modelling notebooks into the production
`src/oqp/options` department and the Ops Live Portfolio Options Hub.

## Notebook Map

| Phase 2 source | Production role | First dashboard surface |
| --- | --- | --- |
| `02_01` martingale pricing | No-arbitrage language, replication, model-risk framing | Quality flags: stale/missing marks, invalid intrinsic/extrinsic, quote bounds |
| `02_02` CRR lattice | American exercise and early-exercise diagnostics | Later: early exercise flag for deep ITM puts/calls around dividends |
| `02_03` BSM PDE | Vanilla baseline price and Greeks | Current: BSM fallback Greeks in live book diagnostics |
| `02_04` implied-vol root finding | IV extraction with quote validation | Current: solved IV where mark, spot, strike, and expiry exist |
| `02_05` Merton jumps | Jump/fat-tail model risk | Later: gap-risk stress and jump sensitivity for short-vol trades |
| `02_06` GBM/fat-tail Monte Carlo | Distributional scenario engine | Existing scanners; later portfolio-level P/L cone |
| `02_07` exotic Monte Carlo | Path-dependent product research | Keep research-only unless exotic positions enter the book |
| `02_08` delta hedging simulation | Discrete hedging error and gamma/theta intuition | Current: gamma 1% P/L, theta/day; later hedge rebalancing study |
| `02_09` vol surface construction | Clean chains, OTM selection, interpolation, static arb checks | Next: surface tab by underlying/expiry |
| `02_10` finite-difference PDE | American and boundary-condition pricing | Later: American option fallback where early exercise matters |
| `02_11` Heston calibration | Stochastic-vol model-risk layer | Offline calibration report, not live gating yet |
| `02_12` SABR calibration | Smile parameterization and term structure | Offline surface feature extraction, not live trading signal yet |
| `02_13` MLMC | Compute-efficient path pricing | Research/performance upgrade for exotic or path-dependent analytics |
| `02_14` Dupire local vol | Arbitrage-aware local-vol surface | Later: local-vol repricing diagnostic after chain store exists |
| `02_15` MC Greeks | Greek estimator selection | Later: MC Greeks for discontinuous/path-dependent payoffs |

## Production Rewire

1. Data contract first:
   option rows need underlying, expiry, right, strike, multiplier, signed
   quantity, mark, entry price, quote source, and timestamp. Chain rows need bid,
   ask, mid, volume/open interest, IV, delta, gamma, theta, vega, and quote time.

2. Vanilla model spine:
   `src/oqp/options/book.py` owns live book diagnostics. It reports moneyness,
   intrinsic/extrinsic value, IV/HV premium, scaled Greeks, and model-quality
   flags. This is the correct bridge from notebooks `02_03`, `02_04`, and
   `02_08` into the live dashboard.

3. Surface layer:
   promote notebook `02_09` into a future `src/oqp/options/surface.py` with
   quote cleaning, OTM quote selection, total-variance calendar repair,
   interpolation, and static-arbitrage checks.

4. Pricing engines:
   keep BSM as the live default. Add CRR/finite-difference for American exercise
   only when position metadata and dividends justify it. Keep Heston, SABR, and
   Dupire as offline calibration diagnostics until the chain store is stable.

5. Trader workflow:
   the Options Hub should move in this order: Book Audit -> Vol & Model Audit ->
   Payoff Lab -> Greeks & Risk -> Surface/Skew -> Scenario/Hedge Plan.

## Desk Standards

- No mark should be trusted without source, timestamp, and bid/ask context.
- No IV should be shown without the pricing convention used to infer it.
- Greeks must be scaled to portfolio units; raw per-contract Greeks are secondary.
- Volatility trades should show IV versus realized/forecast vol before strategy
  scoring.
- Any model beyond BSM should show calibration error and parameter stability, not
  just a prettier price.
