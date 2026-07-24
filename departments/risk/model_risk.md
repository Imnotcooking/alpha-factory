# Risk Model Governance

Last reviewed: 2026-07-17

Risk models support decisions; they do not turn uncertain estimates into facts.
Every displayed metric must state its model, sample, data treatment, and known
failure modes.

## Current Model Inventory

| Model/view | Current use | Principal limitation |
| --- | --- | --- |
| Historical NAV VaR/CVaR | Legacy portfolio summary | Small/mixed NAV history, cash-flow contamination, no option repricing, no conditional regimes. |
| PCA risk breadth | Research market-structure diagnostics | Window, scaling, imputation, and asset-universe sensitivity; component labels are heuristic. |
| Directional/concentration breadth | Research-window context | Equal-weight fallback can differ materially from true market-cap or portfolio weights. |
| Realized volatility | Risk and imputation studies | Sampling frequency, microstructure noise, gaps, and annualization choice. |
| Brownian Bridge imputation | Optional risk reconstruction | Synthetic path, not exchange truth; output depends on seed and variance estimate. |
| BSM price and Greeks | Vanilla option baseline | European/lognormal assumptions, constant volatility, simplified rates/dividends, no early exercise. |
| Option margin helper | Research/backtest reserve estimate | Not broker SPAN/house margin and not a guarantee of available buying power. |
| Terminal Monte Carlo | Option candidate exploration | Calibration, distribution, path, liquidity, and tail assumptions dominate the output. |

## Validation Requirements

Before a model supports a warning or hard limit:

1. Define input schema, unit, timestamp, and missing-data behavior.
2. Test analytical identities and simple hand-calculated fixtures.
3. Compare against an independent implementation where practical.
4. Test sensitivity to window, universe, seed, frequency, and imputation.
5. Measure coverage and exclude unsupported instruments visibly.
6. Record model version and parameter provenance in every result.
7. Establish review frequency and retirement criteria.

## Confidence Language

Component stability, label confidence, data coverage, and calibration fit are not
statistical confidence intervals unless a valid inferential procedure explicitly
constructs one. Dashboard labels must use the precise term represented by the
calculation.

## Synthetic Data

Forward-filled accounting views and Brownian Bridge risk views must retain
freshness and synthetic flags. Synthetic observations cannot update alpha or
execution signals. Risk outputs must expose their synthetic share and provide a
raw/observed-data comparison where material.

## Advanced Option Models

CRR, finite difference, Heston, SABR, local volatility, and Monte Carlo Greeks
remain diagnostic until chain quality, calibration stability, and comparison
evidence are sufficient. A more complex model is not promoted merely because it
produces a smoother surface or a different price.
