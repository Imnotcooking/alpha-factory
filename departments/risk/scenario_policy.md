# Portfolio Scenario Policy

Last reviewed: 2026-07-17

Scenarios answer how the current portfolio changes under deterministic market
shocks. They are not forecasts and must remain reproducible from the same
account snapshot, market-data cut, scenario version, and pricing configuration.

## Required Inputs

- reconciled canonical account positions and NAV;
- native and reporting currencies with approved FX marks;
- instrument-master asset class, sector, multiplier, and underlying identity;
- fresh spot/futures marks and explicit stale-mark flags;
- option contract, expiry, right, strike, mark, IV, and model provenance;
- scenario id, version, parameters, and evaluation timestamp.

## Scenario Families

| Family | Minimum implementation |
| --- | --- |
| Broad market | Parallel and asymmetric spot/index shocks. |
| Sector/industry | Shock selected instrument-master groups while preserving other marks. |
| Futures/commodity | Underlying shock with contract multipliers and limit-move awareness. |
| FX | Native-currency translation shocks by reporting pair. |
| Rates | Bond/discount-rate sensitivity and option repricing where supported. |
| Volatility | Parallel and term/skew IV shocks with option repricing. |
| Combined | Spot, IV, FX, and liquidity shocks applied in a fixed order. |
| Liquidity | Spread widening, stale marks, and conservative liquidation prices. |

No numerical shock is canonical until approved. Dashboard sliders may be used
for exploration but do not become policy scenarios automatically.

## Valuation Rules

- Linear cash instruments may use quantity, multiplier, and shocked mark.
- Futures require contract multiplier and settlement convention.
- Options must be repriced under shocked spot, IV, rates, and remaining time;
  premium-value or delta-only approximations must be labelled as approximations.
- Path-dependent and American exercise features require an appropriate model or
  an explicit unsupported result.
- Missing model inputs produce an excluded/unavailable result, never zero loss.

## Output Contract

Every result should report base value, stressed value, PnL, PnL/NAV, positions
included and excluded, model sources, stale-input count, and largest loss
contributors. Portfolio totals are invalid when excluded value is material and
must expose that fact rather than silently summing partial coverage.

## Promotion Gate

A scenario becomes operational only after unit tests, deterministic fixtures,
option repricing checks, sensitivity sanity checks, and comparison against an
independent calculation. Persistent scenario storage and dashboard wiring follow
the reviewed package contract, not the other way around.
