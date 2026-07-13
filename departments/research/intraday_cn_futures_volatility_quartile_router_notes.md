# CN Futures Intraday Volatility-Quartile Router

## Research question

Can the switching rule in Butt, Kolari, and Sadaqat, *Momentum, Market Volatility, and Reversal*
(SSRN 4342008), improve the allocation between our intraday trend and mean-reversion sleeves?

The source paper is a non-peer-reviewed preprint. Its rule is intentionally simple:

1. Estimate current-month market volatility from daily value-weighted equity-market returns.
2. Rank the current month against a rolling five-year volatility history.
3. Hold momentum next month when volatility is in Q1, Q2, or Q3.
4. Switch fully to short-term reversal next month when volatility is in Q4.

The paper does not optimize four strategy weights. The four quartiles describe the state; the actual
allocation is binary: momentum in Q1-Q3 and reversal in Q4.

## Intraday futures mapping

| Paper | CN futures adaptation |
|---|---|
| One month | One observed futures calendar day |
| Daily value-weighted market return | Equal-weight cross-contract one-minute market return |
| Monthly market volatility | Root-sum-square of one-minute market returns during the day |
| Rolling five years (60 months) | Rolling 252 observed days |
| Month-t state routes month t+1 | Day-t state routes the next observed day |
| Equity momentum | `fac_094` fixed-weight breadth breakout |
| Equity short-term reversal | `fac_093` multi-sleeve intraday mean reversion |

Only consecutive returns separated by at most five minutes enter the market return. A market minute
requires at least 20 contracts. The rolling quartile window includes day t because its volatility is
known after that day, but the state is shifted to the next observed day before it can affect positions.

The engine records the post-midnight part of Friday night trading as Saturday. Because this is not an
independent trading period, Friday's already-known route is carried across the Saturday fragment;
Friday's completed volatility first affects Monday.

## Recorded steps

1. Extracted the paper's methodology and equations from `ssrn-4342008.pdf`.
2. Built the broad minute market-return and daily realized-volatility series.
3. Formed rolling 252-observation Q1/Q2/Q3/Q4 states with a one-period routing lag.
4. Joined those states to the exact 216-day net returns from trend run `run_5ba5a760` and
   mean-reversion run `run_ee854132`.
5. Reproduced the paper's key table: next-period returns of both sleeves inside each volatility quartile.
6. Compared the literal paper switch with each sleeve, static 50/50, and inverse-volatility controls.
7. Added `fac_095` as the signal-level specification and a unit test for its lag discipline.
8. Stopped the combined minute-engine run after the 8 GB workstation exhausted memory. The compact
   artifact router is used for the result below; full regime reconstruction is now opt-in.

## Quartile evidence

All values are average already-costed return in basis points per routed day.

| Volatility state | Days | Trend | Mean reversion | Mean minus trend | Spread t-stat |
|---|---:|---:|---:|---:|---:|
| Q1 | 57 | 1.68 | 3.97 | 2.29 | 0.52 |
| Q2 | 40 | 1.68 | 6.20 | 4.52 | 0.67 |
| Q3 | 41 | -1.34 | 1.56 | 2.90 | 0.46 |
| Q4 | 78 | 2.18 | 2.84 | 0.66 | 0.14 |

The paper's required shape is absent. Mean reversion beats trend in all four quartiles, and its smallest
incremental advantage occurs in Q4. High volatility does not uniquely identify the relative payoff of
these two sleeves. None of the within-quartile sleeve-return spreads is statistically distinguishable
from zero in this short sample; Q4 is the weakest separation of all.

## Portfolio comparison

| Strategy | Annual return | Annual vol | Sharpe | Max drawdown | Compounded total return |
|---|---:|---:|---:|---:|---:|
| Trend | 3.25% | 2.61% | 1.25 | -2.33% | 2.79% |
| Mean reversion | 8.86% | 5.64% | 1.57 | -2.72% | 7.75% |
| Static 50/50 | 6.06% | 3.08% | 1.96 | -2.19% | 5.28% |
| Static inverse-vol, ex-post | 5.02% | 2.50% | 2.01 | -2.04% | 4.37% |
| Paper router | 3.85% | 3.75% | 1.03 | -1.74% | 3.29% |

The paper router used about 281x annual turnover, 1.02 bps average daily modeled cost, and 2.03x maximum
gross leverage in the selected-day execution proxy. It beat the selected same-horizon control, but it
failed the more relevant router benchmarks: both static blends and the standalone mean-reversion sleeve.

## Why the transfer failed

The paper's economic mechanism is a liquidity-supply mechanism. In stressed equity markets, momentum
traders demand immediacy while short-term reversal portfolios supply liquidity and are paid more for it.
Our components are materially different:

- `fac_094` is a long-only, cost-filtered intraday breakout, not a dollar-neutral winner-minus-loser portfolio.
- Much of `fac_093` comes from AU/AG relative value, whose payoff depends on pair convergence rather than
  broad market liquidity provision.
- A single futures volatility index mixes precious metals, industrial commodities, agriculture, rates,
  and equity-index futures. High aggregate volatility can represent a persistent macro trend rather than
  temporary price pressure.
- Daily routing is much faster than the paper's monthly relation and can discard the nearly zero correlation
  benefit that static simultaneous ownership provides.

The low sleeve correlation (`-0.020`) is the decisive fact. Static blending receives diversification every
day without needing to forecast which mechanism will pay. Hard switching gives that up, so the regime
signal needs strong conditional separation to win. The quartile table shows almost none in Q4.

## Decision

Do not promote the literal Q1-Q3 trend / Q4 mean-reversion router. Keep `fac_095` as a documented negative
result and methodology reference. The next defensible router test would use a market-state variable tied
more directly to futures liquidity stress, such as lagged cross-sectional spread/volume deterioration,
while retaining static 50/50 and lagged inverse-volatility allocation as mandatory controls.

## Artifacts

- Factor: `fac_095_Intraday_Volatility_Quartile_Router_Futures_CN.py`
- Reproduction script: `scripts/research/analyze_cn_futures_volatility_quartile_router.py`
- Cached daily series: `runtime/artifacts/research/volatility_quartile_router/cn_futures_volatility_quartile_router_daily.csv`
- Quartile table: `runtime/artifacts/research/volatility_quartile_router/cn_futures_volatility_quartile_sleeve_table.csv`
- Summary: `runtime/artifacts/research/volatility_quartile_router/cn_futures_volatility_quartile_router_summary.json`
