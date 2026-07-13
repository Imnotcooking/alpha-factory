# Slow Momentum / Fast Reversal Volatility Router

## Research question

Does a slower cross-sectional long-short momentum sleeve create the continuation/reversal structure that
the volatility-quartile paper requires, and can the frozen Q1-Q3 momentum / Q4 reversal rule outperform
both sleeves and static allocation?

This experiment follows the pre-declared next step from the 15-minute matched-strategy study. It tests
one-, three-, and five-trade-day momentum formations without changing the volatility classifier.

## Causal specification

- Input: existing cached 15-minute adjusted-main panel. The 16-million-row minute parquet is not scanned.
- Trade-day mapping: timestamps at or after 21:00 belong to the next business day; Friday-night and
  post-midnight Saturday fragments map to Monday.
- Momentum: cross-sectional return over one, three, or five prior trade days, skipping the latest day.
- Reversal: negative of the current one-trade-day close-to-close return.
- Risk normalization: both scores divided by lagged 20-day contract volatility.
- Universe: top 40 contracts by lagged 20-day median notional turnover.
- Portfolio: long top quintile and short bottom quintile, +0.5/-0.5 and 1.0 gross.
- Signal timing: all scores and current-day market volatility are known after day t.
- Execution: selected portfolio is entered for day t+1 and earns next-trade-day open-to-close return.
- Cost: 0.5 tick per side plus InstrumentMaster fees at next open and close.
- Router: momentum in volatility Q1-Q3 and reversal in Q4; rolling 252-trade-day quartiles include day t.
- Controls: standalone sleeves, static 50/50, and 63-day lagged inverse-volatility allocation.
- Development: first routable return through 2025-09-30.
- Holdout: 2025-10-09 through 2026-07-09.

For all three horizons, active decision portfolios were verified at 1.0 gross with numerical net exposure
below `2.3e-8`. Every selected cross-section shared one next execution day.

## Holdout results

Annual returns and Sharpe ratios are after modeled round-trip costs.

| Momentum formation | Momentum | Reversal | Paper router | Static 50/50 |
|---|---:|---:|---:|---:|
| 1 session | 2.14% / 0.19 | 3.46% / 0.33 | **6.69% / 0.62** | 2.80% / 0.35 |
| 3 sessions | **17.37% / 1.64** | 3.99% / 0.36 | 17.12% / 1.56 | 10.68% / 1.43 |
| 5 sessions | **15.24% / 1.18** | 3.65% / 0.34 | 12.42% / 1.10 | 9.45% / 1.11 |

The one-session configuration is the first router in this research path to beat momentum, reversal,
static 50/50, and lagged inverse-volatility allocation in the holdout. It earns 16.09% gross and 6.69%
net annualized return, with 3.73 bps average daily cost.

The three-session result answers the sleeve-design question: a viable long-short trend portfolio exists.
It earns 26.87% gross and 17.37% net in holdout with 1.64 Sharpe. But the router does not improve it because
reversal is not superior in Q4. The five-session result gives the same conclusion.

## One-session routing mechanism

Holdout values are net basis points per execution day.

| Quartile | Days | Momentum | Reversal | Reversal minus momentum |
|---|---:|---:|---:|---:|
| Q1 | 30 | -0.22 | -5.81 | -5.60 |
| Q2 | 38 | 6.91 | -1.01 | -7.92 |
| Q3 | 51 | -3.58 | 1.01 | 4.59 |
| Q4 | 64 | 1.28 | 6.44 | 5.16 |

This is close to the paper's economic shape: lower-volatility Q1-Q2 favor momentum and Q4 favors
reversal. Q3 is the exception; the literal paper rule chooses momentum there and loses approximately
4.59 bps per day relative to reversal.

The router beats momentum entirely through its Q4 switch. It beats reversal because the Q1-Q2 momentum
advantage outweighs the Q3 mistake. Q4 was also much more frequent in holdout (`64/183` days) than in
development (`20/173`), making the switching opportunity economically larger after October 2025.

## Statistical strength

The holdout result is not statistically strong:

- router mean-return t-stat: `0.53`;
- router-minus-momentum t-stat: `0.31`;
- router-minus-reversal t-stat: `0.31`;
- router-minus-static-50/50 t-stat: `0.44`.

None of the one-session quartile spreads has an absolute t-stat above `0.76`. The 183-day holdout is too
short to distinguish the apparent routing gain from sampling noise.

## Stability check

The one-session router fails in development: annual net return is `-11.39%` with `-1.48` Sharpe. The
three-session router is `-5.90%` with `-0.87` Sharpe, and the five-session router is approximately flat.

Across the complete 357-day routable sample, the three-session router earns `5.92%` annualized with
`0.64` Sharpe, ahead of three-session momentum (`4.28%`, `0.48`), reversal (`-0.93%`, `-0.10`), and
static 50/50 (`1.68%`, `0.27`). That aggregate improvement is encouraging, but it is produced by two
subperiods with materially different unconditional strategy returns.

No tested horizon makes the router superior to both sleeves and static allocation in both development
and holdout. Therefore the result is a pilot finding, not a promotion or paper conclusion.

## Interpretation

Slowing momentum solves one major problem from the minute-level study: three- and five-session formation
have strong positive gross and net continuation in the current holdout. The remaining problem is routing.
At those genuinely slower horizons, high volatility does not make reversal better than momentum, so a
Q4 switch removes profitable trend exposure.

The one-session strategies are closer competitors. Their conditional payoff ranking changes across
volatility states, allowing the router to add value. But this relationship appears regime-specific and
is estimated imprecisely.

## Research decision

Do not tune quartile cutoffs or choose the one-session horizon solely from this holdout. Freeze these
results as the pilot table. The next defensible step for a paper is validation on additional history or
an untouched external market/sector panel. A publishable result should require:

1. the same one-session routing rule to beat both sleeves and static 50/50 out of sample;
2. Q1-Q3 momentum and Q4 reversal conditional spreads with stable signs;
3. positive paired router alpha or return spread with stronger statistical evidence;
4. persistence under integer lots, alternative fee assumptions, and sector-neutral construction.

## Artifacts

- Script: `scripts/research/analyze_cn_futures_slow_momentum_fast_reversal_router.py`
- Daily results: `runtime/artifacts/research/slow_momentum_fast_reversal_router/cn_futures_slow_momentum_fast_reversal_daily.csv`
- Quartile table: `runtime/artifacts/research/slow_momentum_fast_reversal_router/cn_futures_slow_momentum_fast_reversal_quartiles.csv`
- Summary: `runtime/artifacts/research/slow_momentum_fast_reversal_router/cn_futures_slow_momentum_fast_reversal_summary.json`
