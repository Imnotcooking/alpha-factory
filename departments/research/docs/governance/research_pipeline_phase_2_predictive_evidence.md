# Research Pipeline Phase 2: Predictive Evidence

## Position in the pipeline

Phase 0 freezes data, universe, timing, costs, parameter permissions, and the
success criterion. Phase 1 defines a pure factor and places it in a compatible
comparison cohort. Phase 2 asks one narrower question:

> Does the factor score at time `t` predict the return that starts only after
> the score could have been observed and executed?

This phase evaluates the factor before routers, portfolio sizing, leverage,
stops, and transaction costs. Those later layers cannot repair a factor that
has no stable predictive relationship.

## Causal target

For product `i` and decision date `t`, the forward return must begin after the
signal is available:

```text
signal available time < executable return start < executable return end
```

The evidence builder rejects a panel unless timestamp columns prove that
ordering or a frozen upstream protocol explicitly attests it. The artifact also
records the execution lag and return assumption. A same-close signal measured
against an unavailable same-close return is not valid Phase 2 evidence.

## Core statistics

For each date with enough valid products:

```text
IC_t     = PearsonCorr(factor_score_i,t, executable_forward_return_i,t+h)
RankIC_t = SpearmanCorr(factor_score_i,t, executable_forward_return_i,t+h)
```

The full, validation, and holdout samples report:

```text
Pearson ICIR = mean(IC_t) / sample_standard_deviation(IC_t)
Rank ICIR    = mean(RankIC_t) / sample_standard_deviation(RankIC_t)
```

These ICIR values are not annualized. They are not Sharpe ratios: the numerator
is predictive correlation rather than portfolio return.

Raw IC is never sign-flipped. If a factor declares `higher_is_bearish`, the
builder additionally records oriented IC as `-1 × raw IC`. The expected-sign
hit rate is the percentage of valid dates on which oriented IC is positive.
This preserves the observed correlation while still answering whether the
factor behaved in its declared direction.

## Evidence panel

Each `factor × market` bundle contains:

1. Mean Pearson IC and mean Rank IC.
2. Pearson ICIR and Rank ICIR.
3. Expected-sign hit rates.
4. Raw rolling and cumulative Pearson and Rank IC.
5. Separate validation and untouched-holdout summaries.
6. Product-level time-series IC distributions.
7. Date, product, row, signal, return, and joint coverage.
8. Year and product concentration diagnostics.

For a cross-sectional factor, the primary test is the correlation across
products on each date. Product-level IC is a limitation diagnostic: it asks
whether a result is broad or dominated by a few contracts. It is not a
substitute for the primary cross-sectional test.

Signal coverage counts non-missing scores. Active signal coverage separately
counts non-zero scores, so a rule that fills most of the panel with neutral
zeros cannot appear fully informative. Forward-return coverage and joint
coverage are reported independently.

## Validation and holdout

The split boundary is frozen before Phase 2. Validation may be used for
diagnosis and model development. Holdout is read only for the final frozen
specification. Repeatedly changing a factor after viewing holdout converts that
period into another validation set; it must not continue to be described as
untouched evidence.

The dashboard shows both samples beside the full result. A positive full-sample
IC with a flat or opposite-sign holdout is evidence of decay, concentration, or
selection risk, not a promotion result.

## Concentration

Year and product concentration use each member's absolute contribution to
oriented Rank IC mass:

```text
member mass = oriented mean Rank IC × valid evidence count
share       = abs(member mass) / sum(abs(all member masses))
```

The bundle reports the largest-year share, largest-product share, top-five
product share, and Herfindahl concentration. These are diagnostics rather than
universal pass/fail thresholds. A factor can have positive average IC while its
entire result comes from one year or a handful of contracts.

## Interpretation boundary

IC is a pre-cost predictive statistic. It does not determine turnover, feasible
contract quantities, margin use, slippage, drawdown, or net P&L. A factor can
have positive IC and still lose money after portfolio translation and costs.
Only later phases may establish tradability.

## Reproducible artifacts

The implementation lives in:

```text
src/oqp/research/predictive_evidence.py
scripts/research/build_predictive_evidence.py
```

Each bundle is written to:

```text
runtime/artifacts/research/predictive_evidence/<factor_id>/<market_vertical>/
```

It contains the frozen configuration and fingerprints, summary JSON, date-level
IC parquet, split and yearly summaries, product IC, and concentration tables.
Factor Drilldown reads these small artifacts directly and does not rerun the
factor or scan the underlying market dataset.

## First completed batch

The first Phase 2 batch uses the five normalized daily Chinese-futures
cross-sectional mean-reversion factors from Phase 1B. The frozen source contains
2,300 evaluation dates and 88 products, with pre-2023 data used as validation
and 2023 onward used as holdout.

`fac_012_Capitulation_Fade` and `fac_014_Range_Amplified_CLV` have stronger Rank
IC in holdout than validation. `fac_001_ST_Reversal`,
`fac_011_Regime_Filtered_Rev`, and `fac_042_Bollinger_Binary` decay materially;
`fac_011` turns negative in holdout. These observations do not overturn the
Phase 1B portfolio result: all five standardized translations were net negative
after costs. The predictive and tradability conclusions must remain separate.
