# Research Pipeline Phase 4: Standalone Sleeve Test

## Position in the pipeline

Phase 2 tests whether a factor predicts a causally available future return.
Phase 3 freezes one explicit translation from score to executable positions.
Phase 4 asks:

> Does that frozen sleeve have credible economics on its own, before a router
> is allowed to choose when it is visible?

The answer must be established without regime selection, parameter optimisation,
or retrospective exclusions. A router cannot be used to rescue a component that
has no standalone economic case.

## Frozen decision rule

The predeclared success profile is
`sleeve_daily_standalone_net_value_v1`. Validation is the decision sample;
holdout is opened only as a final confirmation.

| Test | Frozen rule |
|---|---:|
| Primary metric | Validation net Sharpe greater than cash Sharpe of 0 |
| Net economics | Validation annualised net return greater than 0 |
| Cost coverage | Validation gross edge / estimated cost at least 1.0x |
| Minimum evidence | At least 252 active validation days |
| Holdout confirmation | Net Sharpe and annualised net return both greater than 0 |

All validation conditions must pass before the holdout result can make a sleeve
eligible for router research. Holdout can reject an otherwise valid sleeve, but
it cannot rescue a validation failure. Passing Phase 4 permits router research;
it does not approve production trading.

An **active day** is a date with at least one non-zero executed contract. This
distinction matters for sparse event factors: 1,457 calendar observations do not
constitute 1,457 independent trading opportunities when only 177 dates carry a
position.

## Economic measurements

The Phase 4 engine consumes the saved Phase 3 positions and independently
reconciles daily gross return, fee return, slippage return, net return, and
turnover back to those positions within a tolerance of `1e-12`.

For daily net returns \(r_t^{net}\):

```text
annualised net mean = 252 * mean(r_net)
annualised volatility = sqrt(252) * std(r_net)
net Sharpe = annualised net mean / annualised volatility
maximum drawdown = min(cumulative wealth / running wealth peak - 1)
break-even cost multiple = annualised gross mean / annualised estimated cost
```

The report separates exchange fees from 0.5-tick-per-side slippage. Turnover is
annualised traded notional divided by the frozen CNY 10 million capital base.
No cost is inferred from net return as a residual.

Two hit rates answer different questions:

- **Active-day hit rate:** percentage of active dates with positive sleeve return.
- **Position hit rate:** percentage of non-zero contract positions with positive
  contribution.

Neither is a replacement for return magnitude. A hit rate below 50% can still
work with positively skewed payoffs, while a hit rate above 50% can lose money
if losses are larger than gains.

## Contribution and concentration

Product, sector, and calendar-year contributions are built from the same net
position contributions used by the P&L. The report includes gross contribution,
fees, slippage, net contribution, turnover, active positions, and each member's
share of absolute net contribution.

Position concentration is measured from absolute gross weights:

```text
position HHI = sum(product gross share squared)
effective products = 1 / position HHI
largest position share = max(product gross share)
```

The same construction is applied to sectors. These diagnostics reveal whether a
headline result is carried by one contract, one sector, or one year. They are
not pass/fail gates in this version because numerical limits were not declared
before observing the results.

## Extreme-event protocol

For each date, the market shock score is the cross-sectional median absolute
forward return among available products:

```text
market shock score(t) = median across products of abs(product return(t))
```

At least 10 products are required. The event threshold is the 99th percentile
of that score in validation only, then frozen and applied unchanged to holdout.
For this cohort the threshold is 1.6581%, producing 15 validation events and 5
holdout events. The event study reports mean gross and net sleeve returns from
five sessions before through five sessions after each event, plus summaries for
pre-event, event, post-event, and non-event observations.

This is a diagnostic of event dependence, not a new router. Removing extreme
dates after seeing the result would be a retrospective rule and is prohibited.

## First completed batch

The common default sleeve was tested for the five Phase 1B daily Chinese-futures
mean-reversion factors. No optimisation or routing was used.

| Factor | Validation net mean | Validation net Sharpe | Gross edge / cost | Active validation days | Holdout net Sharpe | Phase 4 status |
|---|---:|---:|---:|---:|---:|---|
| `fac_001_ST_Reversal` | -9.34% | -1.37 | 0.22x | 1,457 | -1.94 | Blocked: economics fail |
| `fac_011_Regime_Filtered_Rev` | -8.51% | -1.43 | 0.26x | 1,457 | -1.76 | Blocked: economics fail |
| `fac_012_Capitulation_Fade` | -10.01% | -1.66 | 0.15x | 1,457 | -0.73 | Blocked: economics fail |
| `fac_014_Range_Amplified_CLV` | -10.60% | -2.05 | 0.03x | 1,457 | -0.57 | Blocked: economics fail |
| `fac_042_Bollinger_Binary` | 0.31% | 0.30 | 1.84x | 177 | -0.22 | Blocked: too few active days; holdout negative |

`fac_001`, `fac_011`, `fac_012`, and `fac_014` trade broadly but fail all three
economic tests: their validation net returns and Sharpes are negative and their
gross edge does not cover estimated cost. Their effective product counts are
approximately 20.6 to 21.8, so the result is not explained by a single-contract
bet. Excluding the validation-defined extreme events also leaves their
full-sample annualised net returns negative. The failure is broad implementation
economics, not a small set of shock dates.

`fac_042` is different. Its sparse sleeve has positive validation net economics
and covers estimated costs, but it activates on only 177 validation dates, below
the frozen 252-day evidence requirement. Its holdout net mean is -0.27% and its
holdout Sharpe is -0.22. It remains an unconfirmed event hypothesis rather than
a standalone sleeve suitable for routing.

## Decision

No sleeve in this batch is eligible for router research. That is the useful
Phase 4 result: the four continuous implementations need a newly hypothesised,
slower or longer-horizon sleeve specification, while `fac_042` needs more
independent events and future confirmation. A volatility router is not tested
against this cohort yet because doing so would let selective exposure disguise
unresolved component weakness.

Phase 5 subsequently describes these same frozen sleeves under transparent
observable conditions. That exploratory analysis does not change any Phase 4
eligibility decision and does not execute a router.

## Reproducible artifacts

The implementation lives in:

```text
src/oqp/research/sleeves/standalone.py
config/research/success_criteria.yaml
scripts/research/build_standalone_sleeve_tests.py
tests/research/test_standalone_sleeve_test.py
```

Each evidence bundle is written to:

```text
runtime/artifacts/research/standalone_sleeve_tests/
  <factor_id>/<market_vertical>/<sleeve_id>/
```

Factor Drilldown reads these saved artifacts and does not rerun the factor or
scan the raw dataset.
