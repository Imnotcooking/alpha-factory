# Phase 0.4: Transaction-Cost Policy

## Position In The Pipeline

This phase runs after dataset identity, liquidity eligibility, and temporal
policy are frozen, but before net factor evaluation, strategy comparison,
routing, optimization, or promotion.

1. Phase 0.1: dataset fingerprint.
2. Phase 0.1B: immutable API materialization.
3. Phase 0.2: asset-aware liquidity eligibility.
4. Phase 0.3: signal decision frequency and holding policy.
5. **Phase 0.4: transaction-cost profile and claim-readiness gate.**
6. Phase 0.5: declarative factor parameters and optimization governance.
7. Later phases: factor evaluation, strategy construction, routing,
   optimization, and promotion.

## Separation Of Responsibilities

The Instrument Master describes the physical instrument: multiplier, tick
size, margin convention, exchange, and product taxonomy.

The central transaction-cost registry describes the route used to trade it:
broker commission, exchange and regulatory charges, clearing fees, and
slippage assumptions. It lives at:

```text
config/execution/transaction_costs.yaml
```

Every profile is versioned, market-specific, and fingerprinted. A run records
the profile ID, SHA-256 fingerprint, registry date, completeness, use case, and
readiness state. Changing a cost profile therefore creates a different
research trial signature.

The Research Home assumptions tab renders only the selected run's market and
cost profile. New runs show the profile frozen with that run. Historical runs
that predate Phase 0.4 show the current default for their own market with an
explicit warning that the profile was not frozen and cannot validate the old
net result.

## Three Claim Levels

| Use case | Meaning | Placeholder allowed? |
|---|---|---|
| `exploratory_gross` | Signal or gross-return diagnosis with no net claim. | Yes |
| `research_net` | A costed research result presented as net performance. | Only an engine-ready verified profile |
| `production` | A production or paper-trading readiness claim. | No; profile must explicitly pass production readiness |

A placeholder does not mean that costs are zero. It means costs are unknown.
The gross-only path must be requested explicitly and the result remains
labelled gross-only.

If a run requests a claim level that its profile cannot support, the terminal
prints a `TRANSACTION COST READINESS BLOCK` with the market, profile, reason,
registry path, missing implementation work, and the explicitly gross-only
fallback. The backtest then stops before producing a misleading net result.

## Current Readiness

| Market | Profile | Current state |
|---|---|---|
| Chinese futures | `cn_futures_broker_v1` | Costed and production-ready under the current internal schedule; 0.5 tick slippage per side. |
| US equities | `ibkr_pro_fixed_us_equity_v1` | Official IBKR Pro fixed and regulatory schedule encoded; exact estimator available; shared backtest adapter pending. |
| US options | `ibkr_pro_us_options_v1` | Premium tiers, order minimum, OCC, CAT, SEC, and FINRA components encoded; venue and ORF route model pending. |
| US futures | `us_futures_broker_pending` | Deliberate placeholder; no physical contract registry and no net execution claim. |
| Chinese equities/options | Pending profiles | Awaiting the Chinese broker schedule. |

IBKR Lite is not used. The initial US profiles use IBKR Pro.

## Why Verified Does Not Always Mean Production-Ready

`verified` means the encoded fee schedule came from an identified source. It
does not imply that every execution cost is present or that the shared engine
implements the schedule exactly.

For US equities, the fixed broker schedule has a per-order minimum and a
trade-value cap. The existing vectorized engine only accepts simple per-unit or
notional rates, so it cannot yet reproduce those order boundaries exactly.

For US options, the broker rate depends on option premium, while exchange and
Options Regulatory Fees depend on the route. Quote data is also required for a
half-spread slippage estimate. Those missing execution inputs block a net or
production claim even though the published broker schedule itself is verified.

## US Futures Safety Rule

US futures are intentionally left as a placeholder. `InstrumentMaster` now
raises an explicit unsupported-market error instead of falling through to
Chinese-futures specifications. This prevents an ES backtest from silently
receiving a Chinese multiplier, tick, margin, or fee schedule.

## Example

```python
from oqp.execution.transaction_costs import attach_transaction_cost_policy

frame = attach_transaction_cost_policy(
    frame,
    market_vertical="FUTURES_CN",
    use_case="research_net",
)
```

An exploratory gross diagnostic for an unfinished market must say so:

```python
frame = attach_transaction_cost_policy(
    frame,
    market_vertical="FUTURES_US",
    use_case="exploratory_gross",
)
```

That second declaration does not authorize an execution backtest because the
US futures physical instrument registry is also pending.
