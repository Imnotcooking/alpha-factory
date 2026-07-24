# Paper Trading Safety Layer

## Status

The current paper trading lane supports monitoring, safety review, dry-run
tickets, human approval, submission preflight, and a guarded IBKR paper broker
submission path. Broker submission is not enabled by default.

Implemented:

- read-only IBKR paper account monitoring
- separate paper SQLite ledger
- unified account ledger snapshots for paper NAV, holdings, returns, and events
- daily Discord paper NAV/P&L report
- proposal safety review and audit logging
- Discord-ready paper execution review messages
- research export and Options Desk draft proposal intake
- automated paper strategy runner that safety-reviews eligible proposals and
  writes dry-run order tickets
- guarded paper broker submitter for approved tickets when
  `ALLOW_PAPER_ORDER_SUBMIT=true`

Not implemented yet:

- fill reconciliation from submitted paper orders
- live-money promotion from paper results

## Active Contracts

Research or dashboard code should produce a `TradeProposal` artifact under:

```text
runtime/artifacts/trade_proposals/
```

The review command is:

```bash
PYTHONPATH=src:. python scripts/trading/review_paper_trade_proposal.py runtime/artifacts/trade_proposals --notify
```

The review writes audit rows to:

```text
runtime/db/paper_trading/paper_trading.db
```

Table:

```text
paper_execution_reviews
```

Paper account snapshots also write to the unified account ledger:

```text
runtime/db/accounts/account_ledger.db
```

The Paper Dashboard and daily Discord report use this ledger for paper NAV,
cash, daily P&L, returns, drawdown, holdings, exposure, and account events.

## Safety Switches

Defaults are conservative:

```bash
ALLOW_PAPER_TRADING=false
ALLOW_PAPER_ORDER_SUBMIT=false
ALLOW_LIVE_TRADING=false
PAPER_MAX_ORDER_NOTIONAL=10000
PAPER_MAX_DAILY_NOTIONAL=50000
PAPER_ALLOWED_SYMBOLS=
PAPER_ALLOWED_ASSET_CLASSES=equity,etf
PAPER_OPTIONS_ENABLED=false
PAPER_OPTION_ALLOWED_UNDERLYINGS=
PAPER_OPTION_ALLOWED_STRATEGIES=
PAPER_OPTION_MAX_CONTRACTS=1
PAPER_OPTION_MAX_PREMIUM=500
PAPER_OPTION_MAX_DEFINED_RISK=1000
PAPER_OPTION_MAX_SPREAD_WIDTH=10
```

Meaning:

- paper execution reviews are blocked until `ALLOW_PAPER_TRADING=true`
- paper broker submission is blocked until `ALLOW_PAPER_ORDER_SUBMIT=true`
- live trading must stay disabled
- proposals must use the IBKR paper profile
- proposals must be marked `paper_only=true`
- market orders are blocked by policy for now
- order intents need a reference/limit/stop price so notional can be calculated
- equities and ETFs are the only default asset classes
- option proposal drafts are visible in the queue but blocked by default
- options require `PAPER_OPTIONS_ENABLED=true`, an underlying allowlist, and a
  strategy allowlist before option-specific risk limits are even considered

`PAPER_ALLOWED_SYMBOLS` is optional. If set, every intent symbol must be present
in the comma-separated allowlist.

For option drafts, `PAPER_OPTION_ALLOWED_UNDERLYINGS` applies to the underlying
symbol, not the generated option contract symbol. `PAPER_OPTION_ALLOWED_STRATEGIES`
accepts normalized strategy ids such as `iron_condor`, `long_call`,
`bear_put_spread`, or `options_iron_condor`.

## Paper Order Submitter

Submission preflight, no broker order:

```bash
PYTHONPATH=src:. python scripts/trading/run_paper_order_submitter.py --record-events
```

Guarded broker submission for tickets already marked `approved_for_submit`:

```bash
PYTHONPATH=src:. python scripts/trading/run_paper_order_submitter.py --submit-approved --notify
```

This path requires all of the following:

- `ALLOW_PAPER_ORDER_SUBMIT=true`
- `ALLOW_LIVE_TRADING=false`
- broker profile `ibkr_paper_submit`
- IBKR paper Gateway connected locally
- ticket status `approved_for_submit`
- strategy registry status `running`
- broker profile is paper and write-enabled
- current implementation supports only equity/ETF limit orders

Use a dedicated submit client id:

```bash
IBKR_PAPER_SUBMIT_CLIENT_ID=121
```

## Next Phase

Before routine broker order placement is scheduled, define and test:

- whether market orders remain blocked or limited to tiny smoke tests
- first symbol allowlist
- per-strategy notional caps
- Discord pre-trade approval behavior
- kill switch procedure
- fill reconciliation process
- broker order id reconciliation
- paper-vs-dashboard P&L reconciliation
