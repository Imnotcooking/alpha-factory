# Paper Trading Safety Layer

## Status

The current paper trading lane is monitoring plus safety review only. It does
not submit broker orders.

Implemented:

- read-only IBKR paper account monitoring
- separate paper SQLite ledger
- daily Discord paper NAV/P&L report
- proposal safety review and audit logging
- Discord-ready paper execution review messages
- research export and Options Desk draft proposal intake

Not implemented yet:

- actual broker order placement
- fill reconciliation from submitted paper orders
- strategy scheduler that creates live paper proposals

## Active Contracts

Research or dashboard code should produce a `TradeProposal` artifact under:

```text
runtime/artifacts/trade_proposals/
```

The review command is:

```bash
PYTHONPATH=src:. python scripts/review_paper_trade_proposal.py runtime/artifacts/trade_proposals --notify
```

The review writes audit rows to:

```text
data/paper_trading/paper_trading.db
```

Table:

```text
paper_execution_reviews
```

## Safety Switches

Defaults are conservative:

```bash
ALLOW_PAPER_TRADING=false
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

## Next Phase

Before broker order placement is added, define:

- whether market orders remain blocked or limited to tiny smoke tests
- first symbol allowlist
- per-strategy notional caps
- Discord pre-trade approval behavior
- kill switch procedure
- fill reconciliation process
