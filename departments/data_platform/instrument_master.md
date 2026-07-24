# Instrument Master Contract

Last reviewed: 2026-07-17

The authoritative executable Instrument Master is
`src/oqp/data/instruments.py`. This document defines ownership and change
rules; it does not duplicate the Python dictionaries.

## Responsibilities

The Instrument Master owns static or slowly changing instrument attributes:

- canonical ticker and accepted symbol aliases;
- asset class, market, exchange, and currency;
- industry or sector classification;
- multiplier, tick size, and settlement conventions;
- fee and margin metadata where the contract supports them;
- option contract multiplier and instrument-family defaults.

Dynamic prices, volumes, open interest, Greeks, positions, and model features
do not belong in the Instrument Master.

## Classification Rule

Dashboards and analytics must resolve industry or sector through
`InstrumentMaster` or its registry classes. They must not maintain private
page-level dictionaries. An unresolved classification must remain `Unknown`;
the UI may report it, but must not guess from ticker text.

For Chinese futures, `ChineseFuturesRegistry` is the current source for the
contract dictionary and sector labels. Ticker normalization is exchange-aware,
so changes must preserve case-sensitive product conventions and existing
aliases.

## Change Procedure

1. Update the relevant registry in `src/oqp/data/instruments.py`.
2. Preserve existing canonical tickers and aliases unless a migration is
   explicitly documented.
3. Add or update assertions in `tests/data/test_instrument_taxonomy.py`.
4. Verify taxonomy and dashboard consumers continue to resolve the profile.
5. Document source and effective date when metadata comes from a vendor or
   exchange rulebook.

## Quality Gates

- Canonical tickers must be unique within an asset class.
- Multipliers and tick sizes must be positive.
- Exchange and currency must be explicit for tradable instruments.
- Sector labels must use the established taxonomy; new labels require review.
- Options metadata must not infer expiry, strike, or right from an ambiguous
  symbol without a validated parser.
