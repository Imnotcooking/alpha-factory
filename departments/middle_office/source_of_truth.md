# Account Source Of Truth

Last reviewed: 2026-07-17

This policy establishes evidence precedence. It does not grant execution
authority and does not allow reconciliation to overwrite a source.

## Precedence

1. A broker snapshot is authoritative for positions, cash, margin, and NAV held
   at that broker at its stated timestamp.
2. An approved manual record is authoritative only for an external holding that
   is outside the connected broker feeds. It is never evidence of broker cash,
   margin, or fills.
3. `runtime/db/accounts/account_ledger.db` is the canonical normalized historical
   record. It preserves the source profile and timestamp; normalization does not
   make it more authoritative than the underlying broker evidence.
4. `unified_live` is the consolidated reporting account. It combines broker
   state and approved manual holdings, so it is not a broker statement and must
   not be compared to one without an explicit allowance for manual additions.
5. `runtime/db/portfolio/portfolio_ledger.db` is a migration and derived-
   valuation lane. It is not the canonical account ledger.

## Authority By Field

| Field | Primary authority | Controlled fallback |
| --- | --- | --- |
| Broker position quantity | Broker snapshot | None; raise a break. |
| Average cost and broker PnL | Broker snapshot | Derived values must be labelled. |
| Broker cash and buying power | Broker snapshot | None; raise a break. |
| Broker NAV and margin | Broker snapshot | Reconstructed NAV is a diagnostic only. |
| External/manual quantity | Approved manual record | None. |
| External/manual price | Approved market mark | Retained stale mark with visible age and source. |
| FX conversion | Approved market FX source | Last valid rate with visible age. |
| Consolidated reporting NAV | `unified_live` calculation | Broker-only NAV shown separately. |
| Trade and fill state | Broker execution evidence | Internal event is pending until reconciled. |

## Timestamp Rules

- Comparisons must use snapshots from the same operational cut or record their
  timestamp difference explicitly.
- Freshness is evaluated per source profile. A fresh aggregate cannot conceal a
  stale component source.
- Manual-file modification time is not market-price freshness.
- Historical restatements create new evidence; they do not silently rewrite the
  original source extract.

## Migration Gate

The legacy portfolio ledger may be retired only after a documented observation
period shows that positions, cash, NAV, currency conversion, and manual holdings
reconcile to the canonical account ledger within approved tolerances. Retirement
must include consumer discovery and rollback instructions.
