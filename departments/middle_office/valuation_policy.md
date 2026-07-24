# Account Valuation Policy

Last reviewed: 2026-07-17

Valuation produces accounting and control views. It does not create alpha
signals and must preserve price, FX, and freshness provenance.

## Mark Hierarchy

1. Broker mark for a broker-owned position at the broker snapshot timestamp.
2. Approved market-data mark for an external/manual or diagnostic position.
3. Last valid mark within an approved stale-price window, clearly flagged.
4. Manual valuation only with owner, reason, timestamp, and evidence.

No silent zero, fabricated trade, or midpoint may replace a missing mark.

## Position Value

```text
native market value = quantity * market price * contract multiplier
reporting value = native market value * approved FX rate
```

If the broker supplies market value, retain it as broker evidence. A locally
reconstructed value is a separate diagnostic used to detect multiplier, FX, or
mark differences.

## NAV And PnL

- Broker NAV is authoritative for that broker account.
- Consolidated NAV combines broker NAV with approved external/manual value and
  must expose the contribution of each source.
- Daily PnL must state whether it is broker-reported, NAV-difference derived, or
  trade-event reconstructed.
- Cash flows, deposits, withdrawals, fees, and FX translation must not be
  mistaken for investment return.
- Legacy `historical_nav` remains a migration diagnostic until parity is proven.

## Freshness

- Accounting may carry the last valid mark for valuation, with `is_fresh=false`
  and a visible age.
- Alpha and execution modules must not treat carried prices as fresh observations.
- A fresh snapshot containing a stale manual mark remains stale for that holding.
- Option valuations must retain contract identity, multiplier, expiry, and mark
  source. Model value and broker liquidation value are separate fields.

## Currency

Store native currency and reporting currency separately. FX source, timestamp,
pair orientation, and fallback status are required evidence. Reconciliation is
performed by native currency before comparing consolidated reporting values.
