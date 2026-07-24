# Paper A - Daily Volatility Routing in Chinese Futures

This is the sanitized public manuscript mirror for the daily cross-market replication:

> **When Volatility Is Not a Sufficient State: A Cross-Market Replication in Chinese Futures**

The project contains no alpha search. It builds only from
`evidence/public_evidence.json`, a reviewed set of rounded aggregate statistics. It does
not import private factor implementations, cached market data, target positions, monthly
return series, contract attribution, cost schedules, or capacity assumptions.

## Research claim

The fixed volatility-only switching rule produces an event-level momentum-crash hedge
echo in one defensible Chinese-futures market proxy, but the relation is not stable across
months or across an equally defensible market proxy. Aggregate volatility measures shock
intensity; in a heterogeneous futures cross-section it does not uniquely encode shock
direction, source, breadth, or stage.

## Build

From the repository root:

```bash
MPLCONFIGDIR=/tmp/oqp-public-mpl /opt/anaconda3/bin/python3 \
  notebooks/Phase_7_Research_Projects/07_06_daily_volatility_router_cn_futures_replication_public/scripts/build_public_figures.py
```

Then from `paper/`:

```bash
/opt/homebrew/bin/tectonic paper_a_volatility_not_sufficient.tex --outdir ../output/pdf
```

The release-candidate manuscript is
`output/pdf/paper_a_volatility_not_sufficient.pdf`. The research and disclosure decisions
made during packaging are recorded in `decision_ledger.md`.

## Evidence status

- Cross-market replication, not a production strategy.
- Methodology fixed before viewing the replication returns.
- Sixty-six evaluable holding months after the five-year state-history requirement.
- No threshold, sleeve, proxy, sector, or universe optimization.
- No untouched confirmatory period; all empirical statements are exploratory.
