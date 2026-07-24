# Public Release Manifest

## Allowed contents

- Rounded aggregate sample counts and date ranges.
- Publicly described momentum, reversal and volatility-routing methodology.
- Aggregate strategy statistics rounded to two decimals.
- Four-state conditional means rounded to two decimals.
- Aggregate proxy agreement, event influence and cross-sectional anatomy diagnostics.
- Figures generated exclusively from `evidence/public_evidence.json`.
- LaTeX manuscript source, bibliography and boundary tests.

## Excluded contents

- Vendor filenames, data exports, credentials or source hashes.
- Private repository paths, private factor imports or internal artifact references.
- Contract identifiers, contract-level attribution, sector PnL attribution or target weights.
- Daily or monthly return series and event dates.
- Exact transaction-cost schedules, capital, integer-lot, capacity or execution parameters.
- Any later profitable mechanism, state threshold or production rule.

## Evidence transformation

Every public empirical value is manually allowlisted and rounded in the JSON evidence
file. Public figures read that file only. The public manuscript does not read or link to
the private evidence source, so a public build cannot accidentally traverse into private
research data.
