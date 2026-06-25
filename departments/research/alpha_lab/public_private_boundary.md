# Alpha Research Public/Private Boundary

Last reviewed: 2026-06-25

This repository can publish the alpha research framework without publishing
the active research edge. Treat the alpha lab as two layers.

## Public By Default

These files are safe candidates for GitHub when they do not contain real
account data, vendor credentials, or proprietary performance results:

- Research framework code.
- Data contracts, schemas, and adapters.
- Dashboard shell code.
- Test fixtures with synthetic data.
- Factor interfaces and reusable contracts, such as `contracts.py`.
- Metadata templates that describe required fields and market suitability.
- Tests that use synthetic demo factors rather than live `fac_*.py` recipes.

## Private By Default

These files should remain local/private unless explicitly sanitized:

- Live factor implementations in `alpha_research_lab/factors/fac_*.py`.
- Candidate, trial, promotion, sweep, and backtest result artifacts.
- Cached market data and vendor exports.
- Execution logs and return series.
- Local trained model files or model JSONs.
- Workbench/archive scripts that reveal current research process or edge.
- Legacy archive factor files, strategy agents, and trained model bundles unless
  they have gone through the retired-factor publication process.

## Retired Factor Rule

Retired factors can be committed only through a deliberate allowlist. Before
publishing one, freeze it, remove proprietary parameters or dataset-specific
tricks, strip performance logs and live comments, mark it as retired or
educational, then move or copy it to a public examples/retired-factors path.

Do not stage retired factors directly from the live `factors/` directory.

## Public Test Rule

Public alpha tests should not import live private factor modules directly. If a
test needs factor behavior, use a synthetic demo factor built inside the test
fixture or a sanitized retired example from the public examples path.
