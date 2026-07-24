# Alpha Research Public/Private Boundary

Last reviewed: 2026-07-24
 
These files are safe candidates for GitHub when they do not contain real
account data, vendor credentials, or proprietary performance results:

- Research framework code.
- Data contracts, schemas, and adapters.
- Dashboard shell code.
- Test fixtures with synthetic data.
- Factor interfaces and reusable contracts, such as `contracts.py`.
- Metadata templates that describe required fields and market suitability.
- Tests that use synthetic demo factors rather than live `fac_*.py` recipes.
- Sanitized research manuscripts whose figures and tables build only from a reviewed,
  rounded aggregate evidence allowlist and contain no contract-level or return-series data.

## Private By Default

These files should remain local/private unless explicitly sanitized:

- Live factor implementations in
  `departments/research/factors/`.
- Candidate, trial, promotion, sweep, and backtest result artifacts.
- Cached market data and vendor exports.
- Execution logs and return series.
- Local trained model files or model JSONs.
- Runtime alpha research matrices and regime outputs under
  `runtime/data/`.
- Local CN futures vendor/static market data under
  `runtime/data/futures_cn/`.
- Local research memory databases such as `research_memory.db` and
  `optimization_memory.db`.
- Tick/latent/regime model artifacts under
  `runtime/artifacts/research/`.
- Workbench/archive scripts that reveal current research process or edge.
- Live router, position-policy, sleeve, risk-overlay, strategy, and
  optimization-study definitions under `departments/research/`.
- Generated report and output folders.
- Legacy archive factor files, strategy agents, and trained model bundles unless
  they have gone through the retired-factor publication process.

## Retired Factor Rule

Retired factors can be committed only through a deliberate allowlist. Before
publishing one, freeze it, remove proprietary parameters or dataset-specific
tricks, strip performance logs and live comments, mark it as retired or
educational, then move or copy it to `departments/research/retired_factors/`.

Do not stage retired factors directly from `departments/research/factors/`.

## Public Test Rule

Public alpha tests should not import live private factor modules directly. If a
test needs factor behavior, use a synthetic demo factor built inside the test
fixture or a sanitized retired example from `departments/research/retired_factors/`.
