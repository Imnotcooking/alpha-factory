# Module 7 Research Program Map

This directory contains several related projects, but they are not one continuously
changing research question. Each project has one claim, one evidence boundary and one
publication role.

## Current execution order

No pathway is currently active. At most one pathway may be active after a written plan
has been approved:

1. **PAUSED - 07_03 intraday microstructure mechanism.** M1 and M1b are frozen. M1b
   identifies session-gap repair as a stronger conditional mechanism, but its economics
   are concentrated after 2025-10; no filter attribution, sleeve change or router work
   has started beyond this checkpoint.
2. **COMPLETE - 07_06 daily public replication paper.** The sanitized English manuscript
   and aggregate evidence contract are frozen; no alpha search occurred in the mirror.
3. **COMPLETE - 07_05 daily vector-state diagnostics.** Breadth, positioning and
   tail-stage mechanisms all failed their frozen gates; no router was unlocked.
4. **LOCKED - threshold and universe optimisation.** A percentile search or universe
   Pareto is permitted only after a router has passed an untouched validation period
   after realistic costs.

## Publication review gate

No new research pathway starts while a completed report is awaiting editorial review.
Analysis completion and publication readiness are separate states. Every report must pass
the evidence, language, figure, layout, reproducibility and confidentiality checks in
`PUBLICATION_REVIEW_CHECKLIST.md`, with a project-local review record, before it is treated
as a finished paper.

## Project ownership

| Project | Role | Current claim | Status |
|---|---|---|---|
| `07_01_daily_latent_regimes_cn_futures` | Pre-existing independent paper | Daily latent-regime research | Separate; untouched |
| `07_04_daily_volatility_router_cn_futures_replication_private` | Faithful baseline experiment | The equity paper's scalar volatility rule does not robustly replicate in Chinese futures | Replication frozen; Chinese report editorially reviewed; manager sign-off pending |
| `07_05_daily_vector_state_router_cn_futures_private` | Private mechanism extension | Breadth, positioning and tail-stage diagnostics did not identify a stable routing mechanism | Frozen complete; no router |
| `07_06_daily_volatility_router_cn_futures_replication_public` | Public daily manuscript mirror | A scalar volatility state is not transportably sufficient across heterogeneous futures | Frozen complete |
| `07_03_intraday_alpha_router_cn_futures_private` | Private intraday laboratory | Determine whether the observed first-bar bounce reflects tradable microstructure or only statistical reversal | Paused at M1b checkpoint |
| `07_02_intraday_volatility_router_cn_futures` | Public manuscript mirror | Publishable lessons on costs, microstructure and failed routing, without private alpha details | Evidence intake frozen |
| `07_07_gtja_alpha191_equity_cn_private` | Private factor-library import | Reproduce the licensed Alpha191 formula collection under local causal semantics, without claiming economic validity | Implementation complete; canonical China-equity data and semantic validation pending |

## Paper pathways

### Paper A: daily cross-market replication and state identification

`07_06` owns the frozen public manuscript and combines sanitized evidence from `07_04`
and `07_05`:

1. faithfully map the equity method to Chinese futures;
2. show why a scalar volatility state is not economically sufficient;
3. test a preregistered, low-dimensional state vector;
4. report either a robust economic improvement or a rigorous failure.

Exact profitable formulas, thresholds, contract attribution, costs and capacity remain
private. The publication mirror builds only from its rounded aggregate evidence
allowlist and never imports private factors or runtime artifacts.

### Paper B: intraday microstructure in Chinese futures

`07_03` owns the research. It begins from the first-bar reversal observation and asks
whether the effect is bid/ask bounce, temporary price pressure, session-gap repair or a
tradable response to order flow. Tick data is used only for targeted events where the
minute bars cannot identify the mechanism. `07_02` is the publication mirror, not a
second alpha-search program.

## Canonical code rule

Projects own hypotheses, frozen configurations, experiment runners, reports and artifact
references. They do not own reusable alpha definitions.

All active signals, state variables, sleeves and routers must have a canonical entry in
`departments/research/factors/catalog.yaml`. New canonical Chinese-futures components
belong in the flat `departments/research/factors/` registry. Historical project-local code may
remain as a frozen reproduction shim, but it must not silently become a second editable
implementation.

Generated data and backtest artifacts remain under `runtime/`; they are never copied into
the paper folders.
