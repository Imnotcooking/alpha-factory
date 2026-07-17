# Research Dashboard Page Roadmap

This document is the working product map for `apps/research_dashboard`. Its job is to make each page understandable before we keep adding functionality.

The dashboard should feel like a guided research pipeline:

1. **Can I trust the data?**
2. **Where are there interesting market events or relationships?**
3. **What regime/risk context am I in?**
4. **Which features/factors deserve promotion?**
5. **Which strategies are worth comparing or simulating?**

Every page should explain what it is answering, how to read its metrics, when a result is valid/promising/weak, and what the user should do next.

## Shared Page Blueprint

Each page should converge toward this structure:

- **Purpose:** one sentence answering "what decision does this page help me make?"
- **Inputs:** 3-6 controls maximum on the first screen.
- **Top metrics:** 3-5 cards that summarize status or opportunity.
- **Primary visual:** the one chart/table the user should read first.
- **Workspace/drilldown:** one focused area for deeper inspection.
- **Explanation toggles:** short expanders beside metrics/widgets, written for a new researcher.
- **Validity rules:** what makes the result valid, weak, promising, or dangerous.
- **Next action:** what page or workflow to use after this one.
- **Public-safe behavior:** clear empty states when private data files are absent.

Preferred tab count: **3-4 top-level tabs maximum**. If a page needs more specialized views, use radios/segmented controls inside a tab.

## Proposed App Flow

| Order | Page | Research Role | Main User Question |
|---:|---|---|---|
| 1 | Data Health | Trust layer | Is the platform/data ready enough to believe anything downstream? |
| 2 | Pattern Lab | Multi-timeframe discovery | Which daily, intraday, or tick patterns deserve a formal hypothesis test? |
| 3 | Intraday Event Study | Microstructure hypothesis testing | Does a specific intraday/tick-level event idea have directional follow-through? |
| 4 | Arbitrage Lab | Relationship/opportunity discovery | Which pairs/spreads look dislocated but still stable enough to research? |
| 5 | Regime Analysis | Market state context | What state is the market/asset in, and can the state model be trusted? |
| 6 | Market Breadth Lab | Portfolio risk geometry | How many independent market dimensions do we really have? |
| 7 | Feature Review | Feature validation | Which features are clean, useful, redundant, or dangerous? |
| 8 | Factor Review | Research governance | Which factors have enough evidence to move forward? |
| 9 | Strategy Comparison | Strategy review | Which completed strategy profile is better after risk, path, and trade checks? |

Potential new landing page later: **Research Overview**, a compact dashboard linking into the nine pages above.

## 01 Data Health

**Purpose:** The trust gate. Use this before believing any other page.

**Current state:** Strong content, but currently has seven tabs: Overview, Core Data, Tick Data, Database & Returns, Latent Artifacts, Model Registry, C++ / ML Infra.

**Recommended UX:** Reduce to four tabs:

1. **Overview**
   - Readiness score.
   - OK/WARN/FAIL counts.
   - Latest run.
   - Area status bar chart.
   - Top blockers table.

2. **Data Sources**
   - Core daily matrices.
   - Tick cache files.
   - Date ranges, rows, assets, size, freshness.
   - Public-safe empty state: "Add a parquet with date/ticker/close."

3. **Research Artifacts**
   - Database schema checks.
   - Return log linkage.
   - Latent artifacts.
   - Model registry artifacts.

4. **Runtime**
   - C++ extension availability.
   - ML model files.
   - Python fallback warnings.

**Key metrics:**

- Readiness score.
- Failure count.
- Warning count.
- Latest data date.
- Missing return-link count.

**How to interpret:**

- **Valid:** readiness high, no FAIL in core data/database, newest files have expected date range.
- **Weak:** warnings in optional model/latent artifacts; downstream pages may still work.
- **Dangerous:** missing feature matrix, missing database columns, missing return logs, or stale tick files.

**Next action:** If core data is healthy, move to discovery pages. If not, fix data/artifacts before researching.

## 02 Pattern Lab

**Purpose:** Find promising patterns across daily, 1-minute, and tick data before forming an alpha hypothesis.

**Current state:** Rich and educational, but it is a long single page with many controls and sections.

**Recommended UX:** Keep it to five focused modes:

1. **Lens Scope**
   - Daily lens for broad asset candidates.
   - 1-minute lens for intraday volatility and session behavior.
   - Tick lens for microstructure/pulse feasibility.

2. **Pulse Radar**
   - File/contract selector.
   - Severity distribution.
   - Pulse/hour and percentile ladder.
   - Candidate event table.

3. **Event Inspector**
   - Pick one pulse event.
   - Price/volume/spread/book-depth window around the event.
   - Pulse fingerprint.
   - Event quality interpretation.

4. **Behavior Preview**
   - What happened after the pulse.
   - Forward move bands.
   - Directionless behavior, not alpha yet.

5. **Cross-Asset Map**
   - Compare pulse severity across downloaded contracts.
   - Severity zones.
   - Best contracts to inspect next.

**Key metrics:**

- Daily/1m realized volatility by asset.
- Session/time-of-day edge.
- Pulses/hour.
- p99/p99.5 move in ticks.
- Median spread in ticks.
- Event severity.
- Post-pulse drift or reversion preview.

**How to interpret:**

- **Valid daily candidate:** enough movement, enough liquidity, and enough coverage to justify deeper work.
- **Valid 1m candidate:** intraday movement appears in tradable sessions rather than only daily gaps.
- **Valid tick candidate:** large tick move, enough rows in sample, reasonable spread, visible event structure.
- **Weak candidate:** one-off spike, sparse sample, stale file, or wide spread.
- **Promising for research:** repeated structure across the right lens, with clean data and a visible behavioral clue.

**Next action:** Promote a good pulse or clock pattern into a hypothesis seed, then test it in Intraday Event Study.

## 03 Intraday Event Study

**Purpose:** Test whether a specific tick-level pulse hypothesis has directional follow-through.

**Current state:** Powered by `TickPulseLabPage`; likely too much workflow in one screen, but the core engine is modular.

**Recommended UX:** Four tabs:

1. **Hypothesis Setup**
   - Select tick file, contract, saved seed or built-in hypothesis.
   - Choose threshold mode.
   - Show what the rule actually means in plain English.

2. **Evidence Board**
   - Base rate vs event rate.
   - Hit rate, lift, expected move, sample size.
   - Validity warnings for small samples or unstable thresholds.

3. **Event Examples**
   - Successful event.
   - Failed event.
   - Middle/typical event.
   - Visual explanation of what the rule saw.

4. **Cross-Asset Sweep**
   - Test the same hypothesis across downloaded assets.
   - Rank by lift, event count, and stability.

**Key metrics:**

- Event count.
- Base rate.
- Event hit rate.
- Lift over base rate.
- Expected move.
- Stable/unstable threshold flag.

**How to interpret:**

- **Valid:** enough events, enough future rows, threshold mode is asset-adaptive or otherwise justified.
- **Weak:** tiny event count, single contract only, no improvement over base rate.
- **Promising:** event hit rate improves over base rate across multiple contracts without spread/liquidity problems.

**Next action:** If robust, save/promote as a research seed or factor candidate.

## 04 Arbitrage Lab

**Purpose:** Find and inspect dislocated but still plausible spread/pair opportunities.

**Current state:** Recently upgraded and simplified to four tabs:

1. **Radar**
   - Scanner table and score.
   - Dislocation vs stability chart.
   - Sector/family opportunity density.

2. **Workspace**
   - Scanner candidate or manual pair.
   - Dual Kalman relationship.
   - Spread construction.
   - Backtest preview.

3. **Maps**
   - Calendar, cross-product, and stat-arb views behind one selector.

4. **Audit**
   - Source schema, asset coverage, contract-level capability.

**Key metrics:**

- Opportunity score.
- Latest z-score.
- Correlation.
- Half-life.
- Beta drift.
- Estimated round-turn cost.

**How to interpret:**

- **Valid:** enough observations, stable beta, reasonable correlation, data source supports the spread type.
- **Weak:** mild z-score, high beta drift, weak half-life, or index data used for a calendar-spread question.
- **Promising:** stretched spread, stable relationship, plausible mean reversion, acceptable cost/liquidity.

**Next action:** If promising, move from preview to proper cost-aware backtest/promotion.

## 05 Regime Analysis

**Purpose:** Understand current market state and whether the regime model is trustworthy.

**Current state:** Four tabs: Timeline, Geometry, Cross-check, Diagnostics. This is close, but needs clearer beginner framing.

**Recommended UX:** Keep four tabs, rename slightly:

1. **Current State**
   - Current regime card.
   - Price/regime probability timeline.
   - Stress z-score patterns.

2. **State Geometry**
   - 3D phase space.
   - Radar/DNA chart.
   - State ranges.

3. **Cross-Checks**
   - VQ-VAE code comparison.
   - Meta-label diagnostics.
   - Future return by state.

4. **Diagnostics**
   - Pipeline audit.
   - GMM density surfaces.
   - State profiler.

**Key metrics:**

- Current dominant state.
- Panic/stress probability.
- State confidence.
- VQ agreement.
- Future return / stress rate by state.

**How to interpret:**

- **Valid:** model probabilities are present, selected asset overlaps with feature matrix, states have distinct behavior.
- **Weak:** low confidence, state labels collapse, VQ disagreement, future outcomes do not differ by state.
- **Promising:** state transitions visibly align with price/risk behavior and future stress diagnostics differ by state.

**Next action:** Use regime context to judge strategies, features, and arbitrage candidates under different market states.

## 06 Market Breadth Lab

**Purpose:** Estimate how many independent market dimensions the selected taxonomy universe really contains.

**Current state:** No top-level tabs, but the page is long. It already has good explanation toggles.

**Recommended UX:** Three tabs:

1. **Breadth Summary**
   - Naive breadth.
   - Valid assets.
   - BR95.
   - Effective rank.
   - Participation ratio.
   - Breadth haircut.

2. **Component Map**
   - Eigen spectrum.
   - Component interpreter.
   - Sector loading heatmap.
   - Top asset loadings.

3. **Rolling Breadth**
   - BR95/effective rank over time.
   - Breadth haircut over time.
   - Stress compression notes.

**Key metrics:**

- BR95.
- Effective rank.
- Participation ratio.
- PC1 variance.
- Breadth haircut.

**How to interpret:**

- **Valid:** enough assets, enough observations, PCA not dominated by missing data.
- **Weak:** fewer than 20 valid assets, short history, or one asset family dominates too much.
- **Promising insight:** breadth expands or compresses in a way that explains diversification/risk capacity.

**Next action:** Use breadth to set realistic expectations for portfolio diversification and factor independence.

## 07 Feature Review

**Purpose:** Decide which features are clean, useful, redundant, or dangerous before model training/promotion.

**Current state:** Powerful but too many tabs: Overview, Correlation, Stability, MDA, Missingness, PCA, Latent, Protocol.

**Recommended UX:** Reduce to four tabs:

1. **Quality Board**
   - Rows/features/assets/date range.
   - Feature family map.
   - Feature quality table.
   - Missingness summary.

2. **Redundancy Map**
   - Spearman correlation heatmap.
   - High-correlation pairs.
   - Cluster representatives.
   - PCA redundancy view.

3. **Predictive Evidence**
   - IC stability.
   - Daily IC/rolling IC.
   - Purged K-Fold MDA.
   - Feature keep/drop recommendation.

4. **Latent & Protocol**
   - VQ-VAE latent cross-check.
   - Codebook usage.
   - Manual feature profile by code.
   - Governance protocol table.

**Key metrics:**

- Quality score.
- Mean IC.
- IC IR.
- Positive day rate.
- Missing percentage.
- Turnover proxy.
- MDA importance.

**How to interpret:**

- **Valid feature:** enough history, low missingness, stable IC, not merely duplicate of another feature.
- **Weak feature:** high missingness, unstable IC, high turnover proxy, no MDA evidence.
- **Promising feature:** stable positive IC, survives purged MDA, low redundancy, interpretable family meaning.

**Next action:** Promote keeper features into model/factor research or remove/retire noisy features.

## 08 Factor Review

**Purpose:** Govern factors from idea to validation/paper-trading candidate.

**Current state:** Already clean: Pipeline Board and Factor Drilldown. Keep two tabs.

**Recommended UX:** Keep two tabs, strengthen explanations:

1. **Pipeline Board**
   - Lifecycle funnel.
   - Promotion board.
   - Filters by stage/category/market.
   - Recent exported candidate snapshots.

2. **Factor Drilldown**
   - Evidence checklist.
   - Run history.
   - Promotion gates.
   - Artifacts.
   - Export research candidate snapshot.

**Key metrics:**

- Promotion score.
- Stage.
- Runs.
- Best Sharpe.
- Best holdout IC.
- Bonferroni p / FDR q.
- Trades.
- Blockers.

**How to interpret:**

- **Valid:** source, metadata, return logs, trades, and diagnostic evidence exist.
- **Weak:** no return log, no trades, non-positive IC, p-value does not survive multiple testing, or market vertical untested.
- **Promising:** passes validation/paper-trading gates with no serious blockers.

**Next action:** Export candidate snapshot only when the factor has passed the relevant gate.

## 09 Strategy Comparison

**Purpose:** Compare completed strategy profiles after returns, risk, path, and trade behavior.

**Current state:** One clean manager-style page. It likely needs clearer workflow grouping but not many tabs.

**Recommended UX:** Three tabs:

1. **Comparison Board**
   - Select heuristic, raw ML, regime/gate strategy, and extra runs.
   - Metric cards.
   - Equity and drawdown.

2. **Risk & Diversification**
   - Metric table.
   - Return correlation.
   - Improvement versus baseline.

3. **Trade X-Ray**
   - Asset PnL contribution.
   - Holding period distribution.
   - Trade ledger diagnostics.

**Key metrics:**

- Annualized return.
- Annualized volatility.
- Sharpe.
- Max drawdown.
- Calmar.
- Average turnover.
- Holdout IC.
- Trades.

**How to interpret:**

- **Valid:** selected runs have return series and comparable date coverage.
- **Weak:** strategy only wins by one lucky spike, high drawdown, tiny trade count, or missing trade ledger.
- **Promising:** better Sharpe/drawdown than baseline, acceptable turnover, diversified PnL, robust path.

**Next action:** Use this page for manager review and simulation prioritization after factor promotion.

## Suggested Analyst Workflow

1. **Data Health**
   Start with the trust gate. If users are lost here, every other page feels suspicious.

2. **Pattern Lab**
   Find raw event structure before writing alpha hypotheses.

3. **Intraday Event Study**
   Turn interesting event structure into a testable microstructure hypothesis.

4. **Arbitrage Lab**
   Scan relationships and spread dislocations that may become candidates.

5. **Regime Analysis**
   Add market-state context before judging any signal or strategy.

6. **Market Breadth Lab**
   Check whether the selected universe really offers independent market capacity.

7. **Feature Review**
   Decide which features are clean, useful, redundant, or dangerous.

8. **Factor Review**  
   Promote only after evidence, artifacts, and gates are defensible.

9. **Strategy Comparison**  
   Review completed strategies after risk, path, and trade checks.

10. **Research Overview**  
   Add last, after the individual pages have stable names and purposes.

## Page Upgrade Definition Of Done

For each page upgrade:

- Top-level tabs are 3-4 or fewer.
- The first screen says what question the page answers.
- Every metric group has an interpretation toggle.
- Every page has explicit valid/weak/promising guidance.
- Empty states work without private data.
- Heavy calculations live in reusable modules or view classes, not one-off page script code.
- Targeted tests cover any new non-UI logic.
