# Research Dashboard Page Roadmap

The dashboard follows the analyst decision sequence rather than exposing every
engine as a separate sidebar page:

1. **Trust the inputs.**
2. **Discover and formalize hypotheses.**
3. **Understand market context.**
4. **Validate features, factors, and completed runs.**
5. **Construct multi-factor and routed strategies.**

Reusable calculations remain in `src/oqp/`. A consolidated page may embed more
than one view, but it must not merge their mathematical contracts.

## Shared Page Blueprint

- State the decision supported by the page in one sentence.
- Keep the first screen to three to six controls and three to five key metrics.
- Prefer three or four tabs; use a segmented mode for distinct workflows.
- Explain validity, weakness, suspicious results, and the next action.
- Keep public-safe empty states when private runtime data is absent.
- Store reusable logic outside the Streamlit entrypoint.

## Page Map

| Order | Page | Workflow role | Primary question |
|---:|---|---|---|
| Home | Research Ledger | Cross-run memory | What has been tested and what evidence exists? |
| 01 | Data Health | Trust | Are data, APIs, artifacts, and infrastructure ready? |
| 02 | Discovery Lab | Signal discovery and hypothesis testing | Which patterns, events, or cross-asset relationships deserve formal validation? |
| 05 | Regime Analysis | State context | What market state is active, and can the state model be trusted? |
| 06 | Market Breadth Lab | Diversification context | Is direction, volatility, concentration, and independent risk genuinely broad? |
| 07 | ML Hub | ML research governance | Are the data, features, experiments, models, and promotion evidence reproducible and safe out of sample? |
| 08 | Research Review | Component governance and strategy construction | Which factors, sleeves, routers, and constructed strategies deserve promotion? |

## 01 Data Health

The trust gate for provider/API readiness, raw data coverage, processed research
artifacts, missingness, and runtime infrastructure. Downstream research should
not proceed when required inputs fail.

## 02 Discovery Lab

Three explicit modes share one discovery stage while preserving separate
mathematical contracts:

- **Pattern Scan:** daily, intraday, and tick lenses; asset ranking; clock
  effects; pulse discovery; event inspection; cross-asset maps.
- **Event Study:** formal hypothesis setup; event/base rates; lift and confidence
  intervals; event examples; cross-asset validation; evidence-ticket capture.
- **Relationship Scan:** pair, calendar-spread, cross-product, and statistical
  relationship discovery; dynamic hedge diagnostics; spread construction;
  cost-aware backtest previews; data audit.

Pattern Scan asks how one asset behaves repeatedly. Event Study asks what
happens after a defined event. Relationship Scan asks whether two assets form a
stable spread. The shared page is a navigation boundary, not a merged engine.

An evidence ticket remains an idea record. It does not enter Research Review
until the idea is implemented as a developed factor with recorded run evidence.

## 05 Regime Analysis

Owns current-state interpretation, state geometry, latent cross-checks, and
model diagnostics. Regime labels provide context; they do not automatically
authorize routing or parameter selection.

## 06 Market Breadth Lab

Owns directional breadth, volatility by asset and industry, concentration
breadth, PCA risk breadth, component interpretation, and rolling compression.
It informs sample selection and risk capacity but does not create alpha by
itself.

## 07 ML Hub

Owns the full machine-learning research lifecycle: taxonomy-scoped feature data,
missingness, redundancy, IC stability, purged MDA, experiment history, runtime
readiness, registered model artifacts, PCA/latent diagnostics, and promotion
protocol. Features remain predictors; they are not tradable factors until a
factor contract turns model output into a score and records backtest evidence.

## 08 Research Review

The component libraries and construction workflow form one governance workspace:

1. **Review Board:** lifecycle funnel, promotion scores, blockers, and candidate
   snapshots.
2. **Factor, Sleeve, and Router Libraries:** reusable components, contracts,
   limitations, and component-level evidence.
3. **Strategy Construction:** assemble frozen components without changing their
   parameters, then apply allocation, overlays, execution, and final costs.
4. **Run Comparison:** comparable metrics, equity/drawdown paths, return
   correlation, improvement versus baseline, and trade X-Ray.

Run Comparison can inspect multiple completed runs after their return artifacts
are available. Strategy Construction is a tab here rather than a duplicate
sidebar page.

## Definition Of Done

- Sidebar exposes only the seven workflow pages above.
- English and Chinese labels are aligned.
- Embedded views do not duplicate page configuration or sidebar controls.
- Evidence provenance records the consolidated source page.
- Page-order, import, AppTest, and relevant engine tests pass.
- Heavy calculations remain independently testable outside Streamlit.
