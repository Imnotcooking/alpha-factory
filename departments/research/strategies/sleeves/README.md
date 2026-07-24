# Research Sleeve Specifications

This registry contains reusable policies that translate one already-computed
factor score into target positions. A sleeve owns cross-sectional selection,
weighting, holding, rebalancing, exposure limits, missing-score treatment, and
execution delay. It does not calculate alpha, choose a regime, or define a
complete entry-and-exit strategy.

Use `slv_*` stable IDs. Keep the underlying alpha in the flat `fac_*` registry,
and keep routing, position policies, and risk overlays in their own registries.

`slv_001_Cross_Sectional_Quintile_Long_Short` is the frozen Phase 3 baseline.
It applies the same simple top/bottom-quintile construction to any eligible
cross-sectional factor. Its default parameters remain frozen. Each active sleeve
also declares a narrow `SLEEVE_PARAMETERS` search schema for a separate Phase 8
candidate study; the study never rewrites the registered default. A promoted
candidate receives a new frozen sleeve specification.

The active registry also contains frozen quartile, 5% tail, and continuous
z-score constructions. These differ only in how they translate the same score
cross-section into positions, which makes their economics directly comparable.
Their selection geometry is not tunable inside these schemas: a quintile study
cannot quietly become the already-registered quartile or 5% tail sleeve.

EMA, MACD, and night-session implementations that generated their own signals
or owned a full trade lifecycle were deleted from this registry. Their factor
definitions and frozen research results remain separate from sleeve construction.

The canonical construction and cost-aware evidence engines live under
`src/oqp/research/sleeves/`. Saved evidence belongs under
`runtime/artifacts/research/sleeve_construction/`; strategy recipes import the
stable `slv_*` component and never copy its allocation logic.
