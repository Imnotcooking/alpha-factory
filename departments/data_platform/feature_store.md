# Local Feature Store Contract

Last reviewed: 2026-07-17

The current feature store is a local, file-based research store under
`runtime/data/feature_store/`. It is not an online feature service and does not
provide automatic serving, registry, or historical replay guarantees.

## Boundaries

- Raw vendor observations belong in the asset-class lanes under
  `runtime/data/`.
- Reference metadata belongs in `runtime/data/metadata/` or the executable
  Instrument Master when it is static and commit-ready.
- Model-ready feature matrices belong in `runtime/data/feature_store/`.
- Trained models, feature importance, diagnostics, and reports belong in
  `runtime/artifacts/`.

## Required Properties

Each feature matrix should declare:

- entity key, event timestamp, and optional availability timestamp;
- source dataset and parent manifest identifiers;
- feature names, types, units, and lookback windows;
- target columns kept separate from predictor columns;
- universe and asset-taxonomy scope;
- missingness/freshness policy;
- point-in-time join policy and publication lag;
- build configuration, code version, and output hash.

Features must be computed from information available at the decision time.
Forward-filled accounting values and Brownian Bridge reconstructions cannot be
silently promoted to alpha features.

## Promotion

A file becomes a canonical feature matrix only after schema, point-in-time,
missingness, and lineage checks pass. Exploratory matrices may remain under
runtime storage, but must not be presented as reproducible promotion evidence
without a manifest.
