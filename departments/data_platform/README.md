# Data Platform Department Map

Last reviewed: 2026-07-17

This folder is the governance and operating layer for market data, reference
data, dataset quality, lineage, vendors, and the local feature store. It does
not contain executable adapters or runtime datasets.

## Ownership Boundary

| Responsibility | Canonical home | Commit posture |
| --- | --- | --- |
| Data contracts, adapters, path discovery, quality views | `src/oqp/data/` | Commit source and tests. |
| Vendor and broker credentials | Environment or approved secret storage | Never commit secrets. |
| Data ownership, quality, lineage, and recovery policy | `departments/data_platform/` | Commit docs and lightweight specifications. |
| Raw and cached market data | `runtime/data/` | Local/private; do not commit vendor extracts. |
| Generated research evidence | `runtime/artifacts/` | Local/private; retain manifests for reproducibility. |
| Operational logs and health evidence | `runtime/logs/` | Local/private. |

Reusable code must remain under `src/oqp/`. Streamlit pages may present data
health, but they must not become a second data-engine implementation.

## Start Here

Use the Research Dashboard Data Health page for visibility, then run the data
tests before changing a contract:

```bash
oqp dashboard research
python -m pytest -q tests -k data
```

## Department Files

- `storage_map.md`: canonical storage boundaries and migration rules.

Additional catalogs, lineage specifications, and recovery runbooks should be
added here only when their implementation and operating owner are committed in
the same change.

## Current Implementation Map

| Capability | Implementation |
| --- | --- |
| Asset taxonomy | `src/oqp/data/asset_taxonomy.py` |
| Instrument reference data | `src/oqp/data/instruments.py` |
| Adapter contracts and DTOs | `src/oqp/data/base.py`, `src/oqp/data/models.py` |
| Vendor factories | `src/oqp/data/registry.py`, `src/oqp/data/vendors/` |
| Runtime path discovery | `src/oqp/data/runtime_paths.py` |
| Missingness and quality views | `src/oqp/data/missingness.py`, `src/oqp/data/views.py` |
| Risk-only Brownian Bridge view | `src/oqp/data/brownian_bridge.py` |
| Operational visibility | `apps/research_dashboard/pages/01_Data_Health.py` |

`src/oqp/data/feeds.py` is a compatibility loader, not the authoritative
vendor-adapter contract. New integrations should use `base.py`, `models.py`,
and `registry.py` unless a migration explicitly requires the legacy feed API.

## Change Procedure

1. Register a new canonical storage lane in `storage_map.md`.
2. Add or update package-owned contracts and adapters under `src/oqp/data/`.
3. Add tests for schema normalization, path discovery, and failure behavior.
4. Verify Data Health reports the lane without making live network calls.
5. Document provider, freshness, and recovery rules before production use.

The catalog describes canonical lanes. The presence of a catalog entry does
not mean data is available, licensed, fresh, or suitable for trading.
