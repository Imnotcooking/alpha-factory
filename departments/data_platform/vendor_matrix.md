# Vendor Matrix

Last reviewed: 2026-07-17

This matrix records intended coverage and adapter readiness. Credential
presence in Data Health confirms configuration only; it is not a live endpoint
test and does not prove entitlement, freshness, or schema completeness.

| Provider | Intended coverage | Executable home | Credential names | Current posture |
| --- | --- | --- | --- | --- |
| Local/static files | CN futures, CN equities, CN options | `runtime/data/`, `src/oqp/data/runtime_paths.py` | None | Active where files exist. |
| FMP | US/HK equities, fundamentals, ratings, news | `src/oqp/data/vendors/fmp.py` | `FMP_API_KEY` | Daily US-equity research materializer active; other endpoints still require endpoint-specific validation. |
| Massive | US options and optional historical flat files | `src/oqp/data/vendors/massive.py` | `MASSIVE_API_KEY`, flat-file access keys | Current option snapshots can be frozen; historical backtests still require time-aligned historical files and entitlement. |
| Yahoo Finance | Public equity history and fallback research data | `src/oqp/data/vendors/yahoo.py` | None | Best-effort fallback, not an execution-grade source. |
| Polygon alias | Legacy name for Massive-compatible options snapshots | `src/oqp/data/vendors/polygon.py` | Massive/options key aliases | Compatibility only; do not register a second data lineage. |
| QMT | Planned CN equity/options market data and execution | Broker connector plus future data adapter | `QMT_API_TOKEN` and connector settings | Planned until account and lane validation are complete. |
| Wind | Planned licensed CN reference and market data | No production adapter yet | `WIND_API_KEY` | Planned. |

## Source Selection

1. Every generated dataset must record its actual provider and request or file
   scope in its manifest.
2. A fallback must never silently replace the primary provider. Record the
   provider change and rerun quality checks.
3. Do not merge observations from vendors without explicit timestamp,
   adjustment, currency, and symbol-normalization rules.
4. API caches are disposable runtime state. The reproducible contract is the
   request configuration, source identity, retrieval time, code version, and
   output hash.
5. Never write credentials, signed URLs, account identifiers, or raw response
   bodies into this department folder.

## Research Materialization

Historical backtests must not call a vendor API directly. Use
`python -m oqp.commands.api_dataset_materialize` to publish a bundle under
`runtime/data/api_materialized/`. Each bundle contains a sanitized request,
compressed raw response, normalized Parquet data, a quality report, and a
dataset-manifest pointer. A refresh creates a new immutable version.

FMP daily-equity bundles are structurally eligible for historical research,
while the manifest separately records whether a point-in-time universe was
supplied. Massive REST chain snapshots are current-only and are rejected by
the historical loader; use them for prospective collection, live research, or
paper trading until a historical option panel is materialized.
