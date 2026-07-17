# OQP Package Ownership

`src/oqp/` is intentionally one level below `src/`. This is the standard
Python `src` layout: `src` is the import root and `oqp` is the installable
namespace. Flattening these folders would turn imports such as
`oqp.research` into top-level names and would break the native extension name
`oqp.native._quant_core`.

## Start Here

Choose the package that owns the business concept, expose a small public API,
and add a focused test in the matching `tests/` domain. The parent
[Source Layout guide](../README.md) explains installation and dependency
direction.

## Package Map

| Package | Owner |
| --- | --- |
| `accounts` | Canonical account, NAV, position, and event records. |
| `artifacts` | Transitional artifact helpers; new behavior needs a domain owner. |
| `brokers` | Broker-neutral contracts and IBKR adapters. |
| `commands` | Transitional CLI implementations; move under domain-owned CLI packages over time. |
| `config` | Settings, paths, credentials, and environment parsing. |
| `contracts` | Cross-domain strategy and artifact contracts. |
| `data` | Data contracts, taxonomy, instruments, vendors, and quality views. |
| `demo` | Deterministic broker-free fixtures, profile isolation, and doctor checks. |
| `domain` | Stable domain types with no dashboard dependency. |
| `execution` | Proposals, review gates, and order safety. |
| `intelligence` | Transitional generic engines being reassigned to accountable domains. |
| `investing` | Discretionary research and directional evidence. |
| `journal` | Transitional journal implementation; future owner is `ops`. |
| `market` | Transitional market-data analytics; future owner is `data.market`. |
| `native` | C++ acceleration source and Python bridge. |
| `ops` | Operational health and reporting services. |
| `options` | Option contracts, lifecycle, analytics, risk, and backtesting. |
| `paper_trading` | Transitional paper execution implementation; future owner is `execution.paper`. |
| `portfolio` | Portfolio construction, ingestion, and reporting. |
| `qmt_connector` | Transitional QMT adapter; future owner is `brokers.qmt`. |
| `research` | Factor contracts, backtesting, ML, regimes, diagnostics, and research artifacts. |
| `risk` | Independent risk measurement, breadth, limits, and scenarios. |
| `storage` | Transitional storage helpers; durable storage belongs to a domain. |
| `ui` | Shared presentation helpers and bilingual text catalogs. |
| `utils` | Transitional generic helpers; new helpers need a concrete owner. |

`research/daily_regimes` is a protected, path-sensitive research package. Move
or rename it only as a dedicated migration with release-lock and source-hash
tests updated in the same change.

## Migration Packages

The generic `intelligence`, `artifacts`, `storage`, and `utils` names are
migration surfaces, not preferred homes for new behavior. Useful code should
move to a concrete owner with its consumers and tests updated in the same
change; empty compatibility packages can then be removed.

## Dependency Rule

Foundational packages must not depend on presentation or orchestration layers.
In particular, `config` and `domain` cannot import `oqp.ui`, and packages outside
`ui` cannot import internal UI modules. Tests in `tests/architecture/test_package_boundaries.py`
keep these ownership constraints executable.

## Verification

```bash
oqp test smoke
python -m pytest -q tests/architecture
```
