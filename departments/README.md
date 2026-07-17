# Department Ownership Map

`departments/` contains governance, policies, operating contracts, and
runbooks. It describes who owns a capability and how it should be operated; it
is not a second source-code tree.

## Start Here

Choose the department that owns the decision you are changing:

| Department | Owns | Executable code normally lives in |
| --- | --- | --- |
| [Data Platform](data_platform/README.md) | Data contracts, vendors, lineage, quality, and storage policy | `src/oqp/data/` |
| [Middle Office](middle_office/README.md) | Account truth, reconciliation, and valuation control | `src/oqp/accounts/`, `src/oqp/portfolio/` |
| [Platform](platform/README.md) | Deployment, schedulers, runtime operations, and repository hygiene | `src/oqp/config/`, `src/oqp/ops/`, `scripts/platform/` |
| [Research](research/README.md) | Factor governance, research protocols, and public/private boundaries | `src/oqp/research/`, `src/oqp/native/` |
| [Risk](risk/README.md) | Risk appetite, limits, scenarios, and model-risk review | `src/oqp/risk/`, `src/oqp/options/` |
| [Trading](trading/README.md) | Order policy, paper execution, approval, and submission controls | `src/oqp/execution/`, `src/oqp/paper_trading/` |

## Rules

- Put reusable implementation in `src/oqp/`, not in a department folder.
- Put operator entrypoints in `scripts/`; keep them thin.
- Put generated evidence, databases, market data, and logs in `runtime/`.
- Never commit credentials, broker exports, live account state, or private
  factor implementations.
- Cross-department contracts must identify a canonical owner rather than being
  copied into multiple folders.

## Change Checklist

1. Identify the owning package and department.
2. Update the reusable implementation and focused tests.
3. Update the owning policy or runbook when behavior changes.
4. Record runtime paths and artifact provenance.
5. Recheck public/private and execution-safety boundaries.

The system-wide dependency and safety boundaries are defined in
[ARCHITECTURE.md](../ARCHITECTURE.md).
