# Source Layout

`src/` is the Python packaging root. The installable product namespace is
`oqp`, so reusable imports consistently begin with `oqp.` in local development,
tests, servers, and built distributions.

## Start Here

Install the package in editable mode, then verify it through the CLI:

```bash
python -m pip install -e ".[dashboard,research,dev]"
oqp doctor
```

The detailed package inventory is in [OQP Package Ownership](oqp/README.md).

## Why The Extra `oqp` Level Exists

The `src` layout prevents Python from importing arbitrary repository files only
because the current directory is the project root. The `oqp` namespace avoids
collisions with third-party modules and gives every domain one stable public
import path.

Do not flatten packages directly into `src/`.

## Dependency Direction

- Foundational contracts and configuration must not import dashboards.
- Domain packages may expose services and data structures to applications.
- Applications and scripts may orchestrate package APIs.
- Department documents describe ownership but are never runtime dependencies.
- Cross-domain imports should use public package exports where practical.

## Verification

```bash
oqp test smoke
```
