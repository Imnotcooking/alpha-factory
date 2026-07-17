# Dependency Management

OQP uses `pyproject.toml` as the canonical declaration of package metadata,
runtime dependencies, optional extras, and the `oqp` console command.

## Start Here

For the repository's complete development environment:

```bash
python -m pip install -r requirements.txt
```

For the smallest broker-free dashboard environment:

```bash
python -m pip install -e ".[dashboard,research,dev]"
```

## File Roles

| File | Role |
| --- | --- |
| `../pyproject.toml` | Canonical package dependencies and optional extras |
| `../requirements.txt` | Full installation entrypoint or compatibility manifest |

## Rules

- Add direct dependencies to the appropriate `pyproject.toml` dependency group.
- Keep broker, notebook, dashboard, and research-heavy packages optional when
  the core package does not require them.
- Add a constraints file only when its update and verification policy is
  documented and committed with it.
- Never depend on a package merely because it happens to exist in a developer's
  global Python environment.
- The optional native extension may accelerate backtests; the Python backend
  must remain importable when the extension is unavailable.

## Verification

```bash
oqp doctor
oqp test smoke
```
