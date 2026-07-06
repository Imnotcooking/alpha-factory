# Retired Public Factors

This folder is the public lane for retired/sanitized alpha research examples.
Code here should teach the framework shape without exposing live research edge.

## Rules

- Use synthetic or broadly known toy logic only.
- Do not copy files directly from `departments/research/factors/`.
- Do not include real performance logs, parameter sweeps, account data, or
  vendor data.
- Mark examples as retired, educational, or synthetic.
- Keep examples small enough that a reader can understand the contract without
  mistaking it for a production trading recipe.

Start from `factor_template_retired_public.py` when preparing a GitHub-safe
retired factor example.

Live factor implementations remain private in `departments/research/factors/`
by default.
