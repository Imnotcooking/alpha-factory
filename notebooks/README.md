# Research Notebooks

`notebooks/` contains reproducible research narratives, experiment records,
figures, and manuscript evidence. A notebook is evidence and exploration, not
the canonical implementation of a reusable engine.

## Start Here

Each committed project should provide its own README with the hypothesis,
dataset contract, protocol, and reproduction command.

Install notebook support separately from the dashboard runtime:

```bash
python -m pip install -e ".[research,notebooks,dev]"
```

## Rules

- Record the dataset identity, universe, date range, parameters, random seed,
  and code revision required to reproduce a result.
- Read local datasets through canonical runtime paths; do not copy vendor data
  into a notebook directory.
- Move reusable calculations into `src/oqp/` and cover them with tests.
- Write generated tables, models, and large figures to `runtime/artifacts/` or
  another ignored project output path.
- Keep private research projects private until their data, factors, and outputs
  have passed the public boundary review.
- A notebook should call a frozen experiment or package API once a protocol is
  promoted; it should not become an alternative production pipeline.

## Promotion Path

1. Explore and record the hypothesis.
2. Freeze dataset and evaluation assumptions.
3. Extract reusable logic into the owning package.
4. Add synthetic or sanitized tests.
5. Run the script or package-owned protocol.
6. Retain the notebook as explanatory evidence.

Research ownership and publication rules are in the
[Research Department guide](../departments/research/README.md).
