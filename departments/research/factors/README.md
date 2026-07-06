# Private Factor Recipes

This folder holds live/private factor recipes. Files under strategy-family
subfolders are private by default and should not be published.

Use `factor_template_private.py` as the starting point for a new live factor:

1. Copy it into the right family folder, such as `daily_signals/` or
   `tick_pulse/`.
2. Rename the copy to a real `fac_*.py` identifier.
3. Fill out `FACTOR_ID`, `FACTOR_METADATA`, and `FACTOR_CONTRACT`.
4. Keep generated data, model files, and backtest output in `runtime/`.
5. Add a synthetic test before promoting any reusable logic.

Do not move live recipes directly into `src/oqp`. Promote only reusable,
sanitized, testable infrastructure there.
