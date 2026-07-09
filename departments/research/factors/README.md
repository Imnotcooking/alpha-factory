# Private Factor Recipes

This folder holds live/private factor recipes. Files under strategy-family
subfolders are private by default and should not be published.

Use `factor_template_private.py` as the starting point for a new live factor:

1. Copy it into the right family folder, such as `daily_signals/` or
   `tick_pulse/`.
2. Rename the copy to a real `fac_*.py` identifier.
3. Fill out `FACTOR_ID`, `FACTOR_METADATA`, and `FACTOR_CONTRACT`.
4. Declare market eligibility with `supported_markets`, for example
   `["FUTURES_CN"]`, `["EQUITY_US"]`, or `["OPTIONS_US"]`.
5. Expose a deterministic `compute(data: pd.DataFrame) -> pd.DataFrame`
   function. The backtest runner calls `compute`, and the returned frame must
   contain `date`, `ticker`, and the contract's `alpha_signal_col`.
   For option factors, prefer `OPTIONS_DAILY_DIRECTIONAL` from
   `oqp.research.factor_presets`; return an underlying-level direction score
   in `factor_score`, and the options CLI will convert sign changes into
   option entry/exit events.
6. Keep allocator choices out of the factor math. If a factor needs a preferred
   execution policy, declare it as runner config:

   ```python
   EXECUTION_MODE_CONFIG = {
       "sizing_modules": ["kelly", "hrp"],
       "kelly_fraction": 0.5,
       "max_gross_leverage": 1.0,
       "max_weight_per_asset": 0.05,
   }
   ```

   `sizing_modules` is optional pipeline infrastructure. Use `["kelly"]`,
   `["hrp"]`, `["kelly", "hrp"]`, or `[]`/`"none"` for raw factor signal plus
   portfolio caps. CLI flags such as `--sizing_modules none` override the
   template for experiments.
7. Keep generated data, model files, and backtest output in `runtime/`.
8. Add a synthetic test before promoting any reusable logic.

Do not move live recipes directly into `src/oqp`. Promote only reusable,
sanitized, testable infrastructure there.
