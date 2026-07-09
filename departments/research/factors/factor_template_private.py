"""Private factor recipe template.

Copy this file into a strategy-family folder and rename the copy to
``fac_###_descriptive_name.py``. Keep this template generic and public-safe; it
should document the expected shape without encoding a live research edge.

Backtest runner expectations:
- expose ``FACTOR_ID``, ``FACTOR_METADATA``, ``FACTOR_CONTRACT``, and ``compute``.
- ``compute(data)`` must return at least ``date``, ``ticker``, and the
  ``FACTOR_CONTRACT["alpha_signal_col"]`` column.
- ``supported_markets`` is a positive allowlist. Anything not listed is blocked.
- allocator modules such as Kelly and HRP are optional execution-stage config,
  not part of the factor alpha calculation.
"""

from __future__ import annotations

import pandas as pd

from oqp.research.factor_presets import (
    CROSS_SECTIONAL_DAILY_NEXT_OPEN as _BASE_FACTOR_CONTRACT,
)


FACTOR_ID = "fac_template_private"
FACTOR_NAME = "Private Factor Template"
CATEGORY = "Template"
COMPLEXITY = 1

FACTOR_CONTRACT = {
    **_BASE_FACTOR_CONTRACT,
    # Change this to the markets where this factor has been designed and tested.
    # Examples: ["FUTURES_CN"], ["EQUITY_US"], ["OPTIONS_US"], ["OPTIONS_CN"].
    "supported_markets": ["FUTURES_CN"],
}

# Optional pipeline policy consumed by the backtest runner. Keep the alpha logic
# inside compute(); keep sizing/allocation choices declarative here or pass them
# from the CLI with --sizing_modules, --kelly_fraction, and leverage caps.
EXECUTION_MODE_CONFIG = {
    # Default risk_desk behavior is ["kelly", "hrp"]. Use [] or "none" to test
    # the raw factor signal with only portfolio caps.
    "sizing_modules": ["kelly", "hrp"],
    "kelly_fraction": 0.5,
    "max_gross_leverage": 1.0,
    "max_weight_per_asset": 0.05,
}

FACTOR_METADATA = {
    "status": "private_template",
    "native_market": "FUTURES_CN",
    "supported_markets": FACTOR_CONTRACT["supported_markets"],
    "experimental_markets": [],
    "required_fields": ["date", "ticker", "close"],
    "optional_fields": [],
    "uses_open_interest": False,
    "requires_shorting": False,
    "requires_continuous_contracts": False,
    "rebalance_frequency": "daily",
    "signal_horizon": "replace_me",
    "execution_style": "close_signal_next_open",
    "execution_mode": "risk_desk",
    "publication_note": "Private recipe template. Replace before use.",
}


def compute(data: pd.DataFrame, *, lookback: int = 20) -> pd.DataFrame:
    """Backtest-runner entrypoint.

    Keep this function deterministic: no file IO, no API calls, no mutation of
    global state. Load data in the runner; compute only the factor signal here.
    """

    return compute_factor(data, lookback=lookback)


def compute_factor(data: pd.DataFrame, *, lookback: int = 20) -> pd.DataFrame:
    """Return a placeholder cross-sectional score from close-to-close returns."""

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    required = {"date", "ticker", "close"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = data.loc[:, ["date", "ticker", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])

    frame["raw_signal"] = frame.groupby("ticker")["close"].pct_change(lookback)
    frame["factor_score"] = frame.groupby("date")["raw_signal"].transform(
        _rank_to_unit_interval
    )
    frame["factor_score"] = frame["factor_score"].fillna(0.0)

    result = frame.loc[:, ["date", "ticker", "factor_score"]].sort_values(
        ["date", "ticker"]
    )
    result.attrs["factor_id"] = FACTOR_ID
    result.attrs["factor_metadata"] = FACTOR_METADATA
    result.attrs["factor_contract"] = FACTOR_CONTRACT
    result.attrs["execution_mode_config"] = EXECUTION_MODE_CONFIG
    return result.reset_index(drop=True)


def _rank_to_unit_interval(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    return (ranks - 0.5) * 2.0
