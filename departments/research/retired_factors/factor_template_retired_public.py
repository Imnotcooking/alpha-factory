"""Retired public factor template.

Use this for GitHub-safe examples that teach the factor contract shape without
publishing active research edge. Keep the logic synthetic, educational, and
small enough to audit quickly.
"""

from __future__ import annotations

import pandas as pd

from oqp.research.factor_presets import (
    CROSS_SECTIONAL_DAILY_NEXT_OPEN as FACTOR_CONTRACT,
)


FACTOR_ID = "fac_template_retired_public"

# Public examples can declare pipeline policy too. This is intentionally simple:
# the factor computes a score; the runner decides whether to add Kelly/HRP.
EXECUTION_MODE_CONFIG = {
    "sizing_modules": ["kelly", "hrp"],
    "kelly_fraction": 0.5,
    "max_gross_leverage": 1.0,
    "max_weight_per_asset": 0.05,
}

FACTOR_METADATA = {
    "status": "retired_public_template",
    "native_market": "FUTURES_CN",
    "supported_markets": ["FUTURES_CN"],
    "experimental_markets": [],
    "required_fields": ["date", "ticker", "close"],
    "optional_fields": [],
    "uses_open_interest": False,
    "requires_shorting": False,
    "requires_continuous_contracts": False,
    "rebalance_frequency": "daily",
    "signal_horizon": "toy_example",
    "execution_style": "close_signal_next_open",
    "execution_mode": "risk_desk",
    "publication_note": "Public template only. Not a production trading signal.",
}


def compute_factor(data: pd.DataFrame, *, lookback: int = 5) -> pd.DataFrame:
    """Return a toy public score from trailing synthetic close momentum."""

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    required = {"date", "ticker", "close"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = data.loc[:, ["date", "ticker", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])

    frame["toy_momentum"] = frame.groupby("ticker")["close"].pct_change(lookback)
    frame["factor_score"] = frame.groupby("date")["toy_momentum"].transform(
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
