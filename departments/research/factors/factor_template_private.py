"""Private factor recipe template.

Copy this file into a strategy-family folder and rename the copy to
``fac_###_descriptive_name.py``. Keep this template generic and public-safe; it
should document the expected shape without encoding a live research edge.
"""

from __future__ import annotations

import pandas as pd

from oqp.research.factor_presets import (
    CROSS_SECTIONAL_DAILY_NEXT_OPEN as FACTOR_CONTRACT,
)


FACTOR_ID = "fac_template_private"

FACTOR_METADATA = {
    "status": "private_template",
    "native_market": "REPLACE_ME",
    "suitable_markets": ["REPLACE_ME"],
    "experimental_markets": [],
    "unsupported_markets": [],
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
    return result.reset_index(drop=True)


def _rank_to_unit_interval(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    return (ranks - 0.5) * 2.0
