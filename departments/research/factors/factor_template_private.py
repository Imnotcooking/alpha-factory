"""Private factor recipe template.

Copy this file in the flat factor registry and rename the copy to
``fac_###_descriptive_name.py``. Keep this template generic and public-safe; it
should document the expected shape without encoding a live research edge.

Backtest runner expectations:
- expose ``FACTOR_ID``, ``FACTOR_METADATA``, ``FACTOR_CONTRACT``, and ``compute``.
- declare every ``compute`` input in ``FACTOR_PARAMETERS``; optimizer logic stays
  outside the factor file.
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
ECONOMIC_HYPOTHESIS = (
    "Replace this text with the causal or behavioral mechanism that should make "
    "the signal predict the declared forward return."
)
ECONOMIC_RATIONALE = ECONOMIC_HYPOTHESIS
SIGNAL_ORIENTATION = "higher_is_bullish"
EXPECTED_HOLDING_HORIZON = {
    "minimum": 1,
    "maximum": 5,
    "unit": "sessions",
    "rationale": "Replace with the horizon implied by the economic mechanism.",
}
KNOWN_LIMITATIONS = (
    "Replace with at least one concrete market, data, capacity, or regime limitation.",
)

FACTOR_CONTRACT = {
    **_BASE_FACTOR_CONTRACT,
    # Change this to the markets where this factor has been designed and tested.
    # Examples: ["FUTURES_CN"], ["EQUITY_US"], ["OPTIONS_US"], ["OPTIONS_CN"].
    "supported_markets": ["FUTURES_CN"],
}

# Signal timing is not inferred from the number of trades produced later.
# Declare when decisions are accepted and how targets persist between them.
TEMPORAL_POLICY = {
    "signal_frequency": "session_close",
    "decision_interval": 1,
    "decision_unit": "sessions",
    "holding_mode": "until_next_decision",
    "holding_unit": "sessions",
    "zero_signal_action": "exit",
}

FACTOR_METADATA = {
    "metadata_schema_version": 1,
    "component_type": "factor",
    "status": "private_template",
    "factor_family": "replace_me",
    "factor_subfamily": "replace_me",
    "native_market": "FUTURES_CN",
    "supported_markets": FACTOR_CONTRACT["supported_markets"],
    "experimental_markets": [],
    "required_fields": ["date", "ticker", "close"],
    "optional_fields": [],
    "uses_open_interest": False,
    "requires_shorting": False,
    "requires_continuous_contracts": False,
    "data_frequency": "daily",
    "signal_frequency": "daily_close",
    "rebalance_frequency": "daily",
    "signal_horizon": "replace_me",
    "execution_style": "close_signal_next_open",
    "execution_mode": "risk_desk",
    "portfolio_layer": "alpha_score",
    "deduplication_cohort": "replace_me",
    "cost_model": "runner_instrument_master_plus_configured_slippage",
    "legacy_ids": [],
    "publication_note": "Private recipe template. Replace before use.",
}

FACTOR_PARAMETERS = {
    "lookback": {
        "default": 20,
        "type": "int",
        "low": 5,
        "high": 120,
        "step": 5,
        "tunable": True,
    },
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
    result.attrs["temporal_policy_overrides"] = TEMPORAL_POLICY
    return result.reset_index(drop=True)


def _rank_to_unit_interval(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    return (ranks - 0.5) * 2.0
