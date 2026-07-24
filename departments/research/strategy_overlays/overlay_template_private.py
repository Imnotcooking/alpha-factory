"""Template for a private causal strategy risk overlay."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


OVERLAY_ID = "ovl_NNN_Descriptive_Name"
OVERLAY_METADATA = {
    "name": "Descriptive strategy risk overlay",
    "component_type": "strategy_risk_overlay",
    "supported_markets": ["FUTURES_CN"],
    "frequency": "daily",
    "status": "hypothesis",
    "economic_rationale": "State why strategy exposure should change causally.",
}
OVERLAY_CONTRACT = {
    "date_col": "date",
    "ticker_col": "ticker",
    "price_col": "close",
    "source_weight_col": "final_target_weight",
    "output_weight_col": "final_target_weight",
    "decision_time": "daily_close",
    "effective_time": "next_open",
    "scope": "portfolio_scalar",
    "allow_sign_flip": False,
    "allow_gross_increase": False,
    "supported_markets": ["FUTURES_CN"],
}
OVERLAY_PARAMETERS = {
    "activation_threshold": {
        "default": 0.0,
        "type": "float",
        "low": -1.0,
        "high": 1.0,
        "step": 0.1,
        "tunable": True,
        "description": "Example only; replace with the overlay's causal threshold.",
    },
}


def apply(
    targets: pd.DataFrame,
    *,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Return the same position grid with an audited target-weight update."""

    raise NotImplementedError
