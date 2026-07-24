"""Template for a private causal strategy router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


ROUTER_ID = "rtr_NNN_Descriptive_Name"
ROUTER_METADATA = {
    "name": "Descriptive router name",
    "component_type": "router",
    "supported_markets": ["FUTURES_CN"],
    "frequency": "daily",
    "status": "hypothesis",
    "economic_claim": (
        "State which observable score predicts sleeve A minus sleeve B over the "
        "next executable holding period."
    ),
    "economic_mechanism": "State why that relative advantage should exist.",
    "score_target": "next_holding_period_net_return_sleeve_a_minus_sleeve_b",
    "score_orientation": "higher_favors_a",
    "threshold": 0.0,
    "hypothesis_frozen_on": "YYYY-MM-DD",
    "score_source_fingerprint": "required-before-testing",
    "oracle_role": "unattainable_upper_bound_only",
}
ROUTER_CONTRACT = {
    "state_col": "state",
    "decision_date_col": "decision_date",
    "effective_date_col": "effective_date",
    "sleeve_col": "sleeve_id",
    "allocation_col": "allocation",
    "decision_lag_periods": 1,
    "allow_partial_allocation": False,
    "allow_negative_allocation": False,
    "supported_markets": ["FUTURES_CN"],
}
ROUTER_PARAMETERS = {
    "decision_threshold": {
        "default": 0.0,
        "type": "float",
        "low": -1.0,
        "high": 1.0,
        "step": 0.05,
        "tunable": True,
        "description": "Frozen score boundary above which allocation favors sleeve A.",
    },
    "switch_buffer": {
        "default": 0.0,
        "type": "float",
        "low": 0.0,
        "high": 0.25,
        "step": 0.01,
        "tunable": True,
        "description": "Hysteresis buffer used to reduce threshold churn.",
    },
}


def route(
    states: pd.DataFrame,
    *,
    sleeve_ids: Sequence[str],
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    """Return decision_date, effective_date, sleeve_id, and allocation."""

    raise NotImplementedError
