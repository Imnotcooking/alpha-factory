"""Template for declaring factor market suitability.

Copy the FACTOR_METADATA shape into real fac_*.py modules. This file is not
picked up by the promotion inventory because it does not start with fac_.
"""

FACTOR_METADATA = {
    "native_market": "FUTURES_CN",
    "suitable_markets": ["FUTURES_CN"],
    "experimental_markets": ["EQUITY_US", "FUTURES_US"],
    "unsupported_markets": ["OPTIONS_US"],
    "required_fields": ["date", "ticker", "open", "close"],
    "optional_fields": ["volume", "open_interest"],
    "uses_open_interest": False,
    "requires_shorting": True,
    "requires_continuous_contracts": True,
    "rebalance_frequency": "daily",
    "signal_horizon": "medium_term",
    "execution_style": "close_signal_next_open",
    "execution_mode": "risk_desk",  # risk_desk | direct | statarb
    "split_policy": {
        "purge_periods": 0,
        "embargo_periods": 0,
        "purge_unit": "auto",  # auto | days | timestamps | rows
    },
}

FACTOR_CONTRACT = {
    "evaluation_geometry": "time_series",  # time_series | cross_sectional
    "execution_mode": "risk_desk",  # risk_desk | direct | statarb
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "next_open",  # same_bar | next_bar | next_open | already_lagged | custom
    "return_assumption": "close_signal_next_open_to_close",
}
