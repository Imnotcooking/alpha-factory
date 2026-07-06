"""Reusable factor contract presets.

Each factor module should still declare ``FACTOR_CONTRACT`` explicitly by
importing one of these presets. The presets keep the declarations consistent
without hiding the factor's intended evaluation and execution semantics.
"""

CROSS_SECTIONAL_DAILY_ALREADY_LAGGED = {
    "evaluation_geometry": "cross_sectional",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "already_lagged",
    "return_assumption": "close_signal_next_open_to_close",
}

CROSS_SECTIONAL_DAILY_NEXT_OPEN = {
    "evaluation_geometry": "cross_sectional",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "next_open",
    "return_assumption": "close_signal_next_open_to_close",
}

TIME_SERIES_DAILY_ALREADY_LAGGED = {
    "evaluation_geometry": "time_series",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "already_lagged",
    "return_assumption": "close_signal_next_open_to_close",
}

TIME_SERIES_DAILY_NEXT_OPEN = {
    "evaluation_geometry": "time_series",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "next_open",
    "return_assumption": "close_signal_next_open_to_close",
}

TIME_SERIES_INTRADAY_FACTOR_SCORE_NEXT_BAR = {
    "evaluation_geometry": "time_series",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "next_bar",
    "return_assumption": "bar_signal_next_bar",
}

TIME_SERIES_INTRADAY_SIGNAL_NEXT_BAR = {
    "evaluation_geometry": "time_series",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "signal",
    "execution_weight_col": "signal",
    "execution_lag": "next_bar",
    "return_assumption": "bar_signal_next_bar",
}

TIME_SERIES_INTRADAY_FACTOR_SCORE_ALREADY_LAGGED = {
    "evaluation_geometry": "time_series",
    "execution_mode": "risk_desk",
    "alpha_signal_col": "factor_score",
    "execution_weight_col": "factor_score",
    "execution_lag": "already_lagged",
    "return_assumption": "bar_signal_next_bar",
}

REGIME_DIAGNOSTIC_DAILY_PNL = {
    "evaluation_geometry": "time_series",
    "execution_mode": "direct",
    "alpha_signal_col": "A_sma_uncond",
    "execution_weight_col": "A_sma_uncond",
    "execution_lag": "custom",
    "return_assumption": "custom_forward_return",
}
