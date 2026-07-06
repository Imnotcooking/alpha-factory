from __future__ import annotations

import pandas as pd

from oqp.intelligence.regime_engine import MarketHMM, build_macro_hmm_emissions


def test_promoted_regime_hmm_is_available_without_lab_wrapper() -> None:
    assert MarketHMM.__module__ == "oqp.intelligence.regime_engine.hmm_regime"


def test_build_macro_hmm_emissions_aggregates_by_date_after_asset_volatility() -> None:
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    feature_matrix = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": list(range(100, 125)) + list(range(200, 225)),
        }
    )

    emissions = build_macro_hmm_emissions(feature_matrix, rolling_vol_window=3)

    assert list(emissions.columns) == ["date", "returns", "volatility"]
    assert emissions["date"].is_monotonic_increasing
    assert emissions["returns"].notna().all()
    assert emissions["volatility"].notna().all()
