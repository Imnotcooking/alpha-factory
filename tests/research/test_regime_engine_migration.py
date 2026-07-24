from __future__ import annotations

import pickle

import pandas as pd

from oqp.research.ml.regimes import (
    MacroHMMTrainingConfig,
    MarketHMM,
    build_macro_hmm_emissions,
)


def test_promoted_regime_hmm_is_available_without_lab_wrapper() -> None:
    assert (
        MarketHMM.__module__
        == "oqp.research.ml.regimes.legacy.hmmlearn_models"
    )


def test_future_macro_hmm_pickles_use_the_canonical_module() -> None:
    config = MacroHMMTrainingConfig()
    payload = pickle.dumps(config)

    assert pickle.loads(payload) == config
    assert b"oqp.research.ml.regimes.legacy.macro_training" in payload


def test_market_hmm_prepares_missing_emissions_without_inventing_state() -> None:
    model = MarketHMM(n_components=2)
    emissions = model._prepare_emissions(
        pd.DataFrame(
            {
                "returns": [0.01, None, -0.02],
                "volatility": [0.10, 0.20, None],
            }
        )
    )

    assert emissions.shape == (3, 2)
    assert float(emissions[1, 0]) == 0.0
    assert float(emissions[2, 1]) == 0.0


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
