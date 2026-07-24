from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.risk import MarketRiskConfig, compute_market_risk_decomposition


def _price_history() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=180, freq="B")
    market_returns = rng.normal(0.0004, 0.011, len(dates))
    asset_a_returns = 1.5 * market_returns + rng.normal(0.0, 0.004, len(dates))
    asset_b_returns = 0.5 * market_returns + rng.normal(0.0, 0.009, len(dates))
    rows: list[pd.DataFrame] = []
    for symbol, returns in {
        "QQQ": market_returns,
        "AAA": asset_a_returns,
        "BBB": asset_b_returns,
    }.items():
        prices = 100.0 * np.cumprod(1.0 + returns)
        rows.append(pd.DataFrame({"symbol": symbol, "date": dates, "close": prices}))
    return pd.concat(rows, ignore_index=True)


def test_market_risk_decomposition_reports_beta_risk_and_missing_coverage() -> None:
    exposure = pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB", "CCC"],
            "Weight": [0.50, 0.30, 0.20],
            "Economic Exposure": [50_000.0, 30_000.0, 20_000.0],
        }
    )

    result = compute_market_risk_decomposition(
        exposure,
        _price_history(),
        config=MarketRiskConfig(min_observations=60, min_covered_weight=0.75),
    )

    assert result["status"] == "live"
    assert result["covered_weight"] == 0.8
    assert 0.80 < result["portfolio_beta"] < 1.00
    assert result["observations"] >= 60
    assert result["total_volatility"] > 0
    assert result["systematic_volatility"] > 0
    assert result["idiosyncratic_volatility"] > 0
    assert np.isclose(
        result["systematic_risk_share"] + result["idiosyncratic_risk_share"],
        1.0,
    )
    missing = result["positions"].set_index("Symbol").loc["CCC"]
    assert missing["Status"] == "insufficient history"
    assert pd.isna(missing["Beta"])


def test_market_risk_decomposition_withholds_beta_when_coverage_is_too_low() -> None:
    exposure = pd.DataFrame(
        {
            "Symbol": ["AAA", "CCC"],
            "Weight": [0.40, 0.60],
        }
    )

    result = compute_market_risk_decomposition(
        exposure,
        _price_history(),
        config=MarketRiskConfig(min_observations=60, min_covered_weight=0.80),
    )

    assert result["status"] == "insufficient_history"
    assert pd.isna(result["portfolio_beta"])
    assert result["covered_weight"] == 0.4
