import numpy as np
import pandas as pd

from oqp.risk import (
    MarketBreadthConfig,
    build_research_window_table,
    compute_concentration_breadth,
    compute_directional_breadth,
    compute_market_structure,
    compute_volatility_map,
)


def _market_panel(days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="B")
    rows = []
    definitions = [
        ("A", "Alpha", "Bank", 100.0, 1.004, 700.0),
        ("B", "Beta", "Bank", 80.0, 1.002, 100.0),
        ("C", "Gamma", "Tech", 120.0, 0.999, 100.0),
        ("D", "Delta", "Tech", 60.0, 1.006, 100.0),
    ]
    for ticker, name, sector, initial, growth, market_cap in definitions:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "close": initial * growth**idx,
                    "volume": 1000 + idx,
                    "market_cap": market_cap,
                }
            )
    return pd.DataFrame(rows)


def test_directional_breadth_counts_active_assets_and_sector_participation():
    result = compute_directional_breadth(_market_panel())

    latest = result["daily"].iloc[-1]
    assert latest["active_assets"] == 4
    assert latest["advancers"] == 3
    assert latest["decliners"] == 1
    assert np.isclose(latest["directional_breadth"], 0.5)

    tech = result["by_sector"].query("sector == 'Tech'").iloc[-1]
    assert tech["advancers"] == 1
    assert tech["decliners"] == 1
    assert np.isclose(tech["directional_breadth"], 0.0)


def test_market_structure_resolves_chinese_futures_labels_through_base_symbols():
    frame = _market_panel().drop(columns=["sector"])
    frame["ticker"] = frame["ticker"].map(
        {"A": "铜(cu)[指数]", "B": "铝(al)[指数]", "C": "原油(sc)[指数]", "D": "PTA(TA)[指数]"}
    )
    result = compute_market_structure(
        frame,
        asset_class="FUTURES_CN",
        config=MarketBreadthConfig(volatility_lookback=20, minimum_observations=10),
    )

    assert set(result["volatility"]["by_sector"]["sector"]) == {"有色", "能源", "化工"}


def test_concentration_breadth_reports_weight_source_and_effective_assets():
    result = compute_concentration_breadth(_market_panel(), asset_class="EQUITY_CN")

    assert result["weight_source"] == "market_cap"
    latest = result["daily"].iloc[-1]
    assert np.isclose(latest["hhi"], 0.52)
    assert np.isclose(latest["effective_assets"], 1.0 / 0.52)
    assert np.isclose(latest["top_5_share"], 1.0)
    assert result["latest_assets"].iloc[0]["ticker"] == "A"


def test_concentration_breadth_uses_equal_weight_fallback_transparently():
    frame = _market_panel().drop(columns=["market_cap", "volume"])
    result = compute_concentration_breadth(frame, asset_class="EQUITY_CN")

    assert result["weight_source"] == "equal_weight_fallback"
    assert np.isclose(result["daily"].iloc[-1]["effective_assets"], 4.0)


def test_volatility_map_aggregates_assets_and_sectors():
    result = compute_volatility_map(
        _market_panel(),
        lookback=20,
        minimum_observations=10,
    )

    assert set(result["by_asset"]["ticker"]) == {"A", "B", "C", "D"}
    assert set(result["by_sector"]["sector"]) == {"Bank", "Tech"}
    assert result["by_sector"]["assets"].sum() == 4
    assert not result["sector_timeline"].empty
    assert not result["market_timeline"].empty


def test_market_structure_and_research_windows_join_all_lenses():
    structure = compute_market_structure(
        _market_panel(days=80),
        asset_class="EQUITY_CN",
        config=MarketBreadthConfig(volatility_lookback=20, minimum_observations=10),
    )
    risk = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"]),
            "breadth_regime": ["Low", "Normal", "High"],
        }
    )
    windows = build_research_window_table(
        structure["directional"]["daily"],
        structure["concentration"]["daily"],
        structure["volatility"]["market_timeline"],
        risk,
    )

    assert not windows.empty
    assert {
        "direction_state",
        "concentration_state",
        "volatility_state",
        "risk_state",
        "research_use_en",
        "research_use_zh",
    }.issubset(windows.columns)
    low_risk = windows[windows["risk_state"].eq("Low")]
    assert not low_risk.empty
    assert low_risk.iloc[0]["research_use_en"].startswith("Structural stress")


def test_constant_equal_weight_breadth_is_not_mislabeled_as_concentrated():
    structure = compute_market_structure(
        _market_panel(days=80).drop(columns=["market_cap", "volume"]),
        asset_class="EQUITY_CN",
        config=MarketBreadthConfig(volatility_lookback=20, minimum_observations=10),
    )
    windows = build_research_window_table(
        structure["directional"]["daily"],
        structure["concentration"]["daily"],
        structure["volatility"]["market_timeline"],
    )

    assert windows["concentration_state"].eq("Normal concentration").all()
