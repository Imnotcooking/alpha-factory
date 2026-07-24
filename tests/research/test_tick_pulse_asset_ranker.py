import numpy as np
import pandas as pd

from oqp.research.tick_pulse import filter_ranked_assets, rank_daily_asset_volatility


def _synthetic_daily_universe(days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="B")
    rows = []
    specs = [
        ("黄金(au)[指数]", 0.018, 900000, 500000),
        ("螺纹钢(rb)[指数]", 0.008, 1200000, 700000),
        ("10年期国债(T)[指数]", 0.002, 300000, 200000),
    ]
    for ticker, daily_amp, volume, oi in specs:
        base = 100.0
        for i, date in enumerate(dates):
            close = base * np.exp(daily_amp * np.sin(i / 3.0) + 0.0005 * i)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close * 0.995,
                    "high": close * (1 + daily_amp),
                    "low": close * (1 - daily_amp),
                    "close": close,
                    "volume": volume + i,
                    "oi": oi + i,
                }
            )
    return pd.DataFrame(rows)


def test_rank_daily_asset_volatility_orders_high_vol_contract_first():
    ranked = rank_daily_asset_volatility(
        _synthetic_daily_universe(),
        lookback_days=126,
        min_observations=60,
    )

    assert not ranked.empty
    assert ranked.sort_values("recent_ann_vol", ascending=False).iloc[0]["base_symbol"] == "au"
    assert {"download_priority_score", "vol_rank", "sector", "coverage"}.issubset(ranked.columns)
    assert ranked["coverage"].between(0, 1).all()


def test_filter_ranked_assets_applies_sector_volume_and_top_n():
    ranked = rank_daily_asset_volatility(
        _synthetic_daily_universe(),
        lookback_days=126,
        min_observations=60,
    )

    filtered = filter_ranked_assets(
        ranked,
        sectors=["贵金属", "黑色"],
        min_volume=800000,
        top_n=1,
        sort_by="recent_ann_vol",
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["sector"] in {"贵金属", "黑色"}
    assert filtered.iloc[0]["avg_daily_volume"] >= 800000
