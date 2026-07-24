from __future__ import annotations

from pathlib import Path

import pandas as pd

from oqp.research.factor_portfolios.data import (
    load_factor_portfolio_data,
    normalize_daily_session_rows,
    normalize_market_frame,
)


def test_factor_portfolio_loader_reads_csv_without_parquet_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "ticker": ["AU", "AU"],
            "open": [500.0, 501.0],
            "close": [501.0, 502.0],
            "volume": [1000.0, 1100.0],
        }
    ).to_csv(source, index=False)

    bundle = load_factor_portfolio_data(
        source,
        market_vertical="FUTURES_CN",
    )

    assert bundle.source_path == source.resolve()
    assert bundle.crisis_period is None
    assert list(bundle.frame["ticker"]) == ["AU", "AU"]


def test_market_frame_normalizes_open_interest_aliases() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "ticker": ["AU"],
            "close": [501.0],
            "open_oi": ["1200"],
        }
    )

    normalized = normalize_market_frame(frame)

    assert normalized.loc[0, "open_interest"] == 1200.0
    assert normalized.loc[0, "open_oi"] == "1200"
    assert normalized.loc[0, "oi"] == 1200.0


def test_daily_session_normalization_prefers_completed_ohlcv_bar() -> None:
    frame = pd.DataFrame(
        {
            "date": [
                "2026-04-23 00:00:00",
                "2026-04-23 14:59:59",
            ],
            "ticker": ["T", "T"],
            "open": [108.801, 108.739],
            "high": [108.829, 108.739],
            "low": [108.716, 108.739],
            "close": [108.739, 108.739],
            "volume": [87_192.0, 0.0],
        }
    )

    normalized = normalize_daily_session_rows(frame)

    assert len(normalized) == 1
    assert normalized.loc[0, "open"] == 108.801
    assert normalized.loc[0, "volume"] == 87_192.0
    assert normalized.loc[0, "date"] == pd.Timestamp("2026-04-23")
    assert normalized.attrs["daily_session_normalization"][
        "collapsed_session_ticker_groups"
    ] == 1
