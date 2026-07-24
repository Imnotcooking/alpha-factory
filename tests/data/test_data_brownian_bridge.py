from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.data import (
    QUALITY_BRIDGE_SYNTHETIC,
    QUALITY_FRESH,
    QUALITY_STALE_EXPIRED,
    build_brownian_bridge_view,
)


def test_brownian_bridge_fills_closed_gap_with_auditable_synthetic_flags() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )

    view = build_brownian_bridge_view(
        raw,
        calendar=pd.date_range("2026-01-01", periods=3, freq="D"),
        max_gap_bars=2,
        sigma_floor=0.0,
    )

    day1 = view[view["date"].eq(pd.Timestamp("2026-01-01"))].iloc[0]
    day2 = view[view["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    day3 = view[view["date"].eq(pd.Timestamp("2026-01-03"))].iloc[0]

    assert day1["close"] == 100.0
    assert day3["close"] == 121.0
    assert np.isclose(day2["close"], 110.0)
    assert bool(day1["is_fresh"]) is True
    assert bool(day2["is_synthetic"]) is True
    assert day2["fill_method"] == "brownian_bridge"
    assert day2["quality_state"] == QUALITY_BRIDGE_SYNTHETIC
    assert day2["bridge_step"] == 1
    assert day2["bridge_steps"] == 1
    assert view.attrs["risk_imputation"] == "brownian_bridge"


def test_brownian_bridge_is_reproducible_for_seeded_risk_paths() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-04", "2026-01-05"]),
            "ticker": ["A", "A", "A"],
            "close": [100.0, 121.0, 115.0],
        }
    )
    calendar = pd.date_range("2026-01-01", periods=5, freq="D")

    first = build_brownian_bridge_view(raw, calendar=calendar, max_gap_bars=3, seed=7)
    second = build_brownian_bridge_view(raw, calendar=calendar, max_gap_bars=3, seed=7)

    assert first["close"].equals(second["close"])
    assert first["fill_method"].equals(second["fill_method"])


def test_brownian_bridge_leaves_too_long_and_open_gaps_unfilled() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-06"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )

    view = build_brownian_bridge_view(
        raw,
        calendar=pd.date_range("2026-01-01", periods=6, freq="D"),
        max_gap_bars=2,
    )

    leading = view[view["date"].eq(pd.Timestamp("2026-01-01"))].iloc[0]
    long_gap = view[view["date"].eq(pd.Timestamp("2026-01-04"))].iloc[0]
    fresh = view[view["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]

    assert pd.isna(leading["close"])
    assert leading["quality_state"] == "missing"
    assert pd.isna(long_gap["close"])
    assert bool(long_gap["is_synthetic"]) is False
    assert long_gap["quality_state"] == QUALITY_STALE_EXPIRED
    assert fresh["quality_state"] == QUALITY_FRESH
