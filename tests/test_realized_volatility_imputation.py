from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.risk import compare_risk_imputation_views, realized_volatility_by_asset


def test_realized_volatility_by_asset_reports_zero_return_pressure() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "ticker": ["A", "A", "A", "A"],
            "close": [100.0, 100.0, 110.0, 121.0],
        }
    )

    summary = realized_volatility_by_asset(frame, annualization=1.0)

    row = summary.iloc[0]
    assert row["ticker"] == "A"
    assert row["observations"] == 4
    assert row["return_observations"] == 3
    assert np.isclose(row["zero_return_pct"], 1 / 3)
    assert row["annualized_vol"] > 0


def test_imputation_comparison_surfaces_bridge_vs_ffill_risk_difference() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )

    result = compare_risk_imputation_views(
        frame,
        calendar=pd.date_range("2026-01-01", periods=3, freq="D"),
        annualization=1.0,
        max_stale_bars=1,
        bridge_max_gap_bars=2,
    )

    asset = result["asset_summary"].iloc[0]
    summary = result["summary"]

    assert asset["ticker"] == "A"
    assert np.isclose(asset["ffill_zero_return_pct"], 0.5)
    assert np.isclose(asset["bridge_zero_return_pct"], 0.0)
    assert summary["bridge_median_rv"] < summary["ffill_median_rv"]
    assert summary["bridge_synthetic_rows"] == 1
    assert summary["bridge_synthetic_pct"] == 1 / 3
