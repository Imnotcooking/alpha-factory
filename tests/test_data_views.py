from __future__ import annotations

import pandas as pd

from oqp.data import build_market_data_views


def test_market_data_views_separate_accounting_alpha_and_risk() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )

    views = build_market_data_views(
        raw,
        max_stale_bars=1,
        calendar=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    accounting_day2 = views.accounting[views.accounting["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    alpha_day2 = views.alpha[views.alpha["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]

    assert views.raw.equals(raw)
    assert accounting_day2["close"] == 100.0
    assert bool(accounting_day2["is_synthetic"]) is True
    assert pd.isna(alpha_day2["close"])
    assert bool(alpha_day2["alpha_can_update"]) is False
    assert views.risk.attrs["risk_imputation"] == "ffill_with_freshness_flags"
    assert views.quality_summary["fresh_rows"] == 2
    assert views.quality_summary["synthetic_rows"] == 1


def test_market_data_views_respect_zero_stale_policy() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )

    views = build_market_data_views(
        raw,
        max_stale_bars=0,
        calendar=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    day2 = views.accounting[views.accounting["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]

    assert pd.isna(day2["close"])
    assert bool(day2["is_synthetic"]) is False
    assert day2["quality_state"] == "stale_expired"


def test_market_data_views_can_use_brownian_bridge_for_risk_only() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )

    views = build_market_data_views(
        raw,
        max_stale_bars=2,
        calendar=pd.date_range("2026-01-01", periods=3, freq="D"),
        risk_imputation="brownian_bridge",
    )

    accounting_day2 = views.accounting[views.accounting["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    alpha_day2 = views.alpha[views.alpha["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    risk_day2 = views.risk[views.risk["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]

    assert accounting_day2["close"] == 100.0
    assert pd.isna(alpha_day2["close"])
    assert 100.0 < risk_day2["close"] < 121.0
    assert risk_day2["fill_method"] == "brownian_bridge"
    assert views.risk.attrs["risk_imputation"] == "brownian_bridge"


def test_market_data_views_reject_unknown_risk_imputation() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["A"],
            "close": [100.0],
        }
    )

    try:
        build_market_data_views(raw, risk_imputation="magic")
    except ValueError as exc:
        assert "risk_imputation" in str(exc)
    else:
        raise AssertionError("Expected unknown risk_imputation to raise ValueError.")
