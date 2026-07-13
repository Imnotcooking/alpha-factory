from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.data import (
    QUALITY_FRESH,
    QUALITY_STALE_EXPIRED,
    QUALITY_STALE_WITHIN_LIMIT,
    build_accounting_view,
    build_alpha_view,
    build_market_data_views,
    require_fresh_alpha_inputs,
)
from oqp.risk.factor_breadth import compute_log_return_matrix


def test_accounting_view_forward_fills_with_audit_flags() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-01", "2026-01-02"]),
            "ticker": ["A", "A", "B", "B"],
            "close": [100.0, 121.0, 200.0, 220.0],
        }
    )
    calendar = pd.date_range("2026-01-01", periods=4, freq="D")

    accounting = build_accounting_view(raw, max_stale_bars=1, calendar=calendar)

    a_day2 = accounting[
        (accounting["ticker"] == "A") & (accounting["date"] == pd.Timestamp("2026-01-02"))
    ].iloc[0]
    b_day4 = accounting[
        (accounting["ticker"] == "B") & (accounting["date"] == pd.Timestamp("2026-01-04"))
    ].iloc[0]

    assert a_day2["close"] == 100.0
    assert bool(a_day2["is_fresh"]) is False
    assert bool(a_day2["is_synthetic"]) is True
    assert a_day2["stale_bars"] == 1
    assert a_day2["fill_method"] == "ffill"
    assert a_day2["quality_state"] == QUALITY_STALE_WITHIN_LIMIT

    assert pd.isna(b_day4["close"])
    assert bool(b_day4["is_synthetic"]) is False
    assert b_day4["stale_bars"] == 2
    assert b_day4["quality_state"] == QUALITY_STALE_EXPIRED


def test_alpha_view_masks_synthetic_accounting_marks() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "ticker": ["A", "A"],
            "close": [100.0, 121.0],
        }
    )
    accounting = build_accounting_view(
        raw,
        max_stale_bars=1,
        calendar=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    alpha = build_alpha_view(accounting)
    alpha_day2 = alpha[alpha["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    fresh_only = require_fresh_alpha_inputs(alpha, required_cols=["close"])

    assert pd.isna(alpha_day2["close"])
    assert bool(alpha_day2["alpha_can_update"]) is False
    assert list(fresh_only["quality_state"]) == [QUALITY_FRESH, QUALITY_FRESH]


def test_market_data_views_keep_raw_and_summarize_quality() -> None:
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

    assert views.raw.equals(raw)
    assert views.accounting.attrs["view_type"] == "accounting"
    assert views.alpha.attrs["view_type"] == "alpha"
    assert views.risk.attrs["view_type"] == "risk"
    assert views.quality_summary["rows"] == 3
    assert views.quality_summary["fresh_rows"] == 2
    assert views.quality_summary["synthetic_rows"] == 1


def test_risk_return_matrix_uses_capped_accounting_marks() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-01", "2026-01-02"]),
            "ticker": ["A", "A", "B", "B"],
            "close": [100.0, 121.0, 200.0, 220.0],
        }
    )

    returns = compute_log_return_matrix(raw, max_stale_bars=1)

    assert np.isclose(returns.loc[pd.Timestamp("2026-01-02"), "A"], 0.0)
    assert np.isclose(returns.loc[pd.Timestamp("2026-01-03"), "A"], np.log(121.0 / 100.0))
    assert np.isclose(returns.loc[pd.Timestamp("2026-01-02"), "B"], np.log(220.0 / 200.0))
