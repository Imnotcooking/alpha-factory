from __future__ import annotations

import pandas as pd
import pytest

from oqp.research.backtesting.margin_policy import apply_margin_utilization_cap


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "ticker": ["AP", "IF"],
            "final_target_weight": [2.0, -2.0],
            "target_weight": [2.0, -2.0],
        }
    )


def test_cn_futures_margin_budget_scales_portfolio_to_thirty_percent() -> None:
    result = apply_margin_utilization_cap(
        _targets(),
        market_vertical="FUTURES_CN",
        source_weight_col="final_target_weight",
        max_margin_utilization=0.30,
    )

    assert result["margin_cap_bound"].all()
    assert result["margin_utilization"].max() == pytest.approx(0.30)
    assert result["final_target_weight"].tolist() == pytest.approx([1.25, -1.25])
    assert result["target_weight"].tolist() == pytest.approx([1.25, -1.25])
    assert result.attrs["minimum_cash_reserve"] == pytest.approx(0.70)


def test_margin_budget_is_a_ceiling_not_a_target() -> None:
    frame = _targets()
    frame[["final_target_weight", "target_weight"]] = 0.25
    result = apply_margin_utilization_cap(
        frame,
        market_vertical="FUTURES_CN",
        source_weight_col="final_target_weight",
        max_margin_utilization=0.30,
    )

    assert result["margin_scale"].eq(1.0).all()
    assert result["final_target_weight"].tolist() == pytest.approx([0.25, 0.25])
    assert result["margin_utilization"].max() == pytest.approx(0.06)


def test_margin_budget_rejects_non_futures_market() -> None:
    with pytest.raises(ValueError, match="support futures only"):
        apply_margin_utilization_cap(
            _targets(),
            market_vertical="EQUITY_US",
            source_weight_col="final_target_weight",
            max_margin_utilization=0.30,
        )
