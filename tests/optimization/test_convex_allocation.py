from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oqp.portfolio.allocation import (
    ConvexAllocationConfig,
    ConvexPortfolioAllocator,
)


def _covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.03, 0.005],
            [0.00, 0.005, 0.02],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


def test_long_only_convex_allocator_honors_constraints() -> None:
    allocator = ConvexPortfolioAllocator(
        ConvexAllocationConfig(
            risk_aversion=2.0,
            gross_limit=1.0,
            max_weight_per_asset=0.60,
            long_only=True,
            net_target=1.0,
        )
    )

    result = allocator.allocate(
        pd.Series({"A": 0.08, "B": 0.05, "C": 0.02}),
        _covariance(),
    )

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.weights.min() >= -1e-8
    assert result.weights.max() <= 0.60 + 1e-6
    assert result.diagnostics["gross_exposure"] <= 1.0 + 1e-6


def test_market_neutral_allocator_respects_eligibility_and_costs() -> None:
    allocator = ConvexPortfolioAllocator(
        ConvexAllocationConfig(
            risk_aversion=1.0,
            turnover_penalty=0.01,
            gross_limit=1.0,
            max_weight_per_asset=0.50,
            long_only=False,
            net_target=0.0,
        )
    )

    result = allocator.allocate(
        pd.Series({"A": 0.08, "B": -0.05, "C": 0.04}),
        _covariance(),
        previous_weights=pd.Series({"A": 0.10, "B": -0.10, "C": 0.0}),
        linear_trading_costs=pd.Series({"A": 0.001, "B": 0.001, "C": 0.50}),
        eligible=pd.Series({"A": True, "B": True, "C": False}),
    )

    assert result.weights.sum() == pytest.approx(0.0, abs=1e-6)
    assert result.weights["C"] == pytest.approx(0.0, abs=1e-8)
    assert np.isfinite(result.estimated_variance)
    assert result.turnover >= 0.0


def test_allocator_rejects_missing_covariance_assets() -> None:
    allocator = ConvexPortfolioAllocator()

    with pytest.raises(ValueError, match="cover every"):
        allocator.allocate(
            pd.Series({"A": 0.1, "B": 0.2}),
            _covariance().loc[["A"], ["A"]],
        )


def test_allocator_rejects_nonfinite_inputs() -> None:
    allocator = ConvexPortfolioAllocator()

    with pytest.raises(ValueError, match="expected_returns must be finite"):
        allocator.allocate(
            pd.Series({"A": np.inf, "B": 0.2}),
            _covariance().loc[["A", "B"], ["A", "B"]],
        )
