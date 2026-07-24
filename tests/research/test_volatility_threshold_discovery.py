from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SCRIPT = REPO_ROOT / "scripts/research/experiments/analyze_cn_futures_all_eligible_threshold_router.py"


def load_module():
    spec = importlib.util.spec_from_file_location("all_eligible_threshold_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capacity_floor_is_derived_from_declared_portfolio_geometry() -> None:
    module = load_module()

    result = module.capacity_liquidity_floor(
        capital=20_000_000.0,
        participation=0.01,
        minimum_cross_section=20,
        tail_fraction=0.20,
    )

    assert result == 250_000_000.0


def test_rolling_percentile_uses_only_the_trailing_window_and_current_value() -> None:
    module = load_module()
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0])

    result = module.rolling_last_percentile(values, lookback=4)

    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == 1.0
    assert result.iloc[4] == 0.25


def test_75_percentile_reproduces_the_rolling_fourth_quartile() -> None:
    module = load_module()
    rng = np.random.default_rng(7)
    volatility = pd.Series(rng.lognormal(mean=-5.0, sigma=0.4, size=420))
    rolling = volatility.rolling(252, min_periods=252)
    q25 = rolling.quantile(0.25)
    q50 = rolling.quantile(0.50)
    q75 = rolling.quantile(0.75)
    quartile = pd.Series(np.nan, index=volatility.index)
    ready = q75.notna()
    quartile.loc[ready & volatility.le(q25)] = 1.0
    quartile.loc[ready & volatility.gt(q25) & volatility.le(q50)] = 2.0
    quartile.loc[ready & volatility.gt(q50) & volatility.le(q75)] = 3.0
    quartile.loc[ready & volatility.gt(q75)] = 4.0
    frame = pd.DataFrame(
        {
            "signal_day": pd.date_range("2020-01-01", periods=len(volatility), freq="B"),
            "market_realized_volatility": volatility,
            "market_volatility_quartile": quartile,
        }
    )

    result = module.add_market_percentile(frame)
    selected = result.loc[result["market_volatility_quartile"].notna()]

    assert selected["market_volatility_percentile"].gt(0.75).equals(
        selected["market_volatility_quartile"].eq(4.0)
    )


def test_threshold_ties_are_resolved_toward_the_paper_anchor() -> None:
    module = load_module()
    curve = pd.DataFrame(
        {
            "threshold_percentile": [70, 75, 80],
            "valid": [True, True, True],
            "router_mean_daily": [0.001, 0.001, 0.001],
        }
    )

    assert module.select_best_threshold(curve, "router_mean_daily") == 75


def test_stationary_bootstrap_indices_are_deterministic_and_in_bounds() -> None:
    module = load_module()
    first = module.stationary_bootstrap_indices(30, 5.0, np.random.default_rng(123))
    second = module.stationary_bootstrap_indices(30, 5.0, np.random.default_rng(123))

    assert np.array_equal(first, second)
    assert first.min() >= 0
    assert first.max() < 30
