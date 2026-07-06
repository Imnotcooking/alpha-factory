import numpy as np
import pandas as pd

from oqp.research.state_space import (
    ARBITRAGE_CROSS_PRODUCT,
    SPREAD_LINEAR_PRICE,
    SPREAD_RETURN_RESIDUAL,
    DataAuditConfig,
    OpportunityScanConfig,
    SpreadModelConfig,
    compute_data_audit,
    construct_pair_spread,
    dislocation_score,
    estimate_half_life,
    interpret_candidate,
    mean_reversion_score,
    run_opportunity_scan,
    score_opportunity,
    simple_spread_backtest,
)


def _pair_frame(rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    x_ret = rng.normal(0.0002, 0.01, size=rows)
    residual = rng.normal(0.0, 0.002, size=rows)
    y_ret = 0.0001 + 1.3 * x_ret + residual
    x_close = 100 * np.exp(np.cumsum(x_ret))
    y_close = 80 * np.exp(np.cumsum(y_ret))
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["x"] * rows + ["y"] * rows,
            "close": list(x_close) + list(y_close),
            "volume": [1000] * rows + [1200] * rows,
        }
    )


def _market_frame(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = rng.normal(0.0002, 0.012, rows)
    au = 100 * np.exp(np.cumsum(base + rng.normal(0, 0.002, rows)))
    ag = 80 * np.exp(np.cumsum(1.2 * base + rng.normal(0, 0.003, rows)))
    rb = 60 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, rows)))
    frames = []
    for ticker, close, volume in [
        ("黄金(au)[指数]", au, 5000),
        ("白银(ag)[指数]", ag, 7000),
        ("螺纹钢(rb)[指数]", rb, 9000),
    ]:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_opportunity_scoring_returns_interpretation():
    assert dislocation_score(-5.0) == 1.0
    assert mean_reversion_score(10.0) == 1.0

    scored = score_opportunity(
        pd.Series(
            {
                "latest_z": 2.6,
                "correlation": 0.82,
                "beta_drift": 0.05,
                "half_life": 12.0,
                "liquidity_score": 0.9,
                "round_turn_cost_bps": 3.0,
            }
        )
    )

    assert scored["opportunity_score"] > 70
    assert "Promising" in scored["interpretation"]
    assert "mean reversion" in interpret_candidate(
        opportunity_score=66.0,
        latest_z=2.4,
        half_life=float("inf"),
        beta_drift=0.1,
    )


def test_construct_return_residual_spread_adds_zscore_and_beta():
    spread = construct_pair_spread(
        _pair_frame(),
        SpreadModelConfig(
            y_ticker="y",
            x_ticker="x",
            method=SPREAD_RETURN_RESIDUAL,
            hedge_lookback=180,
            zscore_window=60,
        ),
    )

    assert {"spread", "spread_z", "hedge_beta", "half_life"}.issubset(spread.columns)
    assert spread["spread_z"].notna().sum() > 100
    assert abs(float(spread["hedge_beta"].iloc[-1]) - 1.3) < 0.25


def test_construct_linear_price_spread_and_backtest():
    spread = construct_pair_spread(
        _pair_frame(),
        SpreadModelConfig(
            y_ticker="y",
            x_ticker="x",
            method=SPREAD_LINEAR_PRICE,
            hedge_lookback=120,
            zscore_window=40,
        ),
    )
    result = simple_spread_backtest(spread, entry_z=1.0, exit_z=0.2, stop_z=3.0, cost_bps=1.0)

    assert spread["spread_units"].iloc[-1] == "price_points"
    assert result["summary"]["observations"] > 100
    assert "equity" in result["curve"].columns


def test_half_life_is_finite_for_mean_reverting_series():
    rng = np.random.default_rng(3)
    values = [0.0]
    for _ in range(180):
        values.append(0.75 * values[-1] + rng.normal(0, 0.2))

    half_life = estimate_half_life(pd.Series(values))

    assert 1.0 < half_life < 20.0


def test_opportunity_scan_ranks_candidates_and_metadata():
    result = run_opportunity_scan(
        _market_frame(),
        OpportunityScanConfig(
            min_observations=180,
            lookback=260,
            zscore_window=60,
            max_assets=5,
            min_abs_correlation=0.1,
        ),
    )

    candidates = result["candidates"]

    assert not candidates.empty
    assert {"opportunity_score", "interpretation", "arbitrage_type"}.issubset(candidates.columns)
    assert candidates["rank"].is_monotonic_increasing
    assert ARBITRAGE_CROSS_PRODUCT in set(candidates["arbitrage_type"])


def test_data_audit_reports_schema_and_eligible_assets():
    audit = compute_data_audit(_market_frame(), DataAuditConfig(min_observations=200))

    assert audit["summary"]["assets"] == 3
    assert audit["summary"]["eligible_assets"] == 3
    assert {"column", "dtype", "non_null"}.issubset(audit["schema"].columns)
