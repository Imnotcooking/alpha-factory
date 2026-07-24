from __future__ import annotations

import pandas as pd
import pytest

from oqp.research.liquidity_eligibility import (
    BELOW_OPEN_INTEREST_NOTIONAL,
    BELOW_TRADED_NOTIONAL,
    ELIGIBLE,
    WARMUP,
    apply_liquidity_gate,
    assess_liquidity_eligibility,
    liquidity_eligible_rows,
    resolve_liquidity_policy,
)
from oqp.research.backtesting.evaluator import AlphaEvaluator


def test_cn_futures_capacity_thresholds_are_capital_and_multiplier_aware() -> None:
    policy = resolve_liquidity_policy(
        "FUTURES_CN",
        initial_capital=10_000_000,
    )

    assert policy.required_daily_traded_notional == pytest.approx(50_000_000)
    assert policy.required_open_interest_notional == pytest.approx(25_000_000)
    assert policy.lookback_sessions == 20
    assert policy.min_observations == 15
    assert policy.decision_lag_sessions == 1


def test_daily_futures_eligibility_is_causal_and_preserves_rows() -> None:
    dates = pd.date_range("2026-01-01", periods=17, freq="D")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "close": 100.0,
                    "volume": volume,
                    "open_interest": open_interest,
                    "signal": 0.10,
                }
            )
            for ticker, volume, open_interest in (
                ("AP", 100_000.0, 50_000.0),
                ("CF", 1_000.0, 50_000.0),
                ("CJ", 100_000.0, 1_000.0),
            )
        ],
        ignore_index=True,
    )
    frame.attrs["initial_capital"] = 10_000_000.0
    policy = resolve_liquidity_policy(
        "FUTURES_CN",
        initial_capital=10_000_000,
    )

    assessed = assess_liquidity_eligibility(frame, policy)

    assert len(assessed) == len(frame)
    day_15 = assessed.loc[assessed["date"].eq(dates[14])].set_index("ticker")
    day_16 = assessed.loc[assessed["date"].eq(dates[15])].set_index("ticker")
    assert day_15["liquidity_reason_code"].eq(WARMUP).all()
    assert day_16.loc["AP", "liquidity_reason_code"] == ELIGIBLE
    assert day_16.loc["CF", "liquidity_reason_code"] == BELOW_TRADED_NOTIONAL
    assert (
        day_16.loc["CJ", "liquidity_reason_code"]
        == BELOW_OPEN_INTEREST_NOTIONAL
    )
    assert day_16.loc["AP", "liquidity_contract_multiplier"] == 10.0
    assert day_16.loc["CF", "liquidity_contract_multiplier"] == 5.0

    metric_rows = liquidity_eligible_rows(assessed)
    assert set(metric_rows["ticker"]) == {"AP"}

    gated = apply_liquidity_gate(assessed)
    gated_day_16 = gated.loc[gated["date"].eq(dates[15])].set_index("ticker")
    assert gated_day_16.loc["AP", "signal"] == pytest.approx(0.10)
    assert gated_day_16.loc["CF", "signal"] == 0.0
    assert gated_day_16.loc["CJ", "signal"] == 0.0
    assert gated_day_16.loc["CF", "pre_liquidity_signal"] == pytest.approx(0.10)


def test_liquidity_gate_normalizes_nullable_object_mask() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "signal": [0.1, 0.1, 0.1, 0.1, 0.1],
            "liquidity_eligible": pd.Series(
                [True, None, "true", "false", 1],
                dtype=object,
            ),
        }
    )

    gated = apply_liquidity_gate(frame)

    assert gated["signal"].tolist() == [0.1, 0.0, 0.1, 0.0, 0.1]
    assert liquidity_eligible_rows(frame)["ticker"].tolist() == ["A", "C", "E"]


def test_intraday_uses_prior_completed_sessions_not_current_day_volume() -> None:
    dates = pd.date_range("2026-01-01", periods=17, freq="D")
    rows = []
    for date in dates:
        rows.extend(
            [
                {
                    "date": date + pd.Timedelta(hours=9),
                    "ticker": "AP",
                    "close": 100.0,
                    "volume": 5_000.0,
                    "open_interest": 50_000.0,
                },
                {
                    "date": date + pd.Timedelta(hours=14),
                    "ticker": "AP",
                    "close": 100.0,
                    "volume": 5_000.0,
                    "open_interest": 50_000.0,
                },
            ]
        )
    frame = pd.DataFrame(rows)
    policy = resolve_liquidity_policy(
        "FUTURES_CN",
        initial_capital=1_000_000,
    )

    baseline = assess_liquidity_eligibility(frame, policy)
    shocked = frame.copy()
    shocked.loc[
        shocked["date"].dt.normalize().eq(dates[15]),
        "volume",
    ] = 10_000_000.0
    changed = assess_liquidity_eligibility(shocked, policy)

    current_session = frame["date"].dt.normalize().eq(dates[15])
    pd.testing.assert_series_equal(
        baseline.loc[current_session, "liquidity_trailing_median_notional"].reset_index(drop=True),
        changed.loc[current_session, "liquidity_trailing_median_notional"].reset_index(drop=True),
    )
    assert baseline.attrs["liquidity_intraday_detected"] is True
    assert baseline.attrs["liquidity_intraday_volume_aggregation"] == "sum"


def test_option_policy_reports_volume_open_interest_and_spread_failures() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 4),
            "ticker": ["A", "B", "C", "D"],
            "bid": [1.00, 1.00, 1.00, 1.00],
            "ask": [1.10, 1.10, 1.10, 2.00],
            "volume": [10.0, 0.0, 10.0, 10.0],
            "open_interest": [100.0, 100.0, 0.0, 100.0],
        }
    )
    policy = resolve_liquidity_policy("OPTIONS_US", initial_capital=100_000)

    assessed = assess_liquidity_eligibility(frame, policy).set_index("ticker")

    assert assessed.loc["A", "liquidity_reason_code"] == ELIGIBLE
    assert assessed.loc["B", "liquidity_reason_code"] == "below_option_volume_floor"
    assert assessed.loc["C", "liquidity_reason_code"] == "below_option_open_interest_floor"
    assert assessed.loc["D", "liquidity_reason_code"] == "option_spread_too_wide"


def test_policy_fingerprint_changes_when_capacity_assumption_changes() -> None:
    base = resolve_liquidity_policy("EQUITY_US", initial_capital=1_000_000)
    larger = resolve_liquidity_policy("EQUITY_US", initial_capital=10_000_000)

    assert base.fingerprint != larger.fingerprint
    assert larger.required_daily_traded_notional == pytest.approx(
        10 * base.required_daily_traded_notional
    )


def test_evaluator_keeps_full_grid_but_gates_execution_targets(tmp_path) -> None:
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    rows = []
    for date_index, date in enumerate(dates):
        for ticker, volume, score in (
            ("AP", 100_000.0, -1.0),
            ("CJ", 200_000.0, 1.0),
            ("CF", 1_000.0, 0.5),
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": 100.0,
                    "volume": volume,
                    "open_interest": 100_000.0,
                    "factor_score": score,
                    "signal": score,
                    "forward_return": score * 0.001 + date_index * 0.000001,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "market_vertical": "FUTURES_CN",
            "initial_capital": 10_000_000.0,
            "capital_currency": "CNY",
            "data_frequency": "daily",
            "data_tradability": "tradable_main_contract",
        }
    )
    evaluator = AlphaEvaluator(
        db_path=tmp_path / "research.db",
        logs_dir=tmp_path / "artifacts",
        asset_class="FUTURES_CN",
    )
    captured: dict[str, pd.DataFrame] = {}

    def capture_log(*args):
        captured["frame"] = args[7]
        return "captured"

    evaluator._log_to_db = capture_log
    run_id = evaluator.run_evaluation(
        "fac_liquidity_integration_test",
        frame,
        crisis_period=("2025-02-01", "2025-02-05"),
        split_mode="ratio",
        validation_fraction=0.60,
        strategy_geometry="cross_sectional",
    )

    assert run_id == "captured"
    executed = captured["frame"]
    assert len(executed) == len(frame)
    late = executed["date"].ge(dates[15])
    blocked = late & executed["ticker"].eq("CF")
    eligible = late & executed["ticker"].eq("CJ")
    assert executed.loc[blocked, "signal"].eq(0.0).all()
    assert executed.loc[blocked, "pre_temporal_signal"].eq(0.5).all()
    assert executed.loc[blocked, "pre_liquidity_signal"].eq(0.0).all()
    assert executed.loc[eligible, "signal"].eq(1.0).all()
    assert executed.attrs["liquidity_gated_weight_columns"]
