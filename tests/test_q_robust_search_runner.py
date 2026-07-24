from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_FILE = (
    REPO_ROOT
    / "notebooks/Phase_7_Research_Projects/"
    "07_04_daily_volatility_router_cn_futures_replication_private/"
    "experiments/10_q_robust_technical_factor_sleeve_search/run_search.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "qrobust_search_runner_for_tests", RUNNER_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_forward_compound_return_uses_current_and_next_four_sessions() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.50])

    observed = RUNNER.forward_compound_return(returns, horizon=5)

    expected_first = np.prod(1.0 + returns.iloc[:5]) - 1.0
    expected_second = np.prod(1.0 + returns.iloc[1:6]) - 1.0
    assert np.isclose(observed.iloc[0], expected_first)
    assert np.isclose(observed.iloc[1], expected_second)
    assert observed.iloc[2:].isna().all()


def test_execution_tradability_never_depends_on_forward_return() -> None:
    dates = pd.to_datetime(["2021-01-04", "2021-01-05"])
    base = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A"],
            "symbol": ["A1", "A1"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [2000.0, 2100.0],
            "open_interest": [200.0, 210.0],
            "open_oi": [200.0, 210.0],
            "sector": ["Test", "Test"],
            "next_date": [dates[1], pd.NaT],
            "common_eligible_close": [True, True],
            "trailing_volatility": [0.02, 0.02],
            "execution_open": [100.0, 101.0],
            "open_to_next_open_return": [0.01, np.nan],
            "five_session_open_return": [np.nan, np.nan],
            "entry_cost_ratio": [0.001, 0.001],
            "close_cost_current_ratio": [0.001, 0.001],
            "close_cost_prior_ratio": [0.001, 0.001],
            "roll_flag": [False, False],
        }
    )

    def compute(data: pd.DataFrame) -> pd.DataFrame:
        return data[["date", "ticker"]].assign(factor_score=1.0)

    module = SimpleNamespace(
        FACTOR_ID="fac_test",
        compute=compute,
    )
    protocol = {"sample": {"q_labeled_evaluation_end": "2021-01-05"}}

    rows = RUNNER.build_factor_signal_rows(base, module, protocol)

    assert len(rows) == 1
    assert bool(rows.iloc[0]["rank_eligible"])
    assert bool(rows.iloc[0]["tradable"])
    assert bool(rows.iloc[0]["actual_execution_row"])
    assert bool(rows.iloc[0]["can_open"])
    assert bool(rows.iloc[0]["can_close_incumbent"])
    assert pd.isna(rows.iloc[0]["forward_return"])
    assert rows.attrs["eligibility_uses_forward_returns"] is False


def test_trial_ledger_is_idempotent_but_preserves_changed_attempt(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "trial_ledger.csv"
    original = pd.DataFrame(
        [
            {
                "trial_id": "trial_a",
                "factor_id": "fac_a",
                "sleeve_id": "slv_a",
                "status": "execution_error",
                "trial_fingerprint": "fingerprint_1",
            }
        ]
    )

    first = RUNNER.append_preserving_trial_ledger(original, ledger)
    repeated = RUNNER.append_preserving_trial_ledger(original, ledger)
    changed = original.assign(
        status="evaluated", trial_fingerprint="fingerprint_2"
    )
    appended = RUNNER.append_preserving_trial_ledger(changed, ledger)

    stored = pd.read_csv(ledger)
    assert first.iloc[0]["trial_id"] == "trial_a"
    assert repeated.iloc[0]["trial_id"] == "trial_a"
    assert appended.iloc[0]["trial_id"] == "trial_a__attempt_002"
    assert stored["trial_id"].tolist() == [
        "trial_a",
        "trial_a__attempt_002",
    ]
    assert stored["status"].tolist() == ["execution_error", "evaluated"]


def test_no_bearish_forecast_is_not_turned_into_a_directional_short_tail() -> None:
    long_only_factor = SimpleNamespace(
        FACTOR_METADATA={"requires_shorting": False}
    )
    tail_sleeve = SimpleNamespace(
        SLEEVE_CONTRACT={"construction": "decile_equal"}
    )
    hedge_sleeve = SimpleNamespace(
        SLEEVE_CONTRACT={"construction": "top_decile_broad_hedge"}
    )
    long_only_sleeve = SimpleNamespace(
        SLEEVE_CONTRACT={"construction": "sparse_positive_event"}
    )

    compatible, reason = RUNNER.pair_compatibility(
        long_only_factor, tail_sleeve
    )

    assert compatible is False
    assert reason == "incompatible_no_bearish_forecast"
    assert RUNNER.pair_compatibility(long_only_factor, hedge_sleeve)[0] is True
    assert (
        RUNNER.pair_compatibility(long_only_factor, long_only_sleeve)[0]
        is True
    )


def test_pair_evaluation_keeps_cost_engine_gross_exposure_column() -> None:
    dates = pd.bdate_range("2021-01-04", periods=15)
    tickers = [f"T{index:02d}" for index in range(20)]
    index = pd.MultiIndex.from_product(
        [dates, tickers], names=["execution_date", "ticker"]
    ).to_frame(index=False)
    ticker_number = index["ticker"].str.removeprefix("T").astype(int)
    signal_rows = index.assign(
        information_date=index["execution_date"] - pd.Timedelta(days=1),
        alpha_score=ticker_number.astype(float),
        rank_eligible=True,
        tradable=True,
        actual_execution_row=True,
        can_open=True,
        can_close_incumbent=True,
        trailing_volatility=0.01 + ticker_number * 0.0001,
        forward_return=0.0002 * (ticker_number - 9.5),
        entry_cost_ratio=0.0001,
        close_cost_current_ratio=0.0001,
        close_cost_prior_ratio=0.0001,
        roll_flag=False,
        execution_sector="Test",
        symbol=index["ticker"] + "1",
    )
    signal_rows.attrs["causal_signal_alignment_verified"] = True
    state_map = pd.DataFrame(
        {
            "holding_month": ["2021-01"],
            RUNNER.PRIMARY_STATE_COL: ["Q1"],
            RUNNER.ROBUSTNESS_STATE_COL: ["Q1"],
        }
    )
    protocol = {
        "sample": {
            "sleeve_state_warmup_start": str(dates[0].date()),
            "q_labeled_evaluation_start": str(dates[0].date()),
            "q_labeled_evaluation_end": str(dates[-1].date()),
        },
        "statistics": {
            "annualization_days": 252,
            "return_hac_lags": 2,
            "month_block_bootstrap_draws": 3,
            "month_block_bootstrap_seed": 7,
            "month_block_confidence_level": 0.90,
        },
    }
    sleeve_id = "slv_005_Decile_Equal_5D_Staggered_Long_Short"
    sleeve_module = RUNNER.load_sleeve_module(sleeve_id)

    result = RUNNER.evaluate_pair(
        signal_rows,
        state_map,
        "fac_test",
        "higher_is_bullish",
        sleeve_id,
        sleeve_module,
        protocol,
        pd.DatetimeIndex(dates),
    )

    assert "gross_exposure" in result.daily
    assert "sleeve_target_gross_exposure" in result.daily
    assert result.daily["gross_exposure"].gt(0.0).any()
    assert result.summary["active_day_fraction"] > 0.0


def _desired_execution_rows() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=4)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "desired_target_weight": [0.5, 0.0, 0.0, 0.0],
            "actual_execution_row": True,
            "can_open": True,
            "can_close_incumbent": True,
            "roll_flag": False,
            "forward_return": [0.01, 0.02, 0.03, np.nan],
            "entry_cost_ratio": 0.01,
            "close_cost_current_ratio": 0.02,
            "close_cost_prior_ratio": 0.03,
        }
    )


def test_rank_loss_closes_once_and_charges_one_close_cost() -> None:
    desired = _desired_execution_rows()

    realized = RUNNER.realize_executable_targets(desired)
    costed = RUNNER.PERSISTENT.attach_target_change_costs(
        realized.rename(columns={"date": "execution_date"})
    )

    assert realized["weight"].tolist() == [0.5, 0.0, 0.0, 0.0]
    assert realized["execution_action"].tolist()[:2] == ["open", "close"]
    assert costed["closing_turnover"].sum() == 0.5
    assert costed["cost_contribution"].sum() == pytest.approx(0.015)


def test_grid_gap_defers_desired_exit_until_next_actual_open() -> None:
    desired = _desired_execution_rows()
    desired.loc[1, "actual_execution_row"] = False

    realized = RUNNER.realize_executable_targets(desired)

    assert realized["date"].tolist() == [
        desired.loc[0, "date"],
        desired.loc[2, "date"],
        desired.loc[3, "date"],
    ]
    assert realized["weight"].tolist() == [0.5, 0.0, 0.0]
    assert realized["execution_action"].tolist() == ["open", "close", "hold"]


def test_missing_close_execution_carries_incumbent_without_phantom_cash() -> None:
    desired = _desired_execution_rows()
    desired.loc[1, "can_close_incumbent"] = False

    realized = RUNNER.realize_executable_targets(desired)

    assert realized["weight"].tolist() == [0.5, 0.5, 0.0, 0.0]
    assert realized["execution_deferred"].tolist() == [False, True, False, False]
    assert realized["execution_action"].tolist() == [
        "open",
        "close_deferred",
        "close",
        "hold",
    ]


def test_terminal_cash_is_checked_after_actual_row_filtering() -> None:
    desired = _desired_execution_rows().iloc[:2].copy()
    desired.loc[1, "actual_execution_row"] = False

    with pytest.raises(RuntimeError, match="terminal cash failed"):
        RUNNER.realize_executable_targets(desired)


def test_unchanged_nonzero_roll_closes_and_reopens_with_full_cost() -> None:
    desired = _desired_execution_rows()
    desired.loc[1, "desired_target_weight"] = 0.5
    desired.loc[1, "roll_flag"] = True

    realized = RUNNER.realize_executable_targets(desired)
    costed = RUNNER.PERSISTENT.attach_target_change_costs(
        realized.rename(columns={"date": "execution_date"})
    )
    roll_row = costed.loc[costed["execution_date"].eq(desired.loc[1, "date"])]

    assert realized.loc[1, "execution_action"] == "roll_close_reopen"
    assert roll_row["opening_turnover"].iloc[0] == pytest.approx(0.5)
    assert roll_row["closing_turnover"].iloc[0] == pytest.approx(0.5)
    assert roll_row["cost_contribution"].iloc[0] == pytest.approx(0.02)


def test_unchanged_nonzero_roll_fails_when_incumbent_cannot_close() -> None:
    desired = _desired_execution_rows()
    desired.loc[1, "desired_target_weight"] = 0.5
    desired.loc[1, "roll_flag"] = True
    desired.loc[1, "can_close_incumbent"] = False

    with pytest.raises(ValueError, match="unexecutable contract roll"):
        RUNNER.realize_executable_targets(desired)


def test_front_loaded_targets_survive_roll_and_finish_in_terminal_cash() -> None:
    dates = pd.bdate_range("2021-01-04", periods=8)
    tickers = [f"T{index:02d}" for index in range(20)]
    frame = pd.MultiIndex.from_product(
        [dates, tickers], names=["date", "ticker"]
    ).to_frame(index=False)
    frame["alpha_score"] = 0.0
    frame.loc[
        frame["date"].eq(dates[0]) & frame["ticker"].eq("T00"),
        "alpha_score",
    ] = -1.0
    frame.loc[
        frame["date"].eq(dates[0]) & frame["ticker"].eq("T19"),
        "alpha_score",
    ] = 1.0
    frame["rank_eligible"] = True
    frame["tradable"] = True
    frame["trailing_volatility"] = 1.0
    frame["actual_execution_row"] = True
    frame["can_open"] = True
    frame["can_close_incumbent"] = True
    frame["roll_flag"] = (
        frame["date"].eq(dates[2])
        & frame["ticker"].isin(["T00", "T19"])
    )
    frame.attrs["causal_signal_alignment_verified"] = True
    sleeve_id = "slv_027_Front_Loaded_Signed_Inverse_Vol_5D"
    module = RUNNER.load_sleeve_module(sleeve_id)
    config = module.build_config(
        "fac_test",
        market_vertical="FUTURES_CN",
        signal_orientation="higher_is_bullish",
    )

    desired = RUNNER.build_persistent_sleeve_targets(
        frame, config
    ).positions
    realized = RUNNER.realize_executable_targets(desired)
    long_rows = realized.loc[realized["ticker"].eq("T19")].sort_values("date")

    assert long_rows["desired_target_weight"].tolist() == pytest.approx(
        [0.032, 0.020, 0.012, 0.0096, 0.0064, 0.0, 0.0, 0.0]
    )
    assert long_rows["execution_action"].iloc[2] == "roll_close_reopen"
    assert long_rows["weight"].iloc[-1] == 0.0
    assert realized.groupby("ticker", sort=False).tail(1)["weight"].eq(0.0).all()


def test_persistent_directional_sleeve_keeps_zero_ties_neutral() -> None:
    dates = pd.bdate_range("2021-01-04", periods=7)
    tickers = [f"T{index:02d}" for index in range(20)]
    frame = pd.MultiIndex.from_product(
        [dates, tickers], names=["date", "ticker"]
    ).to_frame(index=False)
    ticker_number = frame["ticker"].str.removeprefix("T").astype(int)
    frame["alpha_score"] = np.select(
        [ticker_number.eq(0), ticker_number.eq(19)],
        [-1.0, 1.0],
        default=0.0,
    )
    frame["rank_eligible"] = True
    frame["tradable"] = True
    frame.attrs["causal_signal_alignment_verified"] = True
    sleeve_id = "slv_005_Decile_Equal_5D_Staggered_Long_Short"
    module = RUNNER.load_sleeve_module(sleeve_id)
    config = module.build_config(
        "fac_test",
        market_vertical="FUTURES_CN",
        signal_orientation="higher_is_bullish",
    )

    result = RUNNER.build_persistent_sleeve_targets(frame, config)
    first = result.positions.loc[result.positions["date"].eq(dates[0])]
    neutral = first.loc[first["oriented_signal"].eq(0.0)]

    assert config.zero_signal_policy == "neutral"
    assert neutral["selected"].eq(False).all()
    assert neutral["cohort_weight"].eq(0.0).all()
    assert neutral["rank_percentile"].nunique() == 1


def test_calendar_aware_hac_preserves_gaps_between_state_episodes() -> None:
    values = pd.Series([1.0, 2.0, 3.0])
    positions = pd.Series([0, 100, 200])

    observed = RUNNER.calendar_aware_newey_west_mean_test(
        values, positions, max_lag=2
    )

    expected_standard_error = np.sqrt(2.0) / 3.0
    assert np.isclose(observed["mean"], 2.0)
    assert np.isclose(observed["standard_error"], expected_standard_error)


def test_calendar_aware_hac_matches_contiguous_hac_without_gaps() -> None:
    values = pd.Series([0.01, -0.02, 0.03, 0.01, 0.02])

    observed = RUNNER.calendar_aware_newey_west_mean_test(
        values, pd.Series(range(len(values))), max_lag=2
    )
    expected = RUNNER.METRICS.newey_west_mean_test(values, max_lag=2)

    assert np.isclose(observed["standard_error"], expected["standard_error"])
    assert np.isclose(observed["t_stat"], expected["t_stat"])
