from __future__ import annotations

import pandas as pd
import pytest

from oqp.research.temporal_policy import (
    HOLD_FACTOR_MANAGED,
    HOLD_FIXED_PERIOD,
    HOLD_UNTIL_NEXT_DECISION,
    SIGNAL_FIXED_INTERVAL,
    SIGNAL_SESSION_CLOSE,
    SignalHoldingPolicy,
    apply_signal_holding_policy,
    resolve_signal_holding_policy,
    temporal_metric_rows,
)


def test_daily_default_separates_decision_schedule_from_realized_events() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "ticker": "AP",
            "factor_score": [1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    frame.attrs.update(
        {
            "data_frequency": "daily",
            "factor_contract": {"execution_mode": "risk_desk"},
        }
    )
    policy = resolve_signal_holding_policy(frame)
    result = apply_signal_holding_policy(
        frame,
        policy,
        candidate_col="factor_score",
    )

    assert policy.signal_frequency == SIGNAL_SESSION_CLOSE
    assert policy.holding_mode == HOLD_UNTIL_NEXT_DECISION
    assert result["signal_decision_row"].all()
    summary = result.attrs["temporal_policy_summary"]
    assert summary["decision_rows"] == 5
    assert summary["entry_count"] == 1
    assert summary["exit_count"] == 1
    assert summary["target_change_count"] == 2


def test_completed_15_minute_schedule_accepts_only_bucket_closes() -> None:
    timestamps = pd.date_range("2026-01-02 09:00", periods=30, freq="min")
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "ticker": "AP",
            "signal": range(30),
        }
    )
    frame.attrs.update(
        {
            "data_frequency": "intraday",
            "factor_metadata": {"signal_frequency": "completed_15min_bar"},
            "factor_contract": {"execution_mode": "risk_desk"},
        }
    )

    policy = resolve_signal_holding_policy(frame)
    result = apply_signal_holding_policy(frame, policy, candidate_col="signal")

    assert policy.signal_frequency == SIGNAL_FIXED_INTERVAL
    assert policy.decision_interval == 15
    assert policy.decision_unit == "minutes"
    decision_times = result.loc[result["signal_decision_row"], "datetime"].tolist()
    assert decision_times == [timestamps[14], timestamps[29]]
    assert result.loc[13, "signal"] == 0.0
    assert result.loc[14, "signal"] == 14.0
    assert result.loc[28, "signal"] == 14.0
    assert result.loc[29, "signal"] == 29.0
    assert len(temporal_metric_rows(result)) == 2


def test_fixed_holding_period_expires_without_a_new_decision() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-02 09:00", periods=7, freq="min"),
            "ticker": "AP",
            "signal": 1.0,
        }
    )
    policy = SignalHoldingPolicy(
        policy_id="tmp_test_fixed",
        data_frequency="intraday",
        signal_frequency="fixed_interval",
        decision_interval=3,
        decision_unit="bars",
        holding_mode=HOLD_FIXED_PERIOD,
        holding_period=2,
        holding_unit="bars",
    )

    result = apply_signal_holding_policy(frame, policy, candidate_col="signal")

    assert result["signal_decision_row"].tolist() == [True, False, False, True, False, False, True]
    assert result["signal"].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert result["signal_holding_age"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]


def test_factor_managed_holding_preserves_stateful_target() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "ticker": "AP",
            "signal": [0.0, 1.0, 1.0, 1.0, -1.0, 0.0],
            "trend_active_state": [0.0, 1.0, 1.0, 1.0, -1.0, 0.0],
        }
    )
    frame.attrs.update(
        {
            "data_frequency": "daily",
            "factor_contract": {"execution_mode": "direct"},
        }
    )

    policy = resolve_signal_holding_policy(frame)
    result = apply_signal_holding_policy(frame, policy, candidate_col="signal")

    assert policy.holding_mode == HOLD_FACTOR_MANAGED
    assert result["signal"].tolist() == frame["signal"].tolist()
    summary = result.attrs["temporal_policy_summary"]
    assert summary["entry_count"] == 1
    assert summary["reversal_count"] == 1
    assert summary["exit_count"] == 1


def test_liquidity_failure_resets_carried_target_until_next_decision() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-02 09:00", periods=7, freq="min"),
            "ticker": "AP",
            "signal": 1.0,
            "liquidity_eligible": [True, True, False, True, True, True, True],
        }
    )
    policy = SignalHoldingPolicy(
        policy_id="tmp_test_liquidity_reset",
        data_frequency="intraday",
        signal_frequency="fixed_interval",
        decision_interval=4,
        decision_unit="bars",
        holding_mode=HOLD_UNTIL_NEXT_DECISION,
        holding_unit="bars",
    )

    result = apply_signal_holding_policy(frame, policy, candidate_col="signal")

    assert result["signal"].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert result.loc[2, "signal_decision_reason"] == "liquidity_blocked"


def test_nullable_object_liquidity_mask_is_normalized_before_cumsum() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-02 09:00",
                periods=5,
                freq="min",
            ),
            "ticker": "AP",
            "signal": 1.0,
            "liquidity_eligible": pd.Series(
                [True, True, None, "true", "false"],
                dtype=object,
            ),
        }
    )
    policy = SignalHoldingPolicy(
        policy_id="tmp_test_nullable_liquidity",
        data_frequency="intraday",
        signal_frequency="fixed_interval",
        decision_interval=1,
        decision_unit="bars",
        holding_mode=HOLD_UNTIL_NEXT_DECISION,
        holding_unit="bars",
    )

    result = apply_signal_holding_policy(frame, policy, candidate_col="signal")

    assert result["signal"].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert result["signal_decision_reason"].tolist() == [
        "scheduled_decision",
        "scheduled_decision",
        "liquidity_blocked",
        "scheduled_decision",
        "liquidity_blocked",
    ]


def test_invalid_fixed_holding_contract_fails_clearly() -> None:
    with pytest.raises(ValueError, match="holding_period"):
        SignalHoldingPolicy(
            policy_id="tmp_invalid",
            data_frequency="daily",
            signal_frequency="session_close",
            decision_interval=1,
            decision_unit="sessions",
            holding_mode="fixed_period",
        )
