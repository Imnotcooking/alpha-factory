from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from oqp.research.factors import load_factor_module
from oqp.research.sleeves import (
    ExtractedSleeveAlignmentError,
    ExtractedSleeveConfig,
    build_extracted_sleeve_targets,
    load_sleeve_module,
    supports_extracted_sleeve_execution,
)


def _config(
    *,
    signal_orientation: str = "higher_is_bullish",
    execution_supported: bool = True,
    rule_family: str = "opposite_event_state",
) -> ExtractedSleeveConfig:
    return ExtractedSleeveConfig(
        sleeve_id="slv_test_opposite_event",
        factor_id="fac_test_event",
        market_vertical="FUTURES_CN",
        rule_family=rule_family,
        source_factor_ids=("fac_test_event",),
        signal_orientation=signal_orientation,
        parameters={
            "construction_geometry": "time_series_stateful",
            "expression": "directional",
            "construction": "opposite_event_state",
            "normalization": "equal_weight_active_signs",
            "zero_signal_action": "preserve_state",
            "missing_signal_action": "preserve_state",
            "target_gross_exposure": 1.0,
            "max_weight_per_contract": 0.40,
            "rescale_after_contract_cap": False,
        },
        execution_supported=execution_supported,
    )


def _event_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=4)
    signals = {
        "A": [1.0, 0.0, None, -2.0],
        "B": [-1.0, 0.0, 3.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
    }
    rows = [
        {
            "date": date,
            "ticker": ticker,
            "factor_score": values[index],
            "forward_return": 1000.0 + index,
        }
        for ticker, values in signals.items()
        for index, date in enumerate(dates)
    ]
    frame = pd.DataFrame(rows).sample(
        frac=1.0,
        random_state=17,
    ).reset_index(drop=True)
    frame.attrs.update(
        {
            "causal_signal_alignment_verified": True,
            "causal_return_alignment_verified": True,
            "sleeve_alignment_attestation": {
                "verified": True,
                "future_return_used_for_selection": False,
            },
        }
    )
    return frame


def _residual_ttl_config(
    *,
    holding_periods: int = 3,
    signal_orientation: str = "higher_is_bullish",
) -> ExtractedSleeveConfig:
    return ExtractedSleeveConfig(
        sleeve_id="slv_test_residual_ttl",
        factor_id="fac_test_residual",
        market_vertical="FUTURES_CN",
        rule_family="residual_event_ttl",
        source_factor_ids=("fac_test_residual",),
        signal_orientation=signal_orientation,
        parameters={
            "construction_geometry": "time_series_stateful",
            "expression": "directional",
            "construction": "residual_event_ttl",
            "normalization": "equal_weight_active_signs",
            "entry_abs_z": 2.0,
            "exit_abs_z": 0.5,
            "holding_periods": holding_periods,
            "holding_unit": "sessions",
            "same_direction_entry_action": (
                "preserve_state_and_advance_age"
            ),
            "opposite_entry_action": "flip_and_reset_age",
            "missing_signal_action": "preserve_state_and_advance_age",
            "ttl_expiry_timing": "before_next_session_target",
            "target_gross_exposure": 1.0,
            "max_weight_per_contract": 0.40,
            "rescale_after_contract_cap": False,
        },
        execution_supported=True,
    )


def _cross_sectional_z_tail_config(
    *,
    signal_orientation: str = "higher_is_bullish",
    parameters: dict[str, object] | None = None,
) -> ExtractedSleeveConfig:
    frozen_parameters: dict[str, object] = {
        "construction_geometry": "cross_sectional",
        "expression": "directional",
        "construction": "cross_sectional_z_tail",
        "normalization": "equal_weight_active_signs",
        "selection": "inclusive_absolute_z_score_at_least_one",
        "z_threshold": 1.0,
        "threshold_inclusive": True,
        "missing_signal_action": "flat",
        "non_tail_signal_action": "flat",
        "holding_rule": "until_next_decision",
        "state_carry": False,
        "target_gross_exposure": 1.0,
        "net_exposure_policy": "floating_from_tail_count_imbalance",
        "additional_row_shift_periods": 0,
        "contract_cap_owner": "strategy_allocator",
        "max_weight_per_contract": None,
    }
    frozen_parameters.update(parameters or {})
    return ExtractedSleeveConfig(
        sleeve_id="slv_test_cross_sectional_z_tail",
        factor_id="fac_test_cross_sectional_z",
        market_vertical="FUTURES_CN",
        rule_family="cross_sectional_z_tail",
        source_factor_ids=("fac_test_cross_sectional_z",),
        signal_orientation=signal_orientation,
        parameters=frozen_parameters,
        execution_supported=True,
    )


def _causally_attested_panel(
    signals: list[float | None],
    *,
    ticker: str = "A",
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-02-03", periods=len(signals)),
            "ticker": ticker,
            "factor_score": signals,
            "forward_return": [
                1000.0 + index for index in range(len(signals))
            ],
        }
    )
    frame.attrs.update(
        {
            "causal_signal_alignment_verified": True,
            "causal_return_alignment_verified": True,
            "sleeve_alignment_attestation": {
                "verified": True,
                "additional_row_shift_periods": 0,
                "future_return_used_for_selection": False,
            },
        }
    )
    return frame


def _cross_sectional_z_tail_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-03-03", periods=2)
    signals = {
        dates[0]: {
            "A": -1.0,
            "B": -0.999,
            "C": 1.0,
            "D": 2.0,
            "E": None,
        },
        dates[1]: {
            "A": 1.2,
            "B": 0.0,
            "C": None,
            "D": 0.8,
            "E": 0.9,
        },
    }
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "factor_score": score,
                "forward_return": 1000.0 + index,
            }
            for index, (date, ticker, score) in enumerate(
                (
                    (date, ticker, score)
                    for date, cross_section in signals.items()
                    for ticker, score in cross_section.items()
                )
            )
        ]
    ).sample(frac=1.0, random_state=23).reset_index(drop=True)
    frame.attrs.update(
        {
            "causal_signal_alignment_verified": True,
            "causal_return_alignment_verified": True,
            "sleeve_alignment_attestation": {
                "verified": True,
                "additional_row_shift_periods": 0,
                "future_return_used_for_selection": False,
            },
        }
    )
    return frame


def test_cross_sectional_z_tail_is_inclusive_daily_and_all_active_equal() -> None:
    panel = _cross_sectional_z_tail_panel()
    result = build_extracted_sleeve_targets(
        panel,
        _cross_sectional_z_tail_config(),
    )
    positions = result.positions.set_index(["date", "ticker"]).sort_index()
    dates = sorted(panel["date"].unique())

    first = positions.loc[dates[0]]
    assert first["tail_selected"].tolist() == [
        True,
        False,
        True,
        True,
        False,
    ]
    assert first["target_weight"].tolist() == pytest.approx(
        [-1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]
    )
    assert first["target_weight"].abs().sum() == pytest.approx(1.0)
    assert first["target_weight"].sum() == pytest.approx(1.0 / 3.0)
    assert first["tail_threshold_boundary"].sum() == 2

    second = positions.loc[dates[1]]
    assert second["tail_selected"].sum() == 1
    assert second.loc["A", "target_weight"] == pytest.approx(1.0)
    assert second.drop(index="A")["target_weight"].eq(0.0).all()
    second_daily = result.daily_summary.set_index("date").loc[dates[1]]
    assert bool(second_daily["one_sided_selection"])
    assert second_daily["net_exposure"] == pytest.approx(1.0)

    execution = result.positions.attrs["extracted_sleeve_execution"]
    assert execution["additional_row_shift_periods"] == 0
    assert execution["contract_cap_owner"] == "strategy_allocator"
    assert execution["sleeve_contract_cap"] is None
    assert execution["future_return_used"] is False


def test_cross_sectional_z_tail_is_order_invariant_and_ignores_returns() -> None:
    panel = _cross_sectional_z_tail_panel()
    config = _cross_sectional_z_tail_config()
    baseline = build_extracted_sleeve_targets(panel, config).positions

    shocked = panel.sample(frac=1.0, random_state=71).reset_index(drop=True)
    shocked["forward_return"] = -999999.0
    shocked.attrs.update(panel.attrs)
    repeated = build_extracted_sleeve_targets(shocked, config).positions

    columns = ["date", "ticker", "tail_selected", "target_weight"]
    pd.testing.assert_frame_equal(
        baseline[columns].reset_index(drop=True),
        repeated[columns].reset_index(drop=True),
    )


def test_cross_sectional_z_tail_respects_orientation_without_reselecting() -> None:
    panel = _cross_sectional_z_tail_panel()
    bullish = build_extracted_sleeve_targets(
        panel,
        _cross_sectional_z_tail_config(),
    ).positions
    bearish = build_extracted_sleeve_targets(
        panel,
        _cross_sectional_z_tail_config(
            signal_orientation="higher_is_bearish",
        ),
    ).positions

    pd.testing.assert_series_equal(
        bullish["tail_selected"],
        bearish["tail_selected"],
        check_names=False,
    )
    assert bearish["target_weight"].tolist() == pytest.approx(
        (-bullish["target_weight"]).tolist()
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"max_weight_per_contract": 0.05},
            "belongs to the strategy allocator",
        ),
        (
            {"stop_abs_z": 3.0},
            "embeds non-sleeve risk/router fields",
        ),
    ],
)
def test_cross_sectional_z_tail_rejects_embedded_risk_fields(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_extracted_sleeve_targets(
            _cross_sectional_z_tail_panel(),
            _cross_sectional_z_tail_config(parameters=override),
        )


def test_opposite_event_state_preserves_zeros_and_missing_then_flips_same_row() -> None:
    result = build_extracted_sleeve_targets(_event_panel(), _config())
    positions = result.positions.set_index(["ticker", "date"]).sort_index()
    dates = pd.bdate_range("2025-01-02", periods=4)

    assert positions.loc["A", "directional_state"].tolist() == [
        1.0,
        1.0,
        1.0,
        -1.0,
    ]
    assert positions.loc["B", "directional_state"].tolist() == [
        -1.0,
        -1.0,
        1.0,
        1.0,
    ]
    assert positions.loc["C", "directional_state"].tolist() == [
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert positions.loc["A", "state_age_periods"].tolist() == [1, 2, 3, 1]
    assert positions.loc["B", "state_age_periods"].tolist() == [1, 2, 1, 2]
    assert positions.loc["C", "state_age_periods"].tolist() == [0, 1, 2, 3]

    first = result.positions.loc[
        result.positions["date"].eq(dates[0])
    ].set_index("ticker")
    assert first.loc["A", "uncapped_target_weight"] == pytest.approx(0.5)
    assert first.loc["B", "uncapped_target_weight"] == pytest.approx(-0.5)
    assert first.loc["A", "target_weight"] == pytest.approx(0.4)
    assert first.loc["B", "target_weight"] == pytest.approx(-0.4)
    assert first["target_weight"].abs().sum() == pytest.approx(0.8)

    second = result.positions.loc[
        result.positions["date"].eq(dates[1])
    ].set_index("ticker")
    assert second.loc["A", "target_weight"] == pytest.approx(1.0 / 3.0)
    assert second.loc["B", "target_weight"] == pytest.approx(-1.0 / 3.0)
    assert second.loc["C", "target_weight"] == pytest.approx(1.0 / 3.0)

    assert bool(positions.loc[("A", dates[3]), "state_flip_event"])
    assert bool(positions.loc[("B", dates[2]), "state_flip_event"])
    assert bool(
        positions.loc[
            ("A", dates[2]),
            "state_preserved_without_event",
        ]
    )
    assert result.daily_summary.loc[
        result.daily_summary["date"].eq(dates[0]),
        "contract_cap_count",
    ].item() == 2
    assert result.positions.attrs["extracted_sleeve_execution"][
        "future_return_used"
    ] is False
    assert result.positions.attrs["extracted_sleeve_execution"][
        "state_update_timing"
    ] == "same_decision_row"


def test_residual_event_ttl_freezes_entry_decay_flip_and_expiry_order() -> None:
    panel = _causally_attested_panel(
        [2.0, 2.5, None, 1.0, 2.5, -2.2, -0.5]
    )
    result = build_extracted_sleeve_targets(
        panel,
        _residual_ttl_config(),
    )
    positions = result.positions.set_index("date").sort_index()
    dates = panel["date"].tolist()

    assert positions["directional_state"].tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        -1.0,
        0.0,
    ]
    assert positions["state_age_periods"].tolist() == [1, 2, 3, 0, 1, 1, 0]
    assert bool(positions.loc[dates[0], "state_entry_event"])
    assert bool(positions.loc[dates[1], "same_direction_entry_ignored"])
    assert bool(positions.loc[dates[2], "state_preserved_on_missing"])
    assert bool(positions.loc[dates[3], "ttl_exit_event"])
    assert positions.loc[dates[3], "lifecycle_exit_reason"] == "ttl_expiry"
    assert bool(positions.loc[dates[5], "state_flip_event"])
    assert positions.loc[dates[5], "lifecycle_exit_reason"] == (
        "opposite_entry_flip"
    )
    assert bool(positions.loc[dates[6], "decay_exit_event"])
    assert positions.loc[dates[6], "lifecycle_exit_reason"] == (
        "decay_threshold"
    )
    assert positions.loc[dates[0], "target_weight"] == pytest.approx(0.40)
    assert positions.loc[dates[3], "target_weight"] == 0.0
    assert positions.loc[dates[5], "target_weight"] == pytest.approx(-0.40)
    assert result.daily_summary["contract_cap_count"].sum() == 5
    assert result.positions.attrs["extracted_sleeve_execution"][
        "additional_row_shift_periods"
    ] == 0
    assert result.positions.attrs["extracted_sleeve_execution"][
        "future_return_used"
    ] is False

    shocked = panel.copy()
    shocked["forward_return"] = -999999.0
    shocked.attrs.update(panel.attrs)
    shocked_positions = build_extracted_sleeve_targets(
        shocked,
        _residual_ttl_config(),
    ).positions
    pd.testing.assert_series_equal(
        result.positions["target_weight"],
        shocked_positions["target_weight"],
        check_names=False,
    )


def test_slv034_twenty_session_ttl_is_flat_before_age_twenty_one() -> None:
    sleeve = load_sleeve_module("slv_034_Residual_Event_20D_TTL")
    panel = _causally_attested_panel([2.0, *([1.0] * 20)])
    config = sleeve.build_config(
        "fac_007_StatArb_Pairs",
        market_vertical="FUTURES_CN",
        signal_orientation="higher_is_bullish",
    )
    result = build_extracted_sleeve_targets(panel, config)
    positions = result.positions.sort_values("date").reset_index(drop=True)

    assert positions.loc[:19, "directional_state"].eq(1.0).all()
    assert positions.loc[:19, "state_age_periods"].tolist() == list(
        range(1, 21)
    )
    assert positions.loc[:19, "target_weight"].eq(0.12).all()
    assert positions.loc[20, "directional_state"] == 0.0
    assert positions.loc[20, "state_age_periods"] == 0
    assert bool(positions.loc[20, "ttl_exit_event"])
    assert positions.loc[20, "target_weight"] == 0.0


def test_residual_ttl_rejects_an_embedded_stop_rule() -> None:
    config = _residual_ttl_config()
    config = replace(
        config,
        parameters={
            **dict(config.parameters or {}),
            "stop_abs_z": 4.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="stop_abs_z must be implemented as a separate risk overlay",
    ):
        build_extracted_sleeve_targets(
            _causally_attested_panel([2.0, 1.0]),
            config,
        )


def test_opposite_event_state_ignores_forward_returns_and_respects_orientation() -> None:
    panel = _event_panel()
    bullish = build_extracted_sleeve_targets(panel, _config()).positions

    shocked = panel.copy()
    shocked["forward_return"] = -999999.0
    shocked.attrs.update(panel.attrs)
    shocked_result = build_extracted_sleeve_targets(
        shocked,
        _config(),
    ).positions
    pd.testing.assert_series_equal(
        bullish["target_weight"],
        shocked_result["target_weight"],
        check_names=False,
    )

    bearish = build_extracted_sleeve_targets(
        panel,
        _config(signal_orientation="higher_is_bearish"),
    ).positions
    first_date = bearish["date"].min()
    first = bearish.loc[bearish["date"].eq(first_date)].set_index("ticker")
    assert first.loc["A", "directional_state"] == -1.0
    assert first.loc["B", "directional_state"] == 1.0


def test_extracted_execution_requires_both_opt_in_and_registered_rule() -> None:
    disabled = _config(execution_supported=False)
    unregistered = _config(rule_family="fixed_event_hold")

    assert supports_extracted_sleeve_execution(_config())
    assert not supports_extracted_sleeve_execution(disabled)
    assert not supports_extracted_sleeve_execution(unregistered)

    with pytest.raises(ValueError, match="execution_supported=False"):
        build_extracted_sleeve_targets(_event_panel(), disabled)
    with pytest.raises(ValueError, match="no registered execution adapter"):
        build_extracted_sleeve_targets(_event_panel(), unregistered)


def test_extracted_execution_requires_causal_attestation() -> None:
    panel = _event_panel()
    panel.attrs.clear()

    with pytest.raises(
        ExtractedSleeveAlignmentError,
        match="upstream causal signal attestation",
    ):
        build_extracted_sleeve_targets(panel, _config())


def test_fac004_breakout_event_becomes_one_persistent_active_state() -> None:
    factor = load_factor_module("fac_004_Donchian_Breakout")
    sleeve = load_sleeve_module(
        "slv_029_Opposite_Event_Directional_State"
    )
    dates = pd.bdate_range("2025-01-02", periods=60)
    rows: list[dict] = []
    for ticker in ("AU", "AG"):
        for index, date in enumerate(dates):
            high, low, close = 101.0, 99.0, 100.0
            if ticker == "AU" and index == 50:
                high, close = 106.0, 105.0
            if ticker == "AU" and index == 54:
                low, close = 94.0, 95.0
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )

    events = factor.compute(pd.DataFrame(rows))
    events.attrs.update(
        {
            "causal_signal_alignment_verified": True,
            "causal_return_alignment_verified": True,
            "sleeve_alignment_attestation": {
                "verified": True,
                "additional_row_shift_periods": 0,
            },
        }
    )
    config = sleeve.build_config(
        factor.FACTOR_ID,
        market_vertical="FUTURES_CN",
        signal_orientation=factor.SIGNAL_ORIENTATION,
    )
    result = build_extracted_sleeve_targets(events, config)
    au = (
        result.positions.loc[result.positions["ticker"].eq("AU")]
        .set_index("date")
        .sort_index()
    )
    daily = result.daily_summary.set_index("date")

    assert daily.loc[dates[50], "active_products"] == 0
    assert daily.loc[dates[51], "active_products"] == 1
    assert daily.loc[dates[59], "active_products"] == 1
    assert au.loc[dates[51], "directional_state"] == 1.0
    assert au.loc[dates[54], "directional_state"] == 1.0
    assert au.loc[dates[55], "directional_state"] == -1.0
    assert bool(au.loc[dates[55], "state_flip_event"])
    assert au.loc[dates[51], "target_weight"] == pytest.approx(0.12)
