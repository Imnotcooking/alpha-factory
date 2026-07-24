from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from oqp.research.factor_portfolios import (
    FactorPortfolioConfig,
    FactorPortfolioRunner,
    FactorSpec,
)
from oqp.research.sleeves import (
    ExtractedSleeveConfig,
    PersistentSleeveConfig,
    SleeveConstructionConfig,
    load_sleeve_module,
)


FACTOR_ID = "fac_test_daily_score"
RETURN_HORIZON = "close_signal_next_open_to_close"


def _runner(
    *,
    max_weight_per_asset: float = 0.50,
) -> FactorPortfolioRunner:
    return FactorPortfolioRunner(
        FactorPortfolioConfig(
            strategy_id="str_test_daily_factor_sleeve",
            name="Test daily factor sleeve",
            market_vertical="FUTURES_CN",
            factors=(FactorSpec(FACTOR_ID),),
            max_gross_leverage=1.0,
            max_weight_per_asset=max_weight_per_asset,
            return_horizon=RETURN_HORIZON,
        )
    )


def _daily_panel(*, date_count: int = 4, ticker_count: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=date_count)
    rows: list[dict] = []
    for day_index, date in enumerate(dates):
        for ticker_index in range(ticker_count):
            close = (
                100.0
                + ticker_index * 2.0
                + day_index
                + (ticker_index + 1) * 0.015 * day_index**2
            )
            rows.append(
                {
                    "date": date,
                    "ticker": f"P{ticker_index:02d}",
                    "open": close - 0.25,
                    "high": close + 0.50,
                    "low": close - 0.50,
                    "close": close,
                    "sector": "Metals" if ticker_index < ticker_count / 2 else "Energy",
                    "forward_return": 0.001 * (ticker_index - ticker_count / 2),
                    "liquidity_eligible": True,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "market_vertical": "FUTURES_CN",
            "data_frequency": "daily",
            "return_horizon": RETURN_HORIZON,
            "execution_assumption": RETURN_HORIZON,
        }
    )
    return frame


def _factor_module(
    *,
    missing_key: tuple[pd.Timestamp, str] | None = None,
    evaluation_geometry: str = "cross_sectional",
) -> SimpleNamespace:
    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        # Deliberately return a pure score panel. The runner must restore market
        # columns only after factor computation.
        out = frame[["date", "ticker"]].copy()
        ticker_number = pd.to_numeric(
            out["ticker"].str.removeprefix("P"),
            errors="coerce",
        )
        out["factor_score"] = ticker_number
        if missing_key is not None:
            date, ticker = missing_key
            out.loc[
                out["date"].eq(date) & out["ticker"].eq(ticker),
                "factor_score",
            ] = np.nan
        return out

    return SimpleNamespace(
        FACTOR_ID=FACTOR_ID,
        SIGNAL_ORIENTATION="higher_is_bullish",
        FACTOR_METADATA={
            "supported_markets": ["FUTURES_CN"],
            "data_frequency": "daily",
            "signal_orientation": "higher_is_bullish",
        },
        FACTOR_CONTRACT={
            "evaluation_geometry": evaluation_geometry,
            "execution_mode": "risk_desk",
            "alpha_signal_col": "factor_score",
            "execution_weight_col": "factor_score",
            "execution_lag": "next_open",
            "return_assumption": RETURN_HORIZON,
            "supported_markets": ["FUTURES_CN"],
        },
        compute=compute,
    )


def _signed_factor_module() -> SimpleNamespace:
    module = _factor_module()

    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["date", "ticker"]].copy()
        ticker_number = pd.to_numeric(
            out["ticker"].str.removeprefix("P"),
            errors="coerce",
        )
        out["factor_score"] = ticker_number - 2.0
        return out

    module.compute = compute
    return module


def _standard_sleeve_module() -> SimpleNamespace:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> SleeveConstructionConfig:
        return SleeveConstructionConfig(
            sleeve_id="slv_test_standard",
            factor_id=factor_id,
            market_vertical=market_vertical,
            construction_geometry="cross_sectional",
            expression="long_short",
            construction="top_bottom_quantile",
            normalization="equal_weight",
            return_assumption=RETURN_HORIZON,
            signal_orientation=signal_orientation,
            long_fraction=0.20,
            short_fraction=0.20,
            max_weight_per_contract=0.50,
            minimum_cross_section=3,
            minimum_distinct_signals=2,
            signal_timing="already_lagged",
            execution_delay_periods=0,
        )

    return SimpleNamespace(build_config=build_config)


def _time_series_sleeve_module() -> SimpleNamespace:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> SleeveConstructionConfig:
        return SleeveConstructionConfig(
            sleeve_id="slv_test_time_series",
            factor_id=factor_id,
            market_vertical=market_vertical,
            construction_geometry="time_series",
            expression="directional",
            construction="time_series_sign",
            normalization="equal_weight",
            return_assumption=RETURN_HORIZON,
            signal_orientation=signal_orientation,
            winsor_lower_quantile=None,
            winsor_upper_quantile=None,
            max_weight_per_contract=0.50,
            minimum_cross_section=3,
            minimum_distinct_signals=2,
            signal_timing="already_lagged",
            execution_delay_periods=0,
        )

    return SimpleNamespace(build_config=build_config)


def _persistent_sleeve_module() -> SimpleNamespace:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> PersistentSleeveConfig:
        return PersistentSleeveConfig(
            sleeve_id="slv_test_persistent",
            factor_id=factor_id,
            market_vertical=market_vertical,
            construction="quintile_inverse_vol",
            signal_orientation=signal_orientation,
            long_fraction=0.20,
            short_fraction=0.20,
            holding_periods=2,
            max_weight_per_contract=0.50,
            minimum_cross_section=5,
            minimum_distinct_signals=2,
            terminal_cash=True,
        )

    return SimpleNamespace(build_config=build_config)


def _opposite_event_factor_module() -> SimpleNamespace:
    module = _factor_module(evaluation_geometry="time_series")

    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["date", "ticker"]].copy()
        dates = out["date"].drop_duplicates().sort_values().tolist()
        out["factor_score"] = 0.0
        event_map = {
            ("P00", dates[0]): 1.0,
            ("P00", dates[3]): -1.0,
            ("P01", dates[0]): -1.0,
            ("P01", dates[2]): 1.0,
        }
        for (ticker, date), event in event_map.items():
            out.loc[
                out["ticker"].eq(ticker) & out["date"].eq(date),
                "factor_score",
            ] = event
        return out

    module.compute = compute
    return module


def _opposite_event_sleeve_module() -> SimpleNamespace:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> ExtractedSleeveConfig:
        return ExtractedSleeveConfig(
            sleeve_id="slv_test_opposite_event",
            factor_id=factor_id,
            market_vertical=market_vertical,
            rule_family="opposite_event_state",
            source_factor_ids=(factor_id,),
            signal_orientation=signal_orientation,
            parameters={
                "construction_geometry": "time_series_stateful",
                "expression": "directional",
                "construction": "opposite_event_state",
                "normalization": "equal_weight_active_signs",
                "zero_signal_action": "preserve_state",
                "missing_signal_action": "preserve_state",
                "target_gross_exposure": 1.0,
                "max_weight_per_contract": 0.50,
                "rescale_after_contract_cap": False,
            },
            execution_supported=True,
        )

    return SimpleNamespace(build_config=build_config)


def _residual_ttl_factor_module() -> SimpleNamespace:
    module = _factor_module(evaluation_geometry="time_series")
    module.FACTOR_CONTRACT = {
        **module.FACTOR_CONTRACT,
        "execution_lag": "already_lagged",
    }

    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["date", "ticker"]].copy()
        dates = out["date"].drop_duplicates().sort_values().tolist()
        out["factor_score"] = 0.0
        signal_map = {
            ("P00", dates[0]): 2.2,
            ("P00", dates[1]): 1.0,
            ("P00", dates[2]): 0.5,
            ("P01", dates[1]): -2.4,
            ("P01", dates[2]): -1.0,
            ("P01", dates[3]): 0.4,
        }
        for (ticker, date), signal in signal_map.items():
            out.loc[
                out["ticker"].eq(ticker) & out["date"].eq(date),
                "factor_score",
            ] = signal
        return out

    module.compute = compute
    return module


def _residual_ttl_sleeve_module() -> SimpleNamespace:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> ExtractedSleeveConfig:
        return ExtractedSleeveConfig(
            sleeve_id="slv_test_residual_ttl",
            factor_id=factor_id,
            market_vertical=market_vertical,
            rule_family="residual_event_ttl",
            source_factor_ids=(factor_id,),
            signal_orientation=signal_orientation,
            parameters={
                "construction_geometry": "time_series_stateful",
                "expression": "directional",
                "construction": "residual_event_ttl",
                "normalization": "equal_weight_active_signs",
                "entry_abs_z": 2.0,
                "exit_abs_z": 0.5,
                "holding_periods": 20,
                "holding_unit": "sessions",
                "same_direction_entry_action": (
                    "preserve_state_and_advance_age"
                ),
                "opposite_entry_action": "flip_and_reset_age",
                "missing_signal_action": (
                    "preserve_state_and_advance_age"
                ),
                "ttl_expiry_timing": "before_next_session_target",
                "target_gross_exposure": 1.0,
                "max_weight_per_contract": 0.12,
                "rescale_after_contract_cap": False,
            },
            execution_supported=True,
        )

    return SimpleNamespace(build_config=build_config)


def _z_tail_factor_module() -> SimpleNamespace:
    module = _factor_module(evaluation_geometry="cross_sectional")

    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["date", "ticker"]].copy()
        score_by_ticker = {
            "P00": -1.0,
            "P01": -0.999,
            "P02": 1.0,
            "P03": 2.0,
        }
        out["factor_score"] = out["ticker"].map(score_by_ticker).fillna(0.0)
        return out

    module.compute = compute
    return module


def _z_tail_sleeve_module() -> SimpleNamespace:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> ExtractedSleeveConfig:
        return ExtractedSleeveConfig(
            sleeve_id="slv_test_cross_sectional_z_tail",
            factor_id=factor_id,
            market_vertical=market_vertical,
            rule_family="cross_sectional_z_tail",
            source_factor_ids=(factor_id,),
            signal_orientation=signal_orientation,
            parameters={
                "construction_geometry": "cross_sectional",
                "expression": "directional",
                "construction": "cross_sectional_z_tail",
                "normalization": "equal_weight_active_signs",
                "selection": (
                    "inclusive_absolute_z_score_at_least_one"
                ),
                "z_threshold": 1.0,
                "threshold_inclusive": True,
                "missing_signal_action": "flat",
                "non_tail_signal_action": "flat",
                "holding_rule": "until_next_decision",
                "state_carry": False,
                "target_gross_exposure": 1.0,
                "net_exposure_policy": (
                    "floating_from_tail_count_imbalance"
                ),
                "additional_row_shift_periods": 0,
                "contract_cap_owner": "strategy_allocator",
                "max_weight_per_contract": None,
            },
            execution_supported=True,
        )

    return SimpleNamespace(build_config=build_config)


def _build(
    panel: pd.DataFrame,
    sleeve_module: SimpleNamespace,
    *,
    factor_module: SimpleNamespace | None = None,
    max_weight_per_asset: float = 0.50,
):
    with (
        patch(
            "oqp.research.factor_portfolios.runner.load_factor_module",
            return_value=factor_module or _factor_module(),
        ),
        patch(
            "oqp.research.factor_portfolios.runner.load_sleeve_module",
            return_value=sleeve_module,
        ),
    ):
        return _runner(
            max_weight_per_asset=max_weight_per_asset,
        ).build_with_sleeve(
            panel,
            factor_id=FACTOR_ID,
            sleeve_id="slv_test",
        )


def test_daily_next_open_standard_sleeve_uses_no_extra_row_shift() -> None:
    panel = _daily_panel()
    result = _build(panel, _standard_sleeve_module())

    first_date = panel["date"].min()
    first_targets = result.frame.loc[
        result.frame["date"].eq(first_date),
        "target_weight",
    ]
    assert first_targets.abs().gt(0.0).any()
    assert result.frame.attrs["execution_lag"] == "next_open"
    assert result.frame.attrs["sleeve_market_panel_columns_restored"]
    assert result.frame.attrs["sleeve_alignment_attestation"] == {
        "schema_version": 1,
        "verified": True,
        "factor_id": FACTOR_ID,
        "sleeve_id": "slv_test_standard",
        "data_frequency": "daily",
        "factor_execution_lag": "next_open",
        "factor_return_assumption": RETURN_HORIZON,
        "dataset_return_horizon": RETURN_HORIZON,
        "signal_row_semantics": "decision_close",
        "target_row_semantics": (
            "same_decision_row_paired_with_attached_forward_return"
        ),
        "additional_row_shift_periods": 0,
        "forward_return_col": "forward_return",
        "future_return_used_for_selection": False,
    }


def test_canonical_proportional_sleeve_runs_without_restandardizing_score() -> None:
    sleeve_module = load_sleeve_module(
        "slv_028_Capped_Proportional_Score"
    )
    result = _build(
        _daily_panel(),
        sleeve_module,
        factor_module=_signed_factor_module(),
        max_weight_per_asset=0.10,
    )
    first_date = result.frame["date"].min()
    first = (
        result.frame.loc[result.frame["date"].eq(first_date)]
        .set_index("ticker")
        .sort_index()
    )
    score = first["factor_score"]
    expected_pre_cap = score / score.abs().sum()
    expected_sleeve = expected_pre_cap.clip(-0.12, 0.12)

    assert first["decision_target_weight"].tolist() == pytest.approx(
        expected_sleeve.tolist()
    )
    assert first["target_weight"].tolist() == pytest.approx(
        expected_sleeve.clip(-0.10, 0.10).tolist()
    )
    assert first["target_weight"].abs().max() == pytest.approx(0.10)
    assert result.frame.attrs["sleeve_config"]["construction"] == (
        "proportional_score"
    )
    assert result.frame.attrs["sleeve_config"]["normalization"] == (
        "absolute_score_to_gross"
    )
    assert result.frame.attrs["sleeve_alignment_attestation"][
        "additional_row_shift_periods"
    ] == 0


def test_daily_next_open_sleeve_rejects_dataset_horizon_mismatch() -> None:
    panel = _daily_panel()
    panel.attrs["return_horizon"] = "close_signal_next_open_to_next_open"
    panel.attrs["execution_assumption"] = "close_signal_next_open_to_next_open"

    with pytest.raises(
        ValueError,
        match="factor and dataset return horizons are not causally aligned",
    ):
        _build(panel, _standard_sleeve_module())


def test_daily_next_open_time_series_factor_uses_matching_standard_sleeve() -> None:
    result = _build(
        _daily_panel(),
        _time_series_sleeve_module(),
        factor_module=_factor_module(evaluation_geometry="time_series"),
    )

    assert result.frame.attrs["evaluation_geometry"] == "time_series"
    assert result.frame.attrs["sleeve_config"]["construction_geometry"] == (
        "time_series"
    )
    assert result.frame["target_weight"].abs().gt(0.0).any()
    assert result.frame.attrs["sleeve_alignment_attestation"][
        "additional_row_shift_periods"
    ] == 0


def test_persistent_sleeve_dispatch_derives_only_causal_support_columns() -> None:
    panel = _daily_panel(date_count=26, ticker_count=20)
    test_date = panel["date"].drop_duplicates().sort_values().iloc[22]
    panel.loc[
        panel["date"].eq(test_date) & panel["ticker"].eq("P01"),
        "liquidity_eligible",
    ] = False
    factor_module = _factor_module(missing_key=(test_date, "P00"))

    result = _build(
        panel,
        _persistent_sleeve_module(),
        factor_module=factor_module,
    )
    indexed = result.frame.set_index(["date", "ticker"])
    missing_signal_row = indexed.loc[(test_date, "P00")]
    illiquid_row = indexed.loc[(test_date, "P01")]
    assert not bool(missing_signal_row["rank_eligible"])
    assert not bool(illiquid_row["rank_eligible"])
    assert not bool(illiquid_row["tradable"])
    assert indexed.loc[(test_date, "P02"), "execution_sector"] == "Metals"

    p02 = (
        panel.loc[panel["ticker"].eq("P02")]
        .sort_values("date")
        .set_index("date")
    )
    expected_volatility = (
        p02["close"]
        .pct_change(fill_method=None)
        .shift(1)
        .rolling(20, min_periods=20)
        .std(ddof=0)
        .loc[test_date]
    )
    assert indexed.loc[
        (test_date, "P02"),
        "trailing_volatility",
    ] == pytest.approx(expected_volatility)
    assert result.frame.attrs["persistent_sleeve_input_contract"][
        "trailing_volatility_lag_sessions"
    ] == 1
    assert result.frame.attrs["persistent_sleeve_input_contract"][
        "future_return_used"
    ] is False

    shocked = panel.copy()
    final_date = shocked["date"].max()
    shocked.loc[
        shocked["date"].eq(final_date) & shocked["ticker"].eq("P02"),
        "close",
    ] *= 10.0
    shocked.attrs.update(panel.attrs)
    shocked_result = _build(
        shocked,
        _persistent_sleeve_module(),
        factor_module=factor_module,
    )
    original_final_vol = indexed.loc[
        (final_date, "P02"),
        "trailing_volatility",
    ]
    shocked_final_vol = (
        shocked_result.frame.set_index(["date", "ticker"])
        .loc[(final_date, "P02"), "trailing_volatility"]
    )
    assert shocked_final_vol == pytest.approx(original_final_vol)


def test_opposite_event_extracted_sleeve_runs_statefully_without_extra_shift() -> None:
    panel = _daily_panel(date_count=4, ticker_count=10)
    result = _build(
        panel,
        _opposite_event_sleeve_module(),
        factor_module=_opposite_event_factor_module(),
        max_weight_per_asset=0.50,
    )
    indexed = result.frame.set_index(["ticker", "date"]).sort_index()
    dates = panel["date"].drop_duplicates().sort_values().tolist()

    assert indexed.loc["P00", "directional_state"].tolist() == [
        1.0,
        1.0,
        1.0,
        -1.0,
    ]
    assert indexed.loc["P01", "directional_state"].tolist() == [
        -1.0,
        -1.0,
        1.0,
        1.0,
    ]
    assert indexed.loc[("P00", dates[0]), "target_weight"] == pytest.approx(
        0.50
    )
    assert indexed.loc[("P00", dates[1]), "target_weight"] == pytest.approx(
        0.50
    )
    assert indexed.loc[("P01", dates[2]), "target_weight"] == pytest.approx(
        0.50
    )
    assert indexed.loc[("P00", dates[3]), "target_weight"] == pytest.approx(
        -0.50
    )
    assert result.frame.attrs["evaluation_geometry"] == "time_series"
    assert result.frame.attrs["return_assumption"] == RETURN_HORIZON
    assert result.frame.attrs["sleeve_config"]["rule_family"] == (
        "opposite_event_state"
    )
    assert result.frame.attrs["sleeve_alignment_attestation"][
        "additional_row_shift_periods"
    ] == 0
    assert result.frame.attrs["extracted_sleeve_execution"][
        "future_return_used"
    ] is False


def test_opposite_event_extracted_sleeve_rejects_cross_sectional_factor() -> None:
    with pytest.raises(
        ValueError,
        match="factor and extracted sleeve evaluation geometries differ",
    ):
        _build(
            _daily_panel(),
            _opposite_event_sleeve_module(),
            factor_module=_factor_module(evaluation_geometry="cross_sectional"),
        )


def test_residual_ttl_runner_preserves_already_lagged_factor_return_row() -> None:
    panel = _daily_panel(date_count=4, ticker_count=10)
    result = _build(
        panel,
        _residual_ttl_sleeve_module(),
        factor_module=_residual_ttl_factor_module(),
        max_weight_per_asset=0.10,
    )
    indexed = result.frame.set_index(["ticker", "date"]).sort_index()
    dates = panel["date"].drop_duplicates().sort_values().tolist()

    first = indexed.loc[("P00", dates[0])]
    assert first["directional_state"] == 1.0
    assert first["state_age_periods"] == 1
    assert first["decision_target_weight"] == pytest.approx(0.12)
    assert first["target_weight"] == pytest.approx(0.10)
    assert indexed.loc[("P00", dates[1]), "directional_state"] == 1.0
    assert indexed.loc[("P00", dates[2]), "directional_state"] == 0.0
    assert bool(indexed.loc[("P00", dates[2]), "decay_exit_event"])
    assert indexed.loc[("P01", dates[1]), "directional_state"] == -1.0
    assert result.frame.attrs["execution_lag"] == "already_lagged"
    assert result.frame.attrs["return_assumption"] == RETURN_HORIZON
    assert result.frame.attrs["sleeve_alignment_attestation"][
        "signal_row_semantics"
    ] == "pre_aligned_execution_decision_row"
    assert result.frame.attrs["sleeve_alignment_attestation"][
        "additional_row_shift_periods"
    ] == 0
    assert result.frame.attrs["extracted_sleeve_execution"][
        "future_return_used"
    ] is False


def test_cross_sectional_z_tail_runner_keeps_cap_in_allocator() -> None:
    panel = _daily_panel(date_count=2, ticker_count=10)
    result = _build(
        panel,
        _z_tail_sleeve_module(),
        factor_module=_z_tail_factor_module(),
        max_weight_per_asset=0.20,
    )
    first_date = panel["date"].min()
    first = (
        result.frame.loc[result.frame["date"].eq(first_date)]
        .set_index("ticker")
        .sort_index()
    )

    assert first.loc["P00", "decision_target_weight"] == pytest.approx(
        -1.0 / 3.0
    )
    assert first.loc["P02", "decision_target_weight"] == pytest.approx(
        1.0 / 3.0
    )
    assert first.loc["P03", "decision_target_weight"] == pytest.approx(
        1.0 / 3.0
    )
    assert first.loc["P01", "decision_target_weight"] == 0.0
    assert first["decision_target_weight"].abs().sum() == pytest.approx(1.0)
    assert first["target_weight"].abs().max() == pytest.approx(0.20)
    assert first["target_weight"].abs().sum() == pytest.approx(0.60)
    assert result.frame.attrs["sleeve_alignment_attestation"][
        "additional_row_shift_periods"
    ] == 0
    assert result.frame.attrs["extracted_sleeve_execution"][
        "contract_cap_owner"
    ] == "strategy_allocator"
    assert result.frame.attrs["execution_lag"] == "next_open"
    assert result.frame.attrs["extracted_sleeve_execution"][
        "future_return_used"
    ] is False


def test_extracted_sleeve_is_rejected_with_explicit_executor_reason() -> None:
    def build_config(
        factor_id: str,
        *,
        market_vertical: str,
        signal_orientation: str,
    ) -> ExtractedSleeveConfig:
        return ExtractedSleeveConfig(
            sleeve_id="slv_test_extracted",
            factor_id=factor_id,
            market_vertical=market_vertical,
            rule_family="proportional_score",
            source_factor_ids=(factor_id,),
            signal_orientation=signal_orientation,
            execution_supported=False,
        )

    with pytest.raises(
        ValueError,
        match=(
            "ExtractedSleeveConfig.*execution_supported=False.*"
            "cannot be executed"
        ),
    ):
        _build(
            _daily_panel(),
            SimpleNamespace(build_config=build_config),
        )
