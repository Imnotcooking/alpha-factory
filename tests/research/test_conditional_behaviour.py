from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apps.research_dashboard.views.conditional_behaviour_panel import (
    load_conditional_behaviour_snapshot,
)
from oqp.research.sleeves import (
    ConditionalBehaviourConfig,
    SleeveConstructionConfig,
    build_conditional_behaviour,
    build_observable_conditions,
    build_sleeve_evidence,
    build_sleeve_targets,
    build_standalone_sleeve_test,
    load_conditional_behaviour_bundle,
    load_observable_conditions_bundle,
    write_conditional_behaviour_bundle,
    write_observable_conditions_bundle,
)


def _phase3_bundle():
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=400)
    for date_index, date in enumerate(dates):
        split = "validation" if date_index < 320 else "holdout"
        for product_index in range(10):
            centered = product_index - 4.5
            rows.append(
                {
                    "date": date,
                    "ticker": f"P{product_index:02d}",
                    "sector": "A" if product_index < 5 else "B",
                    "alpha_score": float(product_index),
                    "research_split": split,
                    "forward_return": centered * 0.0015,
                    "next_symbol": f"P{product_index:02d}01",
                    "next_actual_open": 100.0,
                    "next_multiplier": 10.0,
                    "next_tick_size": 0.01,
                    "next_fee_type": "fixed",
                    "next_fee_open": 0.01,
                    "next_fee_close_today": 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["causal_return_alignment_verified"] = True
    frame.attrs["input_data_fingerprint"] = "data-sha"
    frame.attrs["factor_definition_fingerprint"] = "factor-sha"
    construction = build_sleeve_targets(
        frame,
        SleeveConstructionConfig(
            sleeve_id="slv_test",
            factor_id="fac_test",
            market_vertical="FUTURES_CN",
            max_weight_per_contract=0.30,
        ),
    )
    return build_sleeve_evidence(
        construction,
        capital=1_000_000.0,
        slippage_ticks_per_side=0.5,
    )


def _market_panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=400)
    closes = np.full(10, 100.0)
    for date_index, date in enumerate(dates):
        for product_index in range(10):
            wave = np.sin(date_index / 11.0 + product_index / 4.0)
            daily_return = 0.0002 * (product_index - 4.5) + 0.004 * wave
            closes[product_index] *= 1.0 + daily_return
            symbol_suffix = "02" if product_index == 0 and date_index >= 180 else "01"
            rows.append(
                {
                    "date": date,
                    "ticker": f"P{product_index:02d}",
                    "symbol": f"P{product_index:02d}{symbol_suffix}",
                    "close": closes[product_index],
                    "volume": 1_000.0 + 10.0 * date_index + 50.0 * product_index,
                    "open_interest": 5_000.0 + 3.0 * date_index + product_index,
                }
            )
    return pd.DataFrame(rows)


def _config() -> ConditionalBehaviourConfig:
    return ConditionalBehaviourConfig(
        volatility_window=10,
        volatility_min_periods=5,
        percentile_window=40,
        percentile_min_history=10,
        volume_percentile_window=30,
        volume_percentile_min_history=10,
        shock_window=40,
        shock_min_history=20,
        minimum_cross_section=10,
        hac_max_lag=3,
    )


def _phase5_bundle():
    phase3 = _phase3_bundle()
    phase4 = build_standalone_sleeve_test(phase3)
    observables = build_observable_conditions(
        _market_panel(), _config(), source_fingerprint="market-sha"
    )
    return phase3, phase4, observables, build_conditional_behaviour(
        phase3, phase4, observables
    )


def test_config_prohibits_optimization_and_router_backtest() -> None:
    with pytest.raises(ValueError, match="neither optimization nor router"):
        ConditionalBehaviourConfig(optimization_permitted=True)
    with pytest.raises(ValueError, match="neither optimization nor router"):
        ConditionalBehaviourConfig(router_backtest_permitted=True)


def test_observable_conditions_do_not_change_when_future_data_changes() -> None:
    panel = _market_panel()
    original = build_observable_conditions(panel, _config())
    cutoff = pd.Timestamp("2024-04-01")
    modified_panel = panel.copy()
    future = modified_panel["date"].gt(cutoff)
    modified_panel.loc[future, "close"] *= 5.0
    modified_panel.loc[future, "volume"] *= 10.0
    modified_panel.loc[future, "open_interest"] *= 0.2
    modified = build_observable_conditions(modified_panel, _config())

    product_columns = [
        "date",
        "ticker",
        "contract_volatility_percentile",
        "volume_percentile",
        "open_interest_change",
    ]
    pd.testing.assert_frame_equal(
        original.product_conditions.loc[
            original.product_conditions["date"].le(cutoff), product_columns
        ].reset_index(drop=True),
        modified.product_conditions.loc[
            modified.product_conditions["date"].le(cutoff), product_columns
        ].reset_index(drop=True),
    )
    market_columns = [
        "date",
        "market_volatility_percentile",
        "dispersion_percentile",
        "shock_threshold",
        "shock_age",
    ]
    pd.testing.assert_frame_equal(
        original.market_conditions.loc[
            original.market_conditions["date"].le(cutoff), market_columns
        ].reset_index(drop=True),
        modified.market_conditions.loc[
            modified.market_conditions["date"].le(cutoff), market_columns
        ].reset_index(drop=True),
    )


def test_open_interest_change_is_missing_on_contract_switch() -> None:
    observables = build_observable_conditions(_market_panel(), _config())
    switched = observables.product_conditions.loc[
        observables.product_conditions["ticker"].eq("P00")
        & observables.product_conditions["date"].eq(
            pd.bdate_range("2023-01-02", periods=400)[180]
        ),
        "open_interest_change",
    ]
    assert len(switched) == 1
    assert switched.isna().all()


def test_phase5_has_all_conditions_metrics_and_no_router_decision() -> None:
    _, _, _, bundle = _phase5_bundle()
    assert bundle.bucket_metrics["condition_id"].nunique() == 8
    assert {
        "net_annualized_mean",
        "net_sharpe",
        "annualized_turnover",
        "annualized_cost",
        "date_count",
        "net_annualized_mean_ci_lower",
        "net_annualized_mean_ci_upper",
    }.issubset(bundle.bucket_metrics.columns)
    assert bundle.manifest["optimization_permitted"] is False
    assert bundle.manifest["router_backtest_permitted"] is False
    assert bundle.manifest["confidence_interval"]["multiple_comparison_adjusted"] is False
    assert "router_decision" not in bundle.summary


def test_contract_condition_contributions_reconcile_to_original_sleeve() -> None:
    phase3, _, _, bundle = _phase5_bundle()
    contract = bundle.condition_daily.loc[
        bundle.condition_daily["condition_id"].eq(
            "contract_volatility_percentile"
        )
    ]
    rebuilt = contract.groupby("date", as_index=False)[
        ["gross_return", "net_return", "cost_return", "turnover"]
    ].sum()
    observed = phase3.daily_returns.loc[
        phase3.daily_returns["date"].isin(rebuilt["date"]),
        ["date", "gross_return", "net_return", "cost_return", "turnover"],
    ]
    merged = observed.merge(rebuilt, on="date", suffixes=("_observed", "_rebuilt"))
    for column in ("gross_return", "net_return", "cost_return", "turnover"):
        assert np.allclose(
            merged[f"{column}_observed"],
            merged[f"{column}_rebuilt"],
            atol=1e-12,
        )


def test_phase5_artifacts_round_trip(tmp_path) -> None:
    _, _, observables, bundle = _phase5_bundle()
    observable_path = write_observable_conditions_bundle(
        observables, tmp_path / "common"
    )
    evidence_path = write_conditional_behaviour_bundle(
        bundle, tmp_path / "factor"
    )
    restored_observables = load_observable_conditions_bundle(observable_path)
    restored = load_conditional_behaviour_bundle(evidence_path)

    assert restored_observables.config == observables.config
    assert restored.config == bundle.config
    assert restored.summary == bundle.summary
    assert set(restored.bucket_metrics["condition_id"]) == set(
        bundle.bucket_metrics["condition_id"]
    )
    assert {path.name for path in evidence_path.iterdir()} == {
        "bucket_metrics.csv",
        "condition_daily.parquet",
        "definitions.json",
        "manifest.json",
        "summary.json",
    }


def test_dashboard_loads_only_complete_market_specific_phase5_bundle(tmp_path) -> None:
    _, _, _, bundle = _phase5_bundle()
    artifact_root = tmp_path / "research"
    write_conditional_behaviour_bundle(
        bundle,
        artifact_root
        / "conditional_behaviour"
        / "fac_test"
        / "FUTURES_CN"
        / "slv_004_Default_Cross_Sectional_Long_Short",
    )

    snapshot = load_conditional_behaviour_snapshot(
        str(artifact_root), "fac_test", "FUTURES_CN"
    )

    assert snapshot is not None
    assert snapshot["summary"]["condition_count"] == 8
    assert len(snapshot["definitions"]) == 8
    assert (
        load_conditional_behaviour_snapshot(
            str(artifact_root), "fac_test", "EQUITY_US"
        )
        is None
    )
