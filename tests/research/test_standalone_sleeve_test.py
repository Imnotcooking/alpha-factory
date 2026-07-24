from __future__ import annotations

import pandas as pd
import pytest

from oqp.research.sleeves import (
    SleeveConstructionConfig,
    StandaloneSleeveTestConfig,
    build_sleeve_evidence,
    build_sleeve_targets,
    build_standalone_sleeve_test,
    load_standalone_sleeve_test_bundle,
    write_standalone_sleeve_test_bundle,
)
from oqp.research.success_criteria import SuccessCriterionRegistry


def _phase3_bundle(*, negative_holdout: bool = False):
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=400)
    for date_index, date in enumerate(dates):
        split = "validation" if date_index < 320 else "holdout"
        scale = 1.0 + (date_index % 40) / 20.0
        direction = -1.0 if negative_holdout and split == "holdout" else 1.0
        for product_index in range(10):
            centered = product_index - 4.5
            rows.append(
                {
                    "date": date,
                    "ticker": f"P{product_index:02d}",
                    "sector": "A" if product_index < 5 else "B",
                    "alpha_score": float(product_index),
                    "research_split": split,
                    "forward_return": direction * centered * 0.002 * scale,
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
    config = SleeveConstructionConfig(
        sleeve_id="slv_test",
        factor_id="fac_test",
        market_vertical="FUTURES_CN",
        max_weight_per_contract=0.30,
    )
    construction = build_sleeve_targets(frame, config)
    return build_sleeve_evidence(
        construction,
        capital=1_000_000.0,
        slippage_ticks_per_side=0.5,
    )


def test_registry_contains_predeclared_standalone_sleeve_criterion() -> None:
    criterion = SuccessCriterionRegistry.load().resolve(
        "sleeve_daily_standalone_net_value_v1"
    )
    assert criterion.research_object == "sleeve"
    assert criterion.primary_metric == "validation_net_sharpe"
    assert {gate.metric for gate in criterion.gates} == {
        "validation_annualized_net_return",
        "validation_break_even_cost_multiple",
        "validation_trading_days",
    }


def test_positive_validation_and_holdout_are_router_eligible() -> None:
    bundle = build_standalone_sleeve_test(_phase3_bundle())
    validation = bundle.split_metrics.set_index("research_split").loc["validation"]

    assert bundle.criterion_result.passed
    assert bundle.summary["holdout_confirmation_passed"] is True
    assert bundle.summary["router_eligible"] is True
    assert bundle.summary["standalone_status"] == "eligible_for_router_research"
    assert validation["active_date_count"] == 320
    assert validation["net_sharpe"] > 0
    assert validation["break_even_cost_multiple"] > 1
    assert validation["active_net_hit_rate"] == pytest.approx(1.0)
    assert validation["mean_effective_products"] == pytest.approx(4.0)
    assert bundle.product_contribution.query("research_split == 'full'")[
        "absolute_net_contribution_share"
    ].sum() == pytest.approx(1.0)


def test_holdout_failure_blocks_router_after_validation_pass() -> None:
    bundle = build_standalone_sleeve_test(_phase3_bundle(negative_holdout=True))

    assert bundle.criterion_result.passed
    assert bundle.summary["holdout_confirmation_passed"] is False
    assert bundle.summary["router_eligible"] is False
    assert bundle.summary["standalone_status"] == "blocked_holdout_confirmation"


def test_extreme_threshold_is_frozen_on_validation_and_event_study_is_complete() -> None:
    bundle = build_standalone_sleeve_test(
        _phase3_bundle(),
        StandaloneSleeveTestConfig(
            extreme_event_quantile=0.95,
            extreme_event_pre_periods=3,
            extreme_event_post_periods=4,
        ),
    )
    summary = bundle.summary["extreme_event"]

    assert summary["validation_threshold"] > 0
    assert summary["validation_event_count"] > 0
    assert set(bundle.extreme_event_study["relative_day"]) == set(range(-3, 5))
    assert set(bundle.extreme_window_summary["window"]) == {
        "pre_event",
        "event",
        "post_event",
        "non_event",
    }
    assert bundle.manifest["extreme_event_protocol"]["threshold_sample"] == "validation"
    assert bundle.manifest["extreme_event_protocol"]["use"] == "diagnostic_only_not_router_input"


def test_reconciliation_rejects_modified_phase3_daily_returns() -> None:
    phase3 = _phase3_bundle()
    phase3.daily_returns.loc[0, "net_return"] += 0.01
    with pytest.raises(ValueError, match="does not reconcile"):
        build_standalone_sleeve_test(phase3)


def test_config_blocks_optimization() -> None:
    with pytest.raises(ValueError, match="cannot permit optimization"):
        StandaloneSleeveTestConfig(optimization_permitted=True)


def test_standalone_bundle_round_trip(tmp_path) -> None:
    original = build_standalone_sleeve_test(_phase3_bundle())
    output = write_standalone_sleeve_test_bundle(original, tmp_path / "standalone")
    restored = load_standalone_sleeve_test_bundle(output)

    assert restored.config == original.config
    assert restored.criterion.fingerprint == original.criterion.fingerprint
    assert restored.criterion_result.decision == original.criterion_result.decision
    assert restored.summary["router_eligible"] is True
    pd.testing.assert_frame_equal(restored.daily_diagnostics, original.daily_diagnostics)
    assert {path.name for path in output.iterdir()} == {
        "daily_diagnostics.parquet",
        "extreme_event_study.csv",
        "extreme_events.csv",
        "extreme_window_summary.csv",
        "gate_evaluation.csv",
        "manifest.json",
        "product_contribution.csv",
        "sector_contribution.csv",
        "split_metrics.csv",
        "summary.json",
        "yearly_contribution.csv",
    }
