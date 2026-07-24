from __future__ import annotations

import pandas as pd
import pytest

from oqp.research.success_criteria import (
    CriterionDecision,
    SuccessCriterionRegistry,
    attach_success_criterion_attrs,
    attach_success_criterion_result_attrs,
    evaluate_success_criterion,
    success_criterion_manifest_payload,
)


def test_registry_loads_purpose_specific_profiles() -> None:
    registry = SuccessCriterionRegistry.load()

    assert registry.resolve(
        "factor_cross_sectional_predictive_v1"
    ).research_object == "factor"
    assert registry.resolve("router_incremental_net_value_v1").research_object == "router"
    assert registry.resolve(
        "strategy_daily_internal_net_value_v1"
    ).primary_metric == "validation_net_sharpe"


def test_factor_criterion_passes_only_with_benchmark_and_all_gates() -> None:
    spec = SuccessCriterionRegistry.load().resolve(
        "factor_cross_sectional_predictive_v1"
    )
    result = evaluate_success_criterion(
        spec,
        {
            "validation_mean_rank_ic": 0.035,
            "frozen_baseline_validation_mean_rank_ic": 0.010,
            "validation_rank_icir": 0.40,
            "validation_positive_rank_ic_share": 0.58,
            "validation_valid_ic_periods": 36,
        },
    )

    assert result.decision == CriterionDecision.PASS
    assert result.improvement == pytest.approx(0.025)
    assert all(gate.passed for gate in result.gates)


def test_router_criterion_fails_when_hac_gate_fails() -> None:
    spec = SuccessCriterionRegistry.load().resolve(
        "router_incremental_net_value_v1"
    )
    result = evaluate_success_criterion(
        spec,
        {
            "validation_net_mean_return": 0.012,
            "best_alternative_validation_net_mean_return": 0.010,
            "validation_increment_hac_t": 0.80,
            "validation_net_sharpe": 0.55,
            "validation_valid_routing_periods": 36,
        },
    )

    assert result.decision == CriterionDecision.FAIL
    assert "gate failed: incremental_hac_evidence" in result.failed_reasons


def test_missing_metric_is_incomplete_instead_of_coerced_to_failure() -> None:
    spec = SuccessCriterionRegistry.load().resolve(
        "strategy_daily_internal_net_value_v1"
    )
    result = evaluate_success_criterion(
        spec,
        {
            "validation_net_sharpe": 0.8,
            "validation_annualized_net_return": 0.12,
            "validation_break_even_cost_multiple": 1.5,
            "validation_trading_days": 500,
        },
    )

    assert result.decision == CriterionDecision.INCOMPLETE
    assert result.missing_metrics == (
        "frozen_benchmark_validation_net_sharpe",
    )


def test_attached_profile_and_result_are_manifest_ready() -> None:
    spec = SuccessCriterionRegistry.load().resolve(
        "factor_cross_sectional_predictive_v1"
    )
    result = evaluate_success_criterion(
        spec,
        {
            "validation_mean_rank_ic": 0.02,
            "frozen_baseline_validation_mean_rank_ic": 0.01,
            "validation_rank_icir": 0.2,
            "validation_positive_rank_ic_share": 0.55,
            "validation_valid_ic_periods": 30,
        },
    )
    frame = pd.DataFrame({"value": [1.0]})

    attach_success_criterion_attrs(frame, spec)
    attach_success_criterion_result_attrs(frame, result)
    payload = success_criterion_manifest_payload(frame)

    assert payload["status"] == "pass"
    assert payload["profile_id"] == spec.profile_id
    assert payload["profile_fingerprint"] == spec.fingerprint
    assert payload["evaluation"]["passed"] is True
