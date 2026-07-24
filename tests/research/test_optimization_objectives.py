from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from oqp.optimization import FrozenResearchInputs, SearchBudget
from oqp.research.optional_optimization import (
    Phase8ExperimentSpec,
    Phase8FoldConfig,
)
from oqp.research.optimization_objectives import (
    OptimizationObjectiveRegistry,
    audit_optimization_objectives,
    evaluate_objective_profile_metrics,
    validate_phase8_objective_profile,
    write_optimization_objective_readiness,
)


def _phase8_spec(
    profile_id: str,
    *,
    component_id: str,
    frozen_components: dict[str, str] | None = None,
) -> Phase8ExperimentSpec:
    profile = OptimizationObjectiveRegistry.load().resolve(profile_id)
    return Phase8ExperimentSpec(
        study_id=f"study_{profile.layer}",
        layer=profile.layer,
        component_id=component_id,
        parameter_schema_fingerprint="parameter-schema-v1",
        objective_profile_id=profile.profile_id,
        objective_profile_fingerprint=profile.fingerprint,
        objectives=tuple(
            objective.to_phase8_objective() for objective in profile.objectives
        ),
        selection_priority=profile.selection_priority,
        frozen_inputs=FrozenResearchInputs(
            dataset_fingerprint="dataset-v1",
            holdout_fingerprint="holdout-v1",
        ),
        budget=SearchBudget(max_trials=5),
        fold_config=Phase8FoldConfig(
            fold_count=2,
            minimum_training_periods=10,
            validation_periods=5,
            purge_periods=1,
            embargo_periods=1,
        ),
        holdout_start="2025-01-01",
        frozen_on="2024-12-01",
        constraints=profile.constraints,
        frozen_component_fingerprints=frozen_components or {},
        enabled=True,
    )


def test_registry_has_one_active_profile_per_supported_layer() -> None:
    registry = OptimizationObjectiveRegistry.load()
    assert {profile.layer for profile in registry.profiles.values()} == {
        "factor",
        "sleeve",
        "router",
        "allocator",
    }
    assert all(profile.status == "active" for profile in registry.profiles.values())
    with pytest.raises(KeyError, match="unknown optimization objective profile"):
        registry.resolve("phase9_overlay_risk_v1")


def test_profiles_use_layer_specific_metrics_and_no_universal_score() -> None:
    registry = OptimizationObjectiveRegistry.load()
    factor = registry.resolve("phase9_factor_predictive_v1")
    sleeve = registry.resolve("phase9_sleeve_economics_v1")
    router = registry.resolve("phase9_router_incremental_v1")
    allocator = registry.resolve("phase9_allocator_risk_v1")

    assert {objective.metric for objective in factor.objectives} >= {
        "mean_pearson_ic",
        "mean_rank_ic",
        "pearson_icir",
        "rank_icir",
        "mean_joint_coverage",
    }
    assert sleeve.selection_priority[0] == "net_sharpe"
    assert router.selection_priority[0] == "incremental_value"
    assert allocator.selection_priority[:2] == ("net_sharpe", "tail_risk")
    all_metrics = {
        objective.metric
        for profile in registry.profiles.values()
        for objective in profile.objectives
    }
    assert "universal_score" not in all_metrics


def test_factor_profile_is_compatible_without_upstream_components() -> None:
    registry = OptimizationObjectiveRegistry.load()
    profile = registry.resolve("phase9_factor_predictive_v1")
    spec = _phase8_spec(profile.profile_id, component_id="fac_test")
    validate_phase8_objective_profile(spec, profile)


@pytest.mark.parametrize(
    ("profile_id", "component_id", "frozen_components", "message"),
    [
        (
            "phase9_sleeve_economics_v1",
            "slv_test",
            {},
            "requires 1 frozen upstream",
        ),
        (
            "phase9_router_incremental_v1",
            "rtr_test",
            {"slv_a": "a"},
            "requires 2 frozen upstream",
        ),
        (
            "phase9_allocator_risk_v1",
            "alc_test",
            {"ovl_drawdown": "overlay"},
            "requires 1 frozen upstream",
        ),
    ],
)
def test_profiles_require_the_correct_frozen_upstream_components(
    profile_id: str,
    component_id: str,
    frozen_components: dict[str, str],
    message: str,
) -> None:
    registry = OptimizationObjectiveRegistry.load()
    profile = registry.resolve(profile_id)
    spec = _phase8_spec(
        profile_id,
        component_id=component_id,
        frozen_components=frozen_components,
    )
    with pytest.raises(ValueError, match=message):
        validate_phase8_objective_profile(spec, profile)


def test_profile_fingerprint_and_layer_mismatch_are_rejected() -> None:
    registry = OptimizationObjectiveRegistry.load()
    profile = registry.resolve("phase9_factor_predictive_v1")
    spec = _phase8_spec(profile.profile_id, component_id="fac_test")
    with pytest.raises(ValueError, match="fingerprint changed"):
        validate_phase8_objective_profile(
            replace(spec, objective_profile_fingerprint="stale"), profile
        )
    with pytest.raises(ValueError, match="belongs to factor, not sleeve"):
        validate_phase8_objective_profile(replace(spec, layer="sleeve"), profile)


def test_factor_metric_diagnostics_apply_coverage_gate() -> None:
    profile = OptimizationObjectiveRegistry.load().resolve(
        "phase9_factor_predictive_v1"
    )
    metrics = {
        f"posterior__{objective.metric}": 0.10
        for objective in profile.objectives
    }
    metrics.update(
        {
            f"raw__{objective.metric}": 0.11
            for objective in profile.objectives
        }
    )
    metrics["posterior__mean_joint_coverage"] = 0.49
    diagnostics = evaluate_objective_profile_metrics(profile, metrics)
    assert diagnostics["hard_constraints_satisfied"] is False
    assert diagnostics["objectives"][1]["metric"] == "mean_rank_ic"


def test_readiness_writes_auditable_profile_tables(tmp_path: Path) -> None:
    result = audit_optimization_objectives()
    summary, profiles, objectives, constraints, upstream = result
    assert summary["status"] == "active"
    assert summary["active_profiles"] == 4
    assert summary["universal_score_permitted"] is False
    assert summary["overlay_profile_available"] is False
    destination = write_optimization_objective_readiness(
        summary, profiles, objectives, constraints, upstream, tmp_path
    )
    assert (destination / "readiness.json").exists()
    assert (destination / "objectives.csv").exists()
    assert (destination / "upstream_requirements.csv").exists()
