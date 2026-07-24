from __future__ import annotations

from oqp.optimization import OptimizationMethodRegistry
from oqp.research.optimization_objectives import OptimizationObjectiveRegistry


def test_method_registry_covers_the_supported_optimization_jobs() -> None:
    registry = OptimizationMethodRegistry.load()
    assert set(registry.purposes) == {
        "factor_parameter",
        "sleeve_parameter",
        "router_parameter",
        "allocator_parameter",
        "overlay_parameter",
        "model_hyperparameter",
        "model_weight_training",
        "portfolio_allocation",
        "universe_selection",
    }
    assert all(
        method.status == "implemented" for method in registry.methods.values()
    )


def test_every_purpose_resolves_its_declared_methods() -> None:
    registry = OptimizationMethodRegistry.load()
    for purpose in registry.purposes.values():
        assert registry.resolve_method(purpose.primary_method)
        if purpose.benchmark_method:
            assert registry.resolve_method(purpose.benchmark_method)
        for method_id in purpose.alternative_methods:
            assert registry.resolve_method(method_id)


def test_governed_purposes_reference_active_objective_profiles() -> None:
    methods = OptimizationMethodRegistry.load()
    objectives = OptimizationObjectiveRegistry.load()
    governed = [
        purpose
        for purpose in methods.purposes.values()
        if purpose.status == "governed"
    ]
    assert {purpose.layer for purpose in governed} == {
        "factor",
        "sleeve",
        "router",
        "allocator",
    }
    for purpose in governed:
        profile = objectives.resolve(str(purpose.objective_profile_id))
        assert profile.layer == purpose.layer


def test_distinct_jobs_use_distinct_optimization_families() -> None:
    registry = OptimizationMethodRegistry.load()
    assert (
        registry.resolve_purpose("factor_parameter").primary_method == "tpe"
    )
    assert (
        registry.resolve_purpose("sleeve_parameter").primary_method == "grid"
    )
    assert (
        registry.resolve_purpose("allocator_parameter").primary_method
        == "nsga2"
    )
    assert (
        registry.resolve_purpose("model_weight_training").primary_method
        == "adamw"
    )
    assert (
        registry.resolve_purpose("portfolio_allocation").primary_method
        == "convex_qp"
    )


def test_unready_purposes_are_explicitly_blocked_or_specialized() -> None:
    registry = OptimizationMethodRegistry.load()
    overlay = registry.resolve_purpose("overlay_parameter")
    universe = registry.resolve_purpose("universe_selection")
    assert overlay.status == "blocked"
    assert overlay.objective_profile_id is None
    assert overlay.blocking_reason
    assert universe.status == "experimental"
    assert universe.blocking_reason
