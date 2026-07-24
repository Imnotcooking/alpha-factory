from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from oqp.optimization import (
    ConstraintSpec,
    FrozenResearchInputs,
    ObjectiveSpec,
    OptimizationStudyRunner,
    OptimizationStudySpec,
    OptimizationStudyStore,
    SearchBudget,
    TrialEvaluation,
    resolve_component_parameter_schema,
    resolve_component_parameter_values,
)
from oqp.optimization.samplers import build_optuna_sampler
from oqp.optimization.samplers.exhaustive import grid_search_space
from oqp.optimization.samplers.global_continuous import solve_global_continuous


def _compute_demo(
    frame: pd.DataFrame,
    *,
    x: float = 0.0,
    window: int = 2,
) -> pd.DataFrame:
    return frame


def _demo_module() -> SimpleNamespace:
    return SimpleNamespace(
        FACTOR_ID="fac_optimization_demo",
        FACTOR_PARAMETERS={
            "x": {
                "default": 0.0,
                "type": "float",
                "low": -2.0,
                "high": 2.0,
                "step": 0.5,
                "tunable": True,
            },
            "window": {
                "default": 2,
                "type": "int",
                "low": 1,
                "high": 3,
                "step": 1,
                "tunable": True,
            },
        },
        compute=_compute_demo,
    )


def _router_module() -> SimpleNamespace:
    return SimpleNamespace(
        ROUTER_ID="rtr_optimization_demo",
        ROUTER_PARAMETERS={
            "q4_sleeve": {
                "default": "reversal",
                "type": "categorical",
                "choices": ["reversal", "trend"],
                "tunable": True,
            }
        },
    )


def _spec(
    study_id: str,
    sampler_id: str,
    *,
    trials: int = 8,
    n_jobs: int = 1,
    objectives: tuple[ObjectiveSpec, ...] | None = None,
) -> OptimizationStudySpec:
    return OptimizationStudySpec(
        study_id=study_id,
        purpose="factor_parameter",
        component_id="fac_optimization_demo",
        sampler_id=sampler_id,
        objectives=objectives or (ObjectiveSpec("score", "score", "maximize"),),
        constraints=(ConstraintSpec("turnover_cap", "turnover", "<=", 2.0),),
        frozen_inputs=FrozenResearchInputs(
            dataset_fingerprint="synthetic-development-v1",
            holdout_fingerprint="synthetic-holdout-locked-v1",
        ),
        budget=SearchBudget(
            max_trials=trials,
            max_grid_combinations=100,
            n_jobs=n_jobs,
        ),
        seed=7,
    )


def _evaluation(parameters, context) -> TrialEvaluation:
    context.require_development_only()
    score = -((float(parameters["x"]) - 0.5) ** 2) - 0.01 * abs(
        int(parameters["window"]) - 2
    )
    return TrialEvaluation(
        metrics={
            "score": score,
            "turnover": abs(float(parameters["x"])),
            "complexity": float(parameters["window"]),
        },
        fold_metrics=({"fold": 1, "score": score},),
    )


def test_grid_search_runs_every_guarded_combination() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    search_space = grid_search_space(schema)

    result = OptimizationStudyRunner().run(
        _spec("optimization_grid_test", "grid", trials=27),
        schema,
        _evaluation,
    )

    assert len(search_space["x"]) == 9
    assert len(search_space["window"]) == 3
    assert result.trial_count == 27
    assert result.diagnostics["holdout_locked"] is True
    assert result.candidates[0].feasible is True


def test_grid_guard_rejects_combinatorial_explosion() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    spec = _spec("optimization_grid_guard_test", "grid")
    guarded = OptimizationStudySpec(
        study_id=spec.study_id,
        purpose=spec.purpose,
        component_id=spec.component_id,
        sampler_id=spec.sampler_id,
        objectives=spec.objectives,
        constraints=spec.constraints,
        frozen_inputs=spec.frozen_inputs,
        budget=SearchBudget(max_trials=8, max_grid_combinations=10),
    )

    with pytest.raises(ValueError, match="above the guard"):
        OptimizationStudyRunner().run(guarded, schema, _evaluation)


def test_grid_budget_must_cover_full_grid() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )

    with pytest.raises(ValueError, match="budget must cover all"):
        OptimizationStudyRunner().run(
            _spec("partial_grid_is_not_grid_search", "grid", trials=15),
            schema,
            _evaluation,
        )


def test_categorical_router_parameter_enforces_choices_without_numeric_cast() -> None:
    schema = resolve_component_parameter_schema(
        _router_module(), component_type="router"
    )

    resolved = resolve_component_parameter_values(
        schema,
        {"q4_sleeve": "trend"},
        enforce_search_bounds=True,
    )

    assert resolved["q4_sleeve"] == "trend"


def test_intelligent_search_runs_same_budget_random_baseline() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )

    comparison = OptimizationStudyRunner().run_with_random_baseline(
        _spec("optimization_tpe_comparison_test", "tpe", trials=6),
        schema,
        _evaluation,
    )

    assert comparison.baseline.sampler_id == "random"
    assert comparison.challenger.sampler_id == "tpe"
    assert comparison.baseline.trial_count == comparison.challenger.trial_count == 6


def test_multiobjective_study_returns_feasible_pareto_candidates() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    objectives = (
        ObjectiveSpec("score", "score", "maximize"),
        ObjectiveSpec("complexity", "complexity", "minimize"),
    )

    result = OptimizationStudyRunner().run(
        _spec(
            "optimization_nsga2_test",
            "nsga2",
            trials=8,
            objectives=objectives,
        ),
        schema,
        _evaluation,
    )

    assert result.trial_count == 8
    assert result.candidates
    assert all(candidate.feasible for candidate in result.candidates)
    assert all(len(candidate.objective_values) == 2 for candidate in result.candidates)


def test_cmaes_runs_numeric_population_study() -> None:
    pytest.importorskip("cmaes")
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )

    result = OptimizationStudyRunner().run(
        _spec("optimization_cmaes_test", "cmaes", trials=6),
        schema,
        _evaluation,
    )

    assert result.trial_count == 6
    assert result.candidates


def test_runner_blocks_parallel_low_memory_execution() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )

    with pytest.raises(ValueError, match="n_jobs=1"):
        OptimizationStudyRunner().run(
            _spec("optimization_parallel_block_test", "random", n_jobs=2),
            schema,
            _evaluation,
        )


@pytest.mark.parametrize(
    "purpose, expected",
    [
        ("model_weight_training", "PyTorch optimizer"),
        ("portfolio_allocation", "convex allocator"),
    ],
)
def test_black_box_runner_rejects_dedicated_optimization_purposes(
    purpose: str,
    expected: str,
) -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    original = _spec(f"wrong_runner_{purpose}", "random")
    spec = OptimizationStudySpec(
        study_id=original.study_id,
        purpose=purpose,
        component_id=original.component_id,
        sampler_id=original.sampler_id,
        objectives=original.objectives,
        frozen_inputs=original.frozen_inputs,
        budget=original.budget,
    )

    with pytest.raises(ValueError, match=expected):
        OptimizationStudyRunner().run(spec, schema, _evaluation)


def test_study_store_separates_state_registry_and_artifacts(tmp_path) -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    store = OptimizationStudyStore(
        state_db_path=tmp_path / "state" / "optuna.sqlite3",
        registry_db_path=tmp_path / "research.sqlite3",
        artifact_root=tmp_path / "artifacts",
    )

    result = OptimizationStudyRunner(store).run(
        _spec("optimization_persistence_test", "random", trials=4),
        schema,
        _evaluation,
    )

    assert (tmp_path / "state" / "optuna.sqlite3").exists()
    assert (tmp_path / "research.sqlite3").exists()
    assert result.artifact_path is not None
    assert (tmp_path / "artifacts" / result.study_id / "trials.json").exists()


def test_study_store_rejects_reusing_id_for_different_frozen_inputs(
    tmp_path,
) -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    store = OptimizationStudyStore(
        state_db_path=tmp_path / "state" / "optuna.sqlite3",
        registry_db_path=tmp_path / "research.sqlite3",
        artifact_root=tmp_path / "artifacts",
    )
    original = _spec("immutable_study_id", "random", trials=2)
    store.register_start(original, schema)
    changed = OptimizationStudySpec(
        study_id=original.study_id,
        purpose=original.purpose,
        component_id=original.component_id,
        sampler_id=original.sampler_id,
        objectives=original.objectives,
        constraints=original.constraints,
        frozen_inputs=FrozenResearchInputs(
            dataset_fingerprint="different-development-dataset"
        ),
        budget=original.budget,
    )

    with pytest.raises(ValueError, match="different frozen inputs"):
        store.register_start(changed, schema)


def test_study_store_rejects_mutating_completed_artifact(tmp_path) -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    store = OptimizationStudyStore(
        state_db_path=tmp_path / "state" / "optuna.sqlite3",
        registry_db_path=tmp_path / "research.sqlite3",
        artifact_root=tmp_path / "artifacts",
    )
    spec = _spec("immutable_artifact", "random", trials=2)
    store.persist_result(spec, schema, {"version": 1}, [])

    with pytest.raises(ValueError, match="different content"):
        store.persist_result(spec, schema, {"version": 2}, [])


def test_sampler_registry_builds_every_optuna_family() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )
    budget = SearchBudget(max_trials=20, max_grid_combinations=100)
    for sampler_id in (
        "grid",
        "bruteforce",
        "random",
        "qmc",
        "tpe",
        "gp",
        "cmaes",
        "nsga2",
    ):
        sampler, metadata = build_optuna_sampler(
            sampler_id,
            schema,
            seed=42,
            budget=budget,
        )
        assert sampler is not None
        assert metadata["sampler_id"] == sampler_id


def test_scipy_global_adapter_uses_same_component_schema() -> None:
    schema = resolve_component_parameter_schema(
        _demo_module(), component_type="factor"
    )

    result = solve_global_continuous(
        lambda parameters: -((parameters["x"] - 0.5) ** 2)
        - abs(parameters["window"] - 2),
        schema,
        method="differential_evolution",
        max_evaluations=80,
        seed=3,
    )

    assert result.parameters["window"] in {1, 2, 3}
    assert -2.0 <= result.parameters["x"] <= 2.0
    assert (result.parameters["x"] + 2.0) / 0.5 == pytest.approx(
        round((result.parameters["x"] + 2.0) / 0.5)
    )
    assert result.evaluations > 0
