"""Low-memory, holdout-locked orchestration for black-box optimization studies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from typing import Any, Protocol

import optuna

from oqp.optimization.constraints import (
    evaluate_constraints,
    hard_constraints_satisfied,
    optuna_constraint_values,
)
from oqp.optimization.contracts import (
    OptimizationCandidate,
    OptimizationEvaluationContext,
    OptimizationPurpose,
    OptimizationStudyResult,
    OptimizationStudySpec,
    TrialEvaluation,
)
from oqp.optimization.objectives import extract_objective_values
from oqp.optimization.parameter_spaces import (
    ComponentParameterSchema,
    suggest_component_parameters,
)
from oqp.optimization.samplers import build_optuna_sampler
from oqp.optimization.study_store import OptimizationStudyStore
from oqp.research.parameter_optimization import (
    diagnose_parameter_boundaries,
    diagnose_parameter_surface,
)


class CandidateEvaluator(Protocol):
    def __call__(
        self,
        parameters: dict[str, Any],
        context: OptimizationEvaluationContext,
    ) -> TrialEvaluation: ...


@dataclass(frozen=True, slots=True)
class OptimizationComparisonResult:
    baseline: OptimizationStudyResult
    challenger: OptimizationStudyResult


class OptimizationStudyRunner:
    def __init__(self, store: OptimizationStudyStore | None = None) -> None:
        self.store = store

    def run(
        self,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
        evaluator: CandidateEvaluator,
    ) -> OptimizationStudyResult:
        self._preflight(spec, schema)

        def constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
            values = trial.user_attrs.get("constraint_values", ())
            return tuple(float(value) for value in values)

        sampler, sampler_metadata = build_optuna_sampler(
            spec.sampler_id,
            schema,
            seed=spec.seed,
            budget=spec.budget,
            constraints_func=constraints_func if spec.constraints else None,
        )
        grid_combinations = sampler_metadata.get("grid_combinations")
        if (
            grid_combinations is not None
            and spec.budget.max_trials < int(grid_combinations)
        ):
            raise ValueError(
                "Grid search budget must cover all "
                f"{int(grid_combinations):,} combinations; received "
                f"{spec.budget.max_trials:,} trials"
            )
        if self.store is not None:
            self.store.register_start(spec, schema)
        storage = self.store.optuna_storage_uri if self.store is not None else None
        create_kwargs: dict[str, Any] = {
            "study_name": spec.study_id,
            "sampler": sampler,
            "storage": storage,
            "load_if_exists": True,
        }
        if len(spec.objectives) == 1:
            create_kwargs["direction"] = spec.directions[0]
        else:
            create_kwargs["directions"] = list(spec.directions)
        study = optuna.create_study(**create_kwargs)
        self._validate_or_initialize_study(study, spec, schema)

        complete_trials = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )
        remaining_trials = max(spec.budget.max_trials - complete_trials, 0)

        def objective(trial: optuna.Trial):
            parameters = suggest_component_parameters(trial, schema)
            context = OptimizationEvaluationContext(
                study_id=spec.study_id,
                trial_number=int(trial.number),
                purpose=spec.purpose,
                component_id=spec.component_id,
                development_dataset_fingerprint=(
                    spec.frozen_inputs.dataset_fingerprint
                ),
                holdout_locked=True,
            )
            context.require_development_only()
            try:
                evaluation = evaluator(parameters, context)
            except Exception as exc:
                trial.set_user_attr("resolved_parameters", parameters)
                trial.set_user_attr("failure_type", type(exc).__name__)
                trial.set_user_attr("failure_message", str(exc))
                raise
            if not isinstance(evaluation, TrialEvaluation):
                raise TypeError("candidate evaluator must return TrialEvaluation")
            objective_values = extract_objective_values(
                evaluation.metrics,
                spec.objectives,
            )
            constraint_results = evaluate_constraints(
                evaluation.metrics,
                spec.constraints,
            )
            trial.set_user_attr("resolved_parameters", parameters)
            trial.set_user_attr("metrics", dict(evaluation.metrics))
            trial.set_user_attr("fold_metrics", list(evaluation.fold_metrics))
            trial.set_user_attr("artifacts", dict(evaluation.artifacts))
            trial.set_user_attr("evaluation_metadata", dict(evaluation.metadata))
            trial.set_user_attr(
                "constraint_values",
                list(optuna_constraint_values(constraint_results)),
            )
            trial.set_user_attr(
                "constraints_satisfied",
                hard_constraints_satisfied(constraint_results),
            )
            if len(objective_values) == 1:
                return objective_values[0]
            return objective_values

        if remaining_trials:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=spec.budget.timeout_seconds,
                n_jobs=1,
                gc_after_trial=True,
                show_progress_bar=False,
                catch=(Exception,),
            )
        gc.collect()
        result = self._build_result(
            study,
            spec,
            schema,
            sampler_metadata=sampler_metadata,
        )
        if self.store is not None:
            trial_payload = [self._trial_payload(trial) for trial in study.trials]
            artifact_path = self.store.persist_result(
                spec,
                schema,
                result.to_dict(),
                trial_payload,
            )
            result = replace(result, artifact_path=artifact_path)
            self.store.register_complete(spec, schema, result)
        return result

    def run_with_random_baseline(
        self,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
        evaluator: CandidateEvaluator,
    ) -> OptimizationComparisonResult:
        if spec.sampler_id == "random":
            result = self.run(spec, schema, evaluator)
            return OptimizationComparisonResult(result, result)
        baseline_spec = replace(
            spec,
            study_id=f"{spec.study_id}__random_baseline",
            sampler_id="random",
            metadata={**dict(spec.metadata), "baseline_for": spec.study_id},
        )
        baseline = self.run(baseline_spec, schema, evaluator)
        challenger = self.run(spec, schema, evaluator)
        return OptimizationComparisonResult(baseline, challenger)

    @staticmethod
    def _preflight(
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
    ) -> None:
        if spec.component_id != schema.component_id:
            raise ValueError(
                f"Study component {spec.component_id!r} does not match schema "
                f"{schema.component_id!r}"
            )
        if not schema.tunable_names:
            raise ValueError("optimization requires at least one tunable parameter")
        if spec.purpose == OptimizationPurpose.MODEL_WEIGHT_TRAINING:
            raise ValueError(
                "Model-weight training must use the declared PyTorch optimizer "
                "factory, not the black-box study runner"
            )
        if spec.purpose == OptimizationPurpose.PORTFOLIO_ALLOCATION:
            raise ValueError(
                "Portfolio allocation must use the convex allocator or an "
                "explicit sizing engine, not the black-box study runner"
            )
        if len(spec.objectives) > 1 and spec.sampler_id in {"cmaes", "gp"}:
            raise ValueError(
                f"Sampler {spec.sampler_id!r} does not support this "
                "multi-objective study; use NSGA-II, TPE, random, or grid"
            )
        if spec.budget.n_jobs != 1:
            raise ValueError(
                "Research optimization is intentionally limited to n_jobs=1 "
                "for deterministic, low-memory execution"
            )

    @staticmethod
    def _validate_or_initialize_study(
        study: optuna.Study,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
    ) -> None:
        existing_study = study.user_attrs.get("study_fingerprint")
        existing_schema = study.user_attrs.get("parameter_schema_fingerprint")
        if existing_study and existing_study != spec.fingerprint:
            raise ValueError(
                f"Study ID {spec.study_id!r} already exists with different frozen inputs"
            )
        if existing_schema and existing_schema != schema.fingerprint:
            raise ValueError(
                f"Study ID {spec.study_id!r} already exists with a different parameter schema"
            )
        study.set_user_attr("study_fingerprint", spec.fingerprint)
        study.set_user_attr("parameter_schema_fingerprint", schema.fingerprint)
        study.set_user_attr("holdout_locked", True)

    @staticmethod
    def _build_result(
        study: optuna.Study,
        spec: OptimizationStudySpec,
        schema: ComponentParameterSchema,
        *,
        sampler_metadata: dict[str, Any],
    ) -> OptimizationStudyResult:
        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        feasible = [
            trial
            for trial in completed
            if bool(trial.user_attrs.get("constraints_satisfied", True))
        ]
        selected = _select_candidates(feasible, spec)
        candidates = tuple(_candidate_from_trial(trial) for trial in selected)
        diagnostics: dict[str, Any] = {
            **sampler_metadata,
            "total_trials": len(study.trials),
            "completed_trials": len(completed),
            "feasible_trials": len(feasible),
            "failed_trials": sum(
                trial.state == optuna.trial.TrialState.FAIL
                for trial in study.trials
            ),
            "holdout_locked": True,
        }
        if candidates:
            boundary = diagnose_parameter_boundaries(
                candidates[0].parameters,
                schema.schema,
            )
            diagnostics["boundary"] = [item.to_dict() for item in boundary]
        if len(spec.objectives) == 1 and feasible:
            observations = [
                {
                    "params": trial.user_attrs.get(
                        "resolved_parameters", trial.params
                    ),
                    "value": float(trial.value),
                }
                for trial in feasible
            ]
            surface = diagnose_parameter_surface(
                observations,
                schema.schema,
                direction=spec.directions[0],
            )
            diagnostics["surface"] = surface.to_dict()
        return OptimizationStudyResult(
            study_id=spec.study_id,
            study_fingerprint=spec.fingerprint,
            parameter_schema_fingerprint=schema.fingerprint,
            sampler_id=spec.sampler_id,
            trial_count=len(completed),
            candidates=candidates,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _trial_payload(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
        return {
            "number": trial.number,
            "state": trial.state.name,
            "params": dict(trial.params),
            "resolved_parameters": trial.user_attrs.get(
                "resolved_parameters", {}
            ),
            "values": list(trial.values or ()),
            "user_attrs": dict(trial.user_attrs),
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete,
        }


def _select_candidates(
    feasible_trials: list[optuna.trial.FrozenTrial],
    spec: OptimizationStudySpec,
) -> list[optuna.trial.FrozenTrial]:
    if not feasible_trials:
        return []
    if len(spec.objectives) == 1:
        reverse = spec.directions[0] == "maximize"
        return [
            sorted(
                feasible_trials,
                key=lambda trial: float(trial.value),
                reverse=reverse,
            )[0]
        ]
    selected: list[optuna.trial.FrozenTrial] = []
    for candidate in feasible_trials:
        if not any(
            _dominates(other, candidate, spec.directions)
            for other in feasible_trials
            if other.number != candidate.number
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda trial: trial.number)


def _dominates(
    left: optuna.trial.FrozenTrial,
    right: optuna.trial.FrozenTrial,
    directions: tuple[str, ...],
) -> bool:
    left_values = tuple(float(value) for value in left.values or ())
    right_values = tuple(float(value) for value in right.values or ())
    weakly_better = []
    strictly_better = []
    for left_value, right_value, direction in zip(
        left_values,
        right_values,
        directions,
        strict=True,
    ):
        if direction == "maximize":
            weakly_better.append(left_value >= right_value)
            strictly_better.append(left_value > right_value)
        else:
            weakly_better.append(left_value <= right_value)
            strictly_better.append(left_value < right_value)
    return all(weakly_better) and any(strictly_better)


def _candidate_from_trial(
    trial: optuna.trial.FrozenTrial,
) -> OptimizationCandidate:
    return OptimizationCandidate(
        trial_number=int(trial.number),
        parameters=dict(
            trial.user_attrs.get("resolved_parameters", trial.params)
        ),
        objective_values=tuple(float(value) for value in trial.values or ()),
        metrics={
            str(key): float(value)
            for key, value in trial.user_attrs.get("metrics", {}).items()
        },
        feasible=bool(trial.user_attrs.get("constraints_satisfied", True)),
    )


__all__ = [
    "CandidateEvaluator",
    "OptimizationComparisonResult",
    "OptimizationStudyRunner",
]
