"""Phase 8 wrapper around the shared low-memory Optuna runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from oqp.optimization import (
    ComponentParameterSchema,
    OptimizationCandidate,
    OptimizationStudyResult,
    OptimizationStudyRunner,
    OptimizationStudyStore,
    TrialEvaluation,
    require_dataset_fingerprint,
    stable_optimization_hash,
)
from oqp.research.optional_optimization.contracts import (
    Phase8ExperimentSpec,
    Phase8ObjectiveSpec,
)
from oqp.research.optional_optimization.folds import Phase8Fold, build_phase8_folds
from oqp.research.optimization_objectives import (
    OptimizationObjectiveRegistry,
    evaluate_objective_profile_metrics,
    validate_phase8_objective_profile,
)


class Phase8FoldEvaluator(Protocol):
    def __call__(
        self,
        parameters: dict[str, Any],
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        fold: Phase8Fold,
    ) -> Mapping[str, float]: ...


class Phase8HoldoutEvaluator(Protocol):
    def __call__(
        self,
        parameters: dict[str, Any],
        holdout_data: pd.DataFrame,
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class Phase8SearchResult:
    spec: Phase8ExperimentSpec
    study_result: OptimizationStudyResult
    random_baseline_result: OptimizationStudyResult
    folds: tuple[Phase8Fold, ...]
    selected_candidate: OptimizationCandidate | None
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FrozenPhase8Candidate:
    study_id: str
    phase8_protocol_fingerprint: str
    parameter_schema_fingerprint: str
    mutable_layer: str
    component_id: str
    trial_number: int
    parameters: Mapping[str, Any]
    objective_values: tuple[float, ...]
    holdout_start: str
    holdout_fingerprint: str
    frozen_at: str

    @property
    def fingerprint(self) -> str:
        return stable_optimization_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parameters": dict(self.parameters),
            "objective_values": list(self.objective_values),
        }


def run_phase8_search(
    spec: Phase8ExperimentSpec,
    schema: ComponentParameterSchema,
    data: pd.DataFrame,
    evaluator: Phase8FoldEvaluator,
    *,
    store: OptimizationStudyStore | None = None,
) -> Phase8SearchResult:
    if not spec.enabled:
        raise RuntimeError("Phase 8 optimization is disabled by default")
    if schema.component_id != spec.component_id:
        raise ValueError("parameter schema component does not match Phase 8 spec")
    if schema.component_type != spec.expected_component_type:
        raise ValueError(
            f"{spec.layer.value} optimization requires component type "
            f"{spec.expected_component_type!r}"
        )
    if schema.fingerprint != spec.parameter_schema_fingerprint:
        raise ValueError("the declared Phase 8 search space fingerprint changed")
    objective_profile = OptimizationObjectiveRegistry.load().resolve(
        spec.objective_profile_id
    )
    validate_phase8_objective_profile(spec, objective_profile)
    observed_fingerprint = require_dataset_fingerprint(data)
    if observed_fingerprint != spec.frozen_inputs.dataset_fingerprint:
        raise ValueError("development dataset fingerprint does not match Phase 8 spec")

    folds = build_phase8_folds(data, spec)
    fold_fingerprint = stable_optimization_hash(
        [fold.to_record() for fold in folds]
    )

    def candidate_evaluator(parameters, context) -> TrialEvaluation:
        context.require_development_only()
        fold_rows: list[dict[str, Any]] = []
        for fold in folds:
            observed = evaluator(
                dict(parameters),
                fold.training_data.copy(deep=True),
                fold.validation_data.copy(deep=True),
                fold,
            )
            metrics = {str(key): float(value) for key, value in observed.items()}
            if not metrics:
                raise ValueError(f"{fold.fold_id} returned no metrics")
            if any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"{fold.fold_id} returned non-finite metrics")
            fold_rows.append({**fold.to_record(), **metrics})
        aggregate = _aggregate_fold_metrics(fold_rows, spec.objectives)
        objective_diagnostics = evaluate_objective_profile_metrics(
            objective_profile, aggregate
        )
        return TrialEvaluation(
            metrics=aggregate,
            fold_metrics=tuple(fold_rows),
            metadata={
                "phase8_protocol_fingerprint": spec.fingerprint,
                "inner_fold_fingerprint": fold_fingerprint,
                "holdout_accessed": False,
                "mutable_layer": spec.layer.value,
                "objective_profile": objective_diagnostics,
            },
        )

    comparison = OptimizationStudyRunner(store).run_with_random_baseline(
        spec.to_optimization_study_spec(), schema, candidate_evaluator
    )
    study_result = comparison.challenger
    baseline_result = comparison.baseline
    selected = _select_frozen_candidate(study_result, spec)
    manifest = {
        "schema_version": spec.schema_version,
        "phase": "Phase 8: Optional Optimisation",
        "status": "search_complete" if selected else "search_failed",
        "phase8_protocol_fingerprint": spec.fingerprint,
        "parameter_schema_fingerprint": schema.fingerprint,
        "objective_profile_id": objective_profile.profile_id,
        "objective_profile_fingerprint": objective_profile.fingerprint,
        "inner_fold_fingerprint": fold_fingerprint,
        "mutable_layer": spec.layer.value,
        "mutable_component_id": spec.component_id,
        "frozen_component_fingerprints": dict(
            spec.frozen_component_fingerprints
        ),
        "sampler": f"optuna_{spec.sampler_id}",
        "trial_budget": spec.budget.max_trials,
        "total_trials_recorded": study_result.diagnostics.get("total_trials", 0),
        "failed_trials_recorded": study_result.diagnostics.get("failed_trials", 0),
        "random_baseline_study_id": baseline_result.study_id,
        "random_baseline_trial_budget": spec.budget.max_trials,
        "random_baseline_total_trials_recorded": baseline_result.diagnostics.get(
            "total_trials", 0
        ),
        "random_baseline_failed_trials_recorded": baseline_result.diagnostics.get(
            "failed_trials", 0
        ),
        "random_baseline_pareto_candidate_count": len(
            baseline_result.candidates
        ),
        "pareto_candidate_count": len(study_result.candidates),
        "selected_trial_number": (
            selected.trial_number if selected is not None else None
        ),
        "selection_priority": list(spec.selection_priority),
        "bayesian_shrinkage_applied": True,
        "equal_budget_random_baseline_run": True,
        "holdout_locked_during_search": True,
        "holdout_accessed": False,
        "joint_layer_optimization_permitted": False,
    }
    return Phase8SearchResult(
        spec=spec,
        study_result=study_result,
        random_baseline_result=baseline_result,
        folds=folds,
        selected_candidate=selected,
        manifest=manifest,
    )


def write_phase8_search_result(
    result: Phase8SearchResult, output_dir: str | Path
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(destination / "protocol.json", result.spec.to_dict())
    _write_immutable_json(destination / "search_manifest.json", result.manifest)
    _write_immutable_json(
        destination / "pareto_candidates.json",
        [candidate.to_dict() for candidate in result.study_result.candidates],
    )
    _write_immutable_json(
        destination / "random_baseline_candidates.json",
        [
            candidate.to_dict()
            for candidate in result.random_baseline_result.candidates
        ],
    )
    pd.DataFrame(fold.to_record() for fold in result.folds).to_csv(
        destination / "inner_folds.csv", index=False
    )
    return destination


def freeze_phase8_candidate(
    result: Phase8SearchResult,
    output_dir: str | Path,
    *,
    frozen_at: str | None = None,
) -> FrozenPhase8Candidate:
    selected = result.selected_candidate
    if selected is None:
        raise ValueError("Phase 8 search produced no feasible Pareto candidate")
    timestamp = frozen_at or _utc_now()
    candidate = FrozenPhase8Candidate(
        study_id=result.spec.study_id,
        phase8_protocol_fingerprint=result.spec.fingerprint,
        parameter_schema_fingerprint=result.spec.parameter_schema_fingerprint,
        mutable_layer=result.spec.layer.value,
        component_id=result.spec.component_id,
        trial_number=selected.trial_number,
        parameters=dict(selected.parameters),
        objective_values=selected.objective_values,
        holdout_start=result.spec.holdout_start,
        holdout_fingerprint=result.spec.frozen_inputs.holdout_fingerprint,
        frozen_at=str(timestamp),
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(destination / "frozen_candidate.json", candidate.to_dict())
    return candidate


def evaluate_final_holdout_once(
    candidate: FrozenPhase8Candidate,
    holdout_data: pd.DataFrame,
    evaluator: Phase8HoldoutEvaluator,
    output_dir: str | Path,
) -> dict[str, Any]:
    observed_fingerprint = str(
        holdout_data.attrs.get("holdout_fingerprint") or ""
    ).strip()
    if observed_fingerprint != candidate.holdout_fingerprint:
        raise ValueError("final holdout fingerprint does not match frozen candidate")
    date_col = "date"
    if date_col not in holdout_data.columns:
        raise ValueError("final holdout data is missing the date column")
    dates = pd.to_datetime(holdout_data[date_col], errors="raise").dt.normalize()
    if dates.lt(pd.Timestamp(candidate.holdout_start)).any():
        raise ValueError("final holdout data contains development-period rows")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    access_path = destination / "holdout_access.json"
    result_path = destination / "holdout_result.json"
    if access_path.exists() or result_path.exists():
        raise RuntimeError("the final holdout has already been accessed")
    access = {
        "candidate_fingerprint": candidate.fingerprint,
        "holdout_fingerprint": candidate.holdout_fingerprint,
        "accessed_at": _utc_now(),
        "status": "access_started",
    }
    _write_immutable_json(access_path, access)
    try:
        metrics = {
            str(key): float(value)
            for key, value in evaluator(
                dict(candidate.parameters), holdout_data.copy(deep=True)
            ).items()
        }
        if not metrics or any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("holdout evaluator returned missing or non-finite metrics")
    except Exception as exc:
        _write_immutable_json(
            destination / "holdout_failure.json",
            {
                "candidate_fingerprint": candidate.fingerprint,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "failed_at": _utc_now(),
                "holdout_consumed": True,
            },
        )
        raise
    result = {
        "candidate_fingerprint": candidate.fingerprint,
        "holdout_fingerprint": candidate.holdout_fingerprint,
        "evaluated_at": _utc_now(),
        "metrics": metrics,
        "evaluation_count": 1,
        "holdout_consumed": True,
    }
    _write_immutable_json(result_path, result)
    return result


def bayesian_shrunk_mean(
    values: list[float], objective: Phase8ObjectiveSpec
) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    if clean.size < 2 or not np.isfinite(clean).all():
        raise ValueError("Bayesian shrinkage requires at least two finite folds")
    sample_mean = float(clean.mean())
    sample_std = float(clean.std(ddof=1))
    standard_error_variance = max(
        sample_std**2 / clean.size,
        objective.noise_floor**2,
    )
    prior_variance = objective.prior_std**2
    posterior_variance = 1.0 / (
        1.0 / prior_variance + 1.0 / standard_error_variance
    )
    posterior_mean = posterior_variance * (
        objective.prior_mean / prior_variance
        + sample_mean / standard_error_variance
    )
    return {
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "posterior_mean": float(posterior_mean),
        "posterior_std": float(math.sqrt(posterior_variance)),
    }


def _aggregate_fold_metrics(
    fold_rows: list[dict[str, Any]],
    objectives: tuple[Phase8ObjectiveSpec, ...],
) -> dict[str, float]:
    objective_metrics = {objective.metric for objective in objectives}
    metric_names = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in row.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }
                for row in fold_rows
            )
        )
    )
    aggregate: dict[str, float] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in fold_rows]
        aggregate[f"mean__{metric}"] = float(np.mean(values))
    for objective in objectives:
        if objective.metric not in objective_metrics or any(
            objective.metric not in row for row in fold_rows
        ):
            raise ValueError(
                f"fold metrics are missing objective {objective.metric!r}"
            )
        shrinkage = bayesian_shrunk_mean(
            [float(row[objective.metric]) for row in fold_rows], objective
        )
        aggregate[f"raw__{objective.metric}"] = shrinkage["sample_mean"]
        aggregate[f"std__{objective.metric}"] = shrinkage["sample_std"]
        aggregate[objective.posterior_metric] = shrinkage["posterior_mean"]
        aggregate[f"posterior_std__{objective.metric}"] = shrinkage[
            "posterior_std"
        ]
    return aggregate


def _select_frozen_candidate(
    result: OptimizationStudyResult,
    spec: Phase8ExperimentSpec,
) -> OptimizationCandidate | None:
    if not result.candidates:
        return None
    objective_index = {
        objective.name: index for index, objective in enumerate(spec.objectives)
    }

    def key(candidate: OptimizationCandidate) -> tuple[float, ...]:
        values: list[float] = []
        for name in spec.selection_priority:
            index = objective_index[name]
            value = candidate.objective_values[index]
            direction = spec.objectives[index].direction.value
            values.append(-value if direction == "maximize" else value)
        values.append(float(candidate.trial_number))
        return tuple(values)

    return min(result.candidates, key=key)


def _write_immutable_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"immutable Phase 8 artifact already differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "FrozenPhase8Candidate",
    "Phase8FoldEvaluator",
    "Phase8HoldoutEvaluator",
    "Phase8SearchResult",
    "bayesian_shrunk_mean",
    "evaluate_final_holdout_once",
    "freeze_phase8_candidate",
    "run_phase8_search",
    "write_phase8_search_result",
]
