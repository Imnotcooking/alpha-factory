from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oqp.optimization import (
    FrozenResearchInputs,
    OptimizationStudyStore,
    SearchBudget,
    build_component_parameter_schema,
)
from oqp.research.optional_optimization import (
    Phase8ExperimentSpec,
    Phase8FoldConfig,
    Phase8ObjectiveSpec,
    audit_phase8_readiness,
    bayesian_shrunk_mean,
    build_phase8_folds,
    evaluate_final_holdout_once,
    freeze_phase8_candidate,
    run_phase8_search,
    write_phase8_search_result,
)
from oqp.research.optimization_objectives import OptimizationObjectiveRegistry


def _schema():
    return build_component_parameter_schema(
        "fac_test",
        "factor",
        {
            "lookback": {
                "default": 8,
                "type": "int",
                "low": 5,
                "high": 10,
                "step": 1,
                "tunable": True,
            }
        },
    )


def _data() -> tuple[pd.DataFrame, str]:
    dates = pd.bdate_range("2025-01-01", periods=42)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "forward_return": 0.001,
        }
    )
    frame.attrs["dataset_fingerprint"] = "dataset-fingerprint-v1"
    return frame, dates[36].date().isoformat()


def _spec(*, enabled: bool, budget: int = 5) -> Phase8ExperimentSpec:
    schema = _schema()
    profile = OptimizationObjectiveRegistry.load().resolve(
        "phase9_factor_predictive_v1"
    )
    _, holdout_start = _data()
    return Phase8ExperimentSpec(
        study_id="phase8_factor_test",
        layer="factor",
        component_id="fac_test",
        parameter_schema_fingerprint=schema.fingerprint,
        objective_profile_id=profile.profile_id,
        objective_profile_fingerprint=profile.fingerprint,
        objectives=tuple(
            objective.to_phase8_objective() for objective in profile.objectives
        ),
        selection_priority=profile.selection_priority,
        frozen_inputs=FrozenResearchInputs(
            dataset_fingerprint="dataset-fingerprint-v1",
            holdout_fingerprint="holdout-fingerprint-v1",
        ),
        budget=SearchBudget(max_trials=budget),
        fold_config=Phase8FoldConfig(
            fold_count=3,
            minimum_training_periods=12,
            validation_periods=5,
            purge_periods=2,
            embargo_periods=1,
        ),
        holdout_start=holdout_start,
        frozen_on="2024-12-31",
        constraints=profile.constraints,
        frozen_component_fingerprints={"slv_locked": "sleeve-fingerprint"},
        enabled=enabled,
    )


def _store(tmp_path: Path) -> OptimizationStudyStore:
    return OptimizationStudyStore(
        state_db_path=tmp_path / "state" / "optuna.sqlite3",
        registry_db_path=tmp_path / "state" / "registry.sqlite3",
        artifact_root=tmp_path / "generic_artifacts",
    )


def test_phase8_rejects_joint_mutable_layers() -> None:
    with pytest.raises(ValueError, match="exactly one mutable layer"):
        Phase8ExperimentSpec.from_mapping(
            {"optimization": {"mutable_layers": ["factor", "router"]}}
        )


def test_phase8_is_disabled_by_default() -> None:
    data, _ = _data()
    with pytest.raises(RuntimeError, match="disabled by default"):
        run_phase8_search(
            _spec(enabled=False),
            _schema(),
            data,
            lambda parameters, training, validation, fold: {
                "rank_ic": 0.01,
                "turnover": 1.0,
            },
        )


def test_inner_folds_are_purged_embargoed_and_exclude_holdout() -> None:
    data, holdout_start = _data()
    folds = build_phase8_folds(data, _spec(enabled=True))
    assert len(folds) == 3
    for fold in folds:
        assert len(fold.purge_dates) == 2
        assert len(fold.embargo_dates) == 1
        assert pd.Timestamp(fold.training_end) < pd.Timestamp(fold.purge_dates[0])
        assert pd.Timestamp(fold.embargo_dates[-1]) < pd.Timestamp(
            fold.validation_start
        )
        assert pd.Timestamp(fold.validation_end) < pd.Timestamp(holdout_start)
        assert fold.training_data["date"].max() < fold.validation_data["date"].min()


def test_bayesian_shrinkage_pulls_noisy_mean_toward_prior() -> None:
    objective = Phase8ObjectiveSpec(
        name="sharpe",
        metric="net_sharpe",
        prior_mean=0.0,
        prior_std=0.25,
        noise_floor=0.01,
    )
    result = bayesian_shrunk_mean([2.0, -0.5, 1.5, 0.0], objective)
    assert result["sample_mean"] > 0.0
    assert 0.0 < result["posterior_mean"] < result["sample_mean"]


def test_tpe_search_records_failures_and_freezes_one_pareto_candidate(
    tmp_path: Path,
) -> None:
    data, _ = _data()
    calls = {"count": 0}

    def evaluator(parameters, training, validation, fold):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("deliberate first-trial failure")
        lookback = float(parameters["lookback"])
        fold_number = float(fold.fold_id[-2:])
        return {
            "mean_pearson_ic": 0.07 - abs(lookback - 8.0) * 0.01,
            "mean_rank_ic": 0.08 - abs(lookback - 8.0) * 0.01,
            "pearson_icir": 0.50 + fold_number * 0.01,
            "rank_icir": 0.60 + fold_number * 0.01,
            "mean_joint_coverage": 0.90 - fold_number * 0.01,
            "stability_floor": 0.02 + fold_number * 0.001,
        }

    result = run_phase8_search(
        _spec(enabled=True),
        _schema(),
        data,
        evaluator,
        store=_store(tmp_path),
    )
    assert result.random_baseline_result.diagnostics["total_trials"] == 5
    assert result.random_baseline_result.diagnostics["failed_trials"] == 1
    assert result.study_result.diagnostics["total_trials"] == 5
    assert result.study_result.diagnostics["failed_trials"] == 0
    assert result.selected_candidate is not None
    generic_trials = json.loads(
        (
            tmp_path
            / "generic_artifacts"
            / f"{result.spec.study_id}__random_baseline"
            / "trials.json"
        ).read_text(encoding="utf-8")
    )
    assert len(generic_trials) == 5
    failed = [trial for trial in generic_trials if trial["state"] == "FAIL"]
    assert len(failed) == 1
    assert failed[0]["user_attrs"]["failure_type"] == "RuntimeError"

    phase8_dir = tmp_path / "phase8" / result.spec.study_id
    write_phase8_search_result(result, phase8_dir)
    assert (phase8_dir / "random_baseline_candidates.json").exists()
    assert result.manifest["equal_budget_random_baseline_run"] is True
    frozen = freeze_phase8_candidate(
        result, phase8_dir, frozen_at="2025-03-01T00:00:00+00:00"
    )
    assert frozen.trial_number == result.selected_candidate.trial_number
    assert (phase8_dir / "frozen_candidate.json").exists()

    holdout = data.loc[data["date"].ge(result.spec.holdout_start)].copy()
    holdout.attrs["holdout_fingerprint"] = "holdout-fingerprint-v1"
    holdout_result = evaluate_final_holdout_once(
        frozen,
        holdout,
        lambda parameters, frame: {"net_sharpe": 0.5, "rows": len(frame)},
        phase8_dir,
    )
    assert holdout_result["evaluation_count"] == 1
    with pytest.raises(RuntimeError, match="already been accessed"):
        evaluate_final_holdout_once(
            frozen,
            holdout,
            lambda parameters, frame: {"net_sharpe": 0.6},
            phase8_dir,
        )


def test_failed_holdout_evaluation_still_consumes_holdout(tmp_path: Path) -> None:
    data, _ = _data()

    def evaluator(parameters, training, validation, fold):
        return {
            "mean_pearson_ic": 0.02,
            "mean_rank_ic": 0.03,
            "pearson_icir": 0.20,
            "rank_icir": 0.30,
            "mean_joint_coverage": 0.90,
            "stability_floor": 0.01,
        }

    result = run_phase8_search(
        _spec(enabled=True, budget=2),
        _schema(),
        data,
        evaluator,
        store=_store(tmp_path),
    )
    phase8_dir = tmp_path / "failed_holdout"
    frozen = freeze_phase8_candidate(
        result, phase8_dir, frozen_at="2025-03-01T00:00:00+00:00"
    )
    holdout = data.loc[data["date"].ge(result.spec.holdout_start)].copy()
    holdout.attrs["holdout_fingerprint"] = "holdout-fingerprint-v1"
    with pytest.raises(RuntimeError, match="holdout failed"):
        evaluate_final_holdout_once(
            frozen,
            holdout,
            lambda parameters, frame: (_ for _ in ()).throw(
                RuntimeError("holdout failed")
            ),
            phase8_dir,
        )
    assert (phase8_dir / "holdout_failure.json").exists()
    with pytest.raises(RuntimeError, match="already been accessed"):
        evaluate_final_holdout_once(
            frozen,
            holdout,
            lambda parameters, frame: {"net_sharpe": 1.0},
            phase8_dir,
        )


def test_phase8_readiness_is_disabled_without_declared_studies(
    tmp_path: Path,
) -> None:
    summary, studies = audit_phase8_readiness(
        tmp_path / "configs", tmp_path / "artifacts"
    )
    assert summary["status"] == "disabled"
    assert summary["optimization_enabled_by_default"] is False
    assert studies.empty
