"""Phase 8 readiness and state audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.research.optional_optimization.contracts import (
    PHASE8_SCHEMA_VERSION,
    load_phase8_experiment_spec,
)


def audit_phase8_readiness(
    config_root: str | Path,
    artifact_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    configs = Path(config_root).expanduser().resolve()
    artifacts = Path(artifact_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(configs.glob("phase8_*.yaml")):
        try:
            spec = load_phase8_experiment_spec(path)
            study_dir = artifacts / spec.study_id
            search_complete = (study_dir / "search_manifest.json").exists()
            candidate_frozen = (study_dir / "frozen_candidate.json").exists()
            holdout_evaluated = (study_dir / "holdout_result.json").exists()
            holdout_consumed = (study_dir / "holdout_access.json").exists()
            rows.append(
                {
                    "study_id": spec.study_id,
                    "mutable_layer": spec.layer.value,
                    "component_id": spec.component_id,
                    "objective_profile_id": spec.objective_profile_id,
                    "objective_profile_fingerprint": (
                        spec.objective_profile_fingerprint
                    ),
                    "enabled": spec.enabled,
                    "trial_budget": spec.budget.max_trials,
                    "sampler": spec.sampler_id,
                    "search_complete": search_complete,
                    "candidate_frozen": candidate_frozen,
                    "holdout_consumed": holdout_consumed,
                    "holdout_evaluated": holdout_evaluated,
                    "status": _study_status(
                        spec.enabled,
                        search_complete,
                        candidate_frozen,
                        holdout_consumed,
                        holdout_evaluated,
                    ),
                    "reason": "",
                    "config_path": str(path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "study_id": path.stem,
                    "mutable_layer": "",
                    "component_id": "",
                    "objective_profile_id": "",
                    "objective_profile_fingerprint": "",
                    "enabled": False,
                    "trial_budget": 0,
                    "sampler": "",
                    "search_complete": False,
                    "candidate_frozen": False,
                    "holdout_consumed": False,
                    "holdout_evaluated": False,
                    "status": "invalid",
                    "reason": str(exc),
                    "config_path": str(path),
                }
            )
    studies = pd.DataFrame(
        rows,
        columns=[
            "study_id",
            "mutable_layer",
            "component_id",
            "objective_profile_id",
            "objective_profile_fingerprint",
            "enabled",
            "trial_budget",
            "sampler",
            "search_complete",
            "candidate_frozen",
            "holdout_consumed",
            "holdout_evaluated",
            "status",
            "reason",
            "config_path",
        ],
    )
    enabled = int(studies["enabled"].sum()) if not studies.empty else 0
    completed = int(studies["search_complete"].sum()) if not studies.empty else 0
    frozen = int(studies["candidate_frozen"].sum()) if not studies.empty else 0
    evaluated = int(studies["holdout_evaluated"].sum()) if not studies.empty else 0
    invalid = int(studies["status"].eq("invalid").sum()) if not studies.empty else 0
    if enabled == 0 and invalid == 0:
        status = "disabled"
    elif invalid:
        status = "blocked"
    elif evaluated:
        status = "holdout_evaluated"
    elif frozen:
        status = "candidate_frozen"
    elif completed:
        status = "search_complete"
    else:
        status = "enabled"
    summary = {
        "schema_version": PHASE8_SCHEMA_VERSION,
        "phase": "Phase 8: Optional Optimisation",
        "status": status,
        "optimization_enabled_by_default": False,
        "declared_studies": len(studies),
        "enabled_studies": enabled,
        "completed_searches": completed,
        "frozen_candidates": frozen,
        "final_holdout_evaluations": evaluated,
        "invalid_studies": invalid,
        "mutable_layers_per_study": 1,
        "samplers": sorted(
            set(studies["sampler"].dropna().astype(str)).difference({""})
        )
        if not studies.empty
        else [],
        "holdout_evaluation_limit": 1,
        "process": [
            "freeze_search_space_and_budget",
            "purged_embargoed_inner_folds",
            "purpose_appropriate_multi_objective_search",
            "bayesian_shrinkage",
            "pareto_candidate_selection",
            "freeze_selected_candidate",
            "single_final_holdout_evaluation",
            "persist_all_trials_including_failures",
        ],
        "boundary": (
            "Only one layer may be mutable in a study. Joint factor, sleeve, "
            "router, allocator, and overlay optimization is prohibited."
        ),
    }
    return summary, studies


def write_phase8_readiness(
    summary: dict[str, Any], studies: pd.DataFrame, output_dir: str | Path
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "readiness.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    studies.to_csv(destination / "studies.csv", index=False)
    return destination


def _study_status(
    enabled: bool,
    search_complete: bool,
    candidate_frozen: bool,
    holdout_consumed: bool,
    holdout_evaluated: bool,
) -> str:
    if not enabled:
        return "disabled"
    if holdout_evaluated:
        return "holdout_evaluated"
    if holdout_consumed:
        return "holdout_consumed_without_result"
    if candidate_frozen:
        return "candidate_frozen"
    if search_complete:
        return "search_complete"
    return "enabled_not_run"


__all__ = ["audit_phase8_readiness", "write_phase8_readiness"]
