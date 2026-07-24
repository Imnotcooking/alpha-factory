"""Reusable regime models with dependency-lazy public imports.

Family-specific estimators are the educational public entry points.  They all
delegate to one deterministic diagonal-EM backend, so family names remain
clear without forking numerical or artifact semantics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    # State-label canonicalization across otherwise equivalent refits.
    "AlignmentMetric": ("alignment", "AlignmentMetric"),
    "AlignmentTieBreak": ("alignment", "AlignmentTieBreak"),
    "HungarianStateAligner": ("alignment", "HungarianStateAligner"),
    "StateAligner": ("alignment", "StateAligner"),
    "StateAlignmentConfig": ("alignment", "StateAlignmentConfig"),
    "StateAlignmentInput": ("alignment", "StateAlignmentInput"),
    "StateAlignmentResult": ("alignment", "StateAlignmentResult"),
    "StatePermutation": ("alignment", "StatePermutation"),
    "StateSignature": ("alignment", "StateSignature"),
    "align_states": ("alignment", "align_states"),
    "build_state_alignment_cost_matrix": (
        "alignment",
        "build_state_alignment_cost_matrix",
    ),
    "deterministic_hungarian_assignment": (
        "alignment",
        "deterministic_hungarian_assignment",
    ),
    "reorder_candidate_probabilities_to_reference": (
        "alignment",
        "reorder_candidate_probabilities_to_reference",
    ),
    "reorder_candidate_transition_matrix_to_reference": (
        "alignment",
        "reorder_candidate_transition_matrix_to_reference",
    ),
    "reorder_candidate_values_to_reference": (
        "alignment",
        "reorder_candidate_values_to_reference",
    ),
    "state_signatures_from_fitted_hmm": (
        "alignment",
        "state_signatures_from_fitted_hmm",
    ),
    # Exact family-specific estimator APIs.
    "GaussianHMM": ("gaussian_hmm", "GaussianHMM"),
    "GMMHMM": ("gmm_hmm", "GMMHMM"),
    "StudentTHMM": ("student_t_hmm", "StudentTHMM"),
    # Shared model lifecycle and advanced backend API.
    "CausalFilterSession": ("filtering", "CausalFilterSession"),
    "CausalFilterStep": ("filtering", "CausalFilterStep"),
    "CausalFilteringError": ("filtering", "CausalFilteringError"),
    "DiagonalHMMConfig": ("base", "DiagonalHMMConfig"),
    "DiagonalHMMTrainingControls": ("trainer", "DiagonalHMMTrainingControls"),
    "DeterministicDiagonalHMMTrainer": (
        "trainer",
        "DeterministicDiagonalHMMTrainer",
    ),
    "FILTER_CHECKPOINT_VERSION": ("filtering", "FILTER_CHECKPOINT_VERSION"),
    "FITTED_DIAGONAL_HMM_VERSION": ("fitted", "FITTED_DIAGONAL_HMM_VERSION"),
    "FilterCheckpoint": ("filtering", "FilterCheckpoint"),
    "FilterStartMode": ("filtering", "FilterStartMode"),
    "FittedDiagonalHMM": ("fitted", "FittedDiagonalHMM"),
    "HMMFamily": ("base", "HMMFamily"),
    "HMMRestartDiagnostic": ("trainer", "HMMRestartDiagnostic"),
    "HMMTrainingError": ("trainer", "HMMTrainingError"),
    "HMMTrainingFailure": ("trainer", "HMMTrainingFailure"),
    "HMMTrainingResult": ("trainer", "HMMTrainingResult"),
    "ObservationBatch": ("observations", "ObservationBatch"),
    "ObservationSequence": ("observations", "ObservationSequence"),
    "RegimeTrainer": ("base", "RegimeTrainer"),
    "SHARED_HMM_TRAINER_VERSION": ("trainer", "SHARED_HMM_TRAINER_VERSION"),
    "SequenceFilterResult": ("filtering", "SequenceFilterResult"),
    "dump_filter_checkpoint_json": (
        "serialization",
        "dump_filter_checkpoint_json",
    ),
    "dump_fitted_hmm_json": ("serialization", "dump_fitted_hmm_json"),
    "filter_observation_sequence": ("filtering", "filter_observation_sequence"),
    "load_filter_checkpoint_json": (
        "serialization",
        "load_filter_checkpoint_json",
    ),
    "load_fitted_hmm_json": ("serialization", "load_fitted_hmm_json"),
    "log_emission_probabilities": ("fitted", "log_emission_probabilities"),
    "hash_observation_batch": ("trainer", "hash_observation_batch"),
    # Legacy pandas/hmmlearn compatibility surface.
    "MacroHMMTrainingConfig": (
        "legacy.macro_training",
        "MacroHMMTrainingConfig",
    ),
    "MacroHMMTrainingResult": (
        "legacy.macro_training",
        "MacroHMMTrainingResult",
    ),
    "MarketGMMHMM": ("legacy.hmmlearn_models", "MarketGMMHMM"),
    "MarketHMM": ("legacy.hmmlearn_models", "MarketHMM"),
    "build_macro_hmm_emissions": (
        "legacy.macro_training",
        "build_macro_hmm_emissions",
    ),
    "train_macro_hmm": ("legacy.macro_training", "train_macro_hmm"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
