"""Compatibility import for legacy macro-HMM training helpers."""

from oqp.research.ml.regimes.legacy.macro_training import (
    MacroHMMTrainingConfig,
    MacroHMMTrainingResult,
    build_macro_hmm_emissions,
    train_macro_hmm,
)

__all__ = [
    "MacroHMMTrainingConfig",
    "MacroHMMTrainingResult",
    "build_macro_hmm_emissions",
    "train_macro_hmm",
]
