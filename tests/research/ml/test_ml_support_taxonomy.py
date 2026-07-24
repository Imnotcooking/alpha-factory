from __future__ import annotations

from oqp.research.ml.core.optimizers import (
    TrainingOptimizerSpec,
    build_torch_optimizer,
)
from oqp.research.ml.evaluation.oos_mda import PurgedMDAConfig
from oqp.research.ml.features.governance import FeatureGovernanceConfig
from oqp.research.ml.preprocessing import PreprocessingSpec


def test_support_compatibility_paths_preserve_object_identity() -> None:
    from oqp.research.ml.feature_governance import (
        FeatureGovernanceConfig as HistoricalFeatureGovernanceConfig,
    )
    from oqp.research.ml.oos_mda import PurgedMDAConfig as HistoricalMDAConfig
    from oqp.research.ml.training_optimizers import (
        TrainingOptimizerSpec as HistoricalOptimizerSpec,
    )
    from oqp.research.ml.training_optimizers import (
        build_torch_optimizer as historical_build_optimizer,
    )
    from oqp.research.preprocessing import (
        PreprocessingSpec as HistoricalPreprocessingSpec,
    )

    assert HistoricalFeatureGovernanceConfig is FeatureGovernanceConfig
    assert HistoricalMDAConfig is PurgedMDAConfig
    assert HistoricalOptimizerSpec is TrainingOptimizerSpec
    assert historical_build_optimizer is build_torch_optimizer
    assert HistoricalPreprocessingSpec is PreprocessingSpec


def test_generic_feature_governance_excludes_regime_probabilities_by_default() -> None:
    assert FeatureGovernanceConfig().include_prob_features is False
