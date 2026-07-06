"""Machine-learning research governance tools."""

from oqp.research.ml.feature_governance import (
    FeatureGovernanceConfig,
    coerce_numeric_columns,
    compute_feature_governance,
    detect_feature_columns,
    list_matrix_files,
    load_matrix,
    tag_feature_family,
)
from oqp.research.ml.lgbm_model import LGBMModel, LGBMModelConfig
from oqp.research.ml.model_factory import MLModelFactory
from oqp.research.ml.oos_mda import (
    PurgedMDAConfig,
    build_purged_time_folds,
    compute_oos_mda,
    default_xgb_regressor,
    rank_ic_score,
)
from oqp.research.ml.supervised import BaseMLModel, SupervisedModelBase, WalkForwardConfig
from oqp.research.ml.xgboost_model import XGBoostTrainingEngine

__all__ = [
    "BaseMLModel",
    "FeatureGovernanceConfig",
    "LGBMModel",
    "LGBMModelConfig",
    "MLModelFactory",
    "PurgedMDAConfig",
    "SupervisedModelBase",
    "WalkForwardConfig",
    "XGBoostTrainingEngine",
    "build_purged_time_folds",
    "coerce_numeric_columns",
    "compute_feature_governance",
    "compute_oos_mda",
    "default_xgb_regressor",
    "detect_feature_columns",
    "list_matrix_files",
    "load_matrix",
    "rank_ic_score",
    "tag_feature_family",
]
