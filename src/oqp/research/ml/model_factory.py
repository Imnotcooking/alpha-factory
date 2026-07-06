"""Factory for promoted supervised research ML models."""

from __future__ import annotations

from oqp.research.ml.lgbm_model import LGBMModel
from oqp.research.ml.xgboost_model import XGBoostTrainingEngine


class MLModelFactory:
    """Create supervised research model instances by stable type string."""

    _REGISTRY = {
        "LGBM": LGBMModel,
        "LIGHTGBM": LGBMModel,
        "XGB": XGBoostTrainingEngine,
        "XGBOOST": XGBoostTrainingEngine,
    }

    @classmethod
    def create_model(cls, model_type: str, data_path: str, **kwargs):
        target = str(model_type).upper()
        model_cls = cls._REGISTRY.get(target)
        if model_cls is None:
            supported = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(f"Unknown model type: {model_type}. Supported: {supported}")
        return model_cls(data_path, **kwargs)

    @classmethod
    def supported_models(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._REGISTRY))


__all__ = ["MLModelFactory"]
