"""Factory for promoted tree-based regression trainers."""

from __future__ import annotations

from oqp.research.ml.tree_based.lightgbm import LightGBMRegressorTrainer
from oqp.research.ml.tree_based.xgboost import XGBoostRegressorTrainer


class MLModelFactory:
    """Create supervised research model instances by stable type string."""

    _REGISTRY = {
        "lightgbm": LightGBMRegressorTrainer,
        "xgboost": XGBoostRegressorTrainer,
    }
    _ALIASES = {
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
        "xgb": "xgboost",
        "xgboost": "xgboost",
    }

    @classmethod
    def create_model(cls, model_type: str, data_path: str, **kwargs):
        target = cls.normalize_model_type(model_type)
        model_cls = cls._REGISTRY[target]
        return model_cls(data_path, **kwargs)

    @classmethod
    def supported_models(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._REGISTRY))

    @classmethod
    def normalize_model_type(cls, model_type: str) -> str:
        target = str(model_type).strip().lower()
        canonical = cls._ALIASES.get(target)
        if canonical is None:
            supported = ", ".join(cls.supported_models())
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {supported}"
            )
        return canonical


__all__ = ["MLModelFactory"]
