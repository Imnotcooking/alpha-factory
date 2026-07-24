"""Tree-based supervised regression models and their runtime infrastructure."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "LGBMModel": ("lightgbm", "LGBMModel"),
    "LGBMModelConfig": ("lightgbm", "LGBMModelConfig"),
    "LightGBMRegressorTrainer": ("lightgbm", "LightGBMRegressorTrainer"),
    "MLModelFactory": ("factory", "MLModelFactory"),
    "ModelRuntimeStatus": ("runtime", "ModelRuntimeStatus"),
    "XGBoostModelConfig": ("xgboost", "XGBoostModelConfig"),
    "XGBoostRegressorTrainer": ("xgboost", "XGBoostRegressorTrainer"),
    "XGBoostTrainingEngine": ("xgboost", "XGBoostTrainingEngine"),
    "probe_model_runtime": ("runtime", "probe_model_runtime"),
    "require_model_runtime": ("runtime", "require_model_runtime"),
    "resolve_model_artifact_path": ("inference", "resolve_model_artifact_path"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
