"""Inference-side artifact resolution for tree-based factor models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oqp.research.ml.regression.experiments import latest_ml_experiment
from oqp.research.ml.tree_based.factory import MLModelFactory
from oqp.research.model_registry import DEFAULT_RESEARCH_DB_PATH


def resolve_model_artifact_path(
    frame: pd.DataFrame | None = None,
    *,
    factor_id: str | None = None,
    model_type: str,
    fallback_path: str | Path | None = None,
    db_path: str | Path = DEFAULT_RESEARCH_DB_PATH,
) -> Path:
    """Resolve the selected experiment artifact without hardcoded legacy folders."""

    canonical = MLModelFactory.normalize_model_type(model_type)
    candidates: list[str | Path] = []
    if frame is not None:
        frame_model_type = frame.attrs.get("ml_model_type")
        if frame_model_type:
            try:
                matches = (
                    MLModelFactory.normalize_model_type(frame_model_type) == canonical
                )
            except ValueError:
                matches = False
            if matches and frame.attrs.get("ml_model_path"):
                candidates.append(frame.attrs["ml_model_path"])

    registered = latest_ml_experiment(
        db_path,
        model_type=canonical,
        factor_id=factor_id,
    )
    if registered and registered.get("artifact_path"):
        candidates.append(registered["artifact_path"])
    if fallback_path is not None:
        candidates.append(fallback_path)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()

    requested = f" for factor {factor_id}" if factor_id else ""
    raise FileNotFoundError(
        f"No registered {canonical} model artifact was found{requested}. "
        "Run run_ml_backtest.py with --retrain first."
    )


__all__ = ["resolve_model_artifact_path"]
