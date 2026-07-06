from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from oqp.research.artifacts import fingerprint_file, slugify
from oqp.research.state_space.base_filter import StateSpaceArtifact
from oqp.research.state_space.diagnostics import coefficient_columns, summarize_dual_kalman_output
from oqp.research.state_space.dual_kalman_regression import (
    DualKalmanRegression,
    DualKalmanRegressionConfig,
)


__all__ = ["build_dual_kalman_features", "save_dual_kalman_feature_artifact"]


def build_dual_kalman_features(df: pd.DataFrame, config: DualKalmanRegressionConfig) -> pd.DataFrame:
    """Compute deterministic DKF feature columns from an in-memory frame."""

    return DualKalmanRegression(config).fit_transform(df)


def save_dual_kalman_feature_artifact(
    df: pd.DataFrame,
    config: DualKalmanRegressionConfig,
    *,
    artifact_name: str = "dual_kalman_regression",
    source_path: str | Path | None = None,
    output_root: str | Path = "runtime/artifacts/research/state_space",
) -> StateSpaceArtifact:
    """
    Compute and persist DKF features plus metadata for reproducible research.
    """

    features = build_dual_kalman_features(df, config)
    output_dir = Path(output_root) / slugify(artifact_name) / _artifact_id(artifact_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.parquet"
    metadata_path = output_dir / "metadata.json"

    features.to_parquet(output_path, index=False)
    feature_cols = _feature_columns(features, prefix=config.prefix)
    diagnostics = summarize_dual_kalman_output(features, prefix=config.prefix)
    metadata: dict[str, Any] = {
        "artifact_name": artifact_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": _config_dict(config),
        "row_count": int(len(features)),
        "feature_columns": feature_cols,
        "coefficient_columns": coefficient_columns(features, prefix=config.prefix),
        "diagnostics": diagnostics.to_dict(orient="records"),
        "source_fingerprint": asdict(fingerprint_file(source_path, include_hash=True)) if source_path else None,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return StateSpaceArtifact(
        output_path=output_path.as_posix(),
        metadata_path=metadata_path.as_posix(),
        row_count=int(len(features)),
        feature_columns=feature_cols,
        metadata=metadata,
    )


def _feature_columns(features: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in features.columns if col.startswith(f"{prefix}_")]


def _artifact_id(name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{slugify(name)}_{timestamp}_{uuid4().hex[:8]}"


def _config_dict(config: DualKalmanRegressionConfig) -> dict[str, Any]:
    out = asdict(config)
    schema = out.get("schema", {})
    if isinstance(schema.get("x_cols"), tuple):
        schema["x_cols"] = list(schema["x_cols"])
    if isinstance(schema.get("group_cols"), tuple):
        schema["group_cols"] = list(schema["group_cols"])
    return out
