"""Persistence and comparison services used by the dashboard latent labs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oqp.research.ml.latent.diagnostics.codebook import (
    codebook_health_summary,
    compute_gmm_overlap,
    merge_gmm_probabilities,
)


DEFAULT_ARTIFACT_DIR = Path("runtime/artifacts/research/latent_factors")

__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "attach_gmm_diagnostics",
    "load_saved_latents",
    "save_latent_artifacts",
]


def attach_gmm_diagnostics(
    latent: pd.DataFrame,
    gmm_path: str | Path,
) -> dict:
    gmm_path = Path(gmm_path)
    if not gmm_path.exists():
        return {
            "latent_with_gmm": latent.copy(),
            "gmm_counts": pd.DataFrame(),
            "gmm_row_pct": pd.DataFrame(),
        }

    gmm = pd.read_parquet(gmm_path)
    merged = merge_gmm_probabilities(latent, gmm)
    counts, row_pct = compute_gmm_overlap(merged)
    return {
        "latent_with_gmm": merged,
        "gmm_counts": counts,
        "gmm_row_pct": row_pct,
    }


def save_latent_artifacts(
    result: dict,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    prefix: str = "storm_temporal_vqvae",
) -> dict[str, str]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "latent": artifact_dir / f"{prefix}_latents.parquet",
        "codebook": artifact_dir / f"{prefix}_codebook.csv",
        "loss_history": artifact_dir / f"{prefix}_loss_history.csv",
        "usage": artifact_dir / f"{prefix}_usage.csv",
        "encoder": artifact_dir / f"{prefix}_encoder.joblib",
    }
    result["latent"].to_parquet(paths["latent"], index=False)
    result["codebook"].to_csv(paths["codebook"], index=False)
    result["loss_history"].to_csv(paths["loss_history"], index=False)
    result["usage"].to_csv(paths["usage"], index=False)
    result["encoder"].save(paths["encoder"])
    return {key: str(value) for key, value in paths.items()}


def load_saved_latents(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    prefix: str = "storm_temporal_vqvae",
) -> dict:
    artifact_dir = Path(artifact_dir)
    paths = {
        "latent": artifact_dir / f"{prefix}_latents.parquet",
        "codebook": artifact_dir / f"{prefix}_codebook.csv",
        "loss_history": artifact_dir / f"{prefix}_loss_history.csv",
        "usage": artifact_dir / f"{prefix}_usage.csv",
    }
    if not paths["latent"].exists():
        return {}

    result = {
        "latent": pd.read_parquet(paths["latent"]),
        "codebook": pd.read_csv(paths["codebook"]) if paths["codebook"].exists() else pd.DataFrame(),
        "loss_history": pd.read_csv(paths["loss_history"]) if paths["loss_history"].exists() else pd.DataFrame(),
        "usage": pd.read_csv(paths["usage"]) if paths["usage"].exists() else pd.DataFrame(),
    }
    result["health"] = codebook_health_summary(result["usage"])
    return result
