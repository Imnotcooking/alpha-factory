from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oqp.research.latent.codebook_diagnostics import (
    codebook_health_summary,
    compute_code_target_ic,
    compute_code_transition_stats,
    compute_codebook_usage,
    compute_gmm_overlap,
    compute_manual_feature_profile,
    merge_gmm_probabilities,
)
from oqp.research.latent.storm_panel_dataset import (
    TemporalPanelConfig,
    build_temporal_windows,
    detect_storm_feature_columns,
    flatten_temporal_windows,
)
from oqp.research.latent.vqvae_feature_encoder import VQVAEConfig, VQVAEFeatureEncoder


DEFAULT_ARTIFACT_DIR = Path("runtime/artifacts/research/latent_factors")


def train_temporal_vqvae_latents(
    matrix: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "target_1d_rank",
    window_size: int = 20,
    max_samples: int = 20_000,
    num_codes: int = 16,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    epochs: int = 12,
    random_state: int = 42,
) -> dict:
    """
    Train a compact temporal VQ-VAE and return latent features + diagnostics.

    This is a research cross-check, not a production factor. The model is kept
    small so the Streamlit page can run it on demand without surprising the user.
    """
    features = feature_cols or detect_storm_feature_columns(matrix, include_prob_features=True)
    panel_config = TemporalPanelConfig(
        window_size=window_size,
        max_samples=max_samples,
        random_state=random_state,
        include_prob_features=True,
    )
    x, meta, features = build_temporal_windows(matrix, features, panel_config)
    flat = flatten_temporal_windows(x, features)

    vq_config = VQVAEConfig(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_codes=num_codes,
        epochs=epochs,
        random_state=random_state,
    )
    encoder = VQVAEFeatureEncoder(vq_config)
    latent_values = encoder.fit_transform(flat)

    latent = pd.concat([meta.reset_index(drop=True), latent_values], axis=1)
    source_cols = ["date", "ticker", *features]
    optional_cols = [target_col, "sector"]
    source_cols.extend([col for col in optional_cols if col in matrix.columns and col not in source_cols])

    matrix_join = matrix[source_cols].copy()
    matrix_join["date"] = pd.to_datetime(matrix_join["date"])
    matrix_join["ticker"] = matrix_join["ticker"].astype(str)
    latent = latent.merge(matrix_join, on=["date", "ticker"], how="left", suffixes=("", "_source"))

    usage = compute_codebook_usage(latent, num_codes=num_codes)
    transitions = compute_code_transition_stats(latent)
    feature_profile = compute_manual_feature_profile(latent, features)
    code_ic = compute_code_target_ic(latent, target_col=target_col)

    return {
        "latent": latent,
        "encoder": encoder,
        "codebook": encoder.get_codebook(),
        "loss_history": encoder.loss_history_frame(),
        "usage": usage,
        "health": codebook_health_summary(usage),
        "transitions": transitions,
        "feature_profile": feature_profile,
        "code_ic": code_ic,
        "feature_cols": features,
        "config": {
            "window_size": window_size,
            "max_samples": max_samples,
            "num_codes": num_codes,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "random_state": random_state,
        },
    }


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
