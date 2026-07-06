"""Latent feature research utilities."""

from oqp.research.latent.codebook_diagnostics import (
    codebook_health_summary,
    compute_code_target_ic,
    compute_code_transition_stats,
    compute_codebook_usage,
    compute_gmm_overlap,
    compute_manual_feature_profile,
    merge_gmm_probabilities,
)
from oqp.research.latent.storm_feature_comparison import (
    DEFAULT_ARTIFACT_DIR,
    attach_gmm_diagnostics,
    load_saved_latents,
    save_latent_artifacts,
    train_temporal_vqvae_latents,
)
from oqp.research.latent.storm_panel_dataset import (
    TemporalPanelConfig,
    build_cross_sectional_snapshot,
    build_temporal_windows,
    detect_storm_feature_columns,
    flatten_temporal_windows,
)
from oqp.research.latent.vae_feature_encoder import VAEConfig, VAEFeatureEncoder
from oqp.research.latent.vqvae_feature_encoder import VQVAEConfig, VQVAEFeatureEncoder

__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "TemporalPanelConfig",
    "VAEConfig",
    "VAEFeatureEncoder",
    "VQVAEConfig",
    "VQVAEFeatureEncoder",
    "attach_gmm_diagnostics",
    "build_cross_sectional_snapshot",
    "build_temporal_windows",
    "codebook_health_summary",
    "compute_code_target_ic",
    "compute_code_transition_stats",
    "compute_codebook_usage",
    "compute_gmm_overlap",
    "compute_manual_feature_profile",
    "detect_storm_feature_columns",
    "flatten_temporal_windows",
    "load_saved_latents",
    "merge_gmm_probabilities",
    "save_latent_artifacts",
    "train_temporal_vqvae_latents",
]
