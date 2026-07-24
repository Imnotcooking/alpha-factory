from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.research.ml.latent.diagnostics import (
    codebook_health_summary,
    compute_codebook_usage,
)
from oqp.research.ml.latent.temporal import (
    TemporalPanelConfig,
    build_cross_sectional_snapshot,
    build_temporal_windows,
    detect_storm_feature_columns,
    flatten_temporal_windows,
)


def test_codebook_usage_health_marks_active_codes_and_entropy() -> None:
    latent = pd.DataFrame({"vq_code": [0, 0, 1, 3]})

    usage = compute_codebook_usage(latent, num_codes=4)
    health = codebook_health_summary(usage)

    assert usage["sample_count"].tolist() == [2, 1, 0, 1]
    assert usage["is_active"].tolist() == [True, True, False, True]
    assert health["active_codes"] == 3
    assert health["total_codes"] == 4
    assert np.isfinite(health["usage_perplexity"])


def test_temporal_panel_windows_and_flattening_are_reproducible() -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    matrix = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * 6 + ["BBB"] * 6,
            "f_momentum": range(12),
            "prob_panic": np.linspace(0.1, 0.9, 12),
            "target_1d_rank": np.linspace(-1.0, 1.0, 12),
            "sector": ["tech"] * 6 + ["health"] * 6,
        }
    )

    features = detect_storm_feature_columns(matrix, include_prob_features=True)
    x, meta, resolved_features = build_temporal_windows(
        matrix,
        config=TemporalPanelConfig(window_size=3, stride=2, max_samples=None),
    )
    flat = flatten_temporal_windows(x, resolved_features)
    snapshot = build_cross_sectional_snapshot(matrix, resolved_features)

    assert "f_momentum" in features
    assert "prob_panic" in features
    assert x.shape == (4, 3, 2)
    assert len(meta) == 4
    assert flat.shape == (4, 6)
    assert snapshot.shape[0] == 6
