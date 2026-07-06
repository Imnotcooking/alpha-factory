"""Optional SHAP regime-DNA diagnostics for trained XGBoost models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REGIME_COLUMNS = ("p_state_0", "p_state_1", "p_state_2")
DEFAULT_REGIME_MAP = {
    0: "State 0: Trend Highway",
    1: "State 1: Mean-Reverting Chop",
    2: "State 2: Panic / Rough Volatility",
}


def compute_shap_regime_dna(
    *,
    model_path: str | Path,
    matrix_path: str | Path,
    regime_probabilities_path: str | Path,
    output_path: str | Path | None = None,
    oos_start: str = "2024-01-01",
    sample_size: int = 10_000,
    random_state: int = 42,
    regime_columns: tuple[str, ...] = DEFAULT_REGIME_COLUMNS,
    regime_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate mean absolute SHAP values by dominant market regime.

    This helper loads SHAP and XGBoost lazily because they are optional,
    heavyweight diagnostic dependencies.
    """

    import shap
    import xgboost as xgb

    model_path = Path(model_path)
    matrix_path = Path(matrix_path)
    regime_probabilities_path = Path(regime_probabilities_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find XGBoost model at {model_path}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Cannot find feature matrix at {matrix_path}")
    if not regime_probabilities_path.exists():
        raise FileNotFoundError(
            f"Cannot find regime probabilities at {regime_probabilities_path}"
        )

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(model_path))
    expected_features = list(xgb_model.get_booster().feature_names or [])
    if not expected_features:
        raise ValueError("Loaded XGBoost model does not expose feature names.")

    feature_df = pd.read_parquet(matrix_path)
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    oos_df = feature_df[feature_df["date"] >= pd.Timestamp(oos_start)].copy()
    for col in expected_features:
        if col not in oos_df.columns:
            oos_df[col] = 0.0

    probs_df = pd.read_parquet(regime_probabilities_path)
    probs_df["date"] = pd.to_datetime(probs_df["date"], errors="coerce")
    required_regime = {"date", "ticker", *regime_columns}
    missing = sorted(required_regime - set(probs_df.columns))
    if missing:
        raise ValueError(f"Regime probabilities missing columns: {missing}")

    oos_df = oos_df.merge(
        probs_df[["date", "ticker", *regime_columns]],
        on=["date", "ticker"],
        how="inner",
    )
    if oos_df.empty:
        return pd.DataFrame()

    oos_df["dominant_regime"] = oos_df[list(regime_columns)].to_numpy().argmax(axis=1)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(oos_df.index, size=min(sample_size, len(oos_df)), replace=False)
    sample_df = oos_df.loc[sample_idx]
    x_sample = sample_df[expected_features]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(x_sample)
    shap_df = pd.DataFrame(np.abs(shap_values), columns=expected_features)
    shap_df["regime"] = sample_df["dominant_regime"].to_numpy()

    dna = shap_df.groupby("regime").mean().reset_index()
    labels = regime_map or DEFAULT_REGIME_MAP
    dna["regime_name"] = dna["regime"].map(labels)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        dna.to_csv(output, index=False)
    return dna


__all__ = [
    "DEFAULT_REGIME_COLUMNS",
    "DEFAULT_REGIME_MAP",
    "compute_shap_regime_dna",
]
