"""Training utilities for market regime HMM artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from oqp.intelligence.regime_engine.hmm_regime import MarketHMM


@dataclass(frozen=True)
class MacroHMMTrainingConfig:
    """Configuration for reproducible macro HMM training."""

    n_components: int = 3
    rolling_vol_window: int = 20
    covariance_type: str = "full"
    random_state: int = 42
    model_filename: str = "hmm_model.pkl"
    scaler_filename: str = "hmm_scaler.pkl"


@dataclass(frozen=True)
class MacroHMMTrainingResult:
    """Artifacts produced by a macro HMM training run."""

    model: MarketHMM
    scaler: object
    macro_emissions: pd.DataFrame
    model_path: Optional[Path] = None
    scaler_path: Optional[Path] = None


def build_macro_hmm_emissions(
    feature_matrix: pd.DataFrame,
    *,
    rolling_vol_window: int = 20,
) -> pd.DataFrame:
    """Build date-level return/volatility emissions from an asset feature matrix."""

    _require_columns(feature_matrix, ("date", "close"))
    df = feature_matrix.copy()
    if "ticker" in df.columns:
        df["returns"] = df.groupby("ticker")["close"].pct_change()
        df["volatility"] = (
            df.groupby("ticker")["returns"]
            .rolling(rolling_vol_window)
            .std()
            .reset_index(level=0, drop=True)
        )
    else:
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(rolling_vol_window).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["returns", "volatility"])
    return (
        df.groupby("date", as_index=False)
        .agg({"returns": "mean", "volatility": "mean"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def train_macro_hmm(
    feature_matrix: pd.DataFrame | str | Path,
    *,
    output_dir: str | Path | None = None,
    config: MacroHMMTrainingConfig | None = None,
) -> MacroHMMTrainingResult:
    """Train a volatility-aligned macro HMM and optionally persist artifacts."""

    from sklearn.preprocessing import StandardScaler

    config = config or MacroHMMTrainingConfig()
    feature_df = _load_feature_matrix(feature_matrix)
    macro_df = build_macro_hmm_emissions(
        feature_df,
        rolling_vol_window=config.rolling_vol_window,
    )

    scaler = StandardScaler()
    scaled_emissions = scaler.fit_transform(macro_df[["returns", "volatility"]])
    scaled_df = pd.DataFrame(
        {"returns": scaled_emissions[:, 0], "volatility": scaled_emissions[:, 1]}
    )

    model = MarketHMM(
        n_components=config.n_components,
        covariance_type=config.covariance_type,
        random_state=config.random_state,
    )
    model.fit(scaled_df)

    model_path = None
    scaler_path = None
    if output_dir is not None:
        import joblib

        artifact_dir = Path(output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / config.model_filename
        scaler_path = artifact_dir / config.scaler_filename
        model.save(str(model_path))
        joblib.dump(scaler, scaler_path)

    return MacroHMMTrainingResult(
        model=model,
        scaler=scaler,
        macro_emissions=macro_df,
        model_path=model_path,
        scaler_path=scaler_path,
    )


def _load_feature_matrix(feature_matrix: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(feature_matrix, pd.DataFrame):
        return feature_matrix
    path = Path(feature_matrix)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature matrix format: {path.suffix}")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Macro HMM feature matrix missing columns: {', '.join(missing)}")


__all__ = [
    "MacroHMMTrainingConfig",
    "MacroHMMTrainingResult",
    "build_macro_hmm_emissions",
    "train_macro_hmm",
]
