from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional research dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class VAEConfig:
    latent_dim: int = 4
    hidden_dim: int = 32
    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 1e-3
    beta: float = 1.0
    random_state: int = 42
    device: str = "cpu"


if nn is not None:

    class _FeatureVAE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.log_var = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, x):
            hidden = self.encoder(x)
            return self.mu(hidden), self.log_var(hidden)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decoder(z)
            return recon, mu, log_var

else:
    _FeatureVAE = None


class VAEFeatureEncoder:
    """
    Nonlinear latent feature encoder for financial feature matrices.

    This is intentionally separated from feature_engineering.py. The transparent
    engineered features remain the source layer; this module learns optional
    latent variables for later walk-forward comparison against raw and PCA
    baselines.
    """

    def __init__(self, config: VAEConfig | None = None):
        self.config = config or VAEConfig()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_: list[str] = []
        self.loss_history_: list[dict[str, float]] = []

    @staticmethod
    def is_available() -> bool:
        return torch is not None and nn is not None

    def fit(self, x: pd.DataFrame | np.ndarray) -> "VAEFeatureEncoder":
        self._require_torch()
        x_frame = self._to_frame(x)
        self.feature_names_ = list(x_frame.columns)
        x_clean = self._clean_frame(x_frame)

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        x_scaled = self.scaler.fit_transform(x_clean).astype(np.float32)
        tensor = torch.tensor(x_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model = _FeatureVAE(
            input_dim=x_scaled.shape[1],
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
        ).to(self.config.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        self.loss_history_ = []

        for epoch in range(1, self.config.epochs + 1):
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_rows = 0

            for (batch,) in loader:
                batch = batch.to(self.config.device)
                optimizer.zero_grad()
                recon, mu, log_var = self.model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.config.beta * kl_loss
                loss.backward()
                optimizer.step()

                batch_size = int(batch.shape[0])
                total_rows += batch_size
                total_loss += float(loss.detach().cpu())
                total_recon += float(recon_loss.detach().cpu())
                total_kl += float(kl_loss.detach().cpu())

            denom = max(total_rows, 1)
            self.loss_history_.append(
                {
                    "epoch": float(epoch),
                    "loss": total_loss / denom,
                    "reconstruction_loss": total_recon / denom,
                    "kl_loss": total_kl / denom,
                }
            )

        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        self._require_fitted()
        x_frame = self._to_frame(x, feature_names=self.feature_names_)
        x_clean = self._clean_frame(x_frame)
        x_scaled = self.scaler.transform(x_clean).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.config.device)
            mu, _ = self.model.encode(tensor)
            latent = mu.detach().cpu().numpy()

        columns = [f"z_vae_{idx + 1:02d}" for idx in range(latent.shape[1])]
        return pd.DataFrame(latent, columns=columns, index=x_frame.index)

    def fit_transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        return self.fit(x).transform(x)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "feature_names": self.feature_names_,
            "scaler": self.scaler,
            "state_dict": self.model.state_dict(),
            "loss_history": self.loss_history_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "VAEFeatureEncoder":
        if torch is None:
            raise ImportError("PyTorch is required to load a VAEFeatureEncoder artifact.")
        payload = joblib.load(path)
        encoder = cls(payload["config"])
        encoder.feature_names_ = list(payload["feature_names"])
        encoder.scaler = payload["scaler"]
        encoder.loss_history_ = list(payload.get("loss_history", []))
        encoder.model = _FeatureVAE(
            input_dim=len(encoder.feature_names_),
            hidden_dim=encoder.config.hidden_dim,
            latent_dim=encoder.config.latent_dim,
        ).to(encoder.config.device)
        encoder.model.load_state_dict(payload["state_dict"])
        encoder.model.eval()
        return encoder

    def _require_torch(self) -> None:
        if not self.is_available():
            raise ImportError(
                "PyTorch is required for VAEFeatureEncoder. Install torch or use the PCA baseline."
            )

    def _require_fitted(self) -> None:
        self._require_torch()
        if self.model is None or not self.feature_names_:
            raise RuntimeError("VAEFeatureEncoder must be fitted before transform().")

    @staticmethod
    def _to_frame(
        x: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()
        array = np.asarray(x)
        if array.ndim != 2:
            raise ValueError("Expected a 2D feature matrix.")
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(array.shape[1])]
        return pd.DataFrame(array, columns=feature_names)

    @staticmethod
    def _clean_frame(x: pd.DataFrame) -> pd.DataFrame:
        clean = x.replace([np.inf, -np.inf], np.nan).copy()
        numeric_cols = [col for col in clean.columns if pd.api.types.is_numeric_dtype(clean[col])]
        clean = clean[numeric_cols]
        if clean.empty:
            raise ValueError("No numeric columns available for VAE encoding.")
        return clean.fillna(clean.median(numeric_only=True)).fillna(0.0)
