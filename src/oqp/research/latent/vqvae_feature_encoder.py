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
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class VQVAEConfig:
    latent_dim: int = 8
    hidden_dim: int = 64
    num_codes: int = 16
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    commitment_cost: float = 0.25
    orthogonality_cost: float = 0.05
    usage_cost: float = 0.10
    random_state: int = 42
    device: str = "cpu"


if nn is not None:

    class _VectorQuantizer(nn.Module):
        def __init__(self, num_codes: int, latent_dim: int, commitment_cost: float):
            super().__init__()
            self.num_codes = int(num_codes)
            self.latent_dim = int(latent_dim)
            self.commitment_cost = float(commitment_cost)
            self.codebook = nn.Embedding(self.num_codes, self.latent_dim)
            self.codebook.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

        def forward(self, z_e):
            distances = (
                torch.sum(z_e**2, dim=1, keepdim=True)
                + torch.sum(self.codebook.weight**2, dim=1)
                - 2 * torch.matmul(z_e, self.codebook.weight.t())
            )
            code_ids = torch.argmin(distances, dim=1)
            z_q = self.codebook(code_ids)

            codebook_loss = nn.functional.mse_loss(z_q, z_e.detach())
            commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss

            # Straight-through estimator: decoder sees quantized vectors while
            # gradients still flow to the encoder as if the path were identity.
            z_q_st = z_e + (z_q - z_e).detach()
            return z_q_st, code_ids, distances, vq_loss

        def orthogonality_loss(self):
            weights = nn.functional.normalize(self.codebook.weight, p=2, dim=1)
            gram = torch.matmul(weights, weights.t())
            eye = torch.eye(self.num_codes, device=weights.device)
            off_diag = gram - eye
            return torch.mean(off_diag**2)


    class _TabularVQVAE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_codes: int, commitment_cost: float):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.quantizer = _VectorQuantizer(num_codes, latent_dim, commitment_cost)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z_e = self.encoder(x)
            z_q, code_ids, distances, vq_loss = self.quantizer(z_e)
            recon = self.decoder(z_q)
            return recon, z_e, z_q, code_ids, distances, vq_loss

else:
    _TabularVQVAE = None


class VQVAEFeatureEncoder:
    """
    Discrete latent feature encoder.

    Unlike a plain VAE, this model snaps each encoded sample to one entry in a
    learned codebook. For market features, that gives us finite machine
    discovered states that can be compared against manual features and GMM/HMM
    regimes.
    """

    def __init__(self, config: VQVAEConfig | None = None):
        self.config = config or VQVAEConfig()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_: list[str] = []
        self.loss_history_: list[dict[str, float]] = []

    @staticmethod
    def is_available() -> bool:
        return torch is not None and nn is not None

    def fit(self, x: pd.DataFrame | np.ndarray) -> "VQVAEFeatureEncoder":
        self._require_torch()
        x_frame = self._to_frame(x)
        self.feature_names_ = list(x_frame.columns)
        x_clean = self._clean_frame(x_frame)

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        x_scaled = self.scaler.fit_transform(x_clean).astype(np.float32)
        tensor = torch.tensor(x_scaled, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.model = _TabularVQVAE(
            input_dim=x_scaled.shape[1],
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_codes=self.config.num_codes,
            commitment_cost=self.config.commitment_cost,
        ).to(self.config.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_history_ = []

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            totals = {
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "vq_loss": 0.0,
                "orthogonality_loss": 0.0,
                "rows": 0,
            }
            usage_counts = np.zeros(self.config.num_codes, dtype=np.int64)

            for (batch,) in loader:
                batch = batch.to(self.config.device)
                optimizer.zero_grad()
                recon, _, _, code_ids, distances, vq_loss = self.model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction="mean")
                ortho_loss = self.model.quantizer.orthogonality_loss()
                soft_assign = torch.softmax(-distances, dim=1)
                mean_probs = soft_assign.mean(dim=0).clamp_min(1e-8)
                usage_loss = torch.sum(mean_probs * torch.log(mean_probs * self.config.num_codes))
                loss = (
                    recon_loss
                    + vq_loss
                    + self.config.orthogonality_cost * ortho_loss
                    + self.config.usage_cost * usage_loss
                )
                loss.backward()
                optimizer.step()

                rows = int(batch.shape[0])
                totals["rows"] += rows
                totals["loss"] += float(loss.detach().cpu()) * rows
                totals["reconstruction_loss"] += float(recon_loss.detach().cpu()) * rows
                totals["vq_loss"] += float(vq_loss.detach().cpu()) * rows
                totals["orthogonality_loss"] += float(ortho_loss.detach().cpu()) * rows
                totals.setdefault("usage_loss", 0.0)
                totals["usage_loss"] += float(usage_loss.detach().cpu()) * rows
                usage_counts += np.bincount(
                    code_ids.detach().cpu().numpy(),
                    minlength=self.config.num_codes,
                )

            denom = max(totals["rows"], 1)
            active_codes = int((usage_counts > 0).sum())
            probs = usage_counts / max(usage_counts.sum(), 1)
            entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
            self.loss_history_.append(
                {
                    "epoch": float(epoch),
                    "loss": totals["loss"] / denom,
                    "reconstruction_loss": totals["reconstruction_loss"] / denom,
                    "vq_loss": totals["vq_loss"] / denom,
                    "orthogonality_loss": totals["orthogonality_loss"] / denom,
                    "usage_loss": totals.get("usage_loss", 0.0) / denom,
                    "active_codes": float(active_codes),
                    "code_entropy": entropy,
                    "code_perplexity": float(np.exp(entropy)),
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
            _, z_e, z_q, code_ids, distances, _ = self.model(tensor)
            z_q_np = z_q.detach().cpu().numpy()
            z_e_np = z_e.detach().cpu().numpy()
            code_np = code_ids.detach().cpu().numpy()
            min_distance = distances.min(dim=1).values.detach().cpu().numpy()

        out = pd.DataFrame(index=x_frame.index)
        out["vq_code"] = code_np.astype(int)
        out["vq_distance"] = min_distance.astype(float)
        for idx in range(z_q_np.shape[1]):
            out[f"z_vq_{idx + 1:02d}"] = z_q_np[:, idx]
            out[f"z_encoder_{idx + 1:02d}"] = z_e_np[:, idx]
        return out.reset_index(drop=True)

    def fit_transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        return self.fit(x).transform(x)

    def get_codebook(self) -> pd.DataFrame:
        self._require_fitted()
        weights = self.model.quantizer.codebook.weight.detach().cpu().numpy()
        out = pd.DataFrame(
            weights,
            columns=[f"code_dim_{idx + 1:02d}" for idx in range(weights.shape[1])],
        )
        out.insert(0, "vq_code", np.arange(weights.shape[0], dtype=int))
        return out

    def loss_history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_history_)

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
    def load(cls, path: str | Path) -> "VQVAEFeatureEncoder":
        if torch is None:
            raise ImportError("PyTorch is required to load a VQVAEFeatureEncoder artifact.")
        payload = joblib.load(path)
        encoder = cls(payload["config"])
        encoder.feature_names_ = list(payload["feature_names"])
        encoder.scaler = payload["scaler"]
        encoder.loss_history_ = list(payload.get("loss_history", []))
        encoder.model = _TabularVQVAE(
            input_dim=len(encoder.feature_names_),
            hidden_dim=encoder.config.hidden_dim,
            latent_dim=encoder.config.latent_dim,
            num_codes=encoder.config.num_codes,
            commitment_cost=encoder.config.commitment_cost,
        ).to(encoder.config.device)
        encoder.model.load_state_dict(payload["state_dict"])
        encoder.model.eval()
        return encoder

    def _require_torch(self) -> None:
        if not self.is_available():
            raise ImportError(
                "PyTorch is required for VQVAEFeatureEncoder. Install torch to train VQ latent features."
            )

    def _require_fitted(self) -> None:
        self._require_torch()
        if self.model is None or not self.feature_names_:
            raise RuntimeError("VQVAEFeatureEncoder must be fitted before transform().")

    @staticmethod
    def _to_frame(
        x: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()
        array = np.asarray(x)
        if array.ndim != 2:
            raise ValueError("Expected a 2D matrix after temporal flattening.")
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(array.shape[1])]
        return pd.DataFrame(array, columns=feature_names)

    @staticmethod
    def _clean_frame(x: pd.DataFrame) -> pd.DataFrame:
        clean = x.replace([np.inf, -np.inf], np.nan).copy()
        numeric_cols = [col for col in clean.columns if pd.api.types.is_numeric_dtype(clean[col])]
        clean = clean[numeric_cols]
        if clean.empty:
            raise ValueError("No numeric columns available for VQ-VAE encoding.")
        return clean.fillna(clean.median(numeric_only=True)).fillna(0.0)
