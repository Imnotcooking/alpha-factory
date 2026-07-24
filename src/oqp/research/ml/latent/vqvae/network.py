"""PyTorch modules used by the canonical reusable VQ-VAE core.

The module remains import-safe when PyTorch is not installed. Availability is
checked only when a network or trainer is actually requested.
"""

from __future__ import annotations

from typing import Any


try:  # pragma: no cover - availability is environment-specific
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
except Exception:  # pragma: no cover
    torch = None
    nn = None
    torch_functional = None


def torch_available() -> bool:
    return torch is not None and nn is not None and torch_functional is not None


def require_torch() -> None:
    if not torch_available():
        raise ImportError(
            "PyTorch is required for VQ-VAE training and inference; install the research extra"
        )


if nn is not None:

    class VectorQuantizer(nn.Module):
        """Nearest-neighbour codebook with a straight-through gradient path."""

        def __init__(self, codebook_size: int, latent_dimension: int) -> None:
            super().__init__()
            self.codebook_size = int(codebook_size)
            self.latent_dimension = int(latent_dimension)
            self.codebook = nn.Embedding(self.codebook_size, self.latent_dimension)
            self.codebook.weight.data.uniform_(
                -1.0 / self.codebook_size,
                1.0 / self.codebook_size,
            )

        def forward(self, encoded: Any) -> tuple[Any, Any, Any, Any, Any]:
            squared_distances = (
                torch.sum(encoded**2, dim=1, keepdim=True)
                + torch.sum(self.codebook.weight**2, dim=1)
                - 2.0 * torch.matmul(encoded, self.codebook.weight.t())
            )
            codes = torch.argmin(squared_distances, dim=1)
            quantized = self.codebook(codes)
            codebook_loss = torch_functional.mse_loss(
                quantized,
                encoded.detach(),
            )
            commitment_loss = torch_functional.mse_loss(
                encoded,
                quantized.detach(),
            )
            straight_through = encoded + (quantized - encoded).detach()
            return (
                straight_through,
                codes,
                squared_distances,
                codebook_loss,
                commitment_loss,
            )

        def orthogonality_loss(self) -> Any:
            weights = torch_functional.normalize(self.codebook.weight, p=2, dim=1)
            gram = torch.matmul(weights, weights.t())
            identity = torch.eye(
                self.codebook_size,
                dtype=weights.dtype,
                device=weights.device,
            )
            return torch.mean((gram - identity) ** 2)

    class MLPVQVAE(nn.Module):
        """Fully connected VQ-VAE for an already prepared two-dimensional matrix."""

        def __init__(
            self,
            *,
            input_dimension: int,
            hidden_dimensions: tuple[int, ...],
            latent_dimension: int,
            codebook_size: int,
        ) -> None:
            super().__init__()
            encoder_layers: list[Any] = []
            previous = int(input_dimension)
            for width in hidden_dimensions:
                encoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            encoder_layers.append(nn.Linear(previous, latent_dimension))
            self.encoder = nn.Sequential(*encoder_layers)
            self.quantizer = VectorQuantizer(codebook_size, latent_dimension)

            decoder_layers: list[Any] = []
            previous = int(latent_dimension)
            for width in reversed(hidden_dimensions):
                decoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            decoder_layers.append(nn.Linear(previous, input_dimension))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, values: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
            encoded = self.encoder(values)
            (
                quantized,
                codes,
                squared_distances,
                codebook_loss,
                commitment_loss,
            ) = self.quantizer(encoded)
            reconstruction = self.decoder(quantized)
            return (
                reconstruction,
                encoded,
                quantized,
                codes,
                squared_distances,
                codebook_loss,
                commitment_loss,
            )

else:  # pragma: no cover - only exercised without the optional dependency
    VectorQuantizer = None
    MLPVQVAE = None


__all__ = [
    "MLPVQVAE",
    "VectorQuantizer",
    "require_torch",
    "torch",
    "torch_available",
    "torch_functional",
]
