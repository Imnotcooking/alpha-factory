"""Validated configuration for the canonical reusable VQ-VAE core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class VQVAEConfig:
    """One fully resolved VQ-VAE training specification.

    Input cleaning, scaling, temporal window construction, and economic state
    interpretation deliberately live in adapters outside this numerical core.
    """

    hidden_dimensions: tuple[int, ...] = (64, 64)
    latent_dimension: int = 8
    codebook_size: int = 16
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    commitment_weight: float = 0.25
    orthogonality_weight: float = 0.0
    usage_weight: float = 0.0
    random_seed: int = 42
    device: str = "cpu"
    deterministic: bool = True
    shuffle: bool = False

    def __post_init__(self) -> None:
        if type(self.hidden_dimensions) is not tuple:
            raise TypeError("hidden_dimensions must be a tuple")
        if not self.hidden_dimensions or any(
            type(value) is not int or value <= 0 for value in self.hidden_dimensions
        ):
            raise ValueError("hidden_dimensions must contain positive integers")
        for name in (
            "latent_dimension",
            "codebook_size",
            "epochs",
            "batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least two")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if type(self.device) is not str or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if type(self.deterministic) is not bool or type(self.shuffle) is not bool:
            raise TypeError("deterministic and shuffle must be booleans")
        if self.device != "cpu":
            raise ValueError("the VQ-VAE core v1 requires device='cpu'")
        if not self.deterministic:
            raise ValueError("the VQ-VAE core v1 requires deterministic=True")
        for name in (
            "learning_rate",
            "commitment_weight",
            "orthogonality_weight",
            "usage_weight",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            minimum = 0.0 if name != "learning_rate" else 1.0e-300
            if (
                value < minimum
                or value != value
                or value in {float("inf"), -float("inf")}
            ):
                raise ValueError(
                    f"{name} must be finite and {'positive' if name == 'learning_rate' else 'non-negative'}"
                )

    def state_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dimensions"] = list(self.hidden_dimensions)
        return payload

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "VQVAEConfig":
        if not isinstance(state, Mapping):
            raise TypeError("VQ-VAE config state must be a mapping")
        expected = {
            "hidden_dimensions",
            "latent_dimension",
            "codebook_size",
            "epochs",
            "batch_size",
            "learning_rate",
            "commitment_weight",
            "orthogonality_weight",
            "usage_weight",
            "random_seed",
            "device",
            "deterministic",
            "shuffle",
        }
        if set(state) != expected:
            raise ValueError("VQ-VAE config state has unknown or missing fields")
        hidden = state["hidden_dimensions"]
        if not isinstance(hidden, list):
            raise TypeError("hidden_dimensions must be an array")
        return cls(
            hidden_dimensions=tuple(hidden),
            latent_dimension=state["latent_dimension"],
            codebook_size=state["codebook_size"],
            epochs=state["epochs"],
            batch_size=state["batch_size"],
            learning_rate=state["learning_rate"],
            commitment_weight=state["commitment_weight"],
            orthogonality_weight=state["orthogonality_weight"],
            usage_weight=state["usage_weight"],
            random_seed=state["random_seed"],
            device=state["device"],
            deterministic=state["deterministic"],
            shuffle=state["shuffle"],
        )


__all__ = ["VQVAEConfig"]
