"""Declarative PyTorch optimizer construction for model-weight training only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable


SUPPORTED_TORCH_OPTIMIZERS = {
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "lbfgs",
}


@dataclass(frozen=True, slots=True)
class TrainingOptimizerSpec:
    algorithm: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm).strip().lower()
        if algorithm not in SUPPORTED_TORCH_OPTIMIZERS:
            raise ValueError(
                f"algorithm must be one of {sorted(SUPPORTED_TORCH_OPTIMIZERS)}"
            )
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")
        if float(self.weight_decay) < 0:
            raise ValueError("weight_decay cannot be negative")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        parameters = dict(self.parameters)
        reserved = sorted(set(parameters).intersection({"lr", "weight_decay"}))
        if reserved:
            raise ValueError(
                "parameters cannot override declarative fields: "
                + ", ".join(reserved)
            )
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_torch_optimizer(
    model_parameters: Iterable,
    spec: TrainingOptimizerSpec,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for model-weight optimization") from exc

    kwargs = {"lr": spec.learning_rate, **dict(spec.parameters)}
    if spec.algorithm == "lbfgs":
        if spec.weight_decay:
            raise ValueError("PyTorch LBFGS does not support weight_decay")
    else:
        kwargs["weight_decay"] = spec.weight_decay
    builders = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "lbfgs": torch.optim.LBFGS,
    }
    return builders[spec.algorithm](model_parameters, **kwargs)


__all__ = [
    "SUPPORTED_TORCH_OPTIMIZERS",
    "TrainingOptimizerSpec",
    "build_torch_optimizer",
]
