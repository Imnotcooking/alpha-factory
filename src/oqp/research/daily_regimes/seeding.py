"""Deterministic seed utilities for fold, restart, and model isolation."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


MAX_SEED = (2**32) - 1


@dataclass(frozen=True, slots=True)
class SeedReport:
    seed: int
    python_seeded: bool
    numpy_seeded: bool
    torch_available: bool
    torch_seeded: bool
    deterministic_torch_requested: bool
    deterministic_torch_enabled: bool
    python_hash_seed: str


def normalize_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    parsed = int(seed)
    if not 0 <= parsed <= MAX_SEED:
        raise ValueError(f"seed must lie in [0, {MAX_SEED}].")
    return parsed


def derive_seed(base_seed: int, *tokens: Any) -> int:
    """Derive a stable, order-sensitive 32-bit seed from a base seed."""

    normalized_base = normalize_seed(base_seed)
    payload = json.dumps(
        {"base_seed": normalized_base, "tokens": tokens},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def restart_seeds(
    base_seed: int,
    count: int,
    *,
    namespace: str = "model_restart",
) -> tuple[int, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer.")
    return tuple(derive_seed(base_seed, namespace, index) for index in range(count))


def numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(normalize_seed(seed))


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedReport:
    """Seed supported libraries when explicitly invoked by a runner.

    Setting ``PYTHONHASHSEED`` here controls child processes.  The current
    interpreter's hash randomization is fixed at interpreter startup, so callers
    should also export the same value before launching a frozen production run.
    """

    normalized = normalize_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(normalized)
    random.seed(normalized)
    np.random.seed(normalized)

    torch_available = False
    torch_seeded = False
    deterministic_enabled = False
    try:
        import torch

        torch_available = True
        torch.manual_seed(normalized)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized)
        torch_seeded = True
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                deterministic_enabled = True
            except (AttributeError, RuntimeError):
                deterministic_enabled = False
    except Exception:  # pragma: no cover - optional runtime dependency
        torch_available = False

    return SeedReport(
        seed=normalized,
        python_seeded=True,
        numpy_seeded=True,
        torch_available=torch_available,
        torch_seeded=torch_seeded,
        deterministic_torch_requested=deterministic_torch,
        deterministic_torch_enabled=deterministic_enabled,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
    )


__all__ = [
    "MAX_SEED",
    "SeedReport",
    "derive_seed",
    "normalize_seed",
    "numpy_rng",
    "restart_seeds",
    "seed_everything",
]
