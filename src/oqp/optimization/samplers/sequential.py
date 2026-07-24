"""Sequential model-based sampler adapters."""

from __future__ import annotations

from typing import Callable

import optuna


ConstraintFunction = Callable[[optuna.trial.FrozenTrial], tuple[float, ...]]


def build_tpe_sampler(
    *,
    seed: int,
    constraints_func: ConstraintFunction | None = None,
) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=10,
        multivariate=True,
        group=True,
        constant_liar=False,
        constraints_func=constraints_func,
    )


def build_gp_sampler(
    *,
    seed: int,
    constraints_func: ConstraintFunction | None = None,
) -> optuna.samplers.GPSampler:
    return optuna.samplers.GPSampler(
        seed=int(seed),
        n_startup_trials=10,
        constraints_func=constraints_func,
    )


__all__ = ["build_gp_sampler", "build_tpe_sampler"]
