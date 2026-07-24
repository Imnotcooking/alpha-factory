"""Population and evolutionary sampler adapters."""

from __future__ import annotations

from typing import Callable

import optuna

from oqp.optimization.parameter_spaces import ComponentParameterSchema


ConstraintFunction = Callable[[optuna.trial.FrozenTrial], tuple[float, ...]]


def build_cmaes_sampler(
    schema: ComponentParameterSchema,
    *,
    seed: int,
) -> optuna.samplers.CmaEsSampler:
    unsupported = [
        spec.name
        for spec in schema.parameters
        if spec.tunable and spec.parameter_type in {"categorical", "bool"}
    ]
    if unsupported:
        raise ValueError(
            "CMA-ES requires a numerical search space; unsupported: "
            + ", ".join(unsupported)
        )
    return optuna.samplers.CmaEsSampler(seed=int(seed))


def build_nsga2_sampler(
    *,
    seed: int,
    max_trials: int,
    constraints_func: ConstraintFunction | None = None,
) -> optuna.samplers.NSGAIISampler:
    population_size = max(2, min(50, max(2, int(max_trials) // 2)))
    return optuna.samplers.NSGAIISampler(
        population_size=population_size,
        seed=int(seed),
        constraints_func=constraints_func,
    )


__all__ = ["build_cmaes_sampler", "build_nsga2_sampler"]
