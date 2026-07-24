"""Registry for purpose-appropriate optimization sampler adapters."""

from __future__ import annotations

from typing import Any

import optuna

from oqp.optimization.contracts import SearchBudget
from oqp.optimization.parameter_spaces import ComponentParameterSchema
from oqp.optimization.samplers.exhaustive import (
    build_bruteforce_sampler,
    build_grid_sampler,
)
from oqp.optimization.samplers.population import (
    build_cmaes_sampler,
    build_nsga2_sampler,
)
from oqp.optimization.samplers.sequential import (
    build_gp_sampler,
    build_tpe_sampler,
)
from oqp.optimization.samplers.stochastic import (
    build_qmc_sampler,
    build_random_sampler,
)


SUPPORTED_OPTUNA_SAMPLERS = {
    "grid",
    "bruteforce",
    "random",
    "qmc",
    "tpe",
    "gp",
    "cmaes",
    "nsga2",
}


def build_optuna_sampler(
    sampler_id: str,
    schema: ComponentParameterSchema,
    *,
    seed: int,
    budget: SearchBudget,
    constraints_func=None,
) -> tuple[optuna.samplers.BaseSampler, dict[str, Any]]:
    sampler_id = str(sampler_id).strip().lower()
    metadata: dict[str, Any] = {"sampler_id": sampler_id}
    if sampler_id == "grid":
        sampler, combinations = build_grid_sampler(
            schema,
            seed=seed,
            max_combinations=budget.max_grid_combinations,
        )
        metadata["grid_combinations"] = combinations
        return sampler, metadata
    if sampler_id == "bruteforce":
        return build_bruteforce_sampler(seed=seed), metadata
    if sampler_id == "random":
        return build_random_sampler(seed=seed), metadata
    if sampler_id == "qmc":
        return build_qmc_sampler(seed=seed), metadata
    if sampler_id == "tpe":
        return build_tpe_sampler(
            seed=seed,
            constraints_func=constraints_func,
        ), metadata
    if sampler_id == "gp":
        return build_gp_sampler(
            seed=seed,
            constraints_func=constraints_func,
        ), metadata
    if sampler_id == "cmaes":
        return build_cmaes_sampler(schema, seed=seed), metadata
    if sampler_id == "nsga2":
        return build_nsga2_sampler(
            seed=seed,
            max_trials=budget.max_trials,
            constraints_func=constraints_func,
        ), metadata
    raise ValueError(
        f"Unknown sampler {sampler_id!r}; expected one of "
        f"{sorted(SUPPORTED_OPTUNA_SAMPLERS)}"
    )


__all__ = ["SUPPORTED_OPTUNA_SAMPLERS", "build_optuna_sampler"]
