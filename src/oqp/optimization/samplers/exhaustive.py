"""Exhaustive-search sampler adapters with combination-count guards."""

from __future__ import annotations

from decimal import Decimal
import math
from typing import Any

import optuna

from oqp.optimization.parameter_spaces import ComponentParameterSchema


def grid_search_space(schema: ComponentParameterSchema) -> dict[str, list[Any]]:
    search_space: dict[str, list[Any]] = {}
    for spec in schema.parameters:
        if not spec.tunable:
            continue
        if spec.parameter_type == "bool":
            search_space[spec.name] = [False, True]
        elif spec.parameter_type == "categorical":
            search_space[spec.name] = list(spec.choices)
        elif spec.parameter_type == "int":
            if spec.step is None:
                raise ValueError(
                    f"Grid search requires an explicit step for {spec.name!r}"
                )
            search_space[spec.name] = list(
                range(int(spec.low), int(spec.high) + 1, int(spec.step))
            )
        elif spec.parameter_type == "float":
            if spec.step is None or spec.log:
                raise ValueError(
                    f"Grid search requires a linear step for {spec.name!r}"
                )
            search_space[spec.name] = _decimal_grid(
                float(spec.low), float(spec.high), float(spec.step)
            )
        else:
            raise ValueError(
                f"Grid search does not support {spec.parameter_type!r}"
            )
    if not search_space:
        raise ValueError("Grid search requires at least one tunable parameter")
    return search_space


def grid_combination_count(search_space: dict[str, list[Any]]) -> int:
    return math.prod(len(values) for values in search_space.values())


def build_grid_sampler(
    schema: ComponentParameterSchema,
    *,
    seed: int,
    max_combinations: int,
) -> tuple[optuna.samplers.GridSampler, int]:
    search_space = grid_search_space(schema)
    combinations = grid_combination_count(search_space)
    if combinations > int(max_combinations):
        raise ValueError(
            f"Grid contains {combinations:,} combinations, above the guard "
            f"of {int(max_combinations):,}"
        )
    return optuna.samplers.GridSampler(search_space, seed=int(seed)), combinations


def build_bruteforce_sampler(*, seed: int) -> optuna.samplers.BruteForceSampler:
    return optuna.samplers.BruteForceSampler(seed=int(seed))


def _decimal_grid(low: float, high: float, step: float) -> list[float]:
    low_decimal = Decimal(str(low))
    high_decimal = Decimal(str(high))
    step_decimal = Decimal(str(step))
    values: list[float] = []
    value = low_decimal
    while value <= high_decimal:
        values.append(float(value))
        value += step_decimal
    return values


__all__ = [
    "build_bruteforce_sampler",
    "build_grid_sampler",
    "grid_combination_count",
    "grid_search_space",
]
