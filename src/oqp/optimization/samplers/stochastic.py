"""Random and low-discrepancy baseline samplers."""

from __future__ import annotations

import optuna


def build_random_sampler(*, seed: int) -> optuna.samplers.RandomSampler:
    return optuna.samplers.RandomSampler(seed=int(seed))


def build_qmc_sampler(*, seed: int) -> optuna.samplers.QMCSampler:
    return optuna.samplers.QMCSampler(
        qmc_type="sobol",
        scramble=True,
        seed=int(seed),
    )


__all__ = ["build_qmc_sampler", "build_random_sampler"]
