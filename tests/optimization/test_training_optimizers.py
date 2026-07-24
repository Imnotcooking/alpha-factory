from __future__ import annotations

import pytest

from oqp.research.ml.core.optimizers import (
    TrainingOptimizerSpec,
    build_torch_optimizer,
)


def test_training_optimizer_spec_is_fingerprinted() -> None:
    first = TrainingOptimizerSpec(
        algorithm="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    second = TrainingOptimizerSpec(
        algorithm="adamw",
        learning_rate=2e-3,
        weight_decay=1e-4,
    )

    assert len(first.fingerprint) == 64
    assert first.fingerprint != second.fingerprint


def test_torch_optimizer_factory_builds_declared_algorithm() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = build_torch_optimizer(
        [parameter],
        TrainingOptimizerSpec(
            algorithm="sgd",
            learning_rate=0.1,
            parameters={"momentum": 0.9},
        ),
    )

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)


def test_lbfgs_rejects_unsupported_weight_decay() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="does not support weight_decay"):
        build_torch_optimizer(
            [parameter],
            TrainingOptimizerSpec(
                algorithm="lbfgs",
                learning_rate=0.1,
                weight_decay=0.01,
            ),
        )


def test_optimizer_parameters_cannot_override_declared_learning_rate() -> None:
    with pytest.raises(ValueError, match="cannot override"):
        TrainingOptimizerSpec(
            algorithm="adam",
            learning_rate=0.1,
            parameters={"lr": 0.2},
        )
