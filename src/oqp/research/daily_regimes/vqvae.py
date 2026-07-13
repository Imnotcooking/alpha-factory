"""Leakage-safe contracts for the windowed VQ-VAE triangulation model.

This module contains interfaces only.  It makes the paper's restrictions
structural: windows are chronological and product/sequence/fold safe, and only
columns explicitly classified as model features may enter the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from math import isfinite
from typing import Generic, Protocol, TypeVar, runtime_checkable


STAGE_OWNER = 9


class VQInputRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    HMM_PROBABILITY = "hmm_probability"
    FORWARD_RETURN = "forward_return"
    STRATEGY_LABEL = "strategy_label"
    IDENTIFIER = "identifier"


class WindowSamplingMode(str, Enum):
    CHRONOLOGICAL = "chronological"


class WindowBoundaryPolicy(str, Enum):
    SPLIT_SEQUENCE = "split_sequence"
    REJECT_CROSSING = "reject_crossing"


class VQSampleRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


class VQBatchPurpose(str, Enum):
    FIT = "fit"
    ENCODE = "encode"


class VQFailureCode(str, Enum):
    NON_CONVERGENCE = "non_convergence"
    NONFINITE_LOSS = "nonfinite_loss"
    CODEBOOK_COLLAPSE = "codebook_collapse"
    FUTURE_DEPENDENCE = "future_dependence"
    INVALID_INPUT_ROLE = "invalid_input_role"


@dataclass(frozen=True)
class VQInputColumn:
    name: str
    role: VQInputRole

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VQ input column name is required")
        if not isinstance(self.role, VQInputRole):
            raise TypeError("role must be a VQInputRole")


@dataclass(frozen=True)
class VQVAEConfig:
    input_columns: tuple[VQInputColumn, ...]
    window_length_days: int = 20
    codebook_size: int = 3
    latent_dimension: int = 8
    hidden_dimensions: tuple[int, ...] = (32, 16)
    commitment_weight: float = 0.25
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    random_seed: int = 42
    sampling_mode: WindowSamplingMode = WindowSamplingMode.CHRONOLOGICAL
    boundary_policy: WindowBoundaryPolicy = WindowBoundaryPolicy.REJECT_CROSSING
    fit_scope: str = "training_only"

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_mode, WindowSamplingMode):
            raise TypeError("sampling_mode must be a WindowSamplingMode")
        if not isinstance(self.boundary_policy, WindowBoundaryPolicy):
            raise TypeError("boundary_policy must be a WindowBoundaryPolicy")
        if not self.input_columns or len({column.name for column in self.input_columns}) != len(
            self.input_columns
        ):
            raise ValueError("input_columns must be non-empty and unique")
        forbidden = [column.name for column in self.input_columns if column.role is not VQInputRole.FEATURE]
        if forbidden:
            raise ValueError(f"VQ-VAE inputs must all have FEATURE role: {forbidden}")
        if self.window_length_days < 2:
            raise ValueError("window_length_days must be at least 2")
        if self.codebook_size < 2 or self.latent_dimension < 1:
            raise ValueError("codebook_size must be at least 2 and latent_dimension positive")
        if not self.hidden_dimensions or any(width < 1 for width in self.hidden_dimensions):
            raise ValueError("hidden_dimensions must contain positive widths")
        if not isfinite(self.commitment_weight) or self.commitment_weight <= 0.0:
            raise ValueError("commitment_weight must be finite and positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.sampling_mode is not WindowSamplingMode.CHRONOLOGICAL:
            raise ValueError("only chronological VQ-VAE sampling is permitted")
        if self.fit_scope != "training_only":
            raise ValueError("VQ-VAE fitting is restricted to training_only scope")


@dataclass(frozen=True)
class VQObservationRef:
    row_id: str
    product_id: str
    sequence_id: str
    fold_id: str
    trading_date: date
    sample_role: VQSampleRole

    def __post_init__(self) -> None:
        if not all((self.row_id, self.product_id, self.sequence_id, self.fold_id)):
            raise ValueError("VQ observation identifiers are required")
        if not isinstance(self.sample_role, VQSampleRole):
            raise TypeError("sample_role must be a VQSampleRole")


@dataclass(frozen=True)
class VQWindow:
    """One chronological window ending at its final observation."""

    observations: tuple[VQObservationRef, ...]
    values: tuple[tuple[float, ...], ...]
    input_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.observations or len(self.values) != len(self.observations):
            raise ValueError("a VQ window requires one value row per observation")
        if not self.input_columns or len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError("input_columns must be non-empty and unique")
        if len({item.row_id for item in self.observations}) != len(self.observations):
            raise ValueError("a VQ window cannot repeat row_ids")
        if len({item.product_id for item in self.observations}) != 1:
            raise ValueError("a VQ window cannot cross products")
        if len({item.sequence_id for item in self.observations}) != 1:
            raise ValueError("a VQ window cannot cross sequence boundaries")
        if len({item.fold_id for item in self.observations}) != 1:
            raise ValueError("a VQ window cannot cross fold boundaries")
        dates = tuple(item.trading_date for item in self.observations)
        if any(left >= right for left, right in zip(dates, dates[1:])):
            raise ValueError("VQ observations must be in strictly increasing chronological order")
        width = len(self.input_columns)
        for row in self.values:
            if len(row) != width:
                raise ValueError("VQ value width must match input_columns")
            if any(not isfinite(value) for value in row):
                raise ValueError("nonfinite values may not enter VQ-VAE windows")

    @property
    def output_observation(self) -> VQObservationRef:
        return self.observations[-1]

    @property
    def fold_id(self) -> str:
        """The single fold shared by every observation in this window."""

        return self.observations[0].fold_id


@dataclass(frozen=True)
class VQWindowBatch:
    feature_set_id: str
    fold_id: str
    windows: tuple[VQWindow, ...]
    config: VQVAEConfig
    purpose: VQBatchPurpose

    def __post_init__(self) -> None:
        if not self.feature_set_id or not self.fold_id or not self.windows:
            raise ValueError("feature_set_id, fold_id, and at least one VQ window are required")
        if not isinstance(self.purpose, VQBatchPurpose):
            raise TypeError("purpose must be a VQBatchPurpose")
        expected_columns = tuple(column.name for column in self.config.input_columns)
        output_rows: set[str] = set()
        for window in self.windows:
            if window.fold_id != self.fold_id:
                raise ValueError("every VQ window in a batch must match the declared fold_id")
            if len(window.observations) != self.config.window_length_days:
                raise ValueError("window length does not match VQVAEConfig")
            if window.input_columns != expected_columns:
                raise ValueError("window columns do not match the configured ordered inputs")
            output_row = window.output_observation.row_id
            if output_row in output_rows:
                raise ValueError("a batch cannot emit more than one code for an output row")
            output_rows.add(output_row)
            if self.purpose is VQBatchPurpose.FIT and any(
                observation.sample_role is not VQSampleRole.TRAIN
                for observation in window.observations
            ):
                raise ValueError("VQ-VAE fitting windows may contain training rows only")


@dataclass(frozen=True)
class VQTrainingEpoch:
    epoch: int
    reconstruction_loss: float
    codebook_loss: float
    commitment_loss: float

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("epoch is one-based and must be positive")
        values = (self.reconstruction_loss, self.codebook_loss, self.commitment_loss)
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("VQ training losses must be finite and non-negative")


@dataclass(frozen=True)
class VQFitSummary:
    model_id: str
    fold_id: str
    feature_set_id: str
    training_rows_hash: str
    parameter_hash: str | None
    converged: bool
    history: tuple[VQTrainingEpoch, ...]
    failure_codes: tuple[VQFailureCode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not all((self.model_id, self.fold_id, self.feature_set_id, self.training_rows_hash)):
            raise ValueError("VQ fit provenance is required")
        if self.converged and not self.parameter_hash:
            raise ValueError("a converged VQ model requires parameter_hash")
        if self.converged and self.failure_codes:
            raise ValueError("a converged fit cannot also carry failure codes")


ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class VQFitResult(Generic[ModelT]):
    config: VQVAEConfig
    model: ModelT
    summary: VQFitSummary


@dataclass(frozen=True)
class VQCodeRecord:
    observation: VQObservationRef
    model_id: str
    code_index: int
    quantization_distance: float

    def __post_init__(self) -> None:
        if not self.model_id or self.code_index < 0:
            raise ValueError("model_id is required and code_index must be non-negative")
        if not isfinite(self.quantization_distance) or self.quantization_distance < 0.0:
            raise ValueError("quantization_distance must be finite and non-negative")


@dataclass(frozen=True)
class VQCodeBatch:
    model_id: str
    fold_id: str
    codebook_size: int
    records: tuple[VQCodeRecord, ...]

    def __post_init__(self) -> None:
        if not self.model_id or not self.fold_id or self.codebook_size < 2 or not self.records:
            raise ValueError("valid model_id, fold_id, codebook_size, and records are required")
        if any(record.model_id != self.model_id for record in self.records):
            raise ValueError("record model_id does not match batch model_id")
        if any(record.observation.fold_id != self.fold_id for record in self.records):
            raise ValueError("every VQ code record must match the declared fold_id")
        if any(record.code_index >= self.codebook_size for record in self.records):
            raise ValueError("code_index exceeds codebook_size")
        row_ids = [record.observation.row_id for record in self.records]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("VQ code output row_ids must be unique")


@runtime_checkable
class VQWindowBuilder(Protocol):
    def build(
        self,
        feature_rows: object,
        config: VQVAEConfig,
        *,
        purpose: VQBatchPurpose,
    ) -> VQWindowBatch:
        ...


@runtime_checkable
class VQVAEEstimator(Protocol[ModelT]):
    def fit(self, windows: VQWindowBatch, config: VQVAEConfig) -> VQFitResult[ModelT]:
        ...

    def encode(self, fitted: VQFitResult[ModelT], windows: VQWindowBatch) -> VQCodeBatch:
        ...
