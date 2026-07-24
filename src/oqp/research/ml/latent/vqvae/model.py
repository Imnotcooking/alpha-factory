"""OOP training and fitted-inference lifecycle for the canonical VQ-VAE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from .config import VQVAEConfig
from .network import (
    MLPVQVAE,
    VectorQuantizer,
    require_torch,
    torch,
    torch_functional,
)


VQVAE_CORE_VERSION = "oqp_vqvae_core_v1"

# These module-qualified names are part of the v1 parameter-hash wire format.
# Keep them stable even when the Python implementation moves between packages;
# otherwise an organizational refactor would invalidate every existing bundle.
_V1_ARCHITECTURE_TYPE_IDS = {
    MLPVQVAE: "oqp.research.latent.vqvae.network.MLPVQVAE",
    VectorQuantizer: "oqp.research.latent.vqvae.network.VectorQuantizer",
}


class VQVAEError(ValueError):
    """Raised when VQ-VAE inputs, parameters, or artifacts violate the contract."""


@dataclass(frozen=True, slots=True)
class VQTrainingEpoch:
    epoch: int
    loss: float
    reconstruction_loss: float
    codebook_loss: float
    commitment_loss: float
    active_codes: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or type(self.active_codes) is not int:
            raise TypeError("epoch and active_codes must be integers")
        if self.epoch < 1 or self.active_codes < 1:
            raise ValueError("invalid VQ training epoch index or active-code count")
        for name in (
            "loss",
            "reconstruction_loss",
            "codebook_loss",
            "commitment_loss",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class VQEncoding:
    model_id: str
    codes: np.ndarray
    encoder_latents: np.ndarray
    quantized_latents: np.ndarray
    quantization_distances: np.ndarray
    reconstructions: np.ndarray
    reconstruction_mse: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "codes",
            "encoder_latents",
            "quantized_latents",
            "quantization_distances",
            "reconstructions",
            "reconstruction_mse",
        ):
            value = _freeze_array(getattr(self, name))
            object.__setattr__(self, name, value)
        rows = len(self.codes)
        if type(self.model_id) is not str or not self.model_id:
            raise ValueError("model_id is required")
        if (
            self.codes.ndim != 1
            or not np.issubdtype(self.codes.dtype, np.integer)
            or np.any(self.codes < 0)
        ):
            raise ValueError(
                "codes must be a one-dimensional non-negative integer array"
            )
        for value in (
            self.encoder_latents,
            self.quantized_latents,
            self.reconstructions,
        ):
            if value.ndim != 2 or len(value) != rows:
                raise ValueError("encoded and reconstructed matrices must align by row")
            if not np.isfinite(value).all():
                raise ValueError("VQ encoding matrices must be finite")
        for value in (self.quantization_distances, self.reconstruction_mse):
            if value.ndim != 1 or len(value) != rows:
                raise ValueError("VQ distance outputs must align by row")
            if not np.isfinite(value).all() or np.any(value < 0.0):
                raise ValueError("VQ distance outputs must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FittedVQVAE:
    """Immutable tensors plus feature and lineage metadata.

    A live ``torch.nn.Module`` is deliberately not retained or exposed.  Each
    inference call materializes the canonical graph from immutable tensors,
    preventing an accidental module, hook, or compiled-call mutation from
    silently changing a registered fitted artifact.
    """

    config: VQVAEConfig
    feature_names: tuple[str, ...]
    _parameter_tensors: tuple[tuple[str, np.ndarray], ...] = field(repr=False)
    model_id: str
    parameter_hash: str
    training_rows_hash: str
    history: tuple[VQTrainingEpoch, ...]
    version: str = VQVAE_CORE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config, VQVAEConfig):
            raise TypeError("config must be a VQVAEConfig")
        if self.version != VQVAE_CORE_VERSION:
            raise ValueError("unsupported fitted VQ-VAE version")
        if type(self.feature_names) is not tuple:
            raise TypeError("feature_names must be a tuple")
        _validate_feature_names(self.feature_names)
        tensors = _freeze_parameter_tensors(self._parameter_tensors)
        object.__setattr__(self, "_parameter_tensors", tensors)
        _require_sha256(self.parameter_hash, "parameter_hash")
        _require_sha256(self.training_rows_hash, "training_rows_hash")
        expected_id = f"vqvae:{self.config.codebook_size}:{self.parameter_hash[:12]}"
        if self.model_id != expected_id:
            raise ValueError("model_id must be bound to the parameter hash")
        if type(self.history) is not tuple or any(
            not isinstance(item, VQTrainingEpoch) for item in self.history
        ):
            raise TypeError("history must contain VQTrainingEpoch values")
        if len(self.history) != self.config.epochs or tuple(
            item.epoch for item in self.history
        ) != tuple(range(1, self.config.epochs + 1)):
            raise ValueError("training history must contain every configured epoch")
        if any(item.active_codes > self.config.codebook_size for item in self.history):
            raise ValueError("training history exceeds the configured codebook")
        # Strict state loading authenticates tensor names and geometry against
        # the one canonical graph implied by the configuration and schema.
        _materialize_vqvae_network(
            config=self.config,
            input_dimension=len(self.feature_names),
            arrays=self.parameter_arrays(),
        )
        self.require_parameter_integrity()

    @property
    def input_dimension(self) -> int:
        return len(self.feature_names)

    def require_parameter_integrity(self) -> None:
        observed = hash_vqvae_parameters(
            config=self.config,
            feature_names=self.feature_names,
            arrays=self.parameter_arrays(),
            architecture=canonical_vqvae_architecture(
                config=self.config,
                input_dimension=self.input_dimension,
            ),
        )
        if observed != self.parameter_hash:
            raise VQVAEError("fitted VQ-VAE parameters drifted")

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        """Return a detached mapping whose array values remain read-only."""

        return {name: _freeze_array(value) for name, value in self._parameter_tensors}

    def encode(
        self,
        values: np.ndarray,
        *,
        feature_names: Sequence[str],
        batch_size: int | None = None,
    ) -> VQEncoding:
        require_torch()
        matrix = validate_vq_matrix(values, expected_width=self.input_dimension)
        if tuple(feature_names) != self.feature_names:
            raise VQVAEError("inference feature order differs from the fitted schema")
        self.require_parameter_integrity()
        size = self.config.batch_size if batch_size is None else batch_size
        if type(size) is not int or size < 1:
            raise VQVAEError("batch_size must be positive")
        network = _materialize_vqvae_network(
            config=self.config,
            input_dimension=self.input_dimension,
            arrays=self.parameter_arrays(),
        )

        codes = np.empty(len(matrix), dtype=np.int64)
        encoded_rows = np.empty(
            (len(matrix), self.config.latent_dimension), dtype=np.float32
        )
        quantized_rows = np.empty_like(encoded_rows)
        distances = np.empty(len(matrix), dtype=np.float32)
        reconstructions = np.empty_like(matrix, dtype=np.float32)
        reconstruction_mse = np.empty(len(matrix), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(matrix), size):
                stop = min(start + size, len(matrix))
                batch = torch.tensor(matrix[start:stop], dtype=torch.float32)
                (
                    reconstructed,
                    encoded,
                    quantized,
                    code_ids,
                    squared,
                    _,
                    _,
                ) = network(batch)
                minimum_squared = torch.gather(squared, 1, code_ids[:, None]).squeeze(1)
                row_mse = torch.mean((reconstructed - batch) ** 2, dim=1)
                codes[start:stop] = code_ids.cpu().numpy().astype(np.int64)
                encoded_rows[start:stop] = encoded.cpu().numpy().astype(np.float32)
                quantized_rows[start:stop] = quantized.cpu().numpy().astype(np.float32)
                distances[start:stop] = (
                    torch.sqrt(torch.clamp_min(minimum_squared, 0.0))
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                reconstructions[start:stop] = (
                    reconstructed.cpu().numpy().astype(np.float32)
                )
                reconstruction_mse[start:stop] = (
                    row_mse.cpu().numpy().astype(np.float32)
                )
        return VQEncoding(
            model_id=self.model_id,
            codes=codes,
            encoder_latents=encoded_rows,
            quantized_latents=quantized_rows,
            quantization_distances=distances,
            reconstructions=reconstructions,
            reconstruction_mse=reconstruction_mse,
        )

    def assign_codes(
        self,
        values: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        return self.encode(values, feature_names=feature_names).codes

    def reconstruct(
        self,
        values: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        return self.encode(values, feature_names=feature_names).reconstructions

    def codebook(self) -> np.ndarray:
        self.require_parameter_integrity()
        arrays = self.parameter_arrays()
        try:
            weights = arrays["quantizer.codebook.weight"]
        except KeyError as exc:  # guarded by strict materialization at construction
            raise VQVAEError("fitted VQ-VAE codebook tensor is missing") from exc
        return _freeze_array(weights)


class VQVAETrainer:
    """Offline trainer; fitted inference is owned by :class:`FittedVQVAE`."""

    def __init__(self, config: VQVAEConfig | None = None) -> None:
        self.config = config or VQVAEConfig()

    def fit(
        self,
        values: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> FittedVQVAE:
        require_torch()
        names = tuple(feature_names)
        _validate_feature_names(names)
        matrix = validate_vq_matrix(values, expected_width=len(names))
        config = self.config
        return _fit_prepared_matrix(matrix=matrix, names=names, config=config)


def _fit_prepared_matrix(
    *,
    matrix: np.ndarray,
    names: tuple[str, ...],
    config: VQVAEConfig,
) -> FittedVQVAE:
    # Constructing on the meta device avoids the random initializers invoked by
    # ``nn.Linear`` and ``nn.Embedding``.  Every parameter is then initialized
    # from a private generator, so concurrent callers never borrow or restore
    # process-global Torch RNG state.
    with torch.device("meta"):
        network = MLPVQVAE(
            input_dimension=matrix.shape[1],
            hidden_dimensions=config.hidden_dimensions,
            latent_dimension=config.latent_dimension,
            codebook_size=config.codebook_size,
        )
    network = network.to_empty(device="cpu").to(dtype=torch.float32)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(config.random_seed)
    _initialize_vqvae_parameters(network, generator=torch_generator)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
    )
    tensor = torch.tensor(matrix, dtype=torch.float32, device="cpu")
    generator = np.random.default_rng(config.random_seed)
    history: list[VQTrainingEpoch] = []
    network.train()
    for epoch in range(1, config.epochs + 1):
        order = (
            generator.permutation(len(tensor))
            if config.shuffle
            else np.arange(len(tensor), dtype=np.int64)
        )
        totals = np.zeros(4, dtype=np.float64)
        usage = np.zeros(config.codebook_size, dtype=np.int64)
        observed = 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = tensor[torch.as_tensor(indices, dtype=torch.long)]
            optimizer.zero_grad(set_to_none=True)
            (
                reconstruction,
                _,
                _,
                codes,
                squared_distances,
                codebook_loss,
                commitment_loss,
            ) = network(batch)
            reconstruction_loss = torch_functional.mse_loss(
                reconstruction,
                batch,
                reduction="mean",
            )
            orthogonality_loss = network.quantizer.orthogonality_loss()
            soft_assignments = torch.softmax(-squared_distances, dim=1)
            mean_probabilities = soft_assignments.mean(dim=0).clamp_min(1.0e-8)
            usage_loss = torch.sum(
                mean_probabilities
                * torch.log(mean_probabilities * config.codebook_size)
            )
            loss = (
                reconstruction_loss
                + codebook_loss
                + config.commitment_weight * commitment_loss
                + config.orthogonality_weight * orthogonality_loss
                + config.usage_weight * usage_loss
            )
            if not bool(torch.isfinite(loss)):
                raise VQVAEError("VQ-VAE produced a non-finite loss")
            loss.backward()
            optimizer.step()
            rows = len(batch)
            observed += rows
            totals += rows * np.asarray(
                [
                    float(loss.detach().cpu()),
                    float(reconstruction_loss.detach().cpu()),
                    float(codebook_loss.detach().cpu()),
                    float(commitment_loss.detach().cpu()),
                ]
            )
            usage += np.bincount(
                codes.detach().cpu().numpy(),
                minlength=config.codebook_size,
            )
        if observed != len(matrix):
            raise VQVAEError("VQ-VAE training skipped observations")
        averages = totals / observed
        history.append(
            VQTrainingEpoch(
                epoch=epoch,
                loss=float(averages[0]),
                reconstruction_loss=float(averages[1]),
                codebook_loss=float(averages[2]),
                commitment_loss=float(averages[3]),
                active_codes=int(np.count_nonzero(usage)),
            )
        )
    network = network.cpu().eval().requires_grad_(False)
    arrays = network_parameter_arrays(network)
    parameter_hash = hash_vqvae_parameters(
        config=config,
        feature_names=names,
        arrays=arrays,
        architecture=network_architecture_state(network),
    )
    training_hash = hash_vq_training_rows(matrix, names)
    model_id = f"vqvae:{config.codebook_size}:{parameter_hash[:12]}"
    return FittedVQVAE(
        config=config,
        feature_names=names,
        _parameter_tensors=_parameter_tensor_items(arrays),
        model_id=model_id,
        parameter_hash=parameter_hash,
        training_rows_hash=training_hash,
        history=tuple(history),
    )


def _initialize_vqvae_parameters(network: Any, *, generator: Any) -> None:
    """Reproduce the network defaults using only a caller-owned RNG."""

    initialized: set[int] = set()
    for module in network.modules():
        if type(module) is torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(
                module.weight,
                a=sqrt(5.0),
                generator=generator,
            )
            initialized.add(id(module.weight))
            if module.bias is not None:
                bound = 1.0 / sqrt(float(module.in_features))
                torch.nn.init.uniform_(
                    module.bias,
                    -bound,
                    bound,
                    generator=generator,
                )
                initialized.add(id(module.bias))
        elif type(module) is VectorQuantizer:
            bound = 1.0 / float(module.codebook_size)
            torch.nn.init.uniform_(
                module.codebook.weight,
                -bound,
                bound,
                generator=generator,
            )
            initialized.add(id(module.codebook.weight))
    expected = {id(parameter) for parameter in network.parameters()}
    if initialized != expected:
        raise VQVAEError("VQ-VAE contains parameters without a private initializer")


def _require_inference_ready_network(network: Any) -> None:
    if getattr(network, "training", True):
        raise VQVAEError("fitted VQ-VAE network must remain in evaluation mode")
    try:
        parameters = tuple(network.parameters())
    except (AttributeError, TypeError) as exc:
        raise VQVAEError("fitted VQ-VAE network is invalid") from exc
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise VQVAEError("fitted VQ-VAE inference currently requires a CPU network")
    if any(parameter.requires_grad for parameter in parameters):
        raise VQVAEError("fitted VQ-VAE parameters must have gradients disabled")


def _materialize_vqvae_network(
    *,
    config: VQVAEConfig,
    input_dimension: int,
    arrays: Mapping[str, np.ndarray],
) -> Any:
    """Build one canonical, private CPU graph from authenticated tensors."""

    require_torch()
    with torch.device("meta"):
        network = MLPVQVAE(
            input_dimension=input_dimension,
            hidden_dimensions=config.hidden_dimensions,
            latent_dimension=config.latent_dimension,
            codebook_size=config.codebook_size,
        )
    network = network.to_empty(device="cpu").to(dtype=torch.float32)
    try:
        network.load_state_dict(
            {
                name: torch.from_numpy(np.array(value, copy=True))
                for name, value in arrays.items()
            },
            strict=True,
        )
    except RuntimeError as exc:
        raise VQVAEError("VQ-VAE parameter structure is incompatible") from exc
    network = network.eval().requires_grad_(False)
    _require_inference_ready_network(network)
    network_architecture_state(network)
    return network


def canonical_vqvae_architecture(
    *,
    config: VQVAEConfig,
    input_dimension: int,
) -> list[dict[str, Any]]:
    """Describe the sole executable graph permitted by a fitted artifact."""

    require_torch()
    with torch.device("meta"):
        network = MLPVQVAE(
            input_dimension=input_dimension,
            hidden_dimensions=config.hidden_dimensions,
            latent_dimension=config.latent_dimension,
            codebook_size=config.codebook_size,
        )
    return network_architecture_state(network)


def validate_vq_matrix(values: np.ndarray, *, expected_width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != expected_width:
        raise VQVAEError(
            f"expected a two-dimensional matrix with width {expected_width}"
        )
    if len(matrix) < 1:
        raise VQVAEError("VQ-VAE matrix cannot be empty")
    if not np.isfinite(matrix).all():
        raise VQVAEError(
            "VQ-VAE core forbids missing or infinite values; use an upstream adapter"
        )
    return np.ascontiguousarray(matrix)


def network_parameter_arrays(network: Any) -> dict[str, np.ndarray]:
    require_torch()
    return {
        str(name): tensor.detach().cpu().numpy().copy()
        for name, tensor in sorted(network.state_dict().items())
    }


def network_architecture_state(network: Any) -> list[dict[str, Any]]:
    """Describe every executable module, including parameterless layers."""

    require_torch()
    supported_types = {
        MLPVQVAE,
        VectorQuantizer,
        torch.nn.Sequential,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Embedding,
    }
    records: list[dict[str, Any]] = []
    for name, module in network.named_modules():
        module_type = type(module)
        if module_type not in supported_types:
            raise VQVAEError(
                f"unsupported or mutated VQ-VAE module at {name or '<root>'}"
            )
        bound_forward = getattr(module, "forward", None)
        if getattr(bound_forward, "__func__", None) is not module_type.forward:
            raise VQVAEError(f"mutated VQ-VAE forward method at {name or '<root>'}")
        hook_attributes = (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
        if any(getattr(module, attribute, None) for attribute in hook_attributes):
            raise VQVAEError(f"runtime hooks are forbidden at {name or '<root>'}")
        record: dict[str, Any] = {
            "name": name,
            "type": _V1_ARCHITECTURE_TYPE_IDS.get(
                module_type,
                f"{module_type.__module__}.{module_type.__qualname__}",
            ),
        }
        if module_type is torch.nn.Linear:
            record.update(
                {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                }
            )
        elif module_type is torch.nn.ReLU:
            record["inplace"] = module.inplace
        elif module_type is torch.nn.Embedding:
            record.update(
                {
                    "num_embeddings": module.num_embeddings,
                    "embedding_dim": module.embedding_dim,
                    "padding_idx": module.padding_idx,
                    "max_norm": module.max_norm,
                    "norm_type": module.norm_type,
                    "scale_grad_by_freq": module.scale_grad_by_freq,
                    "sparse": module.sparse,
                }
            )
        elif module_type is VectorQuantizer:
            record.update(
                {
                    "codebook_size": module.codebook_size,
                    "latent_dimension": module.latent_dimension,
                }
            )
        records.append(record)
    return records


def hash_vqvae_parameters(
    *,
    config: VQVAEConfig,
    feature_names: Sequence[str],
    arrays: dict[str, np.ndarray],
    architecture: Sequence[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    header = {
        "version": VQVAE_CORE_VERSION,
        "config": config.state_dict(),
        "feature_names": list(feature_names),
        "architecture": list(architecture),
    }
    digest.update(_canonical_json(header))
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_vq_training_rows(
    matrix: np.ndarray,
    feature_names: Sequence[str],
) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(_canonical_json(list(feature_names)))
    digest.update(_canonical_json(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _freeze_array(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value))
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return frozen.reshape(array.shape)


def _parameter_tensor_items(
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple((name, value) for name, value in sorted(arrays.items()))


def _freeze_parameter_tensors(
    items: tuple[tuple[str, np.ndarray], ...],
) -> tuple[tuple[str, np.ndarray], ...]:
    if type(items) is not tuple or not items:
        raise TypeError("_parameter_tensors must be a non-empty tuple")
    parsed: list[tuple[str, np.ndarray]] = []
    names: set[str] = set()
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("parameter tensor entries must be (name, array) tuples")
        name, raw_value = item
        if type(name) is not str or not name or name in names:
            raise ValueError("parameter tensor names must be unique strings")
        value = np.asarray(raw_value)
        if value.dtype != np.dtype(np.float32):
            raise VQVAEError("VQ-VAE parameter tensors must use float32")
        if not np.isfinite(value).all():
            raise VQVAEError("VQ-VAE parameter tensors must be finite")
        names.add(name)
        parsed.append((name, _freeze_array(value)))
    return tuple(sorted(parsed, key=lambda pair: pair[0]))


def _validate_feature_names(names: Sequence[str]) -> None:
    if not names or any(not isinstance(value, str) or not value for value in names):
        raise ValueError("feature_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and ordered")


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = [
    "FittedVQVAE",
    "VQEncoding",
    "VQTrainingEpoch",
    "VQVAE_CORE_VERSION",
    "VQVAEError",
    "VQVAETrainer",
    "canonical_vqvae_architecture",
    "hash_vq_training_rows",
    "hash_vqvae_parameters",
    "network_architecture_state",
    "network_parameter_arrays",
    "validate_vq_matrix",
]
