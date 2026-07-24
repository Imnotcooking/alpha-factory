from __future__ import annotations

import json

import numpy as np
import pytest

from oqp.research.ml.latent.vqvae import (
    VQVAEConfig,
    VQVAEError,
    VQVAETrainer,
    load_vqvae_bundle,
    save_vqvae_bundle,
    torch_available,
)
from oqp.research.ml.latent.vqvae.model import (
    canonical_vqvae_architecture,
    hash_vqvae_parameters,
)
from oqp.research.ml.latent.vqvae.network import MLPVQVAE, VectorQuantizer, torch


requires_torch = pytest.mark.skipif(not torch_available(), reason="PyTorch unavailable")


def _matrix() -> np.ndarray:
    rng = np.random.default_rng(17)
    left = rng.normal(loc=-1.0, scale=0.15, size=(24, 3))
    right = rng.normal(loc=1.0, scale=0.15, size=(24, 3))
    return np.vstack((left, right)).astype(np.float32)


def _config() -> VQVAEConfig:
    return VQVAEConfig(
        hidden_dimensions=(8,),
        latent_dimension=3,
        codebook_size=4,
        epochs=3,
        batch_size=16,
        learning_rate=2.0e-3,
        random_seed=9,
        deterministic=True,
        shuffle=False,
    )


def _load_trusted(bundle, fitted):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return load_vqvae_bundle(
        bundle,
        expected_model_id=fitted.model_id,
        expected_parameter_hash=fitted.parameter_hash,
        expected_bundle_hash=manifest["bundle_hash"],
    )


def test_public_types_have_canonical_module_identities() -> None:
    assert VQVAEConfig.__module__ == "oqp.research.ml.latent.vqvae.config"
    assert VQVAETrainer.__module__ == "oqp.research.ml.latent.vqvae.model"
    assert MLPVQVAE.__module__ == "oqp.research.ml.latent.vqvae.network"
    assert VectorQuantizer.__module__ == "oqp.research.ml.latent.vqvae.network"


@requires_torch
def test_v1_architecture_type_ids_and_parameter_hash_survive_package_move() -> None:
    architecture = canonical_vqvae_architecture(
        config=_config(),
        input_dimension=3,
    )
    custom_types = {
        record["type"]
        for record in architecture
        if record["name"] in {"", "quantizer"}
    }
    # These historical strings are frozen v1 wire-format labels. They remain
    # stable for existing bundle hashes but are no longer importable namespaces.
    assert custom_types == {
        "oqp.research.latent.vqvae.network.MLPVQVAE",
        "oqp.research.latent.vqvae.network.VectorQuantizer",
    }

    fixture_arrays = {
        "fixture.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
        "fixture.bias": np.array([0.25, -0.5], dtype=np.float32),
    }
    assert (
        hash_vqvae_parameters(
            config=_config(),
            feature_names=("gk", "amihud", "ker"),
            arrays=fixture_arrays,
            architecture=architecture,
        )
        == "c53391335475d21ec0af38f7761b92b31b2b316e9c9cc79c604d58eeda5c4cf8"
    )


@requires_torch
def test_training_and_encoding_are_deterministic() -> None:
    values = _matrix()
    first = VQVAETrainer(_config()).fit(
        values,
        feature_names=("gk", "amihud", "ker"),
    )
    second = VQVAETrainer(_config()).fit(
        values,
        feature_names=("gk", "amihud", "ker"),
    )

    assert first.parameter_hash == second.parameter_hash
    schema = ("gk", "amihud", "ker")
    np.testing.assert_array_equal(
        first.assign_codes(values, feature_names=schema),
        second.assign_codes(values, feature_names=schema),
    )
    np.testing.assert_allclose(first.codebook(), second.codebook(), rtol=0.0, atol=0.0)
    encoded = first.encode(values, feature_names=schema)
    assert encoded.encoder_latents.shape == (48, 3)
    assert encoded.reconstructions.shape == values.shape
    assert np.isfinite(encoded.reconstruction_mse).all()
    assert not hasattr(first, "network")


@requires_torch
def test_core_rejects_schema_drift_and_missing_values() -> None:
    values = _matrix()
    fitted = VQVAETrainer(_config()).fit(values, feature_names=("a", "b", "c"))

    with pytest.raises(VQVAEError, match="feature order"):
        fitted.encode(values, feature_names=("b", "a", "c"))
    broken = values.copy()
    broken[0, 0] = np.nan
    with pytest.raises(VQVAEError, match="forbids missing"):
        fitted.encode(broken, feature_names=("a", "b", "c"))

    with pytest.raises(TypeError):
        fitted.encode(values)  # type: ignore[call-arg]


@requires_torch
def test_non_pickle_bundle_roundtrip_and_tamper_rejection(tmp_path) -> None:
    values = _matrix()
    fitted = VQVAETrainer(_config()).fit(values, feature_names=("a", "b", "c"))
    bundle = save_vqvae_bundle(fitted, tmp_path / "vq")
    restored = _load_trusted(bundle, fitted)

    assert restored.model_id == fitted.model_id
    assert restored.parameter_hash == fitted.parameter_hash
    schema = ("a", "b", "c")
    np.testing.assert_array_equal(
        restored.assign_codes(values, feature_names=schema),
        fitted.assign_codes(values, feature_names=schema),
    )
    np.testing.assert_allclose(
        restored.reconstruct(values, feature_names=schema),
        fitted.reconstruct(values, feature_names=schema),
        rtol=0.0,
        atol=0.0,
    )

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parameter_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VQVAEError, match="bundle hash mismatch"):
        load_vqvae_bundle(
            bundle,
            expected_model_id=fitted.model_id,
            expected_parameter_hash=fitted.parameter_hash,
            expected_bundle_hash=manifest["bundle_hash"],
        )


def test_configuration_deserialization_rejects_type_coercion() -> None:
    state = _config().state_dict()
    state["epochs"] = 3.5
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        VQVAEConfig.from_state_dict(state)

    state = _config().state_dict()
    state["learning_rate"] = "0.002"
    with pytest.raises(TypeError, match="learning_rate must be a float"):
        VQVAEConfig.from_state_dict(state)

    with pytest.raises(ValueError, match="hidden_dimensions"):
        VQVAEConfig(hidden_dimensions=(1.5,))  # type: ignore[arg-type]


@requires_torch
def test_training_and_loading_preserve_process_rng_and_determinism(tmp_path) -> None:
    np.random.seed(1234)
    numpy_state = np.random.get_state()
    torch.manual_seed(4321)
    torch_state = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()

    fitted = VQVAETrainer(_config()).fit(_matrix(), feature_names=("a", "b", "c"))
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    observed_numpy_state = np.random.get_state()
    assert observed_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(observed_numpy_state[1], numpy_state[1])
    assert observed_numpy_state[2:] == numpy_state[2:]
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.get_num_threads() == threads

    bundle = save_vqvae_bundle(fitted, tmp_path / "rng-safe")
    torch_state = torch.random.get_rng_state().clone()
    _load_trusted(bundle, fitted)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@requires_torch
def test_encoding_outputs_are_read_only() -> None:
    values = _matrix()
    schema = ("a", "b", "c")
    fitted = VQVAETrainer(_config()).fit(values, feature_names=schema)
    encoded = fitted.encode(values, feature_names=schema)

    assert not encoded.codes.flags.writeable
    assert not encoded.reconstructions.flags.writeable
    with pytest.raises(ValueError):
        encoded.codes[0] = 1
    with pytest.raises(ValueError):
        encoded.codes.setflags(write=True)


@requires_torch
def test_fitted_artifact_exposes_no_mutable_execution_graph() -> None:
    values = _matrix()
    schema = ("a", "b", "c")
    fitted = VQVAETrainer(_config()).fit(
        values,
        feature_names=schema,
    )
    baseline = fitted.assign_codes(values, feature_names=schema)
    detached = fitted.parameter_arrays()
    name = next(name for name, value in detached.items() if value.ndim == 2)
    retained_shape = fitted.parameter_arrays()[name].shape

    with pytest.raises(ValueError):
        detached[name].setflags(write=True)
    detached[name].shape = (detached[name].size,)
    assert detached[name].shape != retained_shape
    detached[name] = np.zeros_like(detached[name])
    assert fitted.parameter_arrays()[name].shape == retained_shape
    assert not hasattr(fitted, "network")
    fitted.require_parameter_integrity()
    np.testing.assert_array_equal(
        fitted.assign_codes(values, feature_names=schema),
        baseline,
    )


@requires_torch
def test_bundle_hash_binds_lineage_and_history(tmp_path) -> None:
    fitted = VQVAETrainer(_config()).fit(_matrix(), feature_names=("a", "b", "c"))
    bundle = save_vqvae_bundle(fitted, tmp_path / "bound")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_rows_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VQVAEError, match="bundle hash mismatch"):
        load_vqvae_bundle(
            bundle,
            expected_model_id=fitted.model_id,
            expected_parameter_hash=fitted.parameter_hash,
            expected_bundle_hash=manifest["bundle_hash"],
        )


@requires_torch
def test_bundle_writes_are_immutable_and_never_overwrite(tmp_path) -> None:
    fitted = VQVAETrainer(_config()).fit(_matrix(), feature_names=("a", "b", "c"))
    bundle = save_vqvae_bundle(fitted, tmp_path / "bundle")
    original_manifest = (bundle / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        save_vqvae_bundle(fitted, bundle)
    assert (bundle / "manifest.json").read_bytes() == original_manifest

    with pytest.raises(VQVAEError, match="immutable"):
        save_vqvae_bundle(fitted, bundle, overwrite=True)
    restored = _load_trusted(bundle, fitted)
    assert restored.parameter_hash == fitted.parameter_hash
