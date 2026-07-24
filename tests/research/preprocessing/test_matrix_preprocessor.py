from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oqp.contracts.regime_state import OrderedFeatureSchema
from oqp.research.ml.preprocessing import (
    FittedMatrixPreprocessor,
    MissingValuePolicy,
    PreprocessingError,
    PreprocessingSpec,
    dump_preprocessor_json,
    fit_matrix_preprocessor,
    hash_numeric_matrix,
    load_preprocessor_json,
)


def _schema(
    schema_id: str = "daily-regime-v1",
    names: tuple[str, ...] = ("gk_gap", "amihud"),
) -> OrderedFeatureSchema:
    return OrderedFeatureSchema(schema_id=schema_id, feature_names=names)


def _training_values() -> np.ndarray:
    return np.array(
        [
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [100.0, 40.0],
        ]
    )


def _fitted() -> FittedMatrixPreprocessor:
    return fit_matrix_preprocessor(
        _training_values(),
        feature_schema=_schema(),
        spec=PreprocessingSpec(
            missing_value_policy=MissingValuePolicy.MEDIAN,
            winsor_quantiles=(0.25, 0.75),
        ),
        artifact_id="m2-validation-2024",
        training_run_id="paper-01-validation-2024",
    )


def test_fit_is_deterministic_and_uses_declared_operation_order() -> None:
    first = _fitted()
    second = _fitted()

    assert first.state_dict() == second.state_dict()
    assert first.lineage_sha256 == second.lineage_sha256
    np.testing.assert_allclose(first.imputation_values, [2.5, 30.0])
    np.testing.assert_allclose(first.lower_bounds, [1.75, 25.0])
    np.testing.assert_allclose(first.upper_bounds, [27.25, 32.5])

    imputed_then_clipped = np.array(
        [[1.75, 25.0], [2.0, 30.0], [3.0, 30.0], [27.25, 32.5]]
    )
    np.testing.assert_allclose(first.centers, imputed_then_clipped.mean(axis=0))
    np.testing.assert_allclose(first.scales, imputed_then_clipped.std(axis=0, ddof=0))

    transformed = first.transform(_training_values(), feature_schema=_schema())
    expected = (imputed_then_clipped - first.centers) / first.scales
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(transformed.std(axis=0), [1.0, 1.0])


def test_fit_and_transform_do_not_mutate_caller_matrices() -> None:
    training = _training_values()
    original_training = training.copy()
    fitted = fit_matrix_preprocessor(
        training,
        feature_schema=_schema(),
        spec=PreprocessingSpec(missing_value_policy=MissingValuePolicy.MEAN),
        artifact_id="immutability-fixture",
    )
    np.testing.assert_equal(training, original_training)

    inference = np.array([[5.0, np.nan]])
    original_inference = inference.copy()
    fitted.transform(inference, feature_schema=_schema())
    np.testing.assert_equal(inference, original_inference)


def test_fitted_statistics_and_outputs_are_deeply_immutable() -> None:
    fitted = _fitted()
    transformed = fitted.transform([[2.0, 20.0]], feature_schema=_schema())

    for array in (
        fitted.imputation_values,
        fitted.lower_bounds,
        fitted.upper_bounds,
        fitted.centers,
        fitted.scales,
        fitted.low_variance_mask,
        transformed,
    ):
        assert array is not None
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_transform_rejects_reordered_or_different_feature_schema() -> None:
    fitted = _fitted()
    reordered = _schema("daily-regime-reordered", ("amihud", "gk_gap"))
    with pytest.raises(PreprocessingError, match="feature schema"):
        fitted.transform([[10.0, 1.0]], feature_schema=reordered)
    with pytest.raises(PreprocessingError, match="width"):
        fitted.transform([[1.0, 2.0, 3.0]], feature_schema=_schema())


def test_reject_policy_fails_on_training_and_inference_nan() -> None:
    with pytest.raises(PreprocessingError, match="training values contain NaN"):
        fit_matrix_preprocessor(
            _training_values(),
            feature_schema=_schema(),
            spec=PreprocessingSpec(),
            artifact_id="reject-training-nan",
        )

    fitted = fit_matrix_preprocessor(
        np.array([[1.0, 10.0], [2.0, 20.0]]),
        feature_schema=_schema(),
        spec=PreprocessingSpec(),
        artifact_id="reject-inference-nan",
    )
    with pytest.raises(PreprocessingError, match="reject missing-value policy"):
        fitted.transform([[np.nan, 11.0]], feature_schema=_schema())


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (MissingValuePolicy.MEAN, [2.0, 20.0]),
        (MissingValuePolicy.MEDIAN, [2.0, 20.0]),
        (MissingValuePolicy.CONSTANT, [-7.0, -7.0]),
    ],
)
def test_declared_imputation_policy_is_fitted_per_feature(
    policy: MissingValuePolicy,
    expected: list[float],
) -> None:
    values = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0]])
    fitted = fit_matrix_preprocessor(
        values,
        feature_schema=_schema(),
        spec=PreprocessingSpec(
            standardize=False,
            missing_value_policy=policy,
            constant_fill_value=-7.0,
        ),
        artifact_id=f"{policy.value}-fixture",
    )
    assert fitted.imputation_values is not None
    if policy is MissingValuePolicy.CONSTANT:
        np.testing.assert_allclose(fitted.imputation_values, expected)
    else:
        np.testing.assert_allclose(fitted.imputation_values, [2.0, 20.0])


def test_constant_policy_supports_an_all_missing_training_feature() -> None:
    fitted = fit_matrix_preprocessor(
        np.array([[np.nan, 1.0], [np.nan, 2.0]]),
        feature_schema=_schema(),
        spec=PreprocessingSpec(
            missing_value_policy=MissingValuePolicy.CONSTANT,
            constant_fill_value=4.0,
        ),
        artifact_id="all-missing-constant",
    )
    assert fitted.low_variance_mask.tolist() == [True, False]
    result = fitted.transform([[np.nan, 1.5]], feature_schema=_schema())
    assert result[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "policy",
    [MissingValuePolicy.MEAN, MissingValuePolicy.MEDIAN],
)
def test_statistical_imputation_rejects_all_missing_training_feature(
    policy: MissingValuePolicy,
) -> None:
    with pytest.raises(PreprocessingError, match="all-missing feature 'gk_gap'"):
        fit_matrix_preprocessor(
            np.array([[np.nan, 1.0], [np.nan, 2.0]]),
            feature_schema=_schema(),
            spec=PreprocessingSpec(missing_value_policy=policy),
            artifact_id="all-missing-invalid",
        )


def test_nonstandardizing_artifact_only_imputes_and_clips() -> None:
    fitted = fit_matrix_preprocessor(
        np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]]),
        feature_schema=_schema(),
        spec=PreprocessingSpec(
            standardize=False,
            winsor_quantiles=(0.25, 0.75),
        ),
        artifact_id="clip-only",
    )
    np.testing.assert_allclose(fitted.centers, [0.0, 0.0])
    np.testing.assert_allclose(fitted.scales, [1.0, 1.0])
    np.testing.assert_allclose(
        fitted.transform([[-100.0, 100.0]], feature_schema=_schema()),
        [[1.0, 25.0]],
    )


def test_low_variance_scale_is_safe_and_explicit() -> None:
    fitted = fit_matrix_preprocessor(
        np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]),
        feature_schema=_schema(),
        spec=PreprocessingSpec(scale_floor=1e-10),
        artifact_id="constant-feature",
    )
    assert fitted.low_variance_mask.tolist() == [True, False]
    assert fitted.scales[0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        fitted.transform([[5.0, 2.0]], feature_schema=_schema()),
        [[0.0, 0.0]],
    )


def test_infinity_and_non_numeric_matrices_are_rejected() -> None:
    with pytest.raises(PreprocessingError, match="infinity"):
        fit_matrix_preprocessor(
            [[1.0, np.inf]],
            feature_schema=_schema(),
            spec=PreprocessingSpec(),
            artifact_id="infinite",
        )
    with pytest.raises(TypeError, match="real numeric"):
        fit_matrix_preprocessor(
            [["1.0", "2.0"]],
            feature_schema=_schema(),
            spec=PreprocessingSpec(),
            artifact_id="text-values",
        )


def test_training_digest_normalizes_nan_payload_and_signed_zero() -> None:
    first = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float64)
    second = first.copy()
    second[0, 0] = -0.0
    bits = second.view(np.uint64)
    bits[0, 1] = np.uint64(0x7FF8000000000001)

    assert hash_numeric_matrix(first, feature_schema=_schema()) == hash_numeric_matrix(
        second, feature_schema=_schema()
    )
    reordered = _schema("reordered", ("amihud", "gk_gap"))
    assert hash_numeric_matrix(first, feature_schema=_schema()) != hash_numeric_matrix(
        first, feature_schema=reordered
    )


def test_artifact_digest_binds_schema_spec_training_and_lineage() -> None:
    base = fit_matrix_preprocessor(
        [[1.0, 2.0], [3.0, 4.0]],
        feature_schema=_schema(),
        spec=PreprocessingSpec(),
        artifact_id="base",
        training_run_id="run-a",
    )
    changed_run = fit_matrix_preprocessor(
        [[1.0, 2.0], [3.0, 4.0]],
        feature_schema=_schema(),
        spec=PreprocessingSpec(),
        artifact_id="base",
        training_run_id="run-b",
    )
    changed_data = fit_matrix_preprocessor(
        [[1.0, 2.0], [3.0, 4.1]],
        feature_schema=_schema(),
        spec=PreprocessingSpec(),
        artifact_id="base",
        training_run_id="run-a",
    )
    assert len(base.lineage_sha256) == 64
    assert base.lineage_sha256 != changed_run.lineage_sha256
    assert base.lineage_sha256 != changed_data.lineage_sha256


def test_json_round_trip_requires_externally_trusted_identity(
    tmp_path: Path,
) -> None:
    fitted = _fitted()
    path = dump_preprocessor_json(fitted, tmp_path / "preprocessor.json")
    raw = path.read_text(encoding="utf-8")
    assert "pickle" not in raw.lower()
    assert "NaN" not in raw

    restored = load_preprocessor_json(
        path,
        expected_artifact_id=fitted.artifact_id,
        expected_artifact_sha256=fitted.lineage_sha256,
    )
    assert restored.state_dict() == fitted.state_dict()
    np.testing.assert_allclose(
        restored.transform([[2.0, np.nan]], feature_schema=_schema()),
        fitted.transform([[2.0, np.nan]], feature_schema=_schema()),
    )
    with pytest.raises(PreprocessingError, match="artifact_id differs"):
        load_preprocessor_json(
            path,
            expected_artifact_id="wrong-artifact",
            expected_artifact_sha256=fitted.lineage_sha256,
        )
    with pytest.raises(PreprocessingError, match="differs from expected"):
        load_preprocessor_json(
            path,
            expected_artifact_id=fitted.artifact_id,
            expected_artifact_sha256="0" * 64,
        )


def test_tampering_and_unknown_json_fields_are_rejected(tmp_path: Path) -> None:
    fitted = _fitted()
    path = dump_preprocessor_json(fitted, tmp_path / "preprocessor.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["statistics"]["centers"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="does not authenticate"):
        load_preprocessor_json(
            path,
            expected_artifact_id=fitted.artifact_id,
            expected_artifact_sha256=fitted.lineage_sha256,
        )

    payload = fitted.state_dict()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="unknown or missing"):
        load_preprocessor_json(
            path,
            expected_artifact_id=fitted.artifact_id,
            expected_artifact_sha256=fitted.lineage_sha256,
        )


def test_json_write_is_atomic_no_clobber_by_default(tmp_path: Path) -> None:
    fitted = _fitted()
    path = dump_preprocessor_json(fitted, tmp_path / "preprocessor.json")
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        dump_preprocessor_json(fitted, path)
    assert path.read_bytes() == original

    dump_preprocessor_json(fitted, path, overwrite=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "spec",
    [
        PreprocessingSpec(missing_value_policy="median"),
        PreprocessingSpec(winsor_quantiles=(0.0, 1.0)),
        PreprocessingSpec(standardize=False),
    ],
)
def test_spec_state_round_trip_is_exact(spec: PreprocessingSpec) -> None:
    assert PreprocessingSpec.from_state_dict(spec.state_dict()) == spec


def test_artifact_from_state_rejects_wrong_statistics_types() -> None:
    fitted = _fitted()
    state = fitted.state_dict()
    state["statistics"]["low_variance_mask"][0] = 0
    with pytest.raises(PreprocessingError, match="invalid preprocessing artifact"):
        FittedMatrixPreprocessor.from_state_dict(
            state,
            expected_artifact_id=fitted.artifact_id,
            expected_artifact_sha256=fitted.lineage_sha256,
        )


def test_shared_preprocessing_source_has_no_heavy_or_executable_dependencies() -> None:
    package = Path(__file__).parents[3] / "src/oqp/research/preprocessing"
    forbidden = {"joblib", "pandas", "pickle", "sklearn", "torch"}
    imported_roots: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
    assert imported_roots.isdisjoint(forbidden)


def test_focused_package_import_does_not_initialize_heavy_libraries() -> None:
    environment = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import oqp.research.ml.preprocessing; "
                "assert not {'joblib', 'pandas', 'sklearn', 'torch'} "
                "& sys.modules.keys()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
