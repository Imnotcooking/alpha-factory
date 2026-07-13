"""Strict, typed configuration loading for the daily-regime study.

The preregistration file is a research contract rather than a loose bag of
parameters.  This module therefore rejects unknown keys instead of silently
ignoring misspellings, and exposes a canonical hash of the resolved contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


class DailyRegimeConfigError(ValueError):
    """Raised when the daily-regime research contract is invalid."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    id: str
    title: str
    status: str
    frequency: str
    scope_boundary: str

    @classmethod
    def from_mapping(cls, value: Any) -> "ProjectConfig":
        data = _strict_mapping(
            value,
            {"id", "title", "status", "frequency", "scope_boundary"},
            "project",
        )
        return cls(
            id=_required_text(data, "id", "project"),
            title=_required_text(data, "title", "project"),
            status=_required_text(data, "status", "project"),
            frequency=_required_text(data, "frequency", "project"),
            scope_boundary=_required_text(data, "scope_boundary", "project"),
        )


@dataclass(frozen=True, slots=True)
class ClaimsConfig:
    permitted_before_results: tuple[str, ...]
    prohibited_before_results: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "ClaimsConfig":
        data = _strict_mapping(
            value,
            {"permitted_before_results", "prohibited_before_results"},
            "claims",
        )
        return cls(
            permitted_before_results=_string_tuple(
                _required(data, "permitted_before_results", "claims"),
                "claims.permitted_before_results",
                allow_empty=True,
            ),
            prohibited_before_results=_string_tuple(
                _required(data, "prohibited_before_results", "claims"),
                "claims.prohibited_before_results",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class DataConstructionConfig:
    """Frozen point-in-time contract-selection and roll conventions."""

    selector: str
    decision_lag_periods: int
    primary_metric: str
    secondary_metric: str
    minimum_volume: int
    minimum_open_interest: int
    tie_breakers: tuple[str, ...]
    exclude_limit_locked: bool
    exclude_stale_bars: bool
    return_convention: str
    adjustment_convention: str
    continuous_index_base: float
    missing_policy: str

    @classmethod
    def from_mapping(cls, value: Any) -> "DataConstructionConfig":
        context = "data_construction"
        data = _strict_mapping(
            value,
            {
                "selector",
                "decision_lag_periods",
                "primary_metric",
                "secondary_metric",
                "minimum_volume",
                "minimum_open_interest",
                "tie_breakers",
                "exclude_limit_locked",
                "exclude_stale_bars",
                "return_convention",
                "adjustment_convention",
                "continuous_index_base",
                "missing_policy",
            },
            context,
        )
        exclude_limit_locked = _required(data, "exclude_limit_locked", context)
        exclude_stale_bars = _required(data, "exclude_stale_bars", context)
        if not isinstance(exclude_limit_locked, bool) or not isinstance(
            exclude_stale_bars, bool
        ):
            raise DailyRegimeConfigError(
                "data_construction exclusion flags must be booleans."
            )
        index_base = _finite_float(
            _required(data, "continuous_index_base", context),
            f"{context}.continuous_index_base",
        )
        if index_base <= 0.0:
            raise DailyRegimeConfigError(
                "data_construction.continuous_index_base must be positive."
            )
        config = cls(
            selector=_required_text(data, "selector", context),
            decision_lag_periods=_positive_int(
                _required(data, "decision_lag_periods", context),
                f"{context}.decision_lag_periods",
            ),
            primary_metric=_required_text(data, "primary_metric", context),
            secondary_metric=_required_text(data, "secondary_metric", context),
            minimum_volume=_positive_int(
                _required(data, "minimum_volume", context),
                f"{context}.minimum_volume",
            ),
            minimum_open_interest=_positive_int(
                _required(data, "minimum_open_interest", context),
                f"{context}.minimum_open_interest",
            ),
            tie_breakers=_string_tuple(
                _required(data, "tie_breakers", context),
                f"{context}.tie_breakers",
            ),
            exclude_limit_locked=exclude_limit_locked,
            exclude_stale_bars=exclude_stale_bars,
            return_convention=_required_text(data, "return_convention", context),
            adjustment_convention=_required_text(
                data, "adjustment_convention", context
            ),
            continuous_index_base=index_base,
            missing_policy=_required_text(data, "missing_policy", context),
        )
        expected = {
            "selector": "lagged_liquidity",
            "decision_lag_periods": 1,
            "primary_metric": "open_interest",
            "secondary_metric": "volume",
            "tie_breakers": ("earliest_last_trade_date", "contract"),
            "return_convention": "selected_contract_same_contract_close",
            "adjustment_convention": (
                "chained_same_contract_returns_no_history_rewrite"
            ),
            "missing_policy": "flag_no_backfill",
        }
        for name, expected_value in expected.items():
            if getattr(config, name) != expected_value:
                raise DailyRegimeConfigError(
                    f"data_construction.{name} must equal {expected_value!r}."
                )
        return config


@dataclass(frozen=True, slots=True)
class FeatureSetConfig:
    columns: tuple[str, ...] = field(default_factory=tuple)
    status: str | None = None
    source: str | None = None
    components: int | None = None
    fit_scope: str | None = None
    columns_from: tuple[str, ...] = field(default_factory=tuple)
    role: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, context: str) -> "FeatureSetConfig":
        data = _strict_mapping(
            value,
            {"columns", "status", "source", "components", "fit_scope", "columns_from", "role"},
            context,
        )
        config = cls(
            columns=_optional_string_tuple(data.get("columns"), f"{context}.columns"),
            status=_optional_text(data.get("status"), f"{context}.status"),
            source=_optional_text(data.get("source"), f"{context}.source"),
            components=_optional_positive_int(data.get("components"), f"{context}.components"),
            fit_scope=_optional_text(data.get("fit_scope"), f"{context}.fit_scope"),
            columns_from=_optional_string_tuple(
                data.get("columns_from"), f"{context}.columns_from"
            ),
            role=_optional_text(data.get("role"), f"{context}.role"),
        )
        if not (config.columns or config.columns_from or config.source):
            raise DailyRegimeConfigError(
                f"{context} must declare columns, columns_from, or source."
            )
        return config


@dataclass(frozen=True, slots=True)
class TargetsConfig:
    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    tail_threshold_fit_scope: str

    @classmethod
    def from_mapping(cls, value: Any) -> "TargetsConfig":
        data = _strict_mapping(
            value,
            {"primary", "secondary", "tail_threshold_fit_scope"},
            "targets",
        )
        return cls(
            primary=_string_tuple(
                _required(data, "primary", "targets"), "targets.primary"
            ),
            secondary=_string_tuple(
                _required(data, "secondary", "targets"),
                "targets.secondary",
                allow_empty=True,
            ),
            tail_threshold_fit_scope=_required_text(
                data, "tail_threshold_fit_scope", "targets"
            ),
        )


@dataclass(frozen=True, slots=True)
class HypothesisConfig:
    contrasts: tuple[str, ...] = field(default_factory=tuple)
    expected_sign_for_each: str | None = None
    contrast: str | None = None
    expected_sign: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, context: str) -> "HypothesisConfig":
        data = _strict_mapping(
            value,
            {"contrasts", "expected_sign_for_each", "contrast", "expected_sign"},
            context,
        )
        config = cls(
            contrasts=_optional_string_tuple(data.get("contrasts"), f"{context}.contrasts"),
            expected_sign_for_each=_optional_text(
                data.get("expected_sign_for_each"), f"{context}.expected_sign_for_each"
            ),
            contrast=_optional_text(data.get("contrast"), f"{context}.contrast"),
            expected_sign=_optional_text(
                data.get("expected_sign"), f"{context}.expected_sign"
            ),
        )
        if bool(config.contrasts) == bool(config.contrast):
            raise DailyRegimeConfigError(
                f"{context} must declare exactly one of contrast or contrasts."
            )
        if config.contrasts and config.expected_sign_for_each is None:
            raise DailyRegimeConfigError(
                f"{context}.expected_sign_for_each is required for multiple contrasts."
            )
        if config.contrast and config.expected_sign is None:
            raise DailyRegimeConfigError(
                f"{context}.expected_sign is required for a single contrast."
            )
        return config


@dataclass(frozen=True, slots=True)
class ModelSelectionRulesConfig:
    feature_representation_primary: str
    feature_representation_secondary: str
    matched_input_model_family: str
    tie_break: str

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelSelectionRulesConfig":
        context = "models.selection_rules"
        data = _strict_mapping(
            value,
            {
                "feature_representation_primary",
                "feature_representation_secondary",
                "matched_input_model_family",
                "tie_break",
            },
            context,
        )
        return cls(
            feature_representation_primary=_required_text(
                data, "feature_representation_primary", context
            ),
            feature_representation_secondary=_required_text(
                data, "feature_representation_secondary", context
            ),
            matched_input_model_family=_required_text(
                data, "matched_input_model_family", context
            ),
            tie_break=_required_text(data, "tie_break", context),
        )


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    state_counts: tuple[int, ...]
    mixture_counts: tuple[int, ...]
    covariance_type_primary: str
    restarts: int
    minimum_state_occupancy: float
    model_families: tuple[str, ...]
    selection_rules: ModelSelectionRulesConfig

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelsConfig":
        context = "models"
        data = _strict_mapping(
            value,
            {
                "state_counts",
                "mixture_counts",
                "covariance_type_primary",
                "restarts",
                "minimum_state_occupancy",
                "model_families",
                "selection_rules",
            },
            context,
        )
        occupancy = _finite_float(
            _required(data, "minimum_state_occupancy", context),
            f"{context}.minimum_state_occupancy",
        )
        if not 0.0 < occupancy < 1.0:
            raise DailyRegimeConfigError(
                "models.minimum_state_occupancy must lie strictly between zero and one."
            )
        return cls(
            state_counts=_positive_int_tuple(
                _required(data, "state_counts", context), f"{context}.state_counts"
            ),
            mixture_counts=_positive_int_tuple(
                _required(data, "mixture_counts", context), f"{context}.mixture_counts"
            ),
            covariance_type_primary=_required_text(
                data, "covariance_type_primary", context
            ),
            restarts=_positive_int(
                _required(data, "restarts", context), f"{context}.restarts"
            ),
            minimum_state_occupancy=occupancy,
            model_families=_string_tuple(
                _required(data, "model_families", context), f"{context}.model_families"
            ),
            selection_rules=ModelSelectionRulesConfig.from_mapping(
                _required(data, "selection_rules", context)
            ),
        )


@dataclass(frozen=True, slots=True)
class VQVAEStudyConfig:
    role: str
    input_set_primary: str
    window_length_days: int
    codebook_size_rule: str
    forbidden_inputs: tuple[str, ...]
    simple_benchmark: str
    seeds: tuple[int, ...]
    required_diagnostics: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "VQVAEStudyConfig":
        context = "vqvae"
        data = _strict_mapping(
            value,
            {
                "role",
                "input_set_primary",
                "window_length_days",
                "codebook_size_rule",
                "forbidden_inputs",
                "simple_benchmark",
                "seeds",
                "required_diagnostics",
            },
            context,
        )
        return cls(
            role=_required_text(data, "role", context),
            input_set_primary=_required_text(data, "input_set_primary", context),
            window_length_days=_positive_int(
                _required(data, "window_length_days", context),
                f"{context}.window_length_days",
            ),
            codebook_size_rule=_required_text(data, "codebook_size_rule", context),
            forbidden_inputs=_string_tuple(
                _required(data, "forbidden_inputs", context),
                f"{context}.forbidden_inputs",
                allow_empty=True,
            ),
            simple_benchmark=_required_text(data, "simple_benchmark", context),
            seeds=_nonnegative_int_tuple(
                _required(data, "seeds", context), f"{context}.seeds"
            ),
            required_diagnostics=_string_tuple(
                _required(data, "required_diagnostics", context),
                f"{context}.required_diagnostics",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    split_mode: str
    dates: Any
    preprocessing_scope: str
    purge_horizon: str
    decision_delay_periods: int
    filtered_probabilities_only: bool
    block_bootstrap_confidence_level: float
    metrics: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "EvaluationConfig":
        context = "evaluation"
        data = _strict_mapping(
            value,
            {
                "split_mode",
                "dates",
                "preprocessing_scope",
                "purge_horizon",
                "decision_delay_periods",
                "filtered_probabilities_only",
                "block_bootstrap_confidence_level",
                "metrics",
            },
            context,
        )
        confidence = _finite_float(
            _required(data, "block_bootstrap_confidence_level", context),
            f"{context}.block_bootstrap_confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise DailyRegimeConfigError(
                "evaluation.block_bootstrap_confidence_level must lie between zero and one."
            )
        filtered_only = _required(data, "filtered_probabilities_only", context)
        if not isinstance(filtered_only, bool):
            raise DailyRegimeConfigError(
                "evaluation.filtered_probabilities_only must be a boolean."
            )
        dates = _required(data, "dates", context)
        _ensure_json_compatible(dates, f"{context}.dates")
        return cls(
            split_mode=_required_text(data, "split_mode", context),
            dates=dates,
            preprocessing_scope=_required_text(data, "preprocessing_scope", context),
            purge_horizon=_required_text(data, "purge_horizon", context),
            decision_delay_periods=_nonnegative_int(
                _required(data, "decision_delay_periods", context),
                f"{context}.decision_delay_periods",
            ),
            filtered_probabilities_only=filtered_only,
            block_bootstrap_confidence_level=confidence,
            metrics=_string_tuple(
                _required(data, "metrics", context), f"{context}.metrics"
            ),
        )


@dataclass(frozen=True, slots=True)
class RiskThrottleConfig:
    role: str
    benchmark_return_stream: str | float
    annual_volatility_target: str | float
    maximum_gross_multiplier: str | float
    comparison_models: tuple[str, ...]
    require_average_exposure_matching: bool
    prohibit_strategy_routing: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "RiskThrottleConfig":
        context = "risk_throttle"
        data = _strict_mapping(
            value,
            {
                "role",
                "benchmark_return_stream",
                "annual_volatility_target",
                "maximum_gross_multiplier",
                "comparison_models",
                "require_average_exposure_matching",
                "prohibit_strategy_routing",
            },
            context,
        )
        require_matching = _required(data, "require_average_exposure_matching", context)
        prohibit_routing = _required(data, "prohibit_strategy_routing", context)
        if not isinstance(require_matching, bool) or not isinstance(prohibit_routing, bool):
            raise DailyRegimeConfigError(
                "risk_throttle matching and routing flags must be booleans."
            )
        return cls(
            role=_required_text(data, "role", context),
            benchmark_return_stream=_text_or_float(
                _required(data, "benchmark_return_stream", context),
                f"{context}.benchmark_return_stream",
            ),
            annual_volatility_target=_text_or_float(
                _required(data, "annual_volatility_target", context),
                f"{context}.annual_volatility_target",
            ),
            maximum_gross_multiplier=_text_or_float(
                _required(data, "maximum_gross_multiplier", context),
                f"{context}.maximum_gross_multiplier",
            ),
            comparison_models=_string_tuple(
                _required(data, "comparison_models", context),
                f"{context}.comparison_models",
            ),
            require_average_exposure_matching=require_matching,
            prohibit_strategy_routing=prohibit_routing,
        )


@dataclass(frozen=True, slots=True)
class HoldoutConfig:
    status: str
    opening_rule: str
    amendment_policy: str

    @classmethod
    def from_mapping(cls, value: Any) -> "HoldoutConfig":
        context = "holdout"
        data = _strict_mapping(
            value, {"status", "opening_rule", "amendment_policy"}, context
        )
        return cls(
            status=_required_text(data, "status", context),
            opening_rule=_required_text(data, "opening_rule", context),
            amendment_policy=_required_text(data, "amendment_policy", context),
        )


@dataclass(frozen=True, slots=True)
class DailyRegimeConfig:
    project: ProjectConfig
    claims: ClaimsConfig
    data_construction: DataConstructionConfig
    feature_sets: Mapping[str, FeatureSetConfig]
    targets: TargetsConfig
    hypotheses: Mapping[str, HypothesisConfig]
    models: ModelsConfig
    vqvae: VQVAEStudyConfig
    evaluation: EvaluationConfig
    risk_throttle: RiskThrottleConfig
    holdout: HoldoutConfig

    @classmethod
    def from_mapping(cls, value: Any) -> "DailyRegimeConfig":
        data = _strict_mapping(
            value,
            {
                "project",
                "claims",
                "data_construction",
                "feature_sets",
                "targets",
                "hypotheses",
                "models",
                "vqvae",
                "evaluation",
                "risk_throttle",
                "holdout",
            },
            "root",
        )
        feature_data = _mapping(
            _required(data, "feature_sets", "root"), "feature_sets"
        )
        hypothesis_data = _mapping(
            _required(data, "hypotheses", "root"), "hypotheses"
        )
        if not feature_data:
            raise DailyRegimeConfigError("feature_sets cannot be empty.")
        if not hypothesis_data:
            raise DailyRegimeConfigError("hypotheses cannot be empty.")
        feature_sets = {
            _mapping_key(name, "feature_sets"): FeatureSetConfig.from_mapping(
                payload, context=f"feature_sets.{name}"
            )
            for name, payload in feature_data.items()
        }
        hypotheses = {
            _mapping_key(name, "hypotheses"): HypothesisConfig.from_mapping(
                payload, context=f"hypotheses.{name}"
            )
            for name, payload in hypothesis_data.items()
        }
        config = cls(
            project=ProjectConfig.from_mapping(_required(data, "project", "root")),
            claims=ClaimsConfig.from_mapping(_required(data, "claims", "root")),
            data_construction=DataConstructionConfig.from_mapping(
                _required(data, "data_construction", "root")
            ),
            feature_sets=feature_sets,
            targets=TargetsConfig.from_mapping(_required(data, "targets", "root")),
            hypotheses=hypotheses,
            models=ModelsConfig.from_mapping(_required(data, "models", "root")),
            vqvae=VQVAEStudyConfig.from_mapping(_required(data, "vqvae", "root")),
            evaluation=EvaluationConfig.from_mapping(
                _required(data, "evaluation", "root")
            ),
            risk_throttle=RiskThrottleConfig.from_mapping(
                _required(data, "risk_throttle", "root")
            ),
            holdout=HoldoutConfig.from_mapping(_required(data, "holdout", "root")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        known_features = set(self.feature_sets)
        if self.vqvae.input_set_primary not in known_features:
            raise DailyRegimeConfigError(
                "vqvae.input_set_primary must name a declared feature set."
            )
        for name, feature_set in self.feature_sets.items():
            if feature_set.source and feature_set.source not in known_features:
                raise DailyRegimeConfigError(
                    f"feature_sets.{name}.source references unknown set {feature_set.source!r}."
                )
            missing_sources = set(feature_set.columns_from) - known_features
            if missing_sources:
                raise DailyRegimeConfigError(
                    f"feature_sets.{name}.columns_from references unknown sets: "
                    f"{sorted(missing_sources)}"
                )
        if self.evaluation.filtered_probabilities_only is not True:
            raise DailyRegimeConfigError(
                "This study requires evaluation.filtered_probabilities_only=true."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved contract in JSON-compatible form."""

        return _jsonable(asdict(self))

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def stable_hash(self) -> str:
        """Compatibility method for callers that prefer method-style hashing."""

        return self.config_hash


def load_daily_regime_config(path: str | Path) -> DailyRegimeConfig:
    """Load and strictly validate a daily-regime YAML research contract."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Daily-regime config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DailyRegimeConfigError(
            f"Invalid YAML in daily-regime config {config_path}: {exc}"
        ) from exc
    return DailyRegimeConfig.from_mapping(payload)


load_preregistration_config = load_daily_regime_config


def _strict_mapping(value: Any, allowed: set[str], context: str) -> dict[str, Any]:
    data = _mapping(value, context)
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise DailyRegimeConfigError(f"Unknown keys in {context}: {unknown}")
    return dict(data)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DailyRegimeConfigError(f"{context} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise DailyRegimeConfigError(f"{context} keys must be strings.")
    return value


def _mapping_key(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DailyRegimeConfigError(f"{context} contains an empty key.")
    return value.strip()


def _required(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise DailyRegimeConfigError(f"Missing required key {context}.{key}.")
    return data[key]


def _required_text(data: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(data, key, context)
    return _text(value, f"{context}.{key}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DailyRegimeConfigError(f"{context} must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _text(value, context)


def _string_tuple(value: Any, context: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DailyRegimeConfigError(f"{context} must be a sequence of strings.")
    result = tuple(_text(item, f"{context}[{idx}]") for idx, item in enumerate(value))
    if not result and not allow_empty:
        raise DailyRegimeConfigError(f"{context} cannot be empty.")
    if len(set(result)) != len(result):
        raise DailyRegimeConfigError(f"{context} cannot contain duplicates.")
    return result


def _optional_string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _string_tuple(value, context, allow_empty=True)


def _positive_int_tuple(value: Any, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DailyRegimeConfigError(f"{context} must be a sequence of positive integers.")
    result = tuple(_positive_int(item, f"{context}[{idx}]") for idx, item in enumerate(value))
    if not result:
        raise DailyRegimeConfigError(f"{context} cannot be empty.")
    if len(set(result)) != len(result):
        raise DailyRegimeConfigError(f"{context} cannot contain duplicates.")
    return result


def _nonnegative_int_tuple(value: Any, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DailyRegimeConfigError(
            f"{context} must be a sequence of non-negative integers."
        )
    result = tuple(
        _nonnegative_int(item, f"{context}[{idx}]") for idx, item in enumerate(value)
    )
    if not result:
        raise DailyRegimeConfigError(f"{context} cannot be empty.")
    if len(set(result)) != len(result):
        raise DailyRegimeConfigError(f"{context} cannot contain duplicates.")
    return result


def _positive_int(value: Any, context: str) -> int:
    parsed = _integer(value, context)
    if parsed <= 0:
        raise DailyRegimeConfigError(f"{context} must be positive.")
    return parsed


def _optional_positive_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, context)


def _nonnegative_int(value: Any, context: str) -> int:
    parsed = _integer(value, context)
    if parsed < 0:
        raise DailyRegimeConfigError(f"{context} cannot be negative.")
    return parsed


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DailyRegimeConfigError(f"{context} must be an integer.")
    return int(value)


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DailyRegimeConfigError(f"{context} must be numeric.")
    parsed = float(value)
    if not float("-inf") < parsed < float("inf"):
        raise DailyRegimeConfigError(f"{context} must be finite.")
    return parsed


def _text_or_float(value: Any, context: str) -> str | float:
    if isinstance(value, str):
        return _text(value, context)
    return _finite_float(value, context)


def _ensure_json_compatible(value: Any, context: str) -> None:
    try:
        json.dumps(_jsonable(value), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DailyRegimeConfigError(f"{context} must be JSON-compatible.") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "ClaimsConfig",
    "DataConstructionConfig",
    "DailyRegimeConfig",
    "DailyRegimeConfigError",
    "EvaluationConfig",
    "FeatureSetConfig",
    "HoldoutConfig",
    "HypothesisConfig",
    "ModelSelectionRulesConfig",
    "ModelsConfig",
    "ProjectConfig",
    "RiskThrottleConfig",
    "TargetsConfig",
    "VQVAEStudyConfig",
    "load_daily_regime_config",
    "load_preregistration_config",
]
