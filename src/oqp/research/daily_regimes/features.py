"""Causal Stage 4 feature reconstruction for the daily-regime study.

The builder consumes the point-in-time product panel produced by Stage 3.  It
does not learn scaling, clipping, or PCA parameters: those operations belong to
the fold-local preprocessor in :mod:`oqp.research.daily_regimes.preprocessing`.

The historical H7 formulas were reconstructed against the archived feature
matrix and surviving wavelet-Hurst implementation.  Numerical degeneracies are
made explicit as missing values and diagnostics instead of being converted to
large finite artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


STAGE_OWNER = 4
BUILDER_ID = "causal_daily_regime_features_v1"
PREREGISTERED_FEATURE_SET_IDS = ("H7", "HPCA3", "M2", "M3", "H7_M3")

NUMERICAL_EPSILON = float(np.finfo(np.float64).eps)
LEGACY_EPSILON = 1e-8
KER_LOOKBACK = 20
HURST_LOOKBACK = 64

H7_COLUMNS = (
    "f_clv",
    "f_vol_climax",
    "f_ret_skew_5d",
    "f_value_60d",
    "f_mom_z_sector",
    "f_oi_growth_10d",
    "f_macro_hurst",
)
M2_RAW_COLUMNS = ("log_gk_gap_variance", "log_amihud")
M3_RAW_COLUMNS = (*M2_RAW_COLUMNS, "ker_20d")
FEATURE_SET_RAW_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "H7": H7_COLUMNS,
    # HPCA3 is fitted from H7 inside a training fold.  Its raw builder input is H7.
    "HPCA3": H7_COLUMNS,
    "M2": M2_RAW_COLUMNS,
    "M3": M3_RAW_COLUMNS,
    "H7_M3": (*H7_COLUMNS, *M3_RAW_COLUMNS),
}


class FeatureConstructionUnavailableError(NotImplementedError):
    """Backward-compatible error type retained from the Stage 2 interface."""


@dataclass(frozen=True)
class FeatureSpec:
    """Timing-aware declaration of one raw output feature."""

    name: str
    lookback_periods: int
    availability_lag_periods: int = 0
    formula_version: str = "daily_regime_stage4_v1"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature names must be non-empty.")
        if self.lookback_periods < 1:
            raise ValueError("lookback_periods must be at least one.")
        if self.availability_lag_periods < 0:
            raise ValueError("availability_lag_periods cannot be negative.")


@dataclass(frozen=True)
class FeatureSetRequest:
    """A named representation requested from the causal raw-feature builder."""

    feature_set_id: str
    specs: tuple[FeatureSpec, ...]
    product_column: str = "product"
    trading_date_column: str = "trading_date"
    information_date_column: str = "information_date"

    def __post_init__(self) -> None:
        if not self.feature_set_id.strip():
            raise ValueError("feature_set_id must be non-empty.")
        if not self.specs:
            raise ValueError("A feature-set request must contain at least one feature.")
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique within a feature set.")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)


@dataclass(frozen=True)
class FeatureBuildResult:
    """Causal product-date features with explicit availability timing."""

    frame: pd.DataFrame
    feature_set_id: str
    feature_columns: tuple[str, ...]
    builder_id: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_set_id.strip():
            raise ValueError("feature_set_id must be non-empty.")
        if not self.builder_id.strip():
            raise ValueError("builder_id must be non-empty.")
        if not self.feature_columns:
            raise ValueError("feature_columns must be non-empty.")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("feature_columns must be unique.")


@runtime_checkable
class FeatureBuilder(Protocol):
    """Protocol for causal, audited Stage 4 feature implementations."""

    @property
    def builder_id(self) -> str:
        """Stable implementation identifier recorded in run manifests."""

    def build(
        self,
        panel: pd.DataFrame,
        *,
        request: FeatureSetRequest,
    ) -> FeatureBuildResult:
        """Construct raw features using information available by each row's date."""


_LOOKBACKS = {
    "f_clv": 1,
    "f_vol_climax": 20,
    "f_ret_skew_5d": 6,
    "f_value_60d": 60,
    "f_mom_z_sector": 2,
    "f_oi_growth_10d": 11,
    "f_macro_hurst": HURST_LOOKBACK,
    "log_gk_gap_variance": 2,
    "log_amihud": 2,
    "ker_20d": KER_LOOKBACK + 1,
}


def feature_set_request(feature_set_id: str) -> FeatureSetRequest:
    """Resolve a preregistered representation to its pre-fold raw ingredients."""

    if feature_set_id not in FEATURE_SET_RAW_COLUMNS:
        raise KeyError(f"Unknown preregistered feature set: {feature_set_id}")
    return FeatureSetRequest(
        feature_set_id=feature_set_id,
        specs=tuple(
            FeatureSpec(name=name, lookback_periods=_LOOKBACKS[name])
            for name in FEATURE_SET_RAW_COLUMNS[feature_set_id]
        ),
    )


def feature_formula_registry() -> dict[str, dict[str, Any]]:
    """Return the versioned, JSON-safe Stage 4 formula and lineage registry."""

    return {
        "log_gk_gap_variance": {
            "version": "gk_plus_gap_whole_proxy_floor_v1",
            "formula": "log(max(gap^2 + 0.5*log(H/L)^2 - (2*log(2)-1)*log(C/O)^2, eps))",
            "input": "Stage3 selected-contract OHLC and previous_same_contract_close",
            "timing": "available at trading-day close",
        },
        "log_amihud": {
            "version": "futures_amihud_turnover_v1",
            "formula": "log(abs(same_contract_log_return)/turnover + eps)",
            "input": "Stage3 return and turnover; turnover provenance is external metadata",
            "timing": "available at trading-day close",
        },
        "ker_20d": {
            "version": "roll_safe_price_path_v1",
            "formula": "abs(P[t]-P[t-20])/sum(abs(diff(P)), 20)",
            "input": "Stage3 continuous_index within product and sequence_id",
            "timing": "available at trading-day close",
        },
        "f_clv": {
            "version": "legacy_reconstruction_v1",
            "formula": "(2*C-H-L)/(H-L+1e-8)",
            "source_status": "reconstructed_against_archived_feature_matrix",
        },
        "f_vol_climax": {
            "version": "legacy_reconstruction_v1",
            "formula": "volume/(rolling_mean(volume,20)+1e-8)",
            "source_status": "reconstructed_against_archived_feature_matrix",
        },
        "f_ret_skew_5d": {
            "version": "legacy_reconstruction_stable_degeneracy_v1",
            "formula": "population_skew(last_5_same_contract_simple_returns)",
            "source_status": "reconstructed; zero-variance windows are missing instead of unstable finite outliers",
        },
        "f_value_60d": {
            "version": "legacy_reconstruction_roll_safe_v1",
            "formula": "1-continuous_index/rolling_mean(continuous_index,60)",
            "source_status": "legacy close formula adapted to Stage3 roll-safe price index",
        },
        "f_mom_z_sector": {
            "version": "legacy_reconstruction_taxonomy_required_v1",
            "formula": "same_contract_simple_return same-date sector sample z-score",
            "source_status": "legacy artifact used one Macro bucket; Stage4 requires explicit sector taxonomy",
        },
        "f_oi_growth_10d": {
            "version": "legacy_reconstruction_v1",
            "formula": "open_interest/open_interest[t-10]",
            "source_status": "reconstructed_against_archived_feature_matrix",
        },
        "f_macro_hurst": {
            "version": "legacy_wavelet_hurst_64_roll_safe_v1",
            "formula": "Haar detail-variance slope H=(slope-1)/2, clipped to [0.1,0.9]",
            "source_status": "ported from surviving WaveletHurstEstimator; historical name retained",
        },
    }


class DailyRegimeFeatureBuilder:
    """Build preregistered raw representations without fitting future data."""

    builder_id = BUILDER_ID

    def build(
        self,
        panel: pd.DataFrame,
        *,
        request: FeatureSetRequest,
    ) -> FeatureBuildResult:
        _validate_request(request)
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame.")
        if panel.empty:
            raise ValueError("Feature construction requires a non-empty panel.")

        required = {
            request.product_column,
            request.trading_date_column,
            "sequence_id",
        }
        needs_sparse = any(name in M3_RAW_COLUMNS for name in request.feature_columns)
        needs_h7 = any(name in H7_COLUMNS for name in request.feature_columns)
        if needs_sparse:
            required.update(
                {
                    "open",
                    "high",
                    "low",
                    "close",
                    "previous_same_contract_close",
                    "same_contract_log_return",
                    "turnover",
                }
            )
        if "ker_20d" in request.feature_columns or needs_h7:
            required.add("continuous_index")
        if needs_h7:
            required.update(
                {
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                    "same_contract_log_return",
                    "sector",
                }
            )
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise ValueError(f"Stage 4 panel is missing required columns: {missing}")

        product_col = request.product_column
        date_col = request.trading_date_column
        rows = panel.copy(deep=True)
        rows[date_col] = pd.to_datetime(rows[date_col], errors="raise").dt.normalize()
        if rows.duplicated([product_col, date_col]).any():
            raise ValueError("Stage 4 panel contains duplicate product-date rows.")
        rows = rows.sort_values([product_col, date_col], kind="mergesort").reset_index(
            drop=True
        )

        lineage_columns = [
            column
            for column in (
                product_col,
                date_col,
                "contract",
                "selected_contract",
                "source_row_id",
                "sequence_id",
                "roll_flag",
                "chain_reset_flag",
            )
            if column in rows.columns
        ]
        output = rows[lineage_columns].copy()
        output[request.information_date_column] = rows[date_col]

        if needs_sparse:
            _add_sparse_features(
                rows,
                output,
                include_ker="ker_20d" in request.feature_columns,
                product_col=product_col,
            )
        if needs_h7:
            _add_h7_features(rows, output, product_col=product_col, date_col=date_col)

        output = output.replace([np.inf, -np.inf], np.nan)
        result = FeatureBuildResult(
            frame=output,
            feature_set_id=request.feature_set_id,
            feature_columns=request.feature_columns,
            builder_id=self.builder_id,
            diagnostics={
                "input_rows": int(len(panel)),
                "output_rows": int(len(output)),
                "feature_set_id": request.feature_set_id,
                "raw_feature_columns": list(request.feature_columns),
                "missing_by_feature": {
                    name: int(output[name].isna().sum())
                    for name in request.feature_columns
                },
                "quality_flag_counts": {
                    column: int(output[column].fillna(False).astype(bool).sum())
                    for column in output.columns
                    if column.startswith("quality_")
                },
                "formula_registry": feature_formula_registry(),
                "turnover_source": "stage3_panel_as_supplied",
                "synthetic_verification_only": True,
                "scientific_evidence": False,
            },
        )
        validate_feature_result(result)
        return result


def build_features(
    panel: pd.DataFrame,
    *,
    request: FeatureSetRequest,
) -> FeatureBuildResult:
    """Build a preregistered causal raw representation."""

    return DailyRegimeFeatureBuilder().build(panel, request=request)


def validate_feature_result(
    result: FeatureBuildResult,
    *,
    key_columns: Sequence[str] = ("product", "trading_date", "information_date"),
) -> None:
    """Validate keys, timing, numerical finiteness, and representation bounds."""

    if not isinstance(result.frame, pd.DataFrame):
        raise TypeError("FeatureBuildResult.frame must be a pandas DataFrame.")
    required = tuple(key_columns) + result.feature_columns
    missing = [column for column in required if column not in result.frame.columns]
    if missing:
        raise ValueError(f"Feature result is missing columns: {missing}")
    uniqueness_keys = list(tuple(key_columns)[:2])
    if uniqueness_keys and result.frame.duplicated(uniqueness_keys).any():
        raise ValueError("Feature result has duplicate product-date rows.")
    non_numeric = [
        column
        for column in result.feature_columns
        if not is_numeric_dtype(result.frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature columns must be numeric: {non_numeric}")
    for column in result.feature_columns:
        finite = result.frame[column].dropna().to_numpy(dtype=float)
        if finite.size and not np.isfinite(finite).all():
            raise ValueError(f"Feature column {column!r} contains non-finite values.")
    trading = pd.to_datetime(
        result.frame[key_columns[1]], errors="raise"
    ).dt.normalize()
    information = pd.to_datetime(
        result.frame[key_columns[2]], errors="raise"
    ).dt.normalize()
    if (information < trading).any():
        raise ValueError("Feature information_date cannot precede trading_date.")
    if "ker_20d" in result.frame:
        ker = result.frame["ker_20d"].dropna().to_numpy(dtype=float)
        if ker.size and ((ker < -1e-12) | (ker > 1.0 + 1e-12)).any():
            raise ValueError("ker_20d must lie in [0, 1].")


def wavelet_hurst(prices: Sequence[float]) -> float:
    """Port the legacy Haar detail-variance estimator with deterministic OLS."""

    current = np.asarray(prices, dtype=float)
    if current.ndim != 1 or current.size < 2 or not np.isfinite(current).all():
        return np.nan
    log_scales: list[float] = []
    log_variances: list[float] = []
    scale = 1
    while current.size >= 2:
        pair_count = current.size // 2
        paired = current[: 2 * pair_count].reshape(pair_count, 2)
        approximation = (paired[:, 0] + paired[:, 1]) / np.sqrt(2.0)
        detail = (paired[:, 0] - paired[:, 1]) / np.sqrt(2.0)
        variance = float(np.var(detail, ddof=1)) if detail.size > 1 else 0.0
        if variance > 1e-10:
            log_scales.append(float(np.log2(scale)))
            log_variances.append(float(np.log2(variance)))
        current = approximation
        scale *= 2
    if len(log_scales) < 2:
        return 0.5
    x = np.asarray(log_scales, dtype=float)
    y = np.asarray(log_variances, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return 0.5
    slope = float(np.dot(x_centered, y - y.mean()) / denominator)
    return float(np.clip((slope - 1.0) / 2.0, 0.1, 0.9))


def _validate_request(request: FeatureSetRequest) -> None:
    if request.feature_set_id not in FEATURE_SET_RAW_COLUMNS:
        raise ValueError(f"Unknown preregistered feature set: {request.feature_set_id}")
    expected = FEATURE_SET_RAW_COLUMNS[request.feature_set_id]
    if request.feature_columns != expected:
        raise ValueError(
            f"{request.feature_set_id} raw columns are frozen as {expected}; "
            f"received {request.feature_columns}."
        )


def _add_sparse_features(
    rows: pd.DataFrame,
    output: pd.DataFrame,
    *,
    include_ker: bool,
    product_col: str,
) -> None:
    prices = rows[
        ["open", "high", "low", "close", "previous_same_contract_close"]
    ].apply(pd.to_numeric, errors="coerce")
    invalid_prices = prices.isna().any(axis=1) | (prices <= 0.0).any(axis=1)
    gap = np.log(prices["open"] / prices["previous_same_contract_close"])
    high_low = np.log(prices["high"] / prices["low"])
    close_open = np.log(prices["close"] / prices["open"])
    variance = (
        gap.pow(2)
        + 0.5 * high_low.pow(2)
        - (2.0 * np.log(2.0) - 1.0) * close_open.pow(2)
    )
    variance = variance.mask(invalid_prices)
    gk_floor = variance.notna() & variance.le(NUMERICAL_EPSILON)
    output["gk_gap_variance"] = variance
    output["log_gk_gap_variance"] = np.log(variance.clip(lower=NUMERICAL_EPSILON))
    output["quality_invalid_gk_input"] = invalid_prices
    output["quality_gk_floor"] = gk_floor

    returns = pd.to_numeric(rows["same_contract_log_return"], errors="coerce")
    turnover = pd.to_numeric(rows["turnover"], errors="coerce")
    invalid_turnover = turnover.isna() | turnover.le(0.0)
    amihud = returns.abs().divide(turnover).mask(invalid_turnover | returns.isna())
    output["amihud"] = amihud
    output["log_amihud"] = np.log(amihud + NUMERICAL_EPSILON)
    output["quality_nonpositive_turnover"] = invalid_turnover

    if include_ker:
        continuous = pd.to_numeric(rows["continuous_index"], errors="coerce")
        ker = pd.Series(np.nan, index=rows.index, dtype=float)
        zero_path = pd.Series(False, index=rows.index, dtype=bool)
        for _, indices in rows.groupby(
            [product_col, "sequence_id"], sort=False
        ).groups.items():
            index = pd.Index(indices)
            price = continuous.loc[index]
            movement = price.diff().abs()
            path = movement.rolling(KER_LOOKBACK, min_periods=KER_LOOKBACK).sum()
            displacement = (price - price.shift(KER_LOOKBACK)).abs()
            valid = path.gt(0.0) & price.gt(0.0)
            ker.loc[index] = displacement.divide(path).where(valid).clip(0.0, 1.0)
            zero_path.loc[index] = path.eq(0.0) & path.notna()
        output["ker_20d"] = ker
        output["quality_ker_zero_path"] = zero_path


def _add_h7_features(
    rows: pd.DataFrame,
    output: pd.DataFrame,
    *,
    product_col: str,
    date_col: str,
) -> None:
    high = pd.to_numeric(rows["high"], errors="coerce")
    low = pd.to_numeric(rows["low"], errors="coerce")
    close = pd.to_numeric(rows["close"], errors="coerce")
    volume = pd.to_numeric(rows["volume"], errors="coerce")
    open_interest = pd.to_numeric(rows["open_interest"], errors="coerce")
    continuous = pd.to_numeric(rows["continuous_index"], errors="coerce")
    same_contract_return = pd.to_numeric(
        rows["same_contract_log_return"], errors="coerce"
    )
    simple_return = np.expm1(same_contract_return)

    output["f_clv"] = (2.0 * close - high - low) / (high - low + LEGACY_EPSILON)
    output["quality_zero_daily_range"] = (high - low).abs().le(NUMERICAL_EPSILON)

    volume_climax = pd.Series(np.nan, index=rows.index, dtype=float)
    return_skew = pd.Series(np.nan, index=rows.index, dtype=float)
    value = pd.Series(np.nan, index=rows.index, dtype=float)
    oi_growth = pd.Series(np.nan, index=rows.index, dtype=float)
    hurst = pd.Series(np.nan, index=rows.index, dtype=float)
    skew_degenerate = pd.Series(False, index=rows.index, dtype=bool)
    invalid_oi_lag = pd.Series(False, index=rows.index, dtype=bool)

    for _, indices in rows.groupby(
        [product_col, "sequence_id"], sort=False
    ).groups.items():
        index = pd.Index(indices)
        local_volume = volume.loc[index]
        volume_mean = local_volume.rolling(20, min_periods=1).mean()
        volume_climax.loc[index] = local_volume.divide(volume_mean + LEGACY_EPSILON)

        local_return = simple_return.loc[index]
        return_skew.loc[index] = local_return.rolling(5, min_periods=5).apply(
            _population_skew, raw=True
        )
        local_skew_variance = local_return.rolling(5, min_periods=5).var(ddof=0)
        skew_degenerate.loc[index] = (
            local_skew_variance.notna() & local_skew_variance.le(NUMERICAL_EPSILON)
        )

        local_price = continuous.loc[index]
        price_mean = local_price.rolling(60, min_periods=1).mean()
        value.loc[index] = 1.0 - local_price.divide(price_mean)

        local_oi = open_interest.loc[index]
        lagged_oi = local_oi.shift(10)
        valid_lag = lagged_oi.gt(0.0)
        oi_growth.loc[index] = local_oi.divide(lagged_oi).where(valid_lag)
        invalid_oi_lag.loc[index] = lagged_oi.notna() & ~valid_lag

        hurst.loc[index] = local_price.rolling(
            HURST_LOOKBACK, min_periods=HURST_LOOKBACK
        ).apply(wavelet_hurst, raw=True)
        # The historical baseline used a neutral warm-up value.  It is a fixed
        # convention rather than a fitted imputation parameter.
        hurst.loc[index] = hurst.loc[index].fillna(0.5)

    output["f_vol_climax"] = volume_climax
    output["f_ret_skew_5d"] = return_skew
    output["f_value_60d"] = value
    output["f_oi_growth_10d"] = oi_growth
    output["f_macro_hurst"] = hurst
    output["quality_skew_degenerate"] = skew_degenerate
    output["quality_invalid_oi_lag"] = invalid_oi_lag

    sector = rows["sector"].astype("string")
    invalid_sector = sector.isna() | sector.str.strip().eq("")
    momentum = pd.Series(np.nan, index=rows.index, dtype=float)
    grouped = pd.DataFrame(
        {"date": rows[date_col], "sector": sector, "return": simple_return}
    ).groupby(["date", "sector"], sort=False, dropna=False)["return"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    momentum = (simple_return - mean).divide(std + LEGACY_EPSILON)
    momentum = momentum.mask(invalid_sector | std.isna())
    output["f_mom_z_sector"] = momentum
    output["quality_invalid_sector"] = invalid_sector | std.isna()


def _population_skew(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size != 5 or not np.isfinite(array).all():
        return np.nan
    centered = array - array.mean()
    second = float(np.mean(centered**2))
    if second <= NUMERICAL_EPSILON:
        return np.nan
    third = float(np.mean(centered**3))
    return third / (second**1.5)


__all__ = [
    "BUILDER_ID",
    "DailyRegimeFeatureBuilder",
    "FEATURE_SET_RAW_COLUMNS",
    "FeatureBuildResult",
    "FeatureBuilder",
    "FeatureConstructionUnavailableError",
    "FeatureSetRequest",
    "FeatureSpec",
    "H7_COLUMNS",
    "HURST_LOOKBACK",
    "KER_LOOKBACK",
    "M2_RAW_COLUMNS",
    "M3_RAW_COLUMNS",
    "PREREGISTERED_FEATURE_SET_IDS",
    "STAGE_OWNER",
    "build_features",
    "feature_formula_registry",
    "feature_set_request",
    "validate_feature_result",
    "wavelet_hurst",
]
