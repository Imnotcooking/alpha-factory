"""Versioned predictive-evidence panels for independently evaluated factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PREDICTIVE_EVIDENCE_SCHEMA_VERSION = 1
VALID_GEOMETRIES = {"cross_sectional", "time_series", "pairwise"}


class CausalAlignmentError(ValueError):
    """Raised when a return cannot be proven to begin after signal availability."""


@dataclass(frozen=True, slots=True)
class PredictiveEvidenceConfig:
    factor_id: str
    signal_col: str = "factor_score"
    return_col: str = "forward_return"
    date_col: str = "date"
    product_col: str = "ticker"
    split_col: str = "research_split"
    validation_label: str = "validation"
    holdout_label: str = "holdout"
    evaluation_geometry: str = "cross_sectional"
    expected_sign: int = 1
    rolling_window: int = 63
    rolling_min_periods: int = 21
    minimum_cross_section: int = 10
    minimum_product_observations: int = 60
    execution_lag: str = ""
    return_assumption: str = ""
    signal_available_col: str | None = None
    return_start_col: str | None = None
    return_end_col: str | None = None
    require_causal_alignment: bool = True
    require_validation_holdout: bool = True
    schema_version: int = PREDICTIVE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        text_fields = (
            "factor_id",
            "signal_col",
            "return_col",
            "date_col",
            "product_col",
            "split_col",
            "validation_label",
            "holdout_label",
        )
        for field in text_fields:
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} cannot be empty")
        geometry = str(self.evaluation_geometry).strip().lower()
        if geometry not in VALID_GEOMETRIES:
            raise ValueError(
                f"evaluation_geometry must be one of {sorted(VALID_GEOMETRIES)}"
            )
        if int(self.expected_sign) not in {-1, 1}:
            raise ValueError("expected_sign must be -1 or 1")
        if self.rolling_window < 2:
            raise ValueError("rolling_window must be at least 2")
        if not 1 <= self.rolling_min_periods <= self.rolling_window:
            raise ValueError("rolling_min_periods must be within the rolling window")
        if self.minimum_cross_section < 3:
            raise ValueError("minimum_cross_section must be at least 3")
        if self.minimum_product_observations < 3:
            raise ValueError("minimum_product_observations must be at least 3")
        if self.require_causal_alignment:
            if not str(self.execution_lag).strip():
                raise ValueError(
                    "execution_lag is required when causal alignment is required"
                )
            if not str(self.return_assumption).strip():
                raise ValueError(
                    "return_assumption is required when causal alignment is required"
                )
        timestamp_fields = (
            self.signal_available_col,
            self.return_start_col,
            self.return_end_col,
        )
        if any(timestamp_fields) and not all(timestamp_fields):
            raise ValueError(
                "signal_available_col, return_start_col, and return_end_col must be declared together"
            )
        object.__setattr__(self, "factor_id", str(self.factor_id).strip())
        object.__setattr__(self, "evaluation_geometry", geometry)
        object.__setattr__(self, "expected_sign", int(self.expected_sign))
        object.__setattr__(self, "execution_lag", str(self.execution_lag).strip())
        object.__setattr__(
            self, "return_assumption", str(self.return_assumption).strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class PredictiveEvidenceBundle:
    config: PredictiveEvidenceConfig
    summary: dict[str, Any]
    period_ic: pd.DataFrame
    split_summary: pd.DataFrame
    product_ic: pd.DataFrame
    yearly_summary: pd.DataFrame
    concentration: pd.DataFrame
    manifest: dict[str, Any]


def build_predictive_evidence(
    frame: pd.DataFrame,
    config: PredictiveEvidenceConfig,
) -> PredictiveEvidenceBundle:
    """Build predictive evidence without translating the signal into a portfolio."""

    required = {
        config.date_col,
        config.product_col,
        config.signal_col,
        config.return_col,
        config.split_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"predictive evidence is missing columns: {missing}")
    alignment = _verify_causal_alignment(frame, config)

    columns = list(required)
    for column in (
        config.signal_available_col,
        config.return_start_col,
        config.return_end_col,
    ):
        if column and column not in columns:
            columns.append(column)
    work = frame.loc[:, columns].copy()
    work[config.date_col] = pd.to_datetime(
        work[config.date_col], errors="coerce"
    ).dt.normalize()
    work[config.product_col] = work[config.product_col].astype("string").str.strip()
    work[config.signal_col] = pd.to_numeric(
        work[config.signal_col], errors="coerce"
    )
    work[config.return_col] = pd.to_numeric(
        work[config.return_col], errors="coerce"
    )
    work[config.split_col] = work[config.split_col].astype("string").str.strip()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[config.date_col, config.product_col])
    work = work.sort_values([config.date_col, config.product_col]).reset_index(drop=True)
    if work.duplicated([config.date_col, config.product_col]).any():
        raise ValueError("predictive evidence requires unique date/product rows")

    observed_splits = set(work[config.split_col].dropna().astype(str))
    required_splits = {config.validation_label, config.holdout_label}
    if config.require_validation_holdout and not required_splits.issubset(
        observed_splits
    ):
        missing_splits = sorted(required_splits.difference(observed_splits))
        raise ValueError(
            "predictive evidence is missing required split label(s): "
            + ", ".join(missing_splits)
        )

    period_ic = _build_period_ic(work, config)
    product_ic = _build_product_ic(work, config)
    split_summary = _build_split_summary(work, period_ic, product_ic, config)
    yearly_summary = _build_yearly_summary(period_ic, config)
    concentration = _build_concentration(yearly_summary, product_ic)
    summary = _build_summary(
        work,
        split_summary,
        concentration,
        config,
        alignment,
    )
    manifest = {
        "schema_version": PREDICTIVE_EVIDENCE_SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "factor_id": config.factor_id,
        "dataset_manifest_fingerprint": str(
            frame.attrs.get("dataset_manifest_fingerprint") or ""
        ),
        "input_data_fingerprint": str(
            frame.attrs.get("input_data_fingerprint") or ""
        ),
        "factor_definition_fingerprint": str(
            frame.attrs.get("factor_definition_fingerprint") or ""
        ),
        "factor_implementation_fingerprint": str(
            frame.attrs.get("factor_implementation_fingerprint") or ""
        ),
        "causal_alignment": alignment,
        "formulas": {
            "pearson_ic": "cross-sectional Pearson correlation of signal and executable forward return by date",
            "rank_ic": "cross-sectional Spearman correlation of signal and executable forward return by date",
            "icir": "arithmetic mean of period IC divided by sample standard deviation of period IC",
            "rolling_ic": "rolling arithmetic mean of valid raw period IC values",
            "cumulative_ic": "cumulative sum of raw period IC values",
            "oriented_ic": "raw IC multiplied by the factor's declared expected sign",
            "hit_rate": "share of valid periods whose IC has the declared expected sign",
            "concentration": "absolute share of orientation-adjusted Rank IC mass, weighted by valid evidence count",
        },
        "input": {
            "rows": int(len(work)),
            "dates": int(work[config.date_col].nunique()),
            "products": int(work[config.product_col].nunique()),
            "start": _date_text(work[config.date_col].min()),
            "end": _date_text(work[config.date_col].max()),
            "splits": sorted(observed_splits),
        },
    }
    return PredictiveEvidenceBundle(
        config=config,
        summary=summary,
        period_ic=period_ic,
        split_summary=split_summary,
        product_ic=product_ic,
        yearly_summary=yearly_summary,
        concentration=concentration,
        manifest=manifest,
    )


def write_predictive_evidence_bundle(
    bundle: PredictiveEvidenceBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(_json_safe(bundle.summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(_json_safe(bundle.manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    bundle.period_ic.to_parquet(destination / "period_ic.parquet", index=False)
    bundle.split_summary.to_csv(destination / "split_summary.csv", index=False)
    bundle.product_ic.to_csv(destination / "product_ic.csv", index=False)
    bundle.yearly_summary.to_csv(destination / "yearly_summary.csv", index=False)
    bundle.concentration.to_csv(destination / "concentration.csv", index=False)
    return destination


def load_predictive_evidence_bundle(
    output_dir: str | Path,
) -> PredictiveEvidenceBundle:
    source = Path(output_dir).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    config = PredictiveEvidenceConfig(**manifest["config"])
    return PredictiveEvidenceBundle(
        config=config,
        summary=summary,
        period_ic=pd.read_parquet(source / "period_ic.parquet"),
        split_summary=pd.read_csv(source / "split_summary.csv"),
        product_ic=pd.read_csv(source / "product_ic.csv"),
        yearly_summary=pd.read_csv(source / "yearly_summary.csv"),
        concentration=pd.read_csv(source / "concentration.csv"),
        manifest=manifest,
    )


def _verify_causal_alignment(
    frame: pd.DataFrame,
    config: PredictiveEvidenceConfig,
) -> dict[str, Any]:
    if not config.require_causal_alignment:
        return {"verified": False, "method": "explicitly_not_required"}
    if bool(frame.attrs.get("causal_return_alignment_verified")):
        return {
            "verified": True,
            "method": "frame_attestation",
            "execution_lag": config.execution_lag,
            "return_assumption": config.return_assumption,
        }
    timestamp_columns = (
        config.signal_available_col,
        config.return_start_col,
        config.return_end_col,
    )
    if not all(timestamp_columns):
        raise CausalAlignmentError(
            "causal alignment requires verified frame attrs or signal/return timestamp columns"
        )
    missing = [column for column in timestamp_columns if column not in frame.columns]
    if missing:
        raise CausalAlignmentError(
            f"causal timestamp columns are missing: {sorted(missing)}"
        )
    signal_time = pd.to_datetime(frame[config.signal_available_col], errors="coerce")
    return_start = pd.to_datetime(frame[config.return_start_col], errors="coerce")
    return_end = pd.to_datetime(frame[config.return_end_col], errors="coerce")
    valid = signal_time.notna() & return_start.notna() & return_end.notna()
    if not valid.any():
        raise CausalAlignmentError("causal timestamp columns contain no valid rows")
    invalid_start = valid & return_start.le(signal_time)
    invalid_end = valid & return_end.le(return_start)
    if invalid_start.any() or invalid_end.any():
        raise CausalAlignmentError(
            "forward return must start after signal availability and end after return start"
        )
    return {
        "verified": True,
        "method": "timestamp_columns",
        "verified_rows": int(valid.sum()),
        "execution_lag": config.execution_lag,
        "return_assumption": config.return_assumption,
    }


def _build_period_ic(
    frame: pd.DataFrame,
    config: PredictiveEvidenceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby(config.date_col, sort=True):
        splits = group[config.split_col].dropna().astype(str).unique()
        if len(splits) != 1:
            raise ValueError(f"date {date!s} belongs to {len(splits)} research splits")
        signal = group[config.signal_col]
        returns = group[config.return_col]
        signal_valid = signal.notna()
        return_valid = returns.notna()
        joint = signal_valid & return_valid
        valid = group.loc[joint, [config.signal_col, config.return_col]]
        pearson = math.nan
        rank = math.nan
        if (
            len(valid) >= config.minimum_cross_section
            and valid[config.signal_col].nunique() >= 2
            and valid[config.return_col].nunique() >= 2
        ):
            pearson = _finite_correlation(
                valid[config.signal_col], valid[config.return_col], "pearson"
            )
            rank = _finite_correlation(
                valid[config.signal_col], valid[config.return_col], "spearman"
            )
        denominator = max(len(group), 1)
        rows.append(
            {
                "factor_id": config.factor_id,
                "date": pd.Timestamp(date),
                "research_split": str(splits[0]),
                "product_count": int(len(group)),
                "valid_pair_count": int(joint.sum()),
                "signal_coverage": float(signal_valid.sum() / denominator),
                "active_signal_coverage": float(
                    (signal_valid & signal.abs().gt(1e-12)).sum() / denominator
                ),
                "forward_return_coverage": float(return_valid.sum() / denominator),
                "joint_coverage": float(joint.sum() / denominator),
                "pearson_ic": pearson,
                "rank_ic": rank,
                "oriented_pearson_ic": _orient(pearson, config.expected_sign),
                "oriented_rank_ic": _orient(rank, config.expected_sign),
            }
        )
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    result["rolling_pearson_ic"] = result["pearson_ic"].rolling(
        config.rolling_window,
        min_periods=config.rolling_min_periods,
    ).mean()
    result["rolling_rank_ic"] = result["rank_ic"].rolling(
        config.rolling_window,
        min_periods=config.rolling_min_periods,
    ).mean()
    result["cumulative_pearson_ic"] = result["pearson_ic"].fillna(0.0).cumsum()
    result["cumulative_rank_ic"] = result["rank_ic"].fillna(0.0).cumsum()
    result["rolling_oriented_pearson_ic"] = result["oriented_pearson_ic"].rolling(
        config.rolling_window,
        min_periods=config.rolling_min_periods,
    ).mean()
    result["rolling_oriented_rank_ic"] = result["oriented_rank_ic"].rolling(
        config.rolling_window,
        min_periods=config.rolling_min_periods,
    ).mean()
    result["cumulative_oriented_pearson_ic"] = (
        result["oriented_pearson_ic"].fillna(0.0).cumsum()
    )
    result["cumulative_oriented_rank_ic"] = (
        result["oriented_rank_ic"].fillna(0.0).cumsum()
    )
    return result


def _build_product_ic(
    frame: pd.DataFrame,
    config: PredictiveEvidenceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = [("full", frame)]
    samples.extend(
        (str(label), group)
        for label, group in frame.groupby(config.split_col, sort=True)
    )
    for sample, sample_frame in samples:
        for product, group in sample_frame.groupby(config.product_col, sort=True):
            signal = group[config.signal_col]
            returns = group[config.return_col]
            signal_valid = signal.notna()
            return_valid = returns.notna()
            joint = signal_valid & return_valid
            valid = group.loc[joint, [config.signal_col, config.return_col]]
            pearson = math.nan
            rank = math.nan
            if (
                len(valid) >= config.minimum_product_observations
                and valid[config.signal_col].nunique() >= 2
                and valid[config.return_col].nunique() >= 2
            ):
                pearson = _finite_correlation(
                    valid[config.signal_col], valid[config.return_col], "pearson"
                )
                rank = _finite_correlation(
                    valid[config.signal_col], valid[config.return_col], "spearman"
                )
            denominator = max(len(group), 1)
            rows.append(
                {
                    "factor_id": config.factor_id,
                    "research_split": sample,
                    "product": str(product),
                    "observations": int(len(group)),
                    "valid_pairs": int(joint.sum()),
                    "signal_coverage": float(signal_valid.sum() / denominator),
                    "active_signal_coverage": float(
                        (signal_valid & signal.abs().gt(1e-12)).sum() / denominator
                    ),
                    "forward_return_coverage": float(return_valid.sum() / denominator),
                    "joint_coverage": float(joint.sum() / denominator),
                    "pearson_ic": pearson,
                    "rank_ic": rank,
                    "oriented_pearson_ic": _orient(pearson, config.expected_sign),
                    "oriented_rank_ic": _orient(rank, config.expected_sign),
                }
            )
    return pd.DataFrame(rows)


def _build_split_summary(
    frame: pd.DataFrame,
    period_ic: pd.DataFrame,
    product_ic: pd.DataFrame,
    config: PredictiveEvidenceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = [("full", period_ic, frame)]
    samples.extend(
        (
            str(label),
            period_ic.loc[period_ic["research_split"].eq(str(label))],
            group,
        )
        for label, group in frame.groupby(config.split_col, sort=True)
    )
    for sample, periods, sample_frame in samples:
        products = product_ic.loc[product_ic["research_split"].eq(sample)]
        pearson = _distribution_summary(periods["pearson_ic"])
        rank = _distribution_summary(periods["rank_ic"])
        oriented_pearson = _distribution_summary(periods["oriented_pearson_ic"])
        oriented_rank = _distribution_summary(periods["oriented_rank_ic"])
        product_pearson = _distribution_summary(products["pearson_ic"])
        product_rank = _distribution_summary(products["rank_ic"])
        oriented_product_pearson = _distribution_summary(
            products["oriented_pearson_ic"]
        )
        oriented_product_rank = _distribution_summary(products["oriented_rank_ic"])
        rows.append(
            {
                "factor_id": config.factor_id,
                "research_split": sample,
                "mean_pearson_ic": pearson["mean"],
                "mean_rank_ic": rank["mean"],
                "pearson_ic_std": pearson["std"],
                "rank_ic_std": rank["std"],
                "pearson_icir": pearson["icir"],
                "rank_icir": rank["icir"],
                "oriented_mean_pearson_ic": oriented_pearson["mean"],
                "oriented_mean_rank_ic": oriented_rank["mean"],
                "pearson_ic_hit_rate": oriented_pearson["positive_fraction"],
                "rank_ic_hit_rate": oriented_rank["positive_fraction"],
                "valid_pearson_dates": pearson["count"],
                "valid_rank_ic_dates": rank["count"],
                "mean_product_pearson_ic": product_pearson["mean"],
                "mean_product_rank_ic": product_rank["mean"],
                "product_pearson_icir": product_pearson["icir"],
                "product_rank_icir": product_rank["icir"],
                "oriented_mean_product_pearson_ic": oriented_product_pearson[
                    "mean"
                ],
                "oriented_mean_product_rank_ic": oriented_product_rank["mean"],
                "positive_product_pearson_ic_share": oriented_product_pearson[
                    "positive_fraction"
                ],
                "positive_product_rank_ic_share": oriented_product_rank[
                    "positive_fraction"
                ],
                "valid_product_pearson_ics": product_pearson["count"],
                "valid_product_rank_ics": product_rank["count"],
                "date_count": int(sample_frame[config.date_col].nunique()),
                "product_count": int(sample_frame[config.product_col].nunique()),
                "row_count": int(len(sample_frame)),
                "signal_coverage": _coverage(sample_frame[config.signal_col]),
                "active_signal_coverage": _active_coverage(
                    sample_frame[config.signal_col]
                ),
                "forward_return_coverage": _coverage(
                    sample_frame[config.return_col]
                ),
                "joint_coverage": float(
                    (
                        sample_frame[config.signal_col].notna()
                        & sample_frame[config.return_col].notna()
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_yearly_summary(
    period_ic: pd.DataFrame,
    config: PredictiveEvidenceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dated = period_ic.assign(year=period_ic["date"].dt.year)
    for year, group in dated.groupby("year", sort=True):
        pearson = _distribution_summary(group["pearson_ic"])
        rank = _distribution_summary(group["rank_ic"])
        oriented_pearson = _distribution_summary(group["oriented_pearson_ic"])
        oriented_rank = _distribution_summary(group["oriented_rank_ic"])
        rows.append(
            {
                "factor_id": config.factor_id,
                "year": int(year),
                "mean_pearson_ic": pearson["mean"],
                "mean_rank_ic": rank["mean"],
                "pearson_icir": pearson["icir"],
                "rank_icir": rank["icir"],
                "oriented_mean_pearson_ic": oriented_pearson["mean"],
                "oriented_mean_rank_ic": oriented_rank["mean"],
                "pearson_ic_hit_rate": oriented_pearson["positive_fraction"],
                "rank_ic_hit_rate": oriented_rank["positive_fraction"],
                "valid_pearson_dates": pearson["count"],
                "valid_rank_ic_dates": rank["count"],
                "product_count": int(group["product_count"].max()),
                "mean_joint_coverage": float(group["joint_coverage"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_concentration(
    yearly_summary: pd.DataFrame,
    product_ic: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year_mass = (
        yearly_summary["oriented_mean_rank_ic"].fillna(0.0)
        * yearly_summary["valid_rank_ic_dates"].fillna(0.0)
    )
    year_abs_total = float(year_mass.abs().sum())
    year_evidence_total = float(yearly_summary["valid_rank_ic_dates"].sum())
    for index, row in yearly_summary.iterrows():
        mass = float(year_mass.loc[index])
        rows.append(
            {
                "dimension": "year",
                "member": str(int(row["year"])),
                "oriented_rank_ic": row["oriented_mean_rank_ic"],
                "evidence_count": int(row["valid_rank_ic_dates"]),
                "signed_rank_ic_mass": mass,
                "absolute_rank_ic_mass_share": (
                    abs(mass) / year_abs_total if year_abs_total > 0 else math.nan
                ),
                "observation_share": (
                    float(row["valid_rank_ic_dates"]) / year_evidence_total
                    if year_evidence_total > 0
                    else math.nan
                ),
            }
        )
    products = product_ic.loc[product_ic["research_split"].eq("full")].copy()
    product_mass = (
        products["oriented_rank_ic"].fillna(0.0)
        * products["valid_pairs"].fillna(0.0)
    )
    product_abs_total = float(product_mass.abs().sum())
    product_evidence_total = float(products["valid_pairs"].sum())
    for index, row in products.iterrows():
        mass = float(product_mass.loc[index])
        rows.append(
            {
                "dimension": "product",
                "member": str(row["product"]),
                "oriented_rank_ic": row["oriented_rank_ic"],
                "evidence_count": int(row["valid_pairs"]),
                "signed_rank_ic_mass": mass,
                "absolute_rank_ic_mass_share": (
                    abs(mass) / product_abs_total if product_abs_total > 0 else math.nan
                ),
                "observation_share": (
                    float(row["valid_pairs"]) / product_evidence_total
                    if product_evidence_total > 0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    frame: pd.DataFrame,
    split_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    config: PredictiveEvidenceConfig,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    full = split_summary.loc[split_summary["research_split"].eq("full")]
    full_row = full.iloc[0].to_dict() if not full.empty else {}
    year = concentration.loc[concentration["dimension"].eq("year")]
    product = concentration.loc[concentration["dimension"].eq("product")]
    return {
        "schema_version": PREDICTIVE_EVIDENCE_SCHEMA_VERSION,
        "factor_id": config.factor_id,
        "evaluation_geometry": config.evaluation_geometry,
        "primary_evidence_axis": (
            "date_cross_section"
            if config.evaluation_geometry == "cross_sectional"
            else "product_time_series"
        ),
        "expected_sign": config.expected_sign,
        "execution_lag": config.execution_lag,
        "return_assumption": config.return_assumption,
        "causal_alignment_verified": bool(alignment.get("verified")),
        "rows": int(len(frame)),
        "dates": int(frame[config.date_col].nunique()),
        "products": int(frame[config.product_col].nunique()),
        "full_sample": _json_safe(full_row),
        "concentration": {
            "top_year_absolute_rank_ic_mass_share": _largest_share(year, 1),
            "year_rank_ic_mass_hhi": _hhi(year),
            "top_product_absolute_rank_ic_mass_share": _largest_share(product, 1),
            "top_five_product_absolute_rank_ic_mass_share": _largest_share(
                product, 5
            ),
            "product_rank_ic_mass_hhi": _hhi(product),
        },
        "interpretation_boundary": (
            "Predictive IC is measured before transaction costs and does not prove "
            "that a portfolio translation is profitable."
        ),
    }


def _distribution_summary(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {
            "mean": math.nan,
            "std": math.nan,
            "icir": math.nan,
            "positive_fraction": math.nan,
            "count": 0,
        }
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return {
        "mean": mean,
        "std": standard_deviation,
        "icir": (
            mean / standard_deviation
            if math.isfinite(standard_deviation) and standard_deviation > 0
            else math.nan
        ),
        "positive_fraction": float(np.mean(values > 0.0)),
        "count": int(len(values)),
    }


def _finite_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    value = left.corr(right, method=method)
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else math.nan


def _orient(value: float, expected_sign: int) -> float:
    return float(value * expected_sign) if math.isfinite(value) else math.nan


def _coverage(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").notna().mean())


def _active_coverage(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float((values.notna() & values.abs().gt(1e-12)).mean())


def _largest_share(frame: pd.DataFrame, count: int) -> float | None:
    if frame.empty:
        return None
    values = pd.to_numeric(
        frame["absolute_rank_ic_mass_share"], errors="coerce"
    ).dropna()
    return float(values.nlargest(count).sum()) if not values.empty else None


def _hhi(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    values = pd.to_numeric(
        frame["absolute_rank_ic_mass_share"], errors="coerce"
    ).dropna()
    return float(np.square(values).sum()) if not values.empty else None


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat() if pd.notna(value) else ""


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None:
        return None
    if np.isscalar(value) and pd.isna(value):
        return None
    return value


__all__ = [
    "CausalAlignmentError",
    "PREDICTIVE_EVIDENCE_SCHEMA_VERSION",
    "PredictiveEvidenceBundle",
    "PredictiveEvidenceConfig",
    "build_predictive_evidence",
    "load_predictive_evidence_bundle",
    "write_predictive_evidence_bundle",
]
