"""Causal factor-level evaluator for governed Phase 8 searches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from oqp.optimization import resolve_component_parameter_values
from oqp.research.backtesting.return_horizons import attach_return_horizon
from oqp.research.dataset_fingerprints import (
    DatasetManifest,
    attach_dataset_manifest_attrs,
    load_dataset_manifest,
    verify_dataset_manifest,
)
from oqp.research.factors import load_factor_module
from oqp.research.liquidity_eligibility import (
    ensure_liquidity_eligibility,
    liquidity_eligible_rows,
)
from oqp.research.optional_optimization.folds import Phase8Fold
from oqp.research.optional_optimization.study_builder import (
    REPO_ROOT,
    resolve_selected_component_schema,
)
from oqp.research.predictive_evidence import (
    PredictiveEvidenceConfig,
    build_predictive_evidence,
)


def load_factor_development_data(
    manifest_path: str | Path,
    *,
    initial_capital: float = 10_000_000.0,
    capital_currency: str = "CNY",
    max_position_weight: float = 0.05,
    workspace_root: str | Path = REPO_ROOT,
) -> tuple[pd.DataFrame, DatasetManifest]:
    """Load, verify, align, and liquidity-gate one registered factor dataset."""

    root = Path(workspace_root).expanduser().resolve()
    path = Path(manifest_path).expanduser().resolve()
    manifest = load_dataset_manifest(path)
    verification = verify_dataset_manifest(
        manifest,
        workspace_root=root,
        strict=True,
    )
    frames = [
        _read_source_file(
            source.path,
            workspace_root=root,
        )
        for source in manifest.source_files
    ]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    required = {"date", "ticker", "open", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "factor optimization dataset is missing columns: "
            + ", ".join(missing)
        )
    frame = frame.loc[pd.to_numeric(frame["close"], errors="coerce").notna()].copy()
    if manifest.row_count is not None and len(frame) != manifest.row_count:
        raise ValueError(
            "manifest row scope mismatch after applying the frozen non-missing-close "
            f"rule: expected {manifest.row_count}, observed {len(frame)}"
        )
    frame, normalization = _normalize_factor_rows(
        frame,
        data_frequency=manifest.data_frequency,
    )

    frame = attach_dataset_manifest_attrs(
        frame,
        manifest,
        path,
        verified=verification.verified,
        workspace_root=root,
    )
    frame.attrs.update(
        {
            "dataset_manifest_fingerprint": manifest.aggregate_sha256,
            "market_vertical": manifest.market_vertical,
            "data_frequency": manifest.data_frequency,
            "initial_capital": float(initial_capital),
            "capital_currency": str(capital_currency),
            "factor_data_normalization": normalization,
        }
    )
    frame["_next_session"] = frame.groupby("ticker", sort=False)["date"].shift(-1)
    frame = attach_return_horizon(
        frame,
        return_horizon="close_signal_next_open_to_close",
        data_frequency=manifest.data_frequency,
    )
    frame["signal_available_at"] = frame["date"] + pd.Timedelta(hours=16)
    frame["return_start_at"] = frame["_next_session"] + pd.Timedelta(hours=9)
    frame["return_end_at"] = frame["_next_session"] + pd.Timedelta(hours=15)
    assessed = ensure_liquidity_eligibility(
        frame,
        market_vertical=manifest.market_vertical,
        initial_capital=initial_capital,
        capital_currency=capital_currency,
        max_position_weight=max_position_weight,
    )
    eligible = liquidity_eligible_rows(assessed)
    eligible.attrs.update(assessed.attrs)
    eligible.attrs["dataset_fingerprint"] = manifest.aggregate_sha256
    eligible.attrs["dataset_manifest_fingerprint"] = manifest.aggregate_sha256
    return eligible.reset_index(drop=True), manifest


class FactorPredictiveFoldEvaluator:
    """Evaluate one factor formula without constructing positions or PnL."""

    def __init__(
        self,
        component_id: str,
        development_data: pd.DataFrame,
    ) -> None:
        self.component_id = str(component_id).strip()
        self.module = load_factor_module(
            self.component_id,
            include_public_examples=False,
        )
        self.schema = resolve_selected_component_schema(
            "factor_parameter",
            self.component_id,
        )
        self.contract = dict(getattr(self.module, "FACTOR_CONTRACT", {}) or {})
        self.metadata = dict(getattr(self.module, "FACTOR_METADATA", {}) or {})
        self.signal_col = str(
            self.contract.get("alpha_signal_col") or "factor_score"
        )
        self.expected_sign = _expected_sign(
            getattr(self.module, "SIGNAL_ORIENTATION", "higher_is_bullish")
        )
        self.development_data = development_data.copy()
        self.development_data.attrs.update(development_data.attrs)
        self._metric_cache: dict[
            tuple[tuple[tuple[str, Any], ...], str], dict[str, float]
        ] = {}
        self._scored_key: tuple[tuple[str, Any], ...] | None = None
        self._scored_frame: pd.DataFrame | None = None

    def __call__(
        self,
        parameters: dict[str, Any],
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        fold: Phase8Fold,
    ) -> Mapping[str, float]:
        del training_data
        resolved = resolve_component_parameter_values(
            self.schema,
            parameters,
            enforce_search_bounds=True,
        )
        parameter_key = tuple(sorted(resolved.items()))
        cache_key = (parameter_key, fold.fold_id)
        cached = self._metric_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        scored = self._scored_data(parameter_key, resolved)
        validation_dates = pd.Index(
            pd.to_datetime(validation_data["date"], errors="raise")
            .dt.normalize()
            .unique()
        )
        sample = scored.loc[
            pd.to_datetime(scored["date"]).dt.normalize().isin(validation_dates)
        ].copy()
        sample.attrs.update(scored.attrs)
        sample["research_split"] = "validation"
        config = PredictiveEvidenceConfig(
            factor_id=self.component_id,
            signal_col=self.signal_col,
            return_col="forward_return",
            date_col="date",
            product_col="ticker",
            split_col="research_split",
            validation_label="validation",
            holdout_label="holdout",
            evaluation_geometry=str(
                self.contract.get("evaluation_geometry") or "cross_sectional"
            ),
            expected_sign=self.expected_sign,
            rolling_window=min(63, max(fold.validation_periods, 2)),
            rolling_min_periods=min(21, max(fold.validation_periods, 2)),
            minimum_cross_section=10,
            minimum_product_observations=3,
            execution_lag=str(
                self.contract.get("execution_lag") or "already_lagged"
            ),
            return_assumption=str(
                self.contract.get("return_assumption")
                or "close_signal_next_open_to_close"
            ),
            signal_available_col="signal_available_at",
            return_start_col="return_start_at",
            return_end_col="return_end_at",
            require_causal_alignment=True,
            require_validation_holdout=False,
        )
        bundle = build_predictive_evidence(sample, config)
        summary = bundle.split_summary.loc[
            bundle.split_summary["research_split"].eq("validation")
        ].iloc[0]
        metrics = {
            "mean_pearson_ic": self.expected_sign
            * float(summary["mean_pearson_ic"]),
            "mean_rank_ic": self.expected_sign * float(summary["mean_rank_ic"]),
            "pearson_icir": self.expected_sign * float(summary["pearson_icir"]),
            "rank_icir": self.expected_sign * float(summary["rank_icir"]),
            "mean_joint_coverage": float(summary["joint_coverage"]),
            "stability_floor": _stability_floor(
                bundle.period_ic,
                expected_sign=self.expected_sign,
            ),
            "pearson_ic_hit_rate": float(summary["pearson_ic_hit_rate"]),
            "rank_ic_hit_rate": float(summary["rank_ic_hit_rate"]),
            "valid_rank_ic_dates": float(summary["valid_rank_ic_dates"]),
        }
        self._metric_cache[cache_key] = metrics
        return dict(metrics)

    def _scored_data(
        self,
        parameter_key: tuple[tuple[str, Any], ...],
        parameters: Mapping[str, Any],
    ) -> pd.DataFrame:
        if self._scored_key == parameter_key and self._scored_frame is not None:
            return self._scored_frame
        source_attrs = dict(self.development_data.attrs)
        scored = self.module.compute(
            self.development_data.copy(),
            **dict(parameters),
        )
        scored.attrs.update(source_attrs)
        if self.signal_col not in scored.columns:
            raise ValueError(
                f"{self.component_id} did not produce {self.signal_col!r}"
            )
        self._scored_key = parameter_key
        self._scored_frame = scored
        return scored


def _stability_floor(
    period_ic: pd.DataFrame,
    *,
    expected_sign: int,
) -> float:
    valid = period_ic.loc[
        pd.to_numeric(period_ic["rank_ic"], errors="coerce").notna()
    ].sort_values("date")
    if len(valid) < 4:
        return 0.0
    midpoint = len(valid) // 2
    halves = (valid.iloc[:midpoint], valid.iloc[midpoint:])
    return float(
        min(
            expected_sign
            * pd.to_numeric(half["rank_ic"], errors="coerce").mean()
            for half in halves
        )
    )


def _expected_sign(orientation: str) -> int:
    normalized = str(orientation).strip().lower()
    if normalized in {"higher_is_bullish", "higher_is_better", "positive"}:
        return 1
    if normalized in {"higher_is_bearish", "lower_is_better", "negative"}:
        return -1
    raise ValueError(f"unknown factor signal orientation: {orientation!r}")


def _read_source_file(
    source_path: str,
    *,
    workspace_root: Path,
) -> pd.DataFrame:
    path = Path(source_path)
    if not path.is_absolute():
        path = workspace_root / path
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported factor optimization source: {path}")


def _normalize_factor_rows(
    frame: pd.DataFrame,
    *,
    data_frequency: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    out["_source_timestamp"] = pd.to_datetime(out["date"], errors="raise")
    out["ticker"] = out["ticker"].astype("string").str.strip()
    if str(data_frequency).strip().lower() != "daily":
        out["date"] = out["_source_timestamp"]
        out = out.drop(columns=["_source_timestamp"])
        return out.sort_values(["ticker", "date"]).reset_index(drop=True), {
            "policy_id": "preserve_source_timestamp_v1",
            "collision_groups": 0,
            "rows_removed": 0,
        }

    out["date"] = out["_source_timestamp"].dt.normalize()
    duplicates = out.duplicated(["date", "ticker"], keep=False)
    collision_groups = int(
        out.loc[duplicates, ["date", "ticker"]].drop_duplicates().shape[0]
    )
    before = len(out)
    if collision_groups:
        out["_positive_ohlc"] = (
            out[["open", "high", "low", "close"]]
            .apply(pd.to_numeric, errors="coerce")
            .gt(0.0)
            .all(axis=1)
        )
        out["_volume_rank"] = pd.to_numeric(
            out.get("volume"),
            errors="coerce",
        ).fillna(-np.inf)
        out = out.sort_values(
            [
                "ticker",
                "date",
                "_positive_ohlc",
                "_volume_rank",
                "_source_timestamp",
            ],
            ascending=[True, True, False, False, True],
        ).drop_duplicates(["date", "ticker"], keep="first")
        out = out.drop(columns=["_positive_ohlc", "_volume_rank"])
    out = out.drop(columns=["_source_timestamp"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    if out.duplicated(["date", "ticker"]).any():
        raise ValueError("daily normalization did not produce unique date/ticker rows")
    return out, {
        "policy_id": "daily_session_highest_volume_v1",
        "manifest_row_scope": "non_missing_close",
        "collision_groups": collision_groups,
        "rows_removed": int(before - len(out)),
    }


__all__ = [
    "FactorPredictiveFoldEvaluator",
    "load_factor_development_data",
]
