"""Compose aligned factor scores into one auditable strategy signal."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import pandas as pd

from oqp.research.factor_portfolios.contracts import FactorPortfolioConfig
from oqp.research.factor_portfolios.normalization import normalize_factor_signal


@dataclass(frozen=True)
class CompositionResult:
    frame: pd.DataFrame
    normalized_columns: dict[str, str]
    contribution_columns: dict[str, str]
    configured_weights: dict[str, float]


class FactorPortfolioComposer:
    """Align, normalize and blend several factor frames on date and ticker."""

    key_columns = ("date", "ticker")

    def __init__(self, config: FactorPortfolioConfig):
        self.config = config

    def compose(
        self,
        base_frame: pd.DataFrame,
        factor_frames: Mapping[str, pd.DataFrame],
        *,
        signal_columns: Mapping[str, str] | None = None,
    ) -> CompositionResult:
        self._validate_base_frame(base_frame)
        self._validate_factor_set(factor_frames)
        self._validate_compatibility(factor_frames)

        signal_columns = dict(signal_columns or {})
        out = base_frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if out["date"].isna().any():
            raise ValueError("base frame contains invalid dates")

        configured_weights = self.config.normalized_weights
        normalized_columns: dict[str, str] = {}
        contribution_columns: dict[str, str] = {}
        availability_columns: list[str] = []

        for spec in self.config.factors:
            frame = factor_frames[spec.factor_id].copy()
            self._validate_factor_frame(spec.factor_id, frame)
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            signal_col = signal_columns.get(spec.factor_id) or spec.signal_col
            if not signal_col:
                signal_col = self._infer_signal_column(frame)
            if signal_col not in frame.columns:
                raise ValueError(
                    f"{spec.factor_id} is missing configured signal column {signal_col!r}"
                )

            suffix = _column_suffix(spec.factor_id)
            raw_col = f"factor__{suffix}__raw"
            normalized_col = f"factor__{suffix}__normalized"
            availability_col = f"factor__{suffix}__available"
            contribution_col = f"factor__{suffix}__contribution"
            factor_signal = frame[[*self.key_columns, signal_col]].rename(
                columns={signal_col: raw_col}
            )
            out = out.merge(
                factor_signal,
                on=list(self.key_columns),
                how="left",
                validate="one_to_one",
            )
            if "liquidity_eligible" in out.columns:
                out[raw_col] = out[raw_col].where(
                    out["liquidity_eligible"].fillna(False)
                )
            normalization = spec.normalization or self.config.normalization
            out[normalized_col] = normalize_factor_signal(
                out,
                raw_col,
                method=normalization,
                winsor_limit=self.config.winsor_limit,
            ) * spec.orientation
            out[availability_col] = out[normalized_col].notna()
            out[contribution_col] = (
                out[normalized_col] * configured_weights[spec.factor_id]
            )

            normalized_columns[spec.factor_id] = normalized_col
            contribution_columns[spec.factor_id] = contribution_col
            availability_columns.append(availability_col)

        out["available_factor_count"] = out[availability_columns].sum(axis=1).astype(int)
        out["factor_weight_coverage"] = 0.0
        for spec, available_col in zip(self.config.factors, availability_columns):
            out["factor_weight_coverage"] += (
                out[available_col].astype(float) * configured_weights[spec.factor_id]
            )

        raw_composite = out[list(contribution_columns.values())].sum(
            axis=1,
            min_count=1,
        )
        if self.config.missing_policy == "renormalize_available":
            denominator = out["factor_weight_coverage"].replace(0.0, np.nan)
            composite = raw_composite / denominator
        elif self.config.missing_policy == "complete_case":
            composite = raw_composite.where(
                out["available_factor_count"].eq(len(self.config.factors))
            )
        else:
            composite = raw_composite.fillna(0.0)

        minimum_mask = out["available_factor_count"].ge(
            self.config.min_available_factors
        )
        out["composite_score"] = composite.where(minimum_mask)

        if self.config.missing_policy == "renormalize_available":
            denominator = out["factor_weight_coverage"].replace(0.0, np.nan)
            for contribution_col in contribution_columns.values():
                out[contribution_col] = out[contribution_col] / denominator
                out.loc[~minimum_mask, contribution_col] = np.nan

        out.attrs.update(base_frame.attrs)
        out.attrs["strategy_id"] = self.config.strategy_id
        out.attrs["strategy_name"] = self.config.name
        out.attrs["market_vertical"] = self.config.market_vertical
        out.attrs["alpha_signal_col"] = "composite_score"
        out.attrs["factor_portfolio"] = self.config.to_dict()
        out.attrs["factor_params"] = {
            "component_type": "factor_portfolio",
            "factor_portfolio": self.config.to_dict(),
        }
        return CompositionResult(
            frame=out,
            normalized_columns=normalized_columns,
            contribution_columns=contribution_columns,
            configured_weights=configured_weights,
        )

    def _validate_base_frame(self, frame: pd.DataFrame) -> None:
        missing = [col for col in self.key_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"base frame is missing key columns: {', '.join(missing)}")
        if frame.duplicated(list(self.key_columns)).any():
            raise ValueError("base frame must contain one row per date and ticker")

    def _validate_factor_set(self, factor_frames: Mapping[str, pd.DataFrame]) -> None:
        expected = {spec.factor_id for spec in self.config.factors}
        provided = set(factor_frames)
        missing = sorted(expected - provided)
        if missing:
            raise ValueError(f"factor frames are missing: {', '.join(missing)}")

    def _validate_factor_frame(self, factor_id: str, frame: pd.DataFrame) -> None:
        missing = [col for col in self.key_columns if col not in frame.columns]
        if missing:
            raise ValueError(
                f"{factor_id} is missing key columns: {', '.join(missing)}"
            )
        if frame.duplicated(list(self.key_columns)).any():
            raise ValueError(f"{factor_id} must contain one row per date and ticker")

    def _validate_compatibility(
        self,
        factor_frames: Mapping[str, pd.DataFrame],
    ) -> None:
        tracked_fields = (
            "market_vertical",
            "data_frequency",
            "return_horizon",
            "evaluation_geometry",
            "execution_lag",
            "return_assumption",
        )
        for field in tracked_fields:
            declared = {
                str(frame.attrs.get(field)).strip()
                for frame in factor_frames.values()
                if frame.attrs.get(field) not in (None, "", "auto")
            }
            if len(declared) > 1:
                raise ValueError(
                    f"factor portfolio has incompatible {field} values: {sorted(declared)}"
                )
            if field == "market_vertical" and declared:
                actual = next(iter(declared))
                if actual != self.config.market_vertical:
                    raise ValueError(
                        "factor portfolio market mismatch: "
                        f"config={self.config.market_vertical}, frames={actual}"
                    )

    @staticmethod
    def _infer_signal_column(frame: pd.DataFrame) -> str:
        preferred = frame.attrs.get("alpha_signal_col")
        if preferred and preferred in frame.columns:
            return str(preferred)
        for candidate in (
            "factor_score",
            "raw_signal",
            "signal",
            "target_weight",
            "final_target_weight",
        ):
            if candidate in frame.columns:
                return candidate
        raise ValueError("factor frame has no recognizable signal column")


def _column_suffix(factor_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", factor_id).strip("_").lower()


__all__ = ["CompositionResult", "FactorPortfolioComposer"]
