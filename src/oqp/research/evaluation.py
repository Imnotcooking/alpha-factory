from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import scipy.stats as stats


class EvaluationGeometry(str, Enum):
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"


@dataclass(frozen=True)
class EvaluationMetricResult:
    validation_ic: float
    holdout_ic: float
    crisis_ic: float
    metric_name: str
    geometry: EvaluationGeometry
    signal_col: str
    validation_hit_rate: float | None = None
    holdout_hit_rate: float | None = None
    crisis_hit_rate: float | None = None


class StrategyGeometryClassifier:
    """Infers whether the alpha should be judged cross-sectionally or through time."""

    CROSS_SECTIONAL_HINTS = (
        "cross",
        "cross-sectional",
        "cross_sectional",
        "stat",
        "statistical arbitrage",
        "pairs",
        "pair",
        "sector",
        "relative",
        "basis",
        "value",
        "carry",
        "cs_",
    )
    TIME_SERIES_HINTS = (
        "time",
        "time-series",
        "time_series",
        "tsmom",
        "cta",
        "trend following",
        "directional",
        "sma",
        "bollinger",
        "breakout",
        "donchian",
        "atr",
        "tick",
        "microstructure",
        "order flow",
        "intraday",
        "overnight",
        "pulse",
    )

    def infer(
        self,
        df: pd.DataFrame,
        *,
        factor_id: str = "",
        category: str = "",
        explicit: str | EvaluationGeometry | None = None,
    ) -> EvaluationGeometry:
        if explicit:
            return self._normalize(explicit)

        attr_value = df.attrs.get("evaluation_geometry") or df.attrs.get("strategy_geometry")
        if attr_value:
            return self._normalize(attr_value)

        ticker_count = df["ticker"].nunique() if "ticker" in df.columns else 0
        if ticker_count <= 1:
            return EvaluationGeometry.TIME_SERIES

        text = f"{factor_id} {category}".lower()
        if any(hint in text for hint in self.CROSS_SECTIONAL_HINTS):
            return EvaluationGeometry.CROSS_SECTIONAL
        if any(hint in text for hint in self.TIME_SERIES_HINTS):
            return EvaluationGeometry.TIME_SERIES

        if self._has_cross_sectional_breadth(df):
            return EvaluationGeometry.CROSS_SECTIONAL
        return EvaluationGeometry.TIME_SERIES

    @staticmethod
    def _normalize(value: str | EvaluationGeometry) -> EvaluationGeometry:
        if isinstance(value, EvaluationGeometry):
            return value
        value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "cs": EvaluationGeometry.CROSS_SECTIONAL,
            "cross": EvaluationGeometry.CROSS_SECTIONAL,
            "cross_section": EvaluationGeometry.CROSS_SECTIONAL,
            "cross_sectional": EvaluationGeometry.CROSS_SECTIONAL,
            "rank": EvaluationGeometry.CROSS_SECTIONAL,
            "rank_ic": EvaluationGeometry.CROSS_SECTIONAL,
            "ts": EvaluationGeometry.TIME_SERIES,
            "time": EvaluationGeometry.TIME_SERIES,
            "time_series": EvaluationGeometry.TIME_SERIES,
            "timeseries": EvaluationGeometry.TIME_SERIES,
            "pearson": EvaluationGeometry.TIME_SERIES,
            "pearson_ic": EvaluationGeometry.TIME_SERIES,
        }
        if value not in aliases:
            raise ValueError(f"Unknown evaluation geometry: {value!r}")
        return aliases[value]

    @staticmethod
    def _has_cross_sectional_breadth(df: pd.DataFrame) -> bool:
        if "date" not in df.columns or "ticker" not in df.columns:
            return False
        counts = df.groupby("date")["ticker"].nunique()
        return bool((counts >= 3).mean() >= 0.5) if not counts.empty else False


class RankICCalculator:
    metric_name = "rank_ic_spearman"

    def calculate(self, df: pd.DataFrame, signal_col: str, return_col: str = "forward_return") -> float:
        if df.empty or signal_col not in df.columns or return_col not in df.columns or "date" not in df.columns:
            return np.nan

        daily_ics: list[float] = []
        for _, day in df.groupby("date", sort=False):
            valid = self._valid_frame(day, signal_col, return_col)
            if len(valid) < 3:
                continue
            if valid[signal_col].nunique() < 2 or valid[return_col].nunique() < 2:
                continue
            ic, _ = stats.spearmanr(valid[signal_col], valid[return_col])
            if np.isfinite(ic):
                daily_ics.append(float(ic))

        return float(np.mean(daily_ics)) if daily_ics else np.nan

    @staticmethod
    def _valid_frame(df: pd.DataFrame, signal_col: str, return_col: str) -> pd.DataFrame:
        out = df[[signal_col, return_col]].copy()
        out[signal_col] = pd.to_numeric(out[signal_col], errors="coerce")
        out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan).dropna()


class PearsonICCalculator:
    metric_name = "pearson_ic_time_series"

    def calculate(self, df: pd.DataFrame, signal_col: str, return_col: str = "forward_return") -> float:
        if df.empty or signal_col not in df.columns or return_col not in df.columns:
            return np.nan

        if "ticker" not in df.columns:
            return self._pearson(df, signal_col, return_col)

        weighted_sum = 0.0
        total_weight = 0
        for _, group in df.groupby("ticker", sort=False):
            ic = self._pearson(group, signal_col, return_col)
            valid_count = self._valid_count(group, signal_col, return_col)
            if np.isfinite(ic) and valid_count >= 10:
                weighted_sum += ic * valid_count
                total_weight += valid_count

        return float(weighted_sum / total_weight) if total_weight else np.nan

    def _pearson(self, df: pd.DataFrame, signal_col: str, return_col: str) -> float:
        valid = RankICCalculator._valid_frame(df, signal_col, return_col)
        if len(valid) < 10:
            return np.nan
        if valid[signal_col].nunique() < 2 or valid[return_col].nunique() < 2:
            return np.nan
        ic, _ = stats.pearsonr(valid[signal_col], valid[return_col])
        return float(ic) if np.isfinite(ic) else np.nan

    @staticmethod
    def _valid_count(df: pd.DataFrame, signal_col: str, return_col: str) -> int:
        return len(RankICCalculator._valid_frame(df, signal_col, return_col))


class DirectionalHitRateCalculator:
    metric_name = "directional_hit_rate"

    def calculate(self, df: pd.DataFrame, signal_col: str, return_col: str = "forward_return") -> float:
        if df.empty or signal_col not in df.columns or return_col not in df.columns:
            return np.nan
        valid = RankICCalculator._valid_frame(df, signal_col, return_col)
        valid = valid[valid[signal_col].abs() > 1e-12]
        valid = valid[valid[return_col].abs() > 1e-12]
        if len(valid) < 10:
            return np.nan
        hits = np.sign(valid[signal_col]) == np.sign(valid[return_col])
        return float(hits.mean())


class AlphaMetricEvaluator:
    """Chooses and runs the correct alpha evidence metric for the strategy geometry."""

    def __init__(self):
        self.classifier = StrategyGeometryClassifier()
        self.rank_ic = RankICCalculator()
        self.pearson_ic = PearsonICCalculator()
        self.hit_rate = DirectionalHitRateCalculator()

    def evaluate(
        self,
        *,
        factor_id: str,
        df: pd.DataFrame,
        validation_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        crisis_data: pd.DataFrame,
        signal_col: str,
        return_col: str = "forward_return",
        category: str = "",
        explicit_geometry: str | EvaluationGeometry | None = None,
    ) -> EvaluationMetricResult:
        geometry = self.classifier.infer(
            df,
            factor_id=factor_id,
            category=category,
            explicit=explicit_geometry,
        )
        calculator = self.rank_ic if geometry == EvaluationGeometry.CROSS_SECTIONAL else self.pearson_ic

        return EvaluationMetricResult(
            validation_ic=self._finite_or_nan(calculator.calculate(validation_data, signal_col, return_col)),
            holdout_ic=self._finite_or_nan(calculator.calculate(holdout_data, signal_col, return_col)),
            crisis_ic=self._finite_or_nan(calculator.calculate(crisis_data, signal_col, return_col)),
            metric_name=calculator.metric_name,
            geometry=geometry,
            signal_col=signal_col,
            validation_hit_rate=self._finite_or_none(self.hit_rate.calculate(validation_data, signal_col, return_col)),
            holdout_hit_rate=self._finite_or_none(self.hit_rate.calculate(holdout_data, signal_col, return_col)),
            crisis_hit_rate=self._finite_or_none(self.hit_rate.calculate(crisis_data, signal_col, return_col)),
        )

    @staticmethod
    def _finite_or_nan(value: float) -> float:
        return float(value) if np.isfinite(value) else np.nan

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None
