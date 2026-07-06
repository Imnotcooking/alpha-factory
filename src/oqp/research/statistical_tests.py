from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats

from oqp.research.evaluation import EvaluationGeometry, RankICCalculator


@dataclass(frozen=True)
class StatisticalEvidence:
    raw_p_value: float
    metric_p_value: float
    hit_rate_p_value: float
    metric_observations: int
    hit_rate_observations: int
    test_method: str


class AlphaStatisticalTester:
    """Computes one-sided alpha evidence p-values without changing the IC metric."""

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        signal_col: str,
        return_col: str,
        geometry: EvaluationGeometry,
    ) -> StatisticalEvidence:
        if geometry == EvaluationGeometry.CROSS_SECTIONAL:
            metric_p, metric_n = self._cross_sectional_rank_ic_p_value(df, signal_col, return_col)
            method = "daily_rank_ic_ttest_greater"
        else:
            metric_p, metric_n = self._time_series_pearson_p_value(df, signal_col, return_col)
            method = "pearson_ic_ttest_greater"

        hit_p, hit_n = self._directional_hit_rate_p_value(df, signal_col, return_col)
        raw_p = metric_p if np.isfinite(metric_p) else hit_p
        return StatisticalEvidence(
            raw_p_value=float(raw_p) if np.isfinite(raw_p) else np.nan,
            metric_p_value=float(metric_p) if np.isfinite(metric_p) else np.nan,
            hit_rate_p_value=float(hit_p) if np.isfinite(hit_p) else np.nan,
            metric_observations=int(metric_n),
            hit_rate_observations=int(hit_n),
            test_method=method,
        )

    @staticmethod
    def _cross_sectional_rank_ic_p_value(
        df: pd.DataFrame,
        signal_col: str,
        return_col: str,
    ) -> tuple[float, int]:
        if df.empty or "date" not in df.columns:
            return np.nan, 0

        daily_ics: list[float] = []
        for _, day in df.groupby("date", sort=False):
            valid = RankICCalculator._valid_frame(day, signal_col, return_col)
            if len(valid) < 3:
                continue
            if valid[signal_col].nunique() < 2 or valid[return_col].nunique() < 2:
                continue
            ic, _ = stats.spearmanr(valid[signal_col], valid[return_col])
            if np.isfinite(ic):
                daily_ics.append(float(ic))

        if len(daily_ics) < 2:
            return np.nan, len(daily_ics)
        if float(np.std(daily_ics, ddof=1)) < 1e-12:
            mean_ic = float(np.mean(daily_ics))
            if mean_ic > 0:
                return 0.0, len(daily_ics)
            if mean_ic < 0:
                return 1.0, len(daily_ics)
            return 0.5, len(daily_ics)
        result = stats.ttest_1samp(daily_ics, popmean=0.0, alternative="greater")
        p_value = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
        return p_value, len(daily_ics)

    @staticmethod
    def _time_series_pearson_p_value(
        df: pd.DataFrame,
        signal_col: str,
        return_col: str,
    ) -> tuple[float, int]:
        valid = RankICCalculator._valid_frame(df, signal_col, return_col)
        if len(valid) < 10:
            return np.nan, len(valid)
        if valid[signal_col].nunique() < 2 or valid[return_col].nunique() < 2:
            return np.nan, len(valid)

        r = float(valid[signal_col].corr(valid[return_col], method="pearson"))
        if not np.isfinite(r):
            return np.nan, len(valid)
        r = float(np.clip(r, -0.999999999, 0.999999999))
        t_stat = r * np.sqrt((len(valid) - 2) / max(1e-12, 1.0 - r * r))
        p_value = float(stats.t.sf(t_stat, df=len(valid) - 2))
        return p_value, len(valid)

    @staticmethod
    def _directional_hit_rate_p_value(
        df: pd.DataFrame,
        signal_col: str,
        return_col: str,
    ) -> tuple[float, int]:
        valid = RankICCalculator._valid_frame(df, signal_col, return_col)
        valid = valid[valid[signal_col].abs() > 1e-12]
        valid = valid[valid[return_col].abs() > 1e-12]
        if len(valid) < 10:
            return np.nan, len(valid)
        hits = np.sign(valid[signal_col]) == np.sign(valid[return_col])
        wins = int(hits.sum())
        result = stats.binomtest(wins, n=len(valid), p=0.5, alternative="greater")
        return float(result.pvalue), len(valid)


def sharpe_p_value_from_returns(returns: pd.Series | np.ndarray) -> tuple[float, int]:
    """One-sided t-test that average realized return is greater than zero."""

    values = pd.to_numeric(pd.Series(returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values.abs() > 1e-12]
    if len(values) < 3:
        return np.nan, len(values)
    result = stats.ttest_1samp(values.to_numpy(dtype=float), popmean=0.0, alternative="greater")
    p_value = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
    return p_value, len(values)
