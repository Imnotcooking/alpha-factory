"""Portfolio sizing helpers used by research execution modes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    """Allocate risk budgets from a return-correlation tree."""

    def _get_inverse_variance_weights(self, cov, assets):
        variances = np.asarray(np.diag(cov), dtype=float)
        good_variances = variances[np.isfinite(variances) & (variances > 1e-12)]
        if len(good_variances) == 0:
            return pd.Series(1.0 / len(assets), index=assets)

        variance_floor = max(np.median(good_variances), 1e-6)
        safe_variances = np.where(
            np.isfinite(variances) & (variances > 1e-12),
            variances,
            variance_floor,
        )
        ivp = 1.0 / safe_variances
        ivp_sum = ivp.sum()
        if not np.isfinite(ivp_sum) or ivp_sum <= 0:
            return pd.Series(1.0 / len(assets), index=assets)
        ivp /= ivp_sum
        return pd.Series(ivp, index=assets)

    def _get_cluster_var(self, cov, c_items):
        cov_slice = cov.loc[c_items, c_items]
        weights = self._get_inverse_variance_weights(cov_slice, c_items)
        return np.dot(weights.T, np.dot(cov_slice, weights))

    def _get_rec_bipart(self, cov, sort_ix):
        weights = pd.Series(1.0, index=sort_ix)
        clusters = [sort_ix]

        while len(clusters) > 0:
            clusters = [
                item[j:k]
                for item in clusters
                for j, k in ((0, len(item) // 2), (len(item) // 2, len(item)))
                if len(item) > 1
            ]
            for idx in range(0, len(clusters), 2):
                left = clusters[idx]
                right = clusters[idx + 1]
                left_var = self._get_cluster_var(cov, left)
                right_var = self._get_cluster_var(cov, right)
                left_var = left_var if np.isfinite(left_var) and left_var > 0 else 0.0
                right_var = right_var if np.isfinite(right_var) and right_var > 0 else 0.0

                total_var = left_var + right_var
                alpha = 0.5 if total_var <= 0 else 1.0 - left_var / total_var
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
        return weights

    def compute_weights(self, returns_wide: pd.DataFrame) -> pd.Series:
        returns_wide = (
            returns_wide.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        if len(returns_wide.columns) == 1:
            return pd.Series(1.0, index=returns_wide.columns)
        if len(returns_wide.columns) == 0:
            return pd.Series(dtype=float)

        cov = returns_wide.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr = returns_wide.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr_values = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_values, 1.0)
        corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)

        distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        link = linkage(squareform(distance), method="single")
        sort_ix = leaves_list(link)
        ordered_tickers = returns_wide.columns[sort_ix].tolist()
        return self._get_rec_bipart(cov, ordered_tickers)


class KellySizer:
    """Translate signal edge and local variance into fractional Kelly weights."""

    def __init__(self, kelly_fraction=0.5):
        self.kelly_fraction = kelly_fraction

    def compute_weights(self, df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        out = df.copy()
        out["local_ret"] = out.groupby("ticker")["close"].pct_change()
        out["local_var"] = out.groupby("ticker")["local_ret"].transform(
            lambda values: values.rolling(20, min_periods=5).var()
        )
        out["local_var"] = out["local_var"].fillna(0.0001).clip(lower=1e-6)
        out["kelly_weight"] = out[signal_col] / out["local_var"]
        out["kelly_weight"] = out["kelly_weight"] * self.kelly_fraction
        return out.drop(columns=["local_ret", "local_var"])


class CasinoCapEnforcer:
    """Clip per-asset weights and cap portfolio gross leverage."""

    def __init__(self, max_weight_per_asset=0.05, max_gross_leverage=1.0):
        self.max_weight = max_weight_per_asset
        self.max_leverage = max_gross_leverage

    def enforce(self, df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
        out = df.copy()
        weights = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)
        if self.max_weight is not None and self.max_weight > 0:
            weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)
        out["capped_weight"] = weights
        daily_gross = out.groupby("date")["capped_weight"].transform(
            lambda values: values.abs().sum()
        )
        max_leverage = max(float(self.max_leverage), 0.0)
        shrink_factor = np.where(
            daily_gross > max_leverage,
            max_leverage / daily_gross.replace(0.0, np.nan),
            1.0,
        )
        shrink_factor = (
            pd.Series(shrink_factor, index=out.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        out["final_target_weight"] = out["capped_weight"] * shrink_factor
        return out.drop(columns=["capped_weight"])


class PortfolioOptimizer:
    """Fuse Kelly alpha scaling with HRP risk budgets and portfolio caps."""

    def __init__(self, kelly_fraction=0.5, max_weight=0.05, max_gross_leverage=1.0):
        self.hrp = HierarchicalRiskParity()
        self.kelly = KellySizer(kelly_fraction=kelly_fraction)
        self.cap = CasinoCapEnforcer(
            max_weight_per_asset=max_weight,
            max_gross_leverage=max_gross_leverage,
        )

    def optimize(
        self,
        daily_signals_df: pd.DataFrame,
        trailing_returns_wide: pd.DataFrame,
    ) -> pd.DataFrame:
        out = daily_signals_df.copy()
        hrp_budgets = self.hrp.compute_weights(trailing_returns_wide)
        out["hrp_budget"] = out["ticker"].map(hrp_budgets).fillna(0.0)
        out = self.kelly.compute_weights(out, "raw_signal")
        out["synthesized_weight"] = out["kelly_weight"] * out["hrp_budget"]
        return self.cap.enforce(out, "synthesized_weight")


__all__ = [
    "CasinoCapEnforcer",
    "HierarchicalRiskParity",
    "KellySizer",
    "PortfolioOptimizer",
]
