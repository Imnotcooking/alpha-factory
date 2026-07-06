"""Hierarchical risk parity allocation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # deployment-safe fallback is inverse variance if scipy is unavailable
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover - depends on deployment environment
    linkage = None
    squareform = None


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute long-only HRP weights from an asset return matrix."""

    clean = _clean_returns(returns)
    if clean.empty:
        return pd.Series(dtype=float)
    if clean.shape[1] == 1 or linkage is None or squareform is None:
        return inverse_variance_weights(clean)

    cov = clean.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = clean.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    distance = np.sqrt((1.0 - corr) / 2.0)
    link = linkage(squareform(distance.values, checks=False), method="single")
    sort_order = _quasi_diag(link)
    ordered = [clean.columns[i] for i in sort_order]
    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            denom = left_var + right_var
            allocation = 0.5 if denom <= 0 else 1.0 - left_var / denom
            weights[left] *= allocation
            weights[right] *= 1.0 - allocation
            next_clusters.extend([left, right])
        clusters = next_clusters

    total = float(weights.sum())
    if total <= 0:
        return inverse_variance_weights(clean)
    return weights.reindex(clean.columns).fillna(0.0) / total


def inverse_variance_weights(returns: pd.DataFrame) -> pd.Series:
    clean = _clean_returns(returns)
    if clean.empty:
        return pd.Series(dtype=float)
    variance = clean.var().replace(0.0, np.nan)
    inv = (1.0 / variance).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(inv.sum())
    if total <= 0:
        return pd.Series(1.0 / len(clean.columns), index=clean.columns)
    return inv / total


def _cluster_variance(cov: pd.DataFrame, cluster: list[str]) -> float:
    sub_cov = cov.loc[cluster, cluster]
    weights = inverse_variance_weights_from_cov(sub_cov)
    return float(weights.T @ sub_cov.values @ weights)


def inverse_variance_weights_from_cov(cov: pd.DataFrame) -> np.ndarray:
    diag = np.diag(cov.values).astype(float)
    inv = 1.0 / np.where(diag <= 0, np.nan, diag)
    inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(inv.sum())
    if total <= 0:
        return np.repeat(1.0 / len(diag), len(diag))
    return inv / total


def _quasi_diag(link: np.ndarray) -> list[int]:
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= num_items]
        for index, cluster_id in clusters.items():
            children = link[int(cluster_id - num_items), :2]
            sort_ix.loc[index] = children[0]
            sort_ix.loc[index + 1] = children[1]
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return [int(value) for value in sort_ix.tolist()]


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    clean = returns.copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    clean = clean.fillna(0.0)
    return clean.loc[:, clean.std() > 0]
