"""Feature-governance diagnostics for machine-learning research matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_PREFIXES = ("f_", "prob_")
EXPLICIT_ENGINEERED_FEATURES = frozenset(
    {
        "amihud_z",
        "gk_vol_z",
        "ker_20d",
    }
)
DEFAULT_CORR_THRESHOLD = 0.85
DEFAULT_MIN_ASSETS_PER_DAY = 5
DEFAULT_MAX_CORR_ROWS = 50_000
DEFAULT_CORR_MIN_PERIODS = 50


FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Regime Probability", ("prob_", "regime", "state")),
    ("Liquidity", ("amihud", "liq", "illiquid", "spread", "depth")),
    ("Volatility", ("vol", "gk", "hurst", "natr", "range", "variance")),
    ("Momentum", ("mom", "ret", "trend", "breakout")),
    ("Value / Reversion", ("value", "mean", "clv", "reversion", "z_sector")),
    ("Volume / Flow", ("volume", "vol_climax", "turnover", "flow")),
    ("Open Interest", ("oi", "open_interest")),
    ("Efficiency", ("ker", "efficiency")),
)


@dataclass(frozen=True)
class FeatureGovernanceConfig:
    """Configuration for feature-matrix audit diagnostics."""

    target_col: str = "target_1d_rank"
    include_prob_features: bool = True
    corr_threshold: float = DEFAULT_CORR_THRESHOLD
    min_assets_per_day: int = DEFAULT_MIN_ASSETS_PER_DAY
    max_corr_rows: int = DEFAULT_MAX_CORR_ROWS
    corr_min_periods: int = DEFAULT_CORR_MIN_PERIODS
    random_state: int = 42


def list_matrix_files(base_dir: str | Path) -> list[Path]:
    """Return likely feature-matrix parquet files below a research directory."""

    base_path = Path(base_dir)
    patterns = [
        "*Feature_Matrix*.parquet",
        "*Stacked_Matrix*.parquet",
        "ML_Feature_Matrix.parquet",
    ]
    seen: dict[Path, None] = {}
    repo_root = base_path
    runtime_feature_store = repo_root / "runtime" / "data" / "feature_store"
    for pattern in patterns:
        for path in sorted(runtime_feature_store.glob(pattern) if runtime_feature_store.exists() else []):
            if path.is_file():
                seen[path.resolve()] = None
    return list(seen.keys())


def load_matrix(matrix_path: str | Path) -> pd.DataFrame:
    """Load a feature matrix and normalize the date column when present."""

    df = pd.read_parquet(matrix_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def detect_feature_columns(
    df: pd.DataFrame,
    include_prob_features: bool = True,
) -> list[str]:
    """Detect numeric engineered feature columns by naming convention."""

    return sorted(
        col
        for col in df.columns
        if _is_engineered_feature_column(col, include_prob_features)
        and pd.api.types.is_numeric_dtype(df[col])
    )


def tag_feature_family(feature: str) -> str:
    """Map a feature name to a broad, interpretable research family."""

    lower = feature.lower()
    for family, markers in FAMILY_RULES:
        if any(marker in lower for marker in markers):
            return family
    return "Other"


def compute_feature_governance(
    df: pd.DataFrame,
    config: FeatureGovernanceConfig | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Compute feature diagnostics for an ML matrix.

    The raw matrix is never mutated. The returned artifacts are intended for
    dashboards, promotion gates, model-research notebooks, and reproducible
    feature audit records.
    """

    cfg = config or FeatureGovernanceConfig()
    work = _prepare_frame(df)
    features = detect_feature_columns(work, cfg.include_prob_features)
    if not features:
        raise ValueError("No numeric engineered features found.")
    if cfg.target_col not in work.columns:
        raise ValueError(f"Target column not found: {cfg.target_col}")

    summary = _feature_summary(work, features)
    ic_summary, daily_ic = _compute_ic_summary(
        work,
        features,
        cfg.target_col,
        cfg.min_assets_per_day,
    )
    turnover = _compute_turnover_proxy(work, features)
    summary = (
        summary.merge(ic_summary, on="feature", how="left")
        .merge(turnover, on="feature", how="left")
    )
    summary["family"] = summary["feature"].map(tag_feature_family)
    summary["quality_score"] = _quality_score(summary)
    summary = summary.sort_values(
        ["quality_score", "abs_mean_ic"],
        ascending=False,
    ).reset_index(drop=True)

    corr_matrix = _compute_corr_matrix(
        work,
        features,
        cfg.max_corr_rows,
        cfg.random_state,
        cfg.corr_min_periods,
    )
    corr_pairs = _high_corr_pairs(corr_matrix, cfg.corr_threshold)
    clusters = _correlation_clusters(corr_matrix, summary, cfg.corr_threshold)
    pca_variance, pca_loadings = _compute_pca_baseline(
        work,
        features,
        cfg.max_corr_rows,
        cfg.random_state,
    )

    metadata = {
        "rows": int(len(work)),
        "features": int(len(features)),
        "assets": int(work["ticker"].nunique()) if "ticker" in work.columns else 0,
        "target_col": cfg.target_col,
        "date_min": work["date"].min() if "date" in work.columns else None,
        "date_max": work["date"].max() if "date" in work.columns else None,
        "corr_threshold": float(cfg.corr_threshold),
        "min_assets_per_day": int(cfg.min_assets_per_day),
    }

    family_counts = (
        summary.groupby("family", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            avg_abs_ic=("abs_mean_ic", "mean"),
            avg_turnover_proxy=("turnover_proxy", "mean"),
            avg_missing_pct=("missing_pct", "mean"),
        )
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )

    keeper_features = _keeper_shortlist(clusters, summary)

    return {
        "metadata": metadata,
        "summary": summary,
        "daily_ic": daily_ic,
        "corr_matrix": corr_matrix,
        "corr_pairs": corr_pairs,
        "clusters": clusters,
        "family_counts": family_counts,
        "pca_variance": pca_variance,
        "pca_loadings": pca_loadings,
        "keeper_features": keeper_features,
    }


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected columns coerced to numeric dtype."""

    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _is_engineered_feature_column(
    column_name: str,
    include_prob_features: bool = True,
) -> bool:
    if column_name in EXPLICIT_ENGINEERED_FEATURES:
        return True
    if column_name.startswith("f_"):
        return True
    return include_prob_features and column_name.startswith("prob_")


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if "ticker" in work.columns:
        work["ticker"] = work["ticker"].astype(str)
    if {"ticker", "date"}.issubset(work.columns):
        work = work.sort_values(["ticker", "date"]).reset_index(drop=True)
    return work


def _feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": feature,
                "missing_pct": float(finite.isna().mean()),
                "coverage_pct": float(1.0 - finite.isna().mean()),
                "zero_pct": float((finite.fillna(np.nan) == 0).sum() / total),
                "unique_count": int(finite.nunique(dropna=True)),
                "mean": float(finite.mean()) if finite.notna().any() else np.nan,
                "std": float(finite.std()) if finite.notna().any() else np.nan,
                "min": float(finite.min()) if finite.notna().any() else np.nan,
                "max": float(finite.max()) if finite.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _compute_ic_summary(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    min_assets_per_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"date", target_col}.issubset(df.columns):
        empty_summary = pd.DataFrame(
            {
                "feature": features,
                "mean_ic": np.nan,
                "abs_mean_ic": np.nan,
                "ic_std": np.nan,
                "ic_ir": np.nan,
                "positive_day_rate": np.nan,
                "valid_ic_days": 0,
            }
        )
        return empty_summary, pd.DataFrame()

    daily_rows = []
    cols = [*features, target_col]
    for date, day in df[["date", *cols]].groupby("date", sort=False):
        day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col])
        if len(day) < min_assets_per_day or day[target_col].nunique() < 2:
            continue

        active_features = [
            feature
            for feature in features
            if day[feature].dropna().nunique() > 1
        ]
        if not active_features:
            continue

        corr = day[active_features].corrwith(day[target_col], method="spearman")
        for feature, ic in corr.items():
            if pd.notna(ic) and np.isfinite(ic):
                daily_rows.append({"date": date, "feature": feature, "ic": float(ic)})

    daily_ic = pd.DataFrame(daily_rows)
    if daily_ic.empty:
        summary = pd.DataFrame({"feature": features})
        summary["mean_ic"] = np.nan
        summary["abs_mean_ic"] = np.nan
        summary["ic_std"] = np.nan
        summary["ic_ir"] = np.nan
        summary["positive_day_rate"] = np.nan
        summary["valid_ic_days"] = 0
        return summary, daily_ic

    summary = (
        daily_ic.groupby("feature")
        .agg(
            mean_ic=("ic", "mean"),
            ic_std=("ic", "std"),
            positive_day_rate=("ic", lambda x: float((x > 0).mean())),
            valid_ic_days=("ic", "count"),
        )
        .reset_index()
    )
    summary["abs_mean_ic"] = summary["mean_ic"].abs()
    summary["ic_ir"] = np.where(
        summary["ic_std"].abs() > 1e-12,
        summary["mean_ic"] / summary["ic_std"],
        0.0,
    )
    return summary, daily_ic


def _compute_turnover_proxy(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not {"date", "ticker"}.issubset(df.columns):
        return pd.DataFrame({"feature": features, "turnover_proxy": np.nan})

    rows = []
    ordered = df[["date", "ticker", *features]].sort_values(["ticker", "date"])
    for feature in features:
        ranks = (
            ordered[["date", "ticker", feature]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[feature])
            .copy()
        )
        if ranks.empty:
            rows.append({"feature": feature, "turnover_proxy": np.nan})
            continue

        ranks["_rank"] = ranks.groupby("date")[feature].rank(pct=True, method="average")
        rank_change = ranks.groupby("ticker")["_rank"].diff().abs()
        rows.append(
            {
                "feature": feature,
                "turnover_proxy": float(rank_change.mean())
                if rank_change.notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _compute_corr_matrix(
    df: pd.DataFrame,
    features: list[str],
    max_rows: int,
    random_state: int,
    min_periods: int,
) -> pd.DataFrame:
    corr_input = df[features].replace([np.inf, -np.inf], np.nan)
    if len(corr_input) > max_rows:
        corr_input = corr_input.sample(max_rows, random_state=random_state)
    return corr_input.corr(method="spearman", min_periods=min_periods)


def _high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman_corr": float(value),
                        "abs_corr": float(abs(value)),
                    }
                )
    output_columns = ["feature_a", "feature_b", "spearman_corr", "abs_corr"]
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )


def _correlation_clusters(
    corr: pd.DataFrame,
    summary: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    features = list(corr.columns)
    graph = {feature: set() for feature in features}
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                graph[left].add(right)
                graph[right].add(left)

    summary_lookup = summary.set_index("feature")
    visited: set[str] = set()
    rows = []
    cluster_id = 1
    for feature in features:
        if feature in visited:
            continue
        stack = [feature]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(graph[current] - visited))

        comp_df = summary_lookup.loc[component].copy()
        comp_df["_abs_ic_sort"] = comp_df["abs_mean_ic"].fillna(-1.0)
        comp_df["_missing_sort"] = comp_df["missing_pct"].fillna(1.0)
        comp_df = comp_df.sort_values(
            ["_abs_ic_sort", "_missing_sort"],
            ascending=[False, True],
        )
        representative = comp_df.index[0]

        for member in sorted(component):
            peers = [peer for peer in component if peer != member]
            max_peer_corr = (
                float(corr.loc[member, peers].abs().max()) if peers else 0.0
            )
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(component),
                    "representative": representative,
                    "feature": member,
                    "family": tag_feature_family(member),
                    "mean_ic": summary_lookup.loc[member, "mean_ic"],
                    "abs_mean_ic": summary_lookup.loc[member, "abs_mean_ic"],
                    "missing_pct": summary_lookup.loc[member, "missing_pct"],
                    "turnover_proxy": summary_lookup.loc[member, "turnover_proxy"],
                    "max_abs_corr_to_peer": max_peer_corr,
                    "keep_candidate": member == representative,
                }
            )
        cluster_id += 1

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["cluster_size", "cluster_id", "keep_candidate"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )


def _compute_pca_baseline(
    df: pd.DataFrame,
    features: list[str],
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    x = df[features].replace([np.inf, -np.inf], np.nan).copy()
    valid_cols = [
        col
        for col in features
        if x[col].notna().sum() >= 50 and x[col].nunique(dropna=True) > 1
    ]
    if len(valid_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    x = x[valid_cols]
    x = x.fillna(x.median(numeric_only=True))
    if len(x) > max_rows:
        x = x.sample(max_rows, random_state=random_state)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    n_components = min(len(valid_cols), 12)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(x_scaled)

    variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_cols,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    ).reset_index(names="feature")
    return variance, loadings


def _quality_score(summary: pd.DataFrame) -> pd.Series:
    abs_ic = summary["abs_mean_ic"].fillna(0.0)
    coverage = summary["coverage_pct"].fillna(0.0)
    turnover_median = summary["turnover_proxy"].median()
    if pd.isna(turnover_median):
        turnover_median = 0.5
    turnover = summary["turnover_proxy"].fillna(turnover_median)
    turnover_penalty = turnover.rank(pct=True).fillna(0.5)
    return (
        (abs_ic.rank(pct=True) * 0.55)
        + (coverage * 0.25)
        + ((1.0 - turnover_penalty) * 0.20)
    )


def _keeper_shortlist(clusters: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return pd.DataFrame()
    keepers = clusters[clusters["keep_candidate"]].copy()
    keepers = keepers.merge(
        summary[
            [
                "feature",
                "family",
                "mean_ic",
                "abs_mean_ic",
                "ic_ir",
                "positive_day_rate",
                "missing_pct",
                "turnover_proxy",
                "quality_score",
            ]
        ],
        on=["feature", "family"],
        how="left",
        suffixes=("", "_summary"),
    )
    return keepers.sort_values("quality_score", ascending=False).reset_index(drop=True)


__all__ = [
    "DEFAULT_CORR_MIN_PERIODS",
    "DEFAULT_CORR_THRESHOLD",
    "DEFAULT_MAX_CORR_ROWS",
    "DEFAULT_MIN_ASSETS_PER_DAY",
    "EXPLICIT_ENGINEERED_FEATURES",
    "FAMILY_RULES",
    "FEATURE_PREFIXES",
    "FeatureGovernanceConfig",
    "coerce_numeric_columns",
    "compute_feature_governance",
    "detect_feature_columns",
    "list_matrix_files",
    "load_matrix",
    "tag_feature_family",
]
