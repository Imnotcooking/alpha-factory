"""Out-of-sample mean decrease accuracy for ML feature governance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import clone


@dataclass(frozen=True)
class PurgedMDAConfig:
    """Configuration for purged walk-forward MDA feature audits."""

    n_splits: int = 5
    embargo_days: int = 5
    max_rows: int = 50_000
    max_features: int = 25
    permutation_repeats: int = 1
    random_state: int = 42
    min_train_rows: int = 200
    min_test_rows: int = 50
    min_assets_per_day: int = 3


def default_xgb_regressor(random_state: int = 42):
    """Return the default XGBoost regressor, loading xgboost only on demand."""

    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def build_purged_time_folds(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    n_splits: int = 5,
    embargo_days: int = 5,
    min_train_rows: int = 200,
    min_test_rows: int = 50,
) -> list[dict[str, Any]]:
    """Build time folds with an embargo window around each test block."""

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    unique_dates = pd.Index(sorted(dates.dropna().unique()))
    if len(unique_dates) < max(2, n_splits):
        raise ValueError(
            f"Not enough unique dates for {n_splits} folds: {len(unique_dates)}"
        )

    folds = []
    date_blocks = [block for block in np.array_split(unique_dates, n_splits) if len(block)]
    for fold_idx, test_dates in enumerate(date_blocks, start=1):
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])
        embargo_delta = pd.Timedelta(days=int(embargo_days))
        train_mask = (dates < test_start - embargo_delta) | (
            dates > test_end + embargo_delta
        )
        test_mask = dates.isin(test_dates)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) < min_train_rows or len(test_idx) < min_test_rows:
            continue
        folds.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
        )
    if not folds:
        raise ValueError("No valid purged folds after applying embargo/min-row constraints.")
    return folds


def compute_oos_mda(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    estimator=None,
    config: PurgedMDAConfig | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Compute purged out-of-sample MDA for feature importance validation.

    MDA is measured as the drop in out-of-sample rank IC after one feature is
    shuffled inside each held-out fold. Positive values mean the model needed
    that feature on unseen data.
    """

    cfg = config or PurgedMDAConfig()
    features = [feature for feature in feature_cols if feature in df.columns]
    if not features:
        raise ValueError("No requested feature columns exist in the matrix.")
    features = features[: max(1, int(cfg.max_features))]
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    work = _prepare_dataset(
        df,
        features=features,
        target_col=target_col,
        date_col=date_col,
        ticker_col=ticker_col,
        max_rows=cfg.max_rows,
    )
    folds = build_purged_time_folds(
        work,
        date_col=date_col,
        n_splits=cfg.n_splits,
        embargo_days=cfg.embargo_days,
        min_train_rows=cfg.min_train_rows,
        min_test_rows=cfg.min_test_rows,
    )

    rng = np.random.default_rng(cfg.random_state)
    base_estimator = estimator if estimator is not None else default_xgb_regressor(cfg.random_state)
    fold_rows: list[dict] = []
    mda_rows: list[dict] = []
    gain_rows: list[dict] = []

    for fold in folds:
        model = clone(base_estimator)
        train_df = work.iloc[fold["train_idx"]]
        test_df = work.iloc[fold["test_idx"]]
        x_train = train_df[features]
        y_train = train_df[target_col]
        x_test = test_df[features]
        y_test = test_df[target_col]

        model.fit(x_train, y_train)
        baseline_pred = model.predict(x_test)
        baseline_score = rank_ic_score(
            y_test,
            baseline_pred,
            dates=test_df[date_col] if date_col in test_df.columns else None,
            min_assets_per_day=cfg.min_assets_per_day,
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "train_rows": fold["train_rows"],
                "test_rows": fold["test_rows"],
                "baseline_score": baseline_score,
            }
        )

        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(features):
            for feature, importance in zip(features, importances):
                gain_rows.append(
                    {
                        "fold": fold["fold"],
                        "feature": feature,
                        "gain_importance": float(importance),
                    }
                )

        for feature in features:
            for repeat in range(int(cfg.permutation_repeats)):
                x_perm = x_test.copy()
                x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
                perm_pred = model.predict(x_perm)
                permuted_score = rank_ic_score(
                    y_test,
                    perm_pred,
                    dates=test_df[date_col] if date_col in test_df.columns else None,
                    min_assets_per_day=cfg.min_assets_per_day,
                )
                mda_rows.append(
                    {
                        "fold": fold["fold"],
                        "repeat": repeat + 1,
                        "feature": feature,
                        "baseline_score": baseline_score,
                        "permuted_score": permuted_score,
                        "mda": baseline_score - permuted_score
                        if np.isfinite(baseline_score) and np.isfinite(permuted_score)
                        else np.nan,
                    }
                )

    fold_scores = pd.DataFrame(fold_rows)
    mda_detail = pd.DataFrame(mda_rows)
    gain_detail = pd.DataFrame(gain_rows)
    summary = _summarize_mda(mda_detail, gain_detail)
    metadata = {
        "rows": int(len(work)),
        "features": int(len(features)),
        "folds": int(len(folds)),
        "n_splits_requested": int(cfg.n_splits),
        "embargo_days": int(cfg.embargo_days),
        "permutation_repeats": int(cfg.permutation_repeats),
        "target_col": target_col,
        "score": "oos_rank_ic_drop",
        "baseline_score_mean": float(fold_scores["baseline_score"].mean())
        if not fold_scores.empty
        else np.nan,
    }
    return {
        "summary": summary,
        "detail": mda_detail,
        "fold_scores": fold_scores,
        "gain_detail": gain_detail,
        "metadata": metadata,
    }


def rank_ic_score(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    dates: pd.Series | np.ndarray | None = None,
    min_assets_per_day: int = 3,
) -> float:
    """Compute cross-sectional mean Spearman rank IC."""

    score_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if dates is not None:
        score_df["date"] = pd.to_datetime(pd.Series(dates).to_numpy(), errors="coerce")
        daily_scores = []
        for _, day in score_df.dropna().groupby("date", sort=False):
            if len(day) < min_assets_per_day:
                continue
            if day["y_true"].nunique() < 2 or day["y_pred"].nunique() < 2:
                continue
            value, _ = stats.spearmanr(day["y_pred"], day["y_true"])
            if np.isfinite(value):
                daily_scores.append(float(value))
        return float(np.mean(daily_scores)) if daily_scores else np.nan

    score_df = score_df.dropna()
    if (
        len(score_df) < 3
        or score_df["y_true"].nunique() < 2
        or score_df["y_pred"].nunique() < 2
    ):
        return np.nan
    value, _ = stats.spearmanr(score_df["y_pred"], score_df["y_true"])
    return float(value) if np.isfinite(value) else np.nan


def _prepare_dataset(
    df: pd.DataFrame,
    *,
    features: list[str],
    target_col: str,
    date_col: str,
    ticker_col: str,
    max_rows: int,
) -> pd.DataFrame:
    required = [date_col, target_col, *features]
    if ticker_col in df.columns:
        required.insert(1, ticker_col)
    work = df[required].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if ticker_col in work.columns:
        work[ticker_col] = work[ticker_col].astype(str)
        work = work.sort_values([date_col, ticker_col]).reset_index(drop=True)
    else:
        work = work.sort_values(date_col).reset_index(drop=True)
    for col in [target_col, *features]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = (
        work.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[date_col, target_col, *features])
        .reset_index(drop=True)
    )
    if len(work) > max_rows:
        keep_idx = np.linspace(0, len(work) - 1, int(max_rows), dtype=int)
        work = work.iloc[keep_idx].reset_index(drop=True)
    if len(work) < 100:
        raise ValueError(f"Not enough valid rows for OOS MDA: {len(work):,}")
    return work


def _summarize_mda(mda_detail: pd.DataFrame, gain_detail: pd.DataFrame) -> pd.DataFrame:
    if mda_detail.empty:
        return pd.DataFrame()
    summary = (
        mda_detail.groupby("feature")
        .agg(
            mda_mean=("mda", "mean"),
            mda_std=("mda", "std"),
            mda_median=("mda", "median"),
            positive_fold_rate=("mda", lambda x: float((x > 0).mean())),
            baseline_score_mean=("baseline_score", "mean"),
            permuted_score_mean=("permuted_score", "mean"),
            observations=("mda", "count"),
        )
        .reset_index()
    )
    if not gain_detail.empty:
        gain_summary = (
            gain_detail.groupby("feature")
            .agg(gain_importance_mean=("gain_importance", "mean"))
            .reset_index()
        )
        summary = summary.merge(gain_summary, on="feature", how="left")
    else:
        summary["gain_importance_mean"] = np.nan

    summary["mda_rank"] = summary["mda_mean"].rank(ascending=False, method="min")
    summary["gain_rank"] = summary["gain_importance_mean"].rank(
        ascending=False,
        method="min",
    )
    summary["gain_minus_mda_rank"] = summary["gain_rank"] - summary["mda_rank"]
    summary["diagnosis"] = np.select(
        [
            (summary["mda_mean"] > 0) & (summary["positive_fold_rate"] >= 0.6),
            summary["mda_mean"] < 0,
            (summary["gain_importance_mean"].fillna(0) > 0)
            & (summary["mda_mean"].fillna(0) <= 0),
        ],
        ["OOS useful", "Negative MDA", "In-sample suspect"],
        default="Flat / uncertain",
    )
    return summary.sort_values(
        ["mda_mean", "positive_fold_rate"],
        ascending=False,
    ).reset_index(drop=True)


__all__ = [
    "PurgedMDAConfig",
    "build_purged_time_folds",
    "compute_oos_mda",
    "default_xgb_regressor",
    "rank_ic_score",
]
