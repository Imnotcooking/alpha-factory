from __future__ import annotations

import numpy as np
import pandas as pd


def compute_codebook_usage(
    latent_df: pd.DataFrame,
    code_col: str = "vq_code",
    num_codes: int | None = None,
) -> pd.DataFrame:
    if code_col not in latent_df.columns:
        return pd.DataFrame()

    codes = pd.to_numeric(latent_df[code_col], errors="coerce").dropna().astype(int)
    if codes.empty:
        return pd.DataFrame()

    if num_codes is None:
        num_codes = int(codes.max()) + 1

    counts = codes.value_counts().reindex(range(num_codes), fill_value=0).sort_index()
    total = max(int(counts.sum()), 1)
    out = pd.DataFrame(
        {
            code_col: counts.index.astype(int),
            "sample_count": counts.values.astype(int),
            "usage_pct": counts.values / total,
            "is_active": counts.values > 0,
        }
    )
    return out


def codebook_health_summary(
    usage_df: pd.DataFrame,
    code_col: str = "vq_code",
) -> dict:
    if usage_df.empty or "usage_pct" not in usage_df.columns:
        return {
            "active_codes": 0,
            "total_codes": 0,
            "usage_entropy": np.nan,
            "usage_perplexity": np.nan,
            "largest_code_pct": np.nan,
        }

    probs = usage_df["usage_pct"].to_numpy(dtype=float)
    active_probs = probs[probs > 0]
    entropy = float(-(active_probs * np.log(active_probs)).sum()) if len(active_probs) else 0.0
    return {
        "active_codes": int((usage_df["sample_count"] > 0).sum()),
        "total_codes": int(len(usage_df)),
        "usage_entropy": entropy,
        "usage_perplexity": float(np.exp(entropy)),
        "largest_code_pct": float(probs.max()) if len(probs) else np.nan,
    }


def compute_code_transition_stats(
    latent_df: pd.DataFrame,
    code_col: str = "vq_code",
) -> pd.DataFrame:
    required = {"date", "ticker", code_col}
    if not required.issubset(latent_df.columns):
        return pd.DataFrame()

    work = latent_df[list(required)].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["ticker", "date"])
    work["_prev_code"] = work.groupby("ticker")[code_col].shift(1)
    work["_changed"] = (work[code_col] != work["_prev_code"]).astype(float)
    work.loc[work["_prev_code"].isna(), "_changed"] = np.nan

    rows = []
    for code, group in work.groupby(code_col, sort=True):
        valid = group["_changed"].dropna()
        rows.append(
            {
                code_col: int(code),
                "transition_rate": float(valid.mean()) if not valid.empty else np.nan,
                "observations": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def merge_gmm_probabilities(
    latent_df: pd.DataFrame,
    gmm_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"date", "ticker"}
    if not required.issubset(latent_df.columns) or not required.issubset(gmm_df.columns):
        return latent_df.copy()

    probs = [col for col in gmm_df.columns if col.startswith("p_state_")]
    if not probs:
        return latent_df.copy()

    left = latent_df.copy()
    right = gmm_df[["date", "ticker", *probs]].copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    left["ticker"] = left["ticker"].astype(str)
    right["ticker"] = right["ticker"].astype(str)
    merged = left.merge(right, on=["date", "ticker"], how="left")
    dominant = pd.Series(pd.NA, index=merged.index, dtype="string")
    has_any_probability = merged[probs].notna().any(axis=1)
    if has_any_probability.any():
        dominant.loc[has_any_probability] = merged.loc[has_any_probability, probs].idxmax(axis=1).astype("string")
    merged["gmm_dominant_state"] = dominant.str.replace("p_state_", "", regex=False)
    merged["gmm_dominant_state"] = pd.to_numeric(
        merged["gmm_dominant_state"],
        errors="coerce",
    )
    return merged


def compute_gmm_overlap(
    latent_df: pd.DataFrame,
    code_col: str = "vq_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {code_col, "gmm_dominant_state"}
    if not required.issubset(latent_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    work = latent_df.dropna(subset=[code_col, "gmm_dominant_state"]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = pd.crosstab(work[code_col].astype(int), work["gmm_dominant_state"].astype(int))
    row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    counts = counts.reset_index()
    row_pct = row_pct.reset_index()
    return counts, row_pct


def compute_manual_feature_profile(
    latent_df: pd.DataFrame,
    feature_cols: list[str],
    code_col: str = "vq_code",
) -> pd.DataFrame:
    if code_col not in latent_df.columns or not feature_cols:
        return pd.DataFrame()

    available = [col for col in feature_cols if col in latent_df.columns]
    if not available:
        return pd.DataFrame()

    work = latent_df[[code_col, *available]].replace([np.inf, -np.inf], np.nan)
    grouped = work.groupby(code_col, sort=True)
    rows = []
    for code, group in grouped:
        row = {code_col: int(code), "sample_count": int(len(group))}
        for feature in available:
            values = pd.to_numeric(group[feature], errors="coerce")
            row[f"{feature}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{feature}_median"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_code_target_ic(
    latent_df: pd.DataFrame,
    target_col: str = "target_1d_rank",
    code_col: str = "vq_code",
    min_assets_per_day: int = 5,
) -> pd.DataFrame:
    required = {"date", code_col, target_col}
    if not required.issubset(latent_df.columns):
        return pd.DataFrame()

    work = latent_df[["date", code_col, target_col]].replace([np.inf, -np.inf], np.nan)
    work["date"] = pd.to_datetime(work["date"])
    rows = []
    for date, day in work.dropna().groupby("date", sort=False):
        if len(day) < min_assets_per_day:
            continue
        if day[code_col].nunique() < 2 or day[target_col].nunique() < 2:
            continue
        ic = day[code_col].corr(day[target_col], method="spearman")
        if pd.notna(ic) and np.isfinite(ic):
            rows.append({"date": date, "code_ic": float(ic)})

    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(
            [
                {
                    "mean_code_ic": np.nan,
                    "abs_mean_code_ic": np.nan,
                    "code_ic_ir": np.nan,
                    "positive_day_rate": np.nan,
                    "valid_days": 0,
                }
            ]
        )

    mean_ic = float(daily["code_ic"].mean())
    std_ic = float(daily["code_ic"].std()) if len(daily) > 1 else 0.0
    return pd.DataFrame(
        [
            {
                "mean_code_ic": mean_ic,
                "abs_mean_code_ic": abs(mean_ic),
                "code_ic_ir": mean_ic / std_ic if abs(std_ic) > 1e-12 else 0.0,
                "positive_day_rate": float((daily["code_ic"] > 0).mean()),
                "valid_days": int(len(daily)),
            }
        ]
    )
