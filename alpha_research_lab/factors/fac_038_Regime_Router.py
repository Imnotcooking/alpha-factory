import polars as pl
import pandas as pd
import numpy as np

# ==========================================
# 📜 BILINGUAL METADATA (PARETO FRONTIER)
# ==========================================
FACTOR_ID = "fac_038"
FACTOR_NAME = "Layer 4: Probabilistic Router + Vol Tiering / 第四层：概率路由+波动率分层"
CATEGORY = "Meta-Strategy / 元策略"
COMPLEXITY = 6
ECONOMIC_RATIONALE = (
    "EN: Layer 4 meta-strategy that continuously blends trend (SMA) and mean-reversion (Bollinger) using "
    "a probabilistic 252d volatility gauge, then applies cross-sectional volatility tiering and 5% block sizing. / "
    "ZH: 第四层元策略：使用252日波动率概率仪表连续混合趋势(SMA)与均值回归(Bollinger)，再叠加横截面波动率分层与5%离散仓位。"
)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    print(f"   -> [{FACTOR_ID}] Booting Layer 4: Probabilistic Router + Vol Tiering...")
    lf = pl.from_pandas(df).lazy()
    eps = 1e-8

    # ---------------------------------------------------------
    # STEP 1: Continuous, Z-scored SMA + Bollinger signals
    # ---------------------------------------------------------
    lf = lf.with_columns([
        pl.col("close").rolling_mean(window_size=5).over("ticker").alias("sma_5"),
        pl.col("close").rolling_mean(window_size=20).over("ticker").alias("sma_20"),
        pl.col("close").rolling_std(window_size=20).over("ticker").alias("std_20"),
        pl.col("close").rolling_std(window_size=60).over("ticker").alias("asset_vol_60"),
    ])

    lf = lf.with_columns([
        ((pl.col("sma_5") / (pl.col("sma_20") + eps)) - 1.0).alias("sma_spread"),
        ((pl.col("close") - pl.col("sma_20")) / (2.0 * pl.col("std_20") + eps)).alias("boll_spread"),
    ])

    lf = lf.with_columns(
        (
            (pl.col("sma_spread") - pl.col("sma_spread").mean().over("date"))
            / (pl.col("sma_spread").std().over("date") + eps)
        ).clip(-3.0, 3.0).alias("sma_z")
    )

    lf = lf.with_columns(
        (
            (pl.col("boll_spread") - pl.col("boll_spread").mean().over("date"))
            / (pl.col("boll_spread").std().over("date") + eps)
        ).clip(-3.0, 3.0).alias("boll_z")
    )

    lf = lf.with_columns([
        (pl.col("sma_z") / 3.0 * 0.10).alias("w_sma"),
        (pl.col("boll_z") / 3.0 * 0.10).alias("w_boll"),
        ((pl.col("close") / (pl.col("close").shift(1).over("ticker") + eps)) - 1.0).alias("asset_ret"),
        pl.col("asset_vol_60").rank(method="average").over("date").alias("vol_rank"),
        pl.len().over("date").alias("asset_count"),
    ])

    # ---------------------------------------------------------
    # STEP 2: Probabilistic regime gauge (0.0 -> 1.0)
    # ---------------------------------------------------------
    pre = lf.select([
        "date",
        "ticker",
        "w_sma",
        "w_boll",
        "vol_rank",
        "asset_count",
        "asset_ret",
    ]).collect().to_pandas()
    pre = pre.sort_values(["date", "ticker"]).reset_index(drop=True)

    mkt = (
        pre.groupby("date", as_index=False)["asset_ret"]
        .mean()
        .rename(columns={"asset_ret": "mkt_ret"})
        .sort_values("date")
    )
    mkt["market_vol_252"] = mkt["mkt_ret"].rolling(252, min_periods=60).std()
    roll_min = mkt["market_vol_252"].rolling(252, min_periods=60).min()
    roll_max = mkt["market_vol_252"].rolling(252, min_periods=60).max()
    mkt["regime_prob"] = ((mkt["market_vol_252"] - roll_min) / (roll_max - roll_min + eps)).clip(0.0, 1.0)
    mkt["regime_prob"] = mkt["regime_prob"].fillna(0.5)

    out = pre.merge(mkt[["date", "regime_prob"]], on="date", how="left")

    # Smooth blend: low vol -> SMA-heavy, high vol -> Bollinger-heavy
    out["routed_weight"] = ((1.0 - out["regime_prob"]) * out["w_sma"]) + (out["regime_prob"] * out["w_boll"])

    # ---------------------------------------------------------
    # STEP 3: Cross-sectional volatility tiering
    # top 33% vol assets -> 0.5x, bottom 33% vol assets -> 1.5x
    # ---------------------------------------------------------
    low_cut = out["asset_count"] / 3.0
    high_cut = out["asset_count"] * 2.0 / 3.0
    out["tier_multiplier"] = np.where(
        out["vol_rank"] <= low_cut,
        1.5,  # least volatile
        np.where(out["vol_rank"] >= high_cut, 0.5, 1.0),  # most volatile
    )
    out["tiered_weight"] = out["routed_weight"] * out["tier_multiplier"]

    # Inertia before execution sizing
    out["smoothed_weight"] = (
        out.sort_values(["ticker", "date"])
        .groupby("ticker")["tiered_weight"]
        .transform(lambda s: s.ewm(alpha=0.15, adjust=False).mean())
    )

    # ---------------------------------------------------------
    # STEP 4: 5% discretization and lookahead shift
    # ---------------------------------------------------------
    out["discrete_weight"] = (out["smoothed_weight"] / 0.05).round(0) * 0.05
    out["factor_score"] = (
        out.sort_values(["ticker", "date"])
        .groupby("ticker")["discrete_weight"]
        .shift(1)
        .fillna(0.0)
    )

    # Re-attach original columns and append final factor score.
    base_cols = list(df.columns)
    if "factor_score" in base_cols:
        base_cols.remove("factor_score")

    result = df[base_cols].merge(
        out[["date", "ticker", "factor_score"]],
        on=["date", "ticker"],
        how="left",
    )
    return result