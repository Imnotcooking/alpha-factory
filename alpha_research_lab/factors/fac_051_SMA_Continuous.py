import pandas as pd
import polars as pl
import numpy as np

# ==========================================
# 📜 BILINGUAL METADATA (PARETO FRONTIER)
# ==========================================
FACTOR_ID = "fac_051"
FACTOR_NAME = "Cross-Sectional SMA Continuous 5/20 / 横截面SMA连续信号"
CATEGORY = "Trend / 趋势"
COMPLEXITY = 4
ECONOMIC_RATIONALE = (
    "EN: Tests continuous cross-sectional scoring of 5/20 SMA spread with Z-normalization, clipping, and inertia. / "
    "ZH: 测试5/20均线差的横截面连续评分，使用Z标准化、截断与惯性平滑。"
)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    lf = pl.from_pandas(df).lazy()
    eps = 1e-8

    lf = lf.with_columns([
        pl.col("close").rolling_mean(window_size=5).over("ticker").alias("sma_5"),
        pl.col("close").rolling_mean(window_size=20).over("ticker").alias("sma_20"),
    ])

    lf = lf.with_columns([
        ((pl.col("sma_5") / (pl.col("sma_20") + eps)) - 1.0).alias("sma_spread")
    ])

    lf = lf.with_columns([
        (
            (pl.col("sma_spread") - pl.col("sma_spread").mean().over("date"))
            / (pl.col("sma_spread").std().over("date") + eps)
        ).alias("spread_z")
    ])

    lf = lf.with_columns([
        pl.col("spread_z").clip(-3.0, 3.0).alias("spread_z_capped")
    ])

    lf = lf.with_columns([
        (pl.col("spread_z_capped") / 3.0 * 0.10).alias("raw_score")
    ])

    lf = lf.with_columns([
        pl.col("raw_score")
        .ewm_mean(alpha=0.15, ignore_nulls=True)
        .over("ticker")
        .alias("smoothed_score")
    ])

    lf = lf.with_columns([
        pl.col("smoothed_score").shift(1).over("ticker").fill_null(0.0).alias("factor_score")
    ])

    return lf.collect().to_pandas()
