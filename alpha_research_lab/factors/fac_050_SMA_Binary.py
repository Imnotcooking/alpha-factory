import pandas as pd
import polars as pl
import numpy as np

# ==========================================
# 📜 BILINGUAL METADATA (PARETO FRONTIER)
# ==========================================
FACTOR_ID = "fac_050"
FACTOR_NAME = "Cross-Sectional SMA Binary 5/20 / 横截面SMA二值信号"
CATEGORY = "Trend / 趋势"
COMPLEXITY = 3
ECONOMIC_RATIONALE = (
    "EN: Tests pure rank-based binary allocation from 5/20 SMA spread, with instant execution and no smoothing. / "
    "ZH: 测试基于5/20均线差的纯排序二值分配，即时执行且不做平滑。"
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
        pl.col("sma_spread").rank(method="average").over("date").alias("cs_rank"),
        pl.len().over("date").alias("cs_n"),
    ])

    lf = lf.with_columns([
        (pl.col("cs_rank") / (pl.col("cs_n") + eps)).alias("rank_pct")
    ])

    lf = lf.with_columns([
        pl.when(pl.col("rank_pct") >= 0.85).then(pl.lit(0.10))
        .when(pl.col("rank_pct") <= 0.15).then(pl.lit(-0.10))
        .otherwise(pl.lit(0.0))
        .alias("raw_score")
    ])

    lf = lf.with_columns([
        pl.col("raw_score").shift(1).over("ticker").fill_null(0.0).alias("factor_score")
    ])

    return lf.collect().to_pandas()
