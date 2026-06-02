import polars as pl
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 📜 BILINGUAL METADATA (PARETO FRONTIER)
# ==========================================
FACTOR_ID = "fac_AI_001"
FACTOR_NAME = "AI Quiet Trend V1 / 安静趋势策略"
CATEGORY = "Machine Learning / 机器学习"
COMPLEXITY = 5
ECONOMIC_RATIONALE = (
    "EN: Trend-follow momentum and value only in HMM quiet regimes (prob_quiet > 0.60). "
    "Discrete 5% block weights to avoid fractional TCA bleed. / "
    "ZH: 仅在HMM安静状态(prob_quiet>0.60)下顺势交易动量与价值，5%离散仓位降低摩擦。"
)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quiet Trend: ride sector momentum + value when the market is in a quiet regime.

    Signal Logic:
    - LONG (+0.10): f_mom_z_sector > 1.0 AND f_value_60d > 0.5 AND prob_quiet > 0.60
    - SHORT (-0.10): f_mom_z_sector < -1.0 AND f_value_60d < -0.5 AND prob_quiet > 0.60
    - Otherwise: 0.0 (cash)

    Execution:
    - Discretize to nearest 0.05 before the lookahead shift (block trades).
    - shift(1) per ticker for lookahead protection.
    """

    print(f"   -> [{FACTOR_ID}] Building Quiet Trend signals in Polars...")
    lf = pl.from_pandas(df).lazy()

    # f_value_60d is raw carry; cross-sectional Z per date so +/-0.5 thresholds are meaningful
    lf = lf.with_columns([
        (
            (pl.col("f_value_60d") - pl.col("f_value_60d").mean().over("date"))
            / (pl.col("f_value_60d").std().over("date") + 1e-8)
        ).alias("val_z")
    ])

    lf = lf.with_columns([
        pl.when(
            (pl.col("prob_quiet") > 0.60)
            & (pl.col("f_mom_z_sector") > 1.0)
            & (pl.col("val_z") > 0.5)
        )
        .then(pl.lit(0.10))
        .when(
            (pl.col("prob_quiet") > 0.60)
            & (pl.col("f_mom_z_sector") < -1.0)
            & (pl.col("val_z") < -0.5)
        )
        .then(pl.lit(-0.10))
        .otherwise(pl.lit(0.0))
        .alias("raw_score")
    ])

    lf = lf.with_columns([
        ((pl.col("raw_score") / 0.05).round(0) * 0.05).alias("factor_pre")
    ])

    lf = lf.with_columns([
        pl.col("factor_pre").shift(1).over("ticker").alias("factor_score")
    ])

    result = lf.collect().to_pandas()
    if "forward_return" not in result.columns and "forward_return" in df.columns:
        result = result.merge(
            df[["date", "ticker", "forward_return"]],
            on=["date", "ticker"],
            how="left",
        )
    return result
