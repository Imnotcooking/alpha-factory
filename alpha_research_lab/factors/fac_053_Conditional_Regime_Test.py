import pandas as pd
import numpy as np
import polars as pl

# ==========================================
# 📜 BILINGUAL METADATA (PARETO FRONTIER)
# ==========================================
FACTOR_ID = "fac_053"
FACTOR_NAME = "Conditional Regime Validation Test / 条件状态验证测试"
CATEGORY = "Diagnostics / 诊断"
COMPLEXITY = 4
ECONOMIC_RATIONALE = (
    "EN: Validate marginal risk reduction from HMM-style regime conditioning by comparing unconditional vs conditional gross PnL streams. / "
    "ZH: 通过对比无条件与状态条件化毛收益序列，验证HMM状态过滤带来的边际风控收益。"
)


def _calc_sharpe(returns: pd.Series) -> float:
    s = returns.dropna()
    if s.empty or s.std(ddof=0) < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * s.mean() / s.std(ddof=0))


def _calc_max_drawdown(returns: pd.Series) -> float:
    s = returns.fillna(0.0)
    if s.empty:
        return 0.0
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conditional PnL diagnostic for regime validation.

    Outputs daily gross-return arrays:
      A) SMA unconditional
      B) SMA only in Regime 0 (Quiet)
      C) Bollinger unconditional
      D) Bollinger only in Regime 1 (Choppy)
    """
    df_local = df.copy()
    # Some matrices (e.g., ML_Stacked_Matrix.parquet) do not include forward_return.
    # Build it deterministically from close so the diagnostic is portable.
    if "forward_return" not in df_local.columns:
        df_local = df_local.sort_values(["ticker", "date"]).reset_index(drop=True)
        df_local["forward_return"] = (
            df_local.groupby("ticker")["close"].shift(-1) / (df_local["close"] + 1e-8) - 1.0
        )

    lf = pl.from_pandas(df_local).lazy()
    eps = 1e-8

    lf = lf.with_columns([
        pl.col("date").cast(pl.Datetime).alias("date"),
        pl.col("close").cast(pl.Float64).alias("close"),
    ])

    # Ensure daily return exists.
    lf = lf.with_columns([
        (pl.col("close") / (pl.col("close").shift(1).over("ticker") + eps) - 1.0).alias("daily_ret"),
        pl.col("forward_return").alias("fwd_ret"),
    ])

    # Signals per ticker.
    lf = lf.with_columns([
        pl.col("close").rolling_mean(window_size=5).over("ticker").alias("sma_5"),
        pl.col("close").rolling_mean(window_size=20).over("ticker").alias("sma_20"),
        pl.col("close").rolling_std(window_size=20).over("ticker").alias("boll_std_20"),
    ])

    lf = lf.with_columns([
        (pl.col("sma_5") - pl.col("sma_20")).alias("sma_spread"),
        ((pl.col("close") - pl.col("sma_20")) / (pl.col("boll_std_20") + eps)).alias("boll_z"),
    ])

    # 5/20 SMA trend-follow signal.
    lf = lf.with_columns([
        pl.when(pl.col("sma_spread") > 0).then(pl.lit(1.0))
        .when(pl.col("sma_spread") < 0).then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .alias("sma_sig")
    ])

    # 20d Bollinger mean-reversion signal.
    lf = lf.with_columns([
        pl.when(pl.col("boll_z") < -2.0).then(pl.lit(1.0))
        .when(pl.col("boll_z") > 2.0).then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .alias("boll_sig")
    ])

    # Market volatility proxy: 20d rolling std of equal-weight market return.
    lf = lf.with_columns([
        pl.col("daily_ret").mean().over("date").alias("mkt_ret")
    ])
    lf = lf.with_columns([
        pl.col("mkt_ret").rolling_std(window_size=20).over("ticker").alias("mkt_vol_proxy")
    ])
    lf = lf.with_columns([
        pl.col("mkt_vol_proxy").mean().over("date").alias("market_vol")
    ])

    # Collapse to pandas for quantile-based regime assignment and diagnostics.
    base = lf.select([
        "date",
        "ticker",
        "fwd_ret",
        "sma_sig",
        "boll_sig",
        "market_vol",
    ]).collect().to_pandas()

    base = base.sort_values(["date", "ticker"]).reset_index(drop=True)
    vol_by_date = base.groupby("date", as_index=False)["market_vol"].mean()
    q33 = vol_by_date["market_vol"].quantile(0.33)
    q67 = vol_by_date["market_vol"].quantile(0.67)

    vol_by_date["regime"] = np.select(
        [vol_by_date["market_vol"] <= q33, vol_by_date["market_vol"] > q67],
        [0, 2],
        default=1,
    )

    merged = base.merge(vol_by_date[["date", "regime"]], on="date", how="left")
    merged["pnl_sma"] = merged["sma_sig"] * merged["fwd_ret"]
    merged["pnl_boll"] = merged["boll_sig"] * merged["fwd_ret"]

    daily = (
        merged.groupby("date", as_index=False)
        .agg(
            regime=("regime", "first"),
            A_sma_uncond=("pnl_sma", "mean"),
            C_boll_uncond=("pnl_boll", "mean"),
        )
        .sort_values("date")
    )

    daily["B_sma_regime0"] = np.where(daily["regime"] == 0, daily["A_sma_uncond"], 0.0)
    daily["D_boll_regime1"] = np.where(daily["regime"] == 1, daily["C_boll_uncond"], 0.0)

    strategy_cols = [
        ("A) SMA Unconditional", "A_sma_uncond"),
        ("B) SMA | Regime 0", "B_sma_regime0"),
        ("C) Bollinger Unconditional", "C_boll_uncond"),
        ("D) Bollinger | Regime 1", "D_boll_regime1"),
    ]

    rows = []
    for label, col in strategy_cols:
        sharpe = _calc_sharpe(daily[col])
        max_dd = _calc_max_drawdown(daily[col])
        rows.append(
            {
                "Strategy": label,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd,
            }
        )

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print(" CONDITIONAL PNL VALIDATION (GROSS RETURNS) ")
    print("=" * 72)
    print(summary.to_string(index=False, formatters={"Sharpe": "{:.3f}".format, "Max Drawdown": "{:.2%}".format}))
    print("=" * 72 + "\n")

    return daily[["date", "regime", "A_sma_uncond", "B_sma_regime0", "C_boll_uncond", "D_boll_regime1"]]


if __name__ == "__main__":
    input_path = "ML_Stacked_Matrix.parquet"
    df_in = pd.read_parquet(input_path)
    out = compute(df_in)
    print(out.tail(5).to_string(index=False))
