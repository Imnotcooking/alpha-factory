import polars as pl
import pandas as pd
import os

class MacroTrendAgent:
    def __init__(self, max_positions=5):
        self.max_positions = max_positions
        self.agent_name = "Macro_Trend_Desk"

    def apply_factor_math(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        # --- GRAFTING ZONE: TSMOM Factor ---
        factor_expressions = [
            (pl.col("Close").shift(1) / pl.col("Close").shift(120) - 1.0).alias("raw_ret_120"),
            # Daily return standard deviation over 120 days
            ((pl.col("Close").shift(1) / pl.col("Close").shift(2)).log().rolling_std(120)).alias("vol_120"),
        ]
        lf = lf.with_columns(factor_expressions)
        
        lf = lf.with_columns([
            (pl.col("raw_ret_120") / pl.col("vol_120")).alias("risk_adj_mom")
        ])
        
        return lf.with_columns([
            (pl.col("risk_adj_mom").rank() / pl.col("risk_adj_mom").count()).over("Date").alias("factor_signal")
        ])
        # -----------------------------------

    def generate_signals(self, df: pl.DataFrame) -> pd.DataFrame:
        lf = df.lazy()
        lf = self.apply_factor_math(lf)
        scored_df = lf.collect()
        
        latest_date = scored_df["Date"].max()
        today_market = scored_df.filter(pl.col("Date") == latest_date)
        
        valid_signals = today_market.drop_nulls(subset=["factor_signal"])
        ranked = valid_signals.sort("factor_signal", descending=True)
        targets = ranked.head(self.max_positions)
        
        if targets.height == 0:
            return pd.DataFrame()

        result_pd = targets.select(["Ticker", "Close", "factor_signal"]).to_pandas()
        result_pd = result_pd.rename(columns={"Ticker": "symbol", "Close": "close"}) # Align for Optimizer
        result_pd["Target_Weight"] = 1.0 / len(result_pd)
        result_pd["Strategy"] = self.agent_name
        
        return result_pd