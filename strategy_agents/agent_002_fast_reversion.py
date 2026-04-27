import polars as pl
import pandas as pd
import os

class FastReversionAgent:
    def __init__(self, max_positions=5):
        self.max_positions = max_positions
        self.agent_name = "Fast_Reversion_Desk"

    def apply_factor_math(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        # --- GRAFTING ZONE: Bollinger Snap Factor ---
        factor_expressions = [
            pl.col("Close").shift(1).rolling_mean(20).alias("sma_20"),
            pl.col("Close").shift(1).rolling_std(20).alias("std_20"),
        ]
        lf = lf.with_columns(factor_expressions)
        
        lf = lf.with_columns([
            (pl.col("sma_20") - (2.0 * pl.col("std_20"))).alias("lower_band")
        ])
        
        lf = lf.with_columns([
            ((pl.col("lower_band") - pl.col("Close").shift(1)) / pl.col("lower_band")).alias("band_distance")
        ])
        
        return lf.with_columns([
            (pl.col("band_distance").rank() / pl.col("band_distance").count()).over("Date").alias("factor_signal")
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