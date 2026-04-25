import polars as pl

def compute(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Factor Name: Core Macro Momentum & Volatility
    Description: The foundational quantitative State Space. Calculates multi-timeframe 
                 Log Returns, Annualized Volatility, and Mean Reversion proxies.
    """
    
    factor_expressions = [
        # Momentum
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("Ret_1D"),
        (pl.col("Close") / pl.col("Close").shift(5)).log().alias("Ret_5D"),
        (pl.col("Close") / pl.col("Close").shift(20)).log().alias("Ret_20D"),
        (pl.col("Close") / pl.col("Close").shift(60)).log().alias("Ret_60D"),
        
        # Volatility
        ((pl.col("Close") / pl.col("Close").shift(1)).log().rolling_std(window_size=20) * (252 ** 0.5)).alias("Vol_20D"),
        ((pl.col("Close") / pl.col("Close").shift(1)).log().rolling_std(window_size=60) * (252 ** 0.5)).alias("Vol_60D"),
        
        # Mean Reversion (Distance from SMA)
        ((pl.col("Close") / pl.col("Close").rolling_mean(window_size=20)) - 1.0).alias("Dist_SMA_20"),
        ((pl.col("Close") / pl.col("Close").rolling_mean(window_size=50)) - 1.0).alias("Dist_SMA_50"),
        ((pl.col("Close") / pl.col("Close").rolling_mean(window_size=200)) - 1.0).alias("Dist_SMA_200"),
    ]
    
    return lf.with_columns(factor_expressions)