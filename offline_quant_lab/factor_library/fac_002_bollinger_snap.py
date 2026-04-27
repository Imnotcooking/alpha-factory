import polars as pl

# 1. THE RATIONALE
FACTOR_NAME = "Bollinger Band Liquidity Snap"
AUTHOR = "Alpha Factory"
ECONOMIC_RATIONALE = """
Liquidity Provision / Forced Selling. When an asset crashes below its 20-day lower Bollinger Band 
on high volume, it usually indicates forced liquidations or stop-loss hunting. We get paid a premium 
to step in and provide liquidity to these desperate sellers, capturing the 3-day mean-reversion snapback.
"""

# 2. THE MATHEMATICS
def compute(lf: pl.LazyFrame) -> pl.LazyFrame:
    factor_expressions = [
        # 20-Day Moving Average & Standard Deviation
        pl.col("close").shift(1).rolling_mean(20).alias("sma_20"),
        pl.col("close").shift(1).rolling_std(20).alias("std_20"),
    ]
    
    lf = lf.with_columns(factor_expressions)
    
    lf = lf.with_columns([
        # Lower Band Calculation
        (pl.col("sma_20") - (2.0 * pl.col("std_20"))).alias("lower_band")
    ])
    
    # Calculate how far below the lower band we are (Distance)
    # If price is $90 and band is $100, score is positive (we want to buy the biggest dips)
    lf = lf.with_columns([
        ((pl.col("lower_band") - pl.col("close").shift(1)) / pl.col("lower_band")).alias("band_distance")
    ])
    
    return lf.with_columns([
        # Rank the most oversold assets at the top (Score ~ 1.0)
        (pl.col("band_distance").rank() / pl.col("band_distance").count()).over("date").alias("factor_signal")
    ])