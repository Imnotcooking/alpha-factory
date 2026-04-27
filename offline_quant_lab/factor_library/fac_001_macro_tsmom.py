import polars as pl

# 1. THE RATIONALE
FACTOR_NAME = "Risk-Adjusted Time Series Momentum (TSMOM)"
AUTHOR = "Alpha Factory"
ECONOMIC_RATIONALE = """
Risk Premium / Behavioral Herding. Investors under-react to long-term fundamental changes. 
By dividing the 6-month return by the 6-month volatility, we isolate stocks with 'smooth' upward 
momentum (institutional accumulation) and penalize volatile spikes (retail gambling).
"""

# 2. THE MATHEMATICS
def compute(lf: pl.LazyFrame) -> pl.LazyFrame:
    factor_expressions = [
        # 120-Day (6 Month) Return (Shifted to prevent Look-Ahead)
        (pl.col("close").shift(1) / pl.col("close").shift(120) - 1.0).alias("raw_ret_120"),
        
        # 120-Day Volatility
        (pl.col("close").shift(1).pct_change().rolling_std(120)).alias("vol_120"),
    ]
    
    lf = lf.with_columns(factor_expressions)
    
    # Calculate Sharpe-like momentum and Cross-Sectionally Rank it
    lf = lf.with_columns([
        (pl.col("raw_ret_120") / pl.col("vol_120")).alias("risk_adj_mom")
    ])
    
    return lf.with_columns([
        (pl.col("risk_adj_mom").rank() / pl.col("risk_adj_mom").count()).over("date").alias("factor_signal")
    ])