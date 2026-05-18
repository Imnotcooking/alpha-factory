import polars as pl

# 1. THE RATIONALE (Required for Database Logging)
FACTOR_NAME = "Volume_Damped_Price_Curvature"
AUTHOR = "Alpha Factory"
ECONOMIC_RATIONALE = """
This factor captures the 'acceleration' or 'curvature' of price movements, validated 
by volume clustering. Standard momentum measures velocity (how fast price is changing), 
but curvature measures acceleration (is the trend speeding up or exhausting?). 

By taking the discrete second derivative of the price curve and weighting it by an 
abnormal volume ratio, we isolate instances where a price move is structurally accelerating 
backed by heavy institutional participation (volume clustering). 

Why do we get paid? We are exploiting the inertia of market crowds. When price acceleration 
coincides with abnormal volume, it usually triggers retail FOMO (Fear Of Missing Out) or 
forces systemic short-covering. The counterparties losing on this trade are mean-reversion 
traders who are trying to catch a falling knife or step in front of an accelerating freight 
train, underestimating the 'gravitational' pull of the trend.
"""
CATEGORY = "Momentum/Trend"

# 2. THE MATHEMATICS
def compute(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Computes the factor and appends it as a new column named 'factor_signal'.
    """

    # Define your Polars mathematical expressions inside a list
    factor_expressions = [
        
        # Step 1: Calculate the discrete second derivative of price (Curvature / Acceleration)
        # Formula: (Close_t) - 2*(Close_t-1) + (Close_t-2)
        # Step 2: Calculate the Volume Damping Ratio (Current Volume / 20-period Moving Average)
        # Step 3: Multiply Curvature by Volume Ratio.
        # Step 4: Apply .shift(1) to the entire resulting logic to strictly prevent look-ahead bias.
        (
            (
                pl.col("close") - (2 * pl.col("close").shift(1)) + pl.col("close").shift(2)
            ) * (
                pl.col("volume") / pl.col("volume").rolling_mean(20)
            )
        ).shift(1).alias("factor_signal")
        
    ]

    # Append the new columns to the LazyFrame and return it
    return lf.with_columns(factor_expressions)