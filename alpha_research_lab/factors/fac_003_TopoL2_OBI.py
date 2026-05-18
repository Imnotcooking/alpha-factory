import polars as pl

# 1. THE RATIONALE (Required for Database Logging)
FACTOR_NAME = "L2_Topological_Order_Book_Imbalance"
AUTHOR = "Alpha Factory"
ECONOMIC_RATIONALE = """
This factor captures the immediate micro-structural pressure in the order book. 
In a limit order book (like the CSI 500 IC futures), the resting limit orders act as 
gravitational forces on the price. When there is significantly more volume sitting on 
the Bid side across multiple levels compared to the Ask side, it indicates heavy 
institutional buying interest or market-maker skewing. 

Why do we get paid? We are anticipating the 'Takers' (impatient traders). Takers will 
aggressively cross the spread to consume the thin Ask liquidity, pushing the price up. 
The people taking the other side of this trade are either slow market makers who failed 
to cancel/update their stale quotes in time, or directional traders ignoring micro-structural 
imbalances.
"""
CATEGORY = "Volatility/Statistical Arbitrage"

# 2. THE MATHEMATICS
def compute(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Computes the factor and appends it as a new column named 'factor_signal'.
    """

    # Define your Polars mathematical expressions inside a list
    factor_expressions = [
        
        # Calculate the 5-Level Volume Imbalance. 
        # (Total Bid Vol - Total Ask Vol) / (Total Bid Vol + Total Ask Vol)
        # We apply .shift(1) to ensure the signal calculated at time T is used for execution at time T+1.
        (
            (
                (pl.col("bid_vol_1") + pl.col("bid_vol_2") + pl.col("bid_vol_3") + pl.col("bid_vol_4") + pl.col("bid_vol_5")) - 
                (pl.col("ask_vol_1") + pl.col("ask_vol_2") + pl.col("ask_vol_3") + pl.col("ask_vol_4") + pl.col("ask_vol_5"))
            ) / 
            (
                (pl.col("bid_vol_1") + pl.col("bid_vol_2") + pl.col("bid_vol_3") + pl.col("bid_vol_4") + pl.col("bid_vol_5")) + 
                (pl.col("ask_vol_1") + pl.col("ask_vol_2") + pl.col("ask_vol_3") + pl.col("ask_vol_4") + pl.col("ask_vol_5"))
            )
        ).shift(1).alias("factor_signal")
        
    ]

    # Append the new columns to the LazyFrame and return it
    return lf.with_columns(factor_expressions)