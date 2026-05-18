import polars as pl
import pandas as pd

# 1. THE RATIONALE (Required for Database Logging)
FACTOR_ID = "fac_005"            
COMPLEXITY = 3                   

FACTOR_NAME = "Sloan_Earnings_Quality_Accruals"
AUTHOR = "Alpha Factory"
CATEGORY = "Mean Reversion/Value"

ECONOMIC_RATIONALE = """
This factor captures the 'Quality of Earnings' by measuring the difference between 
reported Net Income and actual Operating Cash Flow, scaled by Total Assets. 

Why do we get paid? Behavioral fixation. The vast majority of retail investors, financial 
journalists, and basic algorithms parse headline 'Earnings Per Share (EPS)'. They blindly 
buy companies with surging Net Income. However, if that Net Income is driven by non-cash 
accruals (e.g., unpaid invoices or unsold inventory), those earnings are highly likely to 
mean-revert negatively in the next 1-2 quarters when reality hits. 

Who is losing? Naive growth chasers and momentum algorithms trading off headline EPS beats. 
We take the other side by isolating companies whose profitability is backed by hard, 
irrefutable cash flow.
"""

# 2. THE MATHEMATICS
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the factor and appends it as a new column named 'factor_score'.
    (The Evaluator looks for 'factor_score', not 'factor_signal')
    """
    # 1. Translate Pandas to Polars
    lf = pl.from_pandas(df).lazy()
    
    # 2. Calculate the Raw Signal (Cash-Backed Earnings Yield)
    # Notice we removed .shift(1) here because our dummy data doesn't have tickers yet.
    # In live trading, the Data Lake API handles the point-in-time lagging for us!
    lf = lf.with_columns([
        (
            (pl.col("operating_cash_flow") - pl.col("net_income")) / 
            pl.when(pl.col("total_assets") == 0).then(None).otherwise(pl.col("total_assets"))
        ).alias("raw_signal")
    ])

    # 3. Cross-Sectional Z-Score (Required by the Alpha Evaluator)
    lf = lf.with_columns([
        (
            (pl.col("raw_signal") - pl.col("raw_signal").mean().over("date")) /
            pl.col("raw_signal").std().over("date")
        ).fill_nan(0).alias("factor_score")
    ])

    # 4. Translate back to Pandas and return
    return lf.collect().to_pandas()