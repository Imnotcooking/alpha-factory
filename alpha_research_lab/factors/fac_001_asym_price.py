import pandas as pd
import numpy as np

# ==========================================
# FACTOR METADATA (For the Database)
# ==========================================
FACTOR_ID = "fac_001_asym_price"
FACTOR_NAME = "Asymmetric Price Locator"
CATEGORY = "Momentum"
COMPLEXITY = 3  # Low complexity, just simple arithmetic
# ECONOMIC RATIONALE: ... 
# ==========================================
# THE MATH ENGINE
# ==========================================
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    The mathematical logic of the factor.
    Must return the dataframe with a new column named 'factor_score'.
    """
    # For this example, we will generate dummy OHLC (Open, High, Low, Close) data 
    # since we don't have your live Polars Data Lake hooked up yet.
    df['high'] = df['forward_return'] + 0.05
    df['low'] = df['forward_return'] - 0.05
    df['close'] = df['forward_return'] + np.random.normal(0, 0.01, len(df))
    
    # THE ACTUAL FACTOR MATH: (Close - Low) / (High - Low)
    # A score of 1.0 means it closed at the absolute high. A score of 0.0 means it closed at the low.
    numerator = df['close'] - df['low']
    denominator = df['high'] - df['low']
    
    # Avoid division by zero
    denominator = denominator.replace(0, 0.0001)
    
    # Output MUST be named 'factor_score'
    df['factor_score'] = numerator / denominator
    
    return df