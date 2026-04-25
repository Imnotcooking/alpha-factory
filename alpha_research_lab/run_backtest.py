import argparse
import importlib
import pandas as pd
import numpy as np
import sqlite3
from evaluator import AlphaEvaluator

def main():
    # 1. Setup the Terminal Command Arguments
    parser = argparse.ArgumentParser(description="Alpha Mine Backtest Engine")
    parser.add_argument("--factor", type=str, required=True, help="Name of the factor file (e.g., fac_001_asym_price)")
    args = parser.parse_args()

    factor_module_name = args.factor
    print(f"🚀 Booting Backtest Engine for: {factor_module_name}...")

    # 2. Dynamically Import the Factor Code
    try:
        # This tells Python to look inside the 'factors' folder for the file
        factor_module = importlib.import_module(f"factors.{factor_module_name}")
    except ModuleNotFoundError:
        print(f"❌ ERROR: Could not find factors/{factor_module_name}.py")
        return

    # 3. Register the Factor Metadata into the Database (if it's new)
    conn = sqlite3.connect("research_memory.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO factors (factor_id, name, category, complexity_score)
        VALUES (?, ?, ?, ?)
    ''', (factor_module.FACTOR_ID, factor_module.FACTOR_NAME, factor_module.CATEGORY, factor_module.COMPLEXITY))
    conn.commit()
    conn.close()

    # 4. Load the Data Lake (Using our dummy data generator for now)
    print("   -> Loading Data Lake...")
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="B")
    df = pd.DataFrame({'date': dates})
    df['forward_return'] = np.random.normal(0, 0.01, len(df))

    # 5. Apply the Math
    print("   -> Computing Factor Scores...")
    df = factor_module.compute(df)

    # 6. Pass to the Evaluator (The strict judge from Phase 2)
    evaluator = AlphaEvaluator()
    evaluator.run_evaluation(factor_module.FACTOR_ID, df)

if __name__ == "__main__":
    main()