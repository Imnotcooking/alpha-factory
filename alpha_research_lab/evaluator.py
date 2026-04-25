import sqlite3
import pandas as pd
import numpy as np
import scipy.stats as stats
import uuid
from datetime import datetime

class AlphaEvaluator:
    def __init__(self, db_path="research_memory.db"):
        self.db_path = db_path

    def calculate_rank_ic(self, factor_values, forward_returns):
        """Calculates the Spearman Rank Correlation (Rank IC)"""
        # Drop rows where we have missing data
        valid_idx = ~np.isnan(factor_values) & ~np.isnan(forward_returns)
        if sum(valid_idx) < 2:
            return 0.0
        
        ic, _ = stats.spearmanr(factor_values[valid_idx], forward_returns[valid_idx])
        return ic

    def run_evaluation(self, factor_id, df, split_date="2023-01-01"):
        print(f"🔬 Evaluating Candidate: {factor_id}")
        
        # ==========================================
        # STEP 1: The Out-of-Sample Splitter
        # ==========================================
        # We strictly divide the data. The model is never allowed to see post-2023 data during validation.
        validation_data = df[df['date'] < split_date]
        holdout_data = df[df['date'] >= split_date]
        
        # ==========================================
        # STEP 2: Calculate Predictive Edge (Rank IC)
        # ==========================================
        val_ic = self.calculate_rank_ic(validation_data['factor_score'], validation_data['forward_return'])
        holdout_ic = self.calculate_rank_ic(holdout_data['factor_score'], holdout_data['forward_return'])
        
        print(f"   -> Validation IC: {val_ic:.4f}")
        print(f"   -> Holdout IC:    {holdout_ic:.4f}")
        
        # ==========================================
        # STEP 3: The Diagnostics Engine
        # ==========================================
        failure_code = "NONE"
        if holdout_ic < 0:
            failure_code = "holdout_not_positive"
        elif holdout_ic < (val_ic * 0.4):
            failure_code = "severe_ic_decay"
            
        if failure_code != "NONE":
            print(f"   ⚠️ DIAGNOSTIC FLAG: [{failure_code.upper()}]")
            
        # ==========================================
        # STEP 4: Log to SQLite Database
        # ==========================================
        self._log_to_db(factor_id, val_ic, holdout_ic, failure_code)

    def _log_to_db(self, factor_id, val_ic, holdout_ic, failure_code):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        
        # Ensure the factor exists in the parent table, if not, auto-create it
        cursor.execute("SELECT factor_id FROM factors WHERE factor_id = ?", (factor_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO factors (factor_id, name, category) VALUES (?, ?, ?)", 
                           (factor_id, f"Auto-Gen {factor_id}", "Experimental"))
        
        # Get the current round number (so we can track mutations)
        cursor.execute("SELECT MAX(round_number) FROM backtest_runs WHERE factor_id = ?", (factor_id,))
        res = cursor.fetchone()[0]
        round_number = (res + 1) if res else 1
        
        # Log the backtest metrics
        cursor.execute('''
            INSERT INTO backtest_runs (run_id, factor_id, round_number, validation_ic, holdout_ic)
            VALUES (?, ?, ?, ?, ?)
        ''', (run_id, factor_id, round_number, val_ic, holdout_ic))
        
        # Log the failure reason if the math broke down
        if failure_code != "NONE":
            cursor.execute('''
                INSERT INTO diagnostics (run_id, failure_code, suggested_action)
                VALUES (?, ?, ?)
            ''', (run_id, failure_code, "Reduce mathematical complexity. Check for Look-Ahead Bias."))
            
        conn.commit()
        conn.close()
        print(f"   ✅ Logged to Ledger as Round {round_number}.")

# ==========================================
# THE TEST PROTOCOL
# ==========================================
if __name__ == "__main__":
    # We generate a "Dummy Data Lake" to prove the engine works.
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="B")
    df = pd.DataFrame({'date': dates})
    
    # 1. We create random stock returns
    df['forward_return'] = np.random.normal(0, 0.01, len(df))
    
    # 2. We simulate an OVERFIT Factor. 
    # It predicts the past perfectly (Validation), but produces total garbage in the future (Holdout).
    val_mask = df['date'] < "2023-01-01"
    holdout_mask = df['date'] >= "2023-01-01"
    
    df.loc[val_mask, 'factor_score'] = df.loc[val_mask, 'forward_return'] + np.random.normal(0, 0.02, sum(val_mask))
    df.loc[holdout_mask, 'factor_score'] = np.random.normal(0, 0.02, sum(holdout_mask))

    # Run the Engine!
    evaluator = AlphaEvaluator()
    evaluator.run_evaluation("FAC-001_Range_Repair", df)