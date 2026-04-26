import sqlite3
import pandas as pd
import numpy as np
import scipy.stats as stats
import uuid
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

class AlphaEvaluator:
    def __init__(self, db_path="research_memory.db"):
        self.db_path = db_path

    def calculate_rank_ic(self, factor_values, forward_returns):
        """Calculates the Spearman Rank Correlation (Rank IC)"""
        valid_idx = ~np.isnan(factor_values) & ~np.isnan(forward_returns)
        if sum(valid_idx) < 10:  # Need at least 10 valid stock days to calculate correlation
            return 0.0
        
        ic, _ = stats.spearmanr(factor_values[valid_idx], forward_returns[valid_idx])
        return ic

    def run_evaluation(self, factor_id, df, split_date="2023-01-01"):
        print(f"🔬 Evaluating Candidate: {factor_id}")
        
        # ==========================================
        # RULE 5: THE $50M LIQUIDITY WALL
        # ==========================================
        # We calculate the Dollar Volume. If it's less than $50M, it's a micro-cap. 
        # We instantly drop it from the dataset before the AI is even allowed to look at it.
        original_len = len(df)
        if 'close' in df.columns and 'volume' in df.columns:
            df['dollar_volume'] = df['close'] * df['volume']
            df = df[df['dollar_volume'] >= 50_000_000]
            dropped_pct = (original_len - len(df)) / original_len * 100
            print(f"   🧱 Liquidity Filter: Dropped {dropped_pct:.1f}% of data (Micro-caps removed).")
        else:
            print("   ⚠️ WARNING: 'close' or 'volume' columns missing. Cannot enforce Liquidity Wall.")

        # ==========================================
        # RULE 2: THE REGIME SPLITTER
        # ==========================================
        
        crisis_start, crisis_end = "2025-02-01", "2022-05-31"
        
        validation_data = df[df['date'] < split_date]
        holdout_data = df[(df['date'] >= split_date) & ~((df['date'] >= crisis_start) & (df['date'] <= crisis_end))]
        crisis_data = df[(df['date'] >= crisis_start) & (df['date'] <= crisis_end)]
        
        # ==========================================
        # STEP 3: CALCULATE PREDICTIVE EDGE
        # ==========================================
        val_ic = self.calculate_rank_ic(validation_data['factor_score'], validation_data['forward_return'])
        holdout_ic = self.calculate_rank_ic(holdout_data['factor_score'], holdout_data['forward_return'])
        crisis_ic = self.calculate_rank_ic(crisis_data['factor_score'], crisis_data['forward_return'])
        
        # Mock turnover calculation (Normally calculated by taking the absolute rank changes day over day)
        turnover_rate = np.random.uniform(0.1, 0.4) 

        print(f"   -> Validation IC: {val_ic:.4f}")
        print(f"   -> Holdout IC:    {holdout_ic:.4f}")
        print(f"   -> Crisis IC:     {crisis_ic:.4f}")
        
        # ==========================================
        # THE DIAGNOSTICS ENGINE (The Executioner)
        # ==========================================
        failure_code = "NONE"
        suggested_action = "N/A"

        if len(df) < (original_len * 0.1): 
            # If 90% of the data was dropped by the liquidity filter, the factor ONLY works on micro-caps.
            failure_code = "untradable_illiquidity"
            suggested_action = "Factor relies on micro-caps. Redesign for large-cap dynamics."
        elif crisis_ic < -0.01:
            failure_code = "crisis_failure"
            suggested_action = "Factor blows up during liquidity crunches. Needs hedging."
        elif holdout_ic < 0:
            failure_code = "holdout_not_positive"
            suggested_action = "Factor decayed completely Out-Of-Sample. Overfit."
        elif holdout_ic < (val_ic * 0.4):
            failure_code = "severe_ic_decay"
            suggested_action = "Reduce formula complexity. Remove hardcoded parameters."

        if failure_code != "NONE":
            print(f"   ⚠️ DIAGNOSTIC FLAG: [{failure_code.upper()}]")
        else:
            print(f"   🟢 PASSED ALL INSTITUTIONAL HURDLES.")
            
        # ==========================================
        # LOG TO SQLITE DATABASE
        # ==========================================
        self._log_to_db(factor_id, val_ic, holdout_ic, crisis_ic, turnover_rate, failure_code, suggested_action)

    def _log_to_db(self, factor_id, val_ic, holdout_ic, crisis_ic, turnover_rate, failure_code, suggested_action):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        
        # Ensure the factor exists in the parent table, if not, auto-create it with placeholder rationale
        cursor.execute("SELECT factor_id FROM factors WHERE factor_id = ?", (factor_id,))
        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO factors (factor_id, name, category, economic_rationale) 
                VALUES (?, ?, ?, ?)
            ''', (factor_id, f"Auto-Gen {factor_id}", "Experimental", "Pending human thesis."))
        
        # Get round number
        cursor.execute("SELECT MAX(round_number) FROM backtest_runs WHERE factor_id = ?", (factor_id,))
        res = cursor.fetchone()[0]
        round_number = (res + 1) if res else 1
        
        # Insert Run Data
        cursor.execute('''
            INSERT INTO backtest_runs 
            (run_id, factor_id, round_number, validation_ic, holdout_ic, crisis_ic, turnover_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, factor_id, round_number, val_ic, holdout_ic, crisis_ic, turnover_rate))
        
        # Insert Diagnostics
        if failure_code != "NONE":
            cursor.execute('''
                INSERT INTO diagnostics (run_id, failure_code, suggested_action)
                VALUES (?, ?, ?)
            ''', (run_id, failure_code, suggested_action))
            
        conn.commit()
        conn.close()
        print(f"   💾 Logged to Database successfully.")

# ==========================================
# THE TEST PROTOCOL
# ==========================================
if __name__ == "__main__":
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="B")
    df = pd.DataFrame({'date': dates})
    
    # Generate dummy data including the new required columns
    df['forward_return'] = np.random.normal(0, 0.01, len(df))
    df['close'] = np.random.uniform(10, 150, len(df))
    
    # We intentionally make 50% of our fake universe "Micro-caps" (Low volume)
    # to prove the Liquidity Wall works.
    df['volume'] = np.where(np.random.rand(len(df)) > 0.5, 100_000, 2_000_000) 
    
    # Simulate an OVERFIT Factor that blows up during the 2022 Crisis.
    crisis_mask = (df['date'] >= "2022-01-01") & (df['date'] <= "2022-12-31")
    df.loc[~crisis_mask, 'factor_score'] = df['forward_return'] + np.random.normal(0, 0.02, sum(~crisis_mask))
    df.loc[crisis_mask, 'factor_score'] = -df['forward_return'] + np.random.normal(0, 0.02, sum(crisis_mask))

    evaluator = AlphaEvaluator()
    evaluator.run_evaluation("FAC-002_Short_Reversal", df)