import sqlite3
import pandas as pd
import os
from config import DB_PATH, LOGS_DIR

class DataManager:
    """Handles all data fetching and cleaning for the Alpha Mine UI."""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def _fetch_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Internal method to execute a query safely."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            print(f"Database Error: {e}")
            return pd.DataFrame()

    def get_all_runs(self) -> pd.DataFrame:
        """Fetches the master ledger of all backtest runs."""
        query = """
            SELECT r.run_id, r.factor_id, f.name, r.round_number, r.validation_ic, r.holdout_ic, 
                   r.total_trades, r.annualized_return, r.sharpe_ratio, r.max_drawdown, r.turnover_rate,
                   r.asset_class, r.universe_size, r.traded_tickers,
                   r.timestamp, d.failure_code, d.suggested_action
            FROM backtest_runs r
            JOIN factors f ON r.factor_id = f.factor_id
            LEFT JOIN diagnostics d ON r.run_id = d.run_id
            ORDER BY r.timestamp DESC
        """
        df = self._fetch_query(query)
        # Defensive schema normalization for legacy DBs / stale results
        expected_cols = [
            "run_id", "factor_id", "name", "round_number", "validation_ic", "holdout_ic",
            "total_trades", "annualized_return", "sharpe_ratio", "max_drawdown", "turnover_rate",
            "asset_class", "universe_size", "traded_tickers", "timestamp", "failure_code", "suggested_action"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def get_pareto_data(self, factor_name: str) -> pd.DataFrame:
        """Fetches bulletproofed data for the Pareto Frontier chart."""
        # 🛡️ THE FIX: Use a subquery to find the true factor_id linked to this name!
        query = """
            SELECT r.round_number, r.annualized_return, r.max_drawdown, r.turnover_rate, r.holdout_ic
            FROM backtest_runs r
            WHERE r.factor_id = (SELECT factor_id FROM factors WHERE name = ? LIMIT 1)
              AND r.annualized_return IS NOT NULL
        """
        df = self._fetch_query(query, (factor_name,))
        
        if not df.empty:
            df['Return (%)'] = df['annualized_return'] * 100
            df['Drawdown (%)'] = df['max_drawdown'] * 100
            df['holdout_ic'] = pd.to_numeric(df['holdout_ic'], errors='coerce').fillna(0.01)
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0.0)
            df['bubble_size'] = df['holdout_ic'].abs().clip(lower=0.01)
            
        return df

    def get_run_returns(self, run_id: str) -> pd.DataFrame:
        """Safely loads and cleans the return CSV for a specific run."""
        file_path = os.path.join(LOGS_DIR, "returns", f"returns_{run_id}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Standardize columns so the UI never crashes
        if 'gross_return' not in df.columns: df['gross_return'] = df.get('net_return', 0.0)
        if 'benchmark_return' not in df.columns: df['benchmark_return'] = 0.0
        if 'portfolio_leverage' not in df.columns: df['portfolio_leverage'] = 1.0
        if 'daily_turnover' not in df.columns: df['daily_turnover'] = df.get('turnover', 0.0)
        if 'net_return' not in df.columns: df['net_return'] = df.get('strategy_return', 0.0)
        
        return df
        
    def get_trade_ledger(self, run_id: str) -> pd.DataFrame:
        """Safely loads the Trade DNA ledger."""
        file_path = os.path.join(LOGS_DIR, "trades", f"trades_{run_id}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return pd.read_csv(file_path)

    # --- ADDED: Missing Methods for Correlation and ML Tabs ---
    
    def get_correlation_returns(self, run_dict: dict) -> pd.DataFrame:
        """
        Takes a dict of {factor_name: run_id} and returns a combined 
        DataFrame of their net returns for correlation analysis.
        """
        corr_data = {}
        for name, run_id in run_dict.items():
            df = self.get_run_returns(run_id)
            if not df.empty and 'net_return' in df.columns:
                df = df.set_index('date')
                corr_data[name] = df['net_return']
                
        if corr_data:
            return pd.DataFrame(corr_data).dropna()
        return pd.DataFrame()

    def get_feature_importance(self, run_id: str) -> pd.DataFrame:
        """Safely loads the ML Feature Importance CSV."""
        file_path = os.path.join(LOGS_DIR, "feature_importance", f"feature_importance_{run_id}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return pd.read_csv(file_path)