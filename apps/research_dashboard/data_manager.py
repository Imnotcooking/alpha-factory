import sqlite3
import pandas as pd
import os
import sys

UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import BASE_DIR, DB_PATH, LOGS_DIR

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

    def _get_table_columns(self, table_name: str) -> set[str]:
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            conn.close()
            return {row[1] for row in rows}
        except Exception:
            return set()

    def get_all_runs(self) -> pd.DataFrame:
        """Fetches the master ledger of all backtest runs."""
        run_columns = self._get_table_columns("backtest_runs")

        def run_col(column_name: str) -> str:
            if column_name in run_columns:
                return f"r.{column_name}"
            return f"NULL AS {column_name}"

        order_by = "r.timestamp DESC" if "timestamp" in run_columns else "r.run_id DESC"

        query = f"""
            SELECT r.run_id, r.factor_id, f.name, f.category AS factor_category,
                   {run_col("round_number")},
                   {run_col("validation_ic")},
                   {run_col("holdout_ic")},
                   {run_col("total_trades")},
                   {run_col("annualized_return")},
                   {run_col("sharpe_ratio")},
                   {run_col("max_drawdown")},
                   {run_col("turnover_rate")},
                   {run_col("asset_class")},
                   {run_col("market_vertical")},
                   {run_col("dataset_id")},
                   {run_col("universe_id")},
                   {run_col("data_frequency")},
                   {run_col("data_vendor")},
                   {run_col("execution_assumption")},
                   {run_col("universe_size")},
                   {run_col("validation_rows")},
                   {run_col("holdout_rows")},
                   {run_col("crisis_rows")},
                   {run_col("traded_tickers")},
                   {run_col("returns_file_path")},
                   {run_col("evaluation_geometry")},
                   {run_col("execution_mode")},
                   {run_col("data_execution_reality")},
                   {run_col("stat_research_family")},
                   {run_col("research_family")},
                   {run_col("backtest_engine")},
                   {run_col("runner")},
                   {run_col("runner_name")},
                   {run_col("engine_type")},
                   {run_col("model_type")},
                   {run_col("model_family")},
                   {run_col("strategy_type")},
                   {run_col("timestamp")},
                   d.failure_code, d.suggested_action
            FROM backtest_runs r
            JOIN factors f ON r.factor_id = f.factor_id
            LEFT JOIN diagnostics d ON r.run_id = d.run_id
            ORDER BY {order_by}
        """
        df = self._fetch_query(query)
        # Defensive schema normalization for legacy DBs / stale results
        expected_cols = [
            "run_id", "factor_id", "name", "factor_category", "round_number", "validation_ic", "holdout_ic",
            "total_trades", "annualized_return", "sharpe_ratio", "max_drawdown", "turnover_rate",
            "asset_class", "market_vertical", "dataset_id", "universe_id", "data_frequency",
            "data_vendor", "execution_assumption", "universe_size", "validation_rows", "holdout_rows",
            "crisis_rows", "traded_tickers", "returns_file_path",
            "evaluation_geometry", "execution_mode", "data_execution_reality", "stat_research_family",
            "research_family", "backtest_engine", "runner", "runner_name", "engine_type", "model_type",
            "model_family", "strategy_type",
            "timestamp", "failure_code", "suggested_action"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def get_run_record(self, run_id: str) -> pd.Series:
        """Fetch a wide backtest run row for detail views."""
        query = """
            SELECT r.*, f.name AS factor_name, f.category AS factor_category,
                   f.economic_rationale AS factor_rationale
            FROM backtest_runs r
            LEFT JOIN factors f ON r.factor_id = f.factor_id
            WHERE r.run_id = ?
            LIMIT 1
        """
        df = self._fetch_query(query, (run_id,))
        if df.empty:
            return pd.Series(dtype=object)
        return df.iloc[0]

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

    def get_run_returns(self, run_id: str, returns_path: str | None = None) -> pd.DataFrame:
        """Safely loads and cleans the return CSV for a specific run."""
        file_path = self._resolve_returns_file_path(returns_path)
        if not file_path:
            file_path = self._get_returns_file_path(run_id)

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

    def _resolve_returns_file_path(self, file_path: str | None) -> str:
        if file_path is None or pd.isna(file_path) or not str(file_path).strip():
            return ""

        repo_root = os.fspath(BASE_DIR)
        base_dir = repo_root
        file_path = str(file_path)
        legacy_artifact_root = os.path.join(
            "runtime",
            "artifacts",
            "research",
            "alpha_lab",
        )
        canonical_artifact_root = os.path.join(
            "runtime",
            "artifacts",
            "research",
        )

        def legacy_rewrite(path: str) -> str:
            return path.replace(legacy_artifact_root, canonical_artifact_root)

        candidates = []
        if os.path.isabs(file_path):
            candidates.append(file_path)
            rewritten = legacy_rewrite(file_path)
            if rewritten != file_path:
                candidates.append(rewritten)
        else:
            rewritten = legacy_rewrite(file_path)
            if rewritten != file_path:
                candidates.append(os.path.join(repo_root, rewritten))
        if file_path.startswith("execution_logs/"):
            candidates.append(
                os.path.join(LOGS_DIR, file_path.removeprefix("execution_logs/"))
            )
        candidates.extend(
            [
                os.path.join(repo_root, file_path),
                os.path.join(base_dir, file_path),
                os.path.join(os.path.dirname(LOGS_DIR), file_path),
            ]
        )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0] if candidates else ""

    def _get_returns_file_path(self, run_id: str) -> str:
        path_df = self._fetch_query(
            "SELECT returns_file_path FROM backtest_runs WHERE run_id = ?",
            (run_id,),
        )
        if not path_df.empty and "returns_file_path" in path_df.columns:
            resolved_path = self._resolve_returns_file_path(
                path_df["returns_file_path"].iloc[0]
            )
            if resolved_path:
                return resolved_path

        return os.path.join(LOGS_DIR, "returns", f"returns_{run_id}.csv")
        
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

    def feature_importance_paths(
        self,
        run_id: str,
        factor_id: str | None = None,
        *,
        include_factor_level: bool = True,
    ) -> list[str]:
        """Return candidate feature-importance artifact paths in lookup order."""
        identifiers = [str(run_id)] if run_id else []
        if include_factor_level and factor_id:
            factor_key = str(factor_id)
            if factor_key not in identifiers:
                identifiers.append(factor_key)

        roots = [
            os.path.join(LOGS_DIR, "feature_importance"),
            os.path.join(LOGS_DIR, "returns", "feature_importance"),
        ]
        return [
            os.path.join(root, f"feature_importance_{identifier}.csv")
            for identifier in identifiers
            for root in roots
        ]

    def has_feature_importance(
        self,
        run_id: str,
        factor_id: str | None = None,
        *,
        include_factor_level: bool = True,
    ) -> bool:
        """True when a run or eligible legacy factor-level importance artifact exists."""
        return any(
            os.path.exists(path)
            for path in self.feature_importance_paths(
                run_id,
                factor_id=factor_id,
                include_factor_level=include_factor_level,
            )
        )

    def get_feature_importance(self, run_id: str, factor_id: str | None = None) -> pd.DataFrame:
        """Safely loads the ML Feature Importance CSV."""
        for file_path in self.feature_importance_paths(run_id, factor_id=factor_id):
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
        return pd.DataFrame()

    def get_shap_dna(self) -> pd.DataFrame:
        """Safely loads the Regime-based SHAP DNA matrix for the ML Autopsy."""
        candidates = [
            os.path.join(LOGS_DIR, "diagnostics", "shap_regime_dna.csv"),
            os.path.join(LOGS_DIR, "shap_regime_dna.csv"),
        ]
        for file_path in candidates:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
        return pd.DataFrame()
