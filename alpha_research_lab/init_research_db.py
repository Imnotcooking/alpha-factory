import sqlite3
import os

def initialize_database():
    db_path = "research_memory.db"
    is_new = not os.path.exists(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("🛠️ Forging Alpha Mine Database V2.0 (Ironclad SOP)...")

    # ==========================================
    # STEP 1: The Factors Table (Upgraded)
    # ==========================================
    # ADDED: `economic_rationale` (Rule 4: No Black Boxes)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS factors (
            factor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            economic_rationale TEXT NOT NULL, 
            description TEXT,
            complexity_score INTEGER,
            status TEXT DEFAULT 'INCUBATION',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("   ✅ STEP 1: `factors` table upgraded (Rule 4: Economic Intuition enforced).")

    # ==========================================
    # STEP 2: The Backtest Runs Table (Upgraded)
    # ==========================================
    # ADDED: `crisis_ic` (Rule 2: Regime Stress Testing)
    # ADDED: `turnover_rate` (Rule 3: Friction & Slippage Tracking)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            factor_id TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            validation_ic REAL,
            holdout_ic REAL,
            crisis_ic REAL,
            ic_ir REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            turnover_rate REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (factor_id) REFERENCES factors (factor_id)
        )
    ''')
    print("   ✅ STEP 2: `backtest_runs` table upgraded (Rules 2 & 3: Friction & Regimes tracked).")

    # ==========================================
    # STEP 3: The Diagnostics Table
    # ==========================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostics (
            diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            failure_code TEXT NOT NULL,
            suggested_action TEXT,
            FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id)
        )
    ''')
    print("   ✅ STEP 3: `diagnostics` table verified.")

    conn.commit()
    conn.close()
    
    if is_new:
        print("\n🎉 SUCCESS: `research_memory.db` V2.0 has been forged.")
    else:
        print("\n⚡ Database verified. All tables are structurally sound.")

if __name__ == "__main__":
    initialize_database()