import sqlite3
import os

def initialize_database():
    db_path = "research_memory.db"
    
    # Check if we are overwriting an existing lab
    is_new = not os.path.exists(db_path)
    
    # Connect to SQLite (this automatically creates the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("🛠️ Building the Alpha Mine Database Architecture...")

    # ==========================================
    # STEP 1: The Factors Table
    # ==========================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS factors (
            factor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            complexity_score INTEGER,
            status TEXT DEFAULT 'INCUBATION',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("   ✅ STEP 1: `factors` table created. (Inventory ready)")

    # ==========================================
    # STEP 2: The Backtest Runs Table
    # ==========================================
    # Note the FOREIGN KEY. This ensures every run MUST belong to a valid factor.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            factor_id TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            validation_ic REAL,
            holdout_ic REAL,
            ic_ir REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (factor_id) REFERENCES factors (factor_id)
        )
    ''')
    print("   ✅ STEP 2: `backtest_runs` table created. (Ledger ready)")

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
    print("   ✅ STEP 3: `diagnostics` table created. (Autopsy system ready)")

    # Save changes and close the connection
    conn.commit()
    conn.close()
    
    if is_new:
        print("\n🎉 SUCCESS: `research_memory.db` has been forged.")
    else:
        print("\n⚡ Database verified. All tables are structurally sound.")

if __name__ == "__main__":
    initialize_database()