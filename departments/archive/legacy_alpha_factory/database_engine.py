import sqlite3
import pandas as pd
import os
from datetime import datetime

print("Booting Alpha Factory Database Engine...")

# We place the database in the root folder so both Offline and Online environments can reach it
DB_PATH = "portfolio_memory.db"

def init_db():
    """Initializes the SQLite Database and builds the three core Ledgers."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. THE INVENTORY (What do we currently own?)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS open_positions (
        ticker TEXT PRIMARY KEY,
        strategy TEXT,
        quantity REAL,
        average_price REAL,
        current_price REAL,
        unrealized_pnl REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2. THE LEDGER (Every trade the AI ever makes)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_history (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ticker TEXT,
        action TEXT,
        quantity REAL,
        execution_price REAL,
        strategy TEXT,
        realized_pnl REAL
    )
    ''')

    # 3. THE SCOREBOARD (Daily Account Metrics for Streamlit)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio_metrics (
        date DATE PRIMARY KEY,
        total_value REAL,
        available_cash REAL,
        daily_pnl REAL,
        margin_used REAL
    )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Neural Ledger initialized successfully at {DB_PATH}.")

# --- HELPER FUNCTIONS FOR THE AI ---

def log_trade(ticker, action, quantity, price, strategy, realized_pnl=0.0):
    """The execution engine will call this immediately after IBKR fills an order."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trade_history (ticker, action, quantity, execution_price, strategy, realized_pnl)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, action, quantity, price, strategy, realized_pnl))
    conn.commit()
    conn.close()

def update_position(ticker, strategy, quantity, average_price):
    """Updates the inventory. If quantity hits 0, it removes the row."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if quantity <= 0:
        cursor.execute('DELETE FROM open_positions WHERE ticker = ?', (ticker,))
    else:
        # Upsert logic (Update if exists, Insert if new)
        cursor.execute('''
        INSERT INTO open_positions (ticker, strategy, quantity, average_price)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET 
            quantity=excluded.quantity,
            average_price=excluded.average_price,
            last_updated=CURRENT_TIMESTAMP
        ''', (ticker, strategy, quantity, average_price))
        
    conn.commit()
    conn.close()

def get_all_positions():
    """Streamlit will call this to render the Phase 1 holdings dashboard."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM open_positions", conn)
    conn.close()
    return df

if __name__ == "__main__":
    # When you run this script directly, it simply builds the blank database
    init_db()