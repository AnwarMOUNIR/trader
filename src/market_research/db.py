"""SQLite task state and checkpoint database for Market Research."""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/task_state.db")

def get_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DB_PATH):
    conn = get_db(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS acquisition_tasks (
                task_id TEXT PRIMARY KEY,
                security_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                feed TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                status TEXT NOT NULL, -- PENDING, IN_PROGRESS, COMPLETED, EMPTY, FAILED
                bars_count INTEGER DEFAULT 0,
                file_path TEXT,
                file_sha256 TEXT,
                attempts INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_acq_status ON acquisition_tasks(status);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_acq_sym ON acquisition_tasks(symbol);
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS derivation_tasks (
                task_id TEXT PRIMARY KEY,
                security_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL, -- 20min, 30min
                status TEXT NOT NULL,
                candle_count INTEGER DEFAULT 0,
                file_path TEXT,
                file_sha256 TEXT,
                error_message TEXT,
                updated_at TEXT NOT NULL
            );
        """)
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Initialized task state database at", DB_PATH)
