"""
Database layer for llm-contracts.

Why SQLite? Three reasons:
1. Zero setup — no server to run, no connection string to configure
2. File-based — the version store travels with the repo, can be committed
3. SQL queries — we need joins and aggregates for regression comparison

For a local-first tool, simplicity wins. If this became a hosted
multi-user service, we'd switch to PostgreSQL.
"""
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path.home() / ".llm-contracts" / "contracts.db"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Returns a SQLite connection with row_factory set to Row.
    Row factory means we get dict-like objects back, not plain tuples.
    Accepts both Path and str for convenience.
    """
    if isinstance(db_path, str):
        db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def run_migrations(conn: sqlite3.Connection) -> None:
    """
    Creates tables if they don't exist.
    Uses CREATE TABLE IF NOT EXISTS for safe migrations.
    For a production service we'd use Alembic for versioned migrations.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS contract_definitions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            version       TEXT NOT NULL,
            schema_hash   TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            input_schema  TEXT NOT NULL,   -- JSON stringified
            output_schema TEXT NOT NULL,   -- JSON stringified
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            UNIQUE(name, version)          -- One row per name+version pair
        );

        CREATE TABLE IF NOT EXISTS contract_runs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_name    TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            schema_hash      TEXT NOT NULL,
            input_data       TEXT NOT NULL,  -- JSON stringified
            raw_output       TEXT NOT NULL,
            parsed_output    TEXT NOT NULL,  -- JSON stringified
            retry_count      INTEGER NOT NULL DEFAULT 0,
            latency_ms       INTEGER NOT NULL,
            tokens_used      INTEGER NOT NULL,
            provider         TEXT NOT NULL,
            model            TEXT NOT NULL,
            succeeded        INTEGER NOT NULL,  -- SQLite has no BOOLEAN
            error            TEXT,
            created_at       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_runs_name_version
            ON contract_runs(contract_name, contract_version);

        CREATE INDEX IF NOT EXISTS idx_runs_created_at
            ON contract_runs(created_at DESC);
    """)
    conn.commit()
