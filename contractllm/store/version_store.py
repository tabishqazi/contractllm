"""
Version store: read/write operations for contract definitions and runs.

This is the repository pattern — a clean abstraction over the raw SQL in db.py.
All persistence logic lives here so the rest of the codebase stays SQL-free.
"""
import json
from pathlib import Path
from typing import Any

from contractllm.store.db import get_connection, run_migrations, DEFAULT_DB_PATH
from contractllm.store.models import ContractDefinition, ContractRun


class VersionStore:
    """
    Repository for contract definitions and runs.
    Manages the SQLite database connection and provides
    a clean interface for persistence operations.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        """
        Initialize the version store, running migrations if needed.
        The store owns the connection lifecycle — connection-per-request pattern.
        For a local tool this is fine; for a web service we'd use a pool.
        """
        self.db_path = db_path
        self._conn = get_connection(db_path)
        run_migrations(self._conn)

    def register_contract(self, definition: ContractDefinition) -> None:
        """
        Register a contract definition (or update if name+version exists).
        Uses INSERT OR REPLACE so re-registering with the same name+version
        updates the existing row. This lets you re-register after code changes.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO contract_definitions
            (name, version, schema_hash, system_prompt, input_schema,
             output_schema, provider, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                definition.name,
                definition.version,
                definition.schema_hash,
                definition.system_prompt,
                json.dumps(definition.input_schema),
                json.dumps(definition.output_schema),
                definition.provider,
                definition.model,
                definition.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_definition(
        self, name: str, version: str
    ) -> ContractDefinition | None:
        """
        Retrieve a contract definition by name and version.
        Returns None if not found.
        """
        row = self._conn.execute(
            """
            SELECT * FROM contract_definitions
            WHERE name = ? AND version = ?
            """,
            (name, version),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_definition(row)

    def list_all(self) -> list[ContractDefinition]:
        """List all registered contract definitions."""
        rows = self._conn.execute(
            "SELECT * FROM contract_definitions ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_definition(row) for row in rows]

    def save_run(self, run: ContractRun) -> int:
        """
        Save a contract run to the database.
        Returns the row ID of the inserted record.
        """
        cursor = self._conn.execute(
            """
            INSERT INTO contract_runs
            (contract_name, contract_version, schema_hash, input_data,
             raw_output, parsed_output, retry_count, latency_ms, tokens_used,
             provider, model, succeeded, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.contract_name,
                run.contract_version,
                run.schema_hash,
                json.dumps(run.input_data),
                run.raw_output,
                json.dumps(run.parsed_output),
                run.retry_count,
                run.latency_ms,
                run.tokens_used,
                run.provider,
                run.model,
                1 if run.succeeded else 0,
                run.error,
                run.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_runs(
        self,
        contract_name: str,
        version: str | None = None,
        limit: int = 10,
    ) -> list[ContractRun]:
        """
        Get recent runs for a contract, optionally filtered by version.
        Ordered by created_at DESC so most recent runs come first.
        """
        if version:
            rows = self._conn.execute(
                """
                SELECT * FROM contract_runs
                WHERE contract_name = ? AND contract_version = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (contract_name, version, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM contract_runs
                WHERE contract_name = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (contract_name, limit),
            ).fetchall()

        return [self._row_to_run(row) for row in rows]

    def _row_to_definition(self, row: Any) -> ContractDefinition:
        """Convert a database row to a ContractDefinition model."""
        return ContractDefinition(
            name=row["name"],
            version=row["version"],
            schema_hash=row["schema_hash"],
            system_prompt=row["system_prompt"],
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]),
            provider=row["provider"],
            model=row["model"],
            created_at=row["created_at"],
        )

    def _row_to_run(self, row: Any) -> ContractRun:
        """Convert a database row to a ContractRun model."""
        return ContractRun(
            contract_name=row["contract_name"],
            contract_version=row["contract_version"],
            schema_hash=row["schema_hash"],
            input_data=json.loads(row["input_data"]),
            raw_output=row["raw_output"],
            parsed_output=json.loads(row["parsed_output"]),
            retry_count=row["retry_count"],
            latency_ms=row["latency_ms"],
            tokens_used=row["tokens_used"],
            provider=row["provider"],
            model=row["model"],
            succeeded=bool(row["succeeded"]),
            error=row["error"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
