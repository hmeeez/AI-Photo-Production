from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class JobStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_size INTEGER NOT NULL,
                    source_mtime_ns INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'in_progress', 'completed', 'partial', 'failed')
                    ),
                    report_path TEXT,
                    error TEXT,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    UNIQUE(source_path, source_size, source_mtime_ns)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_source ON jobs(status, source_path)"
            )
            connection.execute(
                """
                UPDATE jobs SET status = 'pending', updated_utc = ?
                WHERE status = 'in_progress'
                """,
                (self._now(),),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def register(self, source: Path) -> tuple[int, bool]:
        resolved = str(source.resolve())
        stat = source.stat()
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    source_path, source_size, source_mtime_ns, status, created_utc, updated_utc
                ) VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (resolved, stat.st_size, stat.st_mtime_ns, now, now),
            )
            row = connection.execute(
                """
                SELECT id, status FROM jobs
                WHERE source_path = ? AND source_size = ? AND source_mtime_ns = ?
                """,
                (resolved, stat.st_size, stat.st_mtime_ns),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Could not register processing job for {source}.")
            return int(row["id"]), bool(cursor.rowcount)

    def pending(self, input_dir: Path) -> list[tuple[int, Path]]:
        resolved = str(input_dir.resolve())
        prefix = resolved.rstrip("\\/") + os.sep
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, source_path FROM jobs
                WHERE status = 'pending'
                  AND (source_path = ? OR substr(source_path, 1, ?) = ?)
                ORDER BY id
                """,
                (resolved, len(prefix), prefix),
            ).fetchall()
        return [(int(row["id"]), Path(row["source_path"])) for row in rows]

    def retry_failed(self, input_dir: Path) -> int:
        resolved = str(input_dir.resolve())
        prefix = resolved.rstrip("\\/") + os.sep
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status = 'pending', error = NULL, updated_utc = ?
                WHERE status IN ('failed', 'partial')
                  AND (source_path = ? OR substr(source_path, 1, ?) = ?)
                """,
                (self._now(), resolved, len(prefix), prefix),
            )
            return cursor.rowcount

    def set_status(
        self,
        job_id: int,
        status: str,
        report_path: str | None = None,
        error: str | None = None,
    ) -> None:
        if status not in {"pending", "in_progress", "completed", "partial", "failed"}:
            raise ValueError(f"Invalid job status: {status}")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, report_path = COALESCE(?, report_path), error = ?,
                    updated_utc = ?
                WHERE id = ?
                """,
                (status, report_path, error, self._now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Processing job {job_id} does not exist.")

    def counts(self, input_dir: Path) -> dict[str, int]:
        resolved = str(input_dir.resolve())
        prefix = resolved.rstrip("\\/") + os.sep
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM jobs
                WHERE source_path = ? OR substr(source_path, 1, ?) = ?
                GROUP BY status
                """,
                (resolved, len(prefix), prefix),
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}
