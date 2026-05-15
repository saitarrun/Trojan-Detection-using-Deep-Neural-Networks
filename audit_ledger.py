"""
Append-only audit ledger using SQLite WAL mode.
Every scan submission and completion writes one immutable row.
Never UPDATE or DELETE rows — treat this as a blockchain ledger.
"""

import sqlite3
import hashlib
import os
import time
from typing import Optional

DB_PATH = os.environ.get("AUDIT_DB_PATH", "audit_ledger.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            event_type TEXT NOT NULL,  -- 'submitted' | 'completed' | 'failed'
            client_token_hash TEXT,    -- SHA256 of API token (not the token itself)
            model_sha256 TEXT,
            model_size_bytes INTEGER,
            verdict TEXT,
            fusion_score REAL,
            ood_detected INTEGER,      -- 0 or 1
            report_hash TEXT,          -- SHA256 of the full JSON report
            timestamp_utc REAL NOT NULL,
            extra_json TEXT            -- JSON blob for extensibility
        )
    """)
    conn.commit()
    return conn


def log_submission(
    task_id: str,
    client_token: str,
    model_sha256: str,
    model_size_bytes: int,
) -> None:
    """Call when a scan task is submitted."""
    token_hash = hashlib.sha256(client_token.encode()).hexdigest() if client_token else None
    conn = _get_conn()
    conn.execute(
        """INSERT INTO audit_log
           (task_id, event_type, client_token_hash, model_sha256, model_size_bytes, timestamp_utc)
           VALUES (?, 'submitted', ?, ?, ?, ?)""",
        (task_id, token_hash, model_sha256, model_size_bytes, time.time()),
    )
    conn.commit()
    conn.close()


def log_completion(
    task_id: str,
    verdict: str,
    fusion_score: float,
    ood_detected: bool,
    report_hash: str,
) -> None:
    """Call when a scan task completes successfully."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO audit_log
           (task_id, event_type, verdict, fusion_score, ood_detected, report_hash, timestamp_utc)
           VALUES (?, 'completed', ?, ?, ?, ?, ?)""",
        (task_id, verdict, fusion_score, int(ood_detected), report_hash, time.time()),
    )
    conn.commit()
    conn.close()


def log_failure(task_id: str, error_trace_id: str) -> None:
    """Call when a scan task fails."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO audit_log
           (task_id, event_type, extra_json, timestamp_utc)
           VALUES (?, 'failed', ?, ?)""",
        (task_id, f'{{"trace_id":"{error_trace_id}"}}', time.time()),
    )
    conn.commit()
    conn.close()


def query_log(since_timestamp: Optional[float] = None, limit: int = 100) -> list[dict]:
    """Return audit rows. Never allows DELETE or UPDATE."""
    conn = _get_conn()
    if since_timestamp:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE timestamp_utc >= ? ORDER BY id DESC LIMIT ?",
            (since_timestamp, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM audit_log LIMIT 0").description or
            [("id",), ("task_id",), ("event_type",), ("client_token_hash",),
             ("model_sha256",), ("model_size_bytes",), ("verdict",),
             ("fusion_score",), ("ood_detected",), ("report_hash",),
             ("timestamp_utc",), ("extra_json",)]]
    conn.close()
    # Safer column name extraction
    conn2 = _get_conn()
    cursor = conn2.execute("SELECT * FROM audit_log LIMIT 0")
    col_names = [d[0] for d in cursor.description]
    conn2.close()
    return [dict(zip(col_names, row)) for row in rows]
