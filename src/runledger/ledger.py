from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

from runledger.schema import Prediction
from runledger.util import IntegrityError, canonical_json, digest, utc_now

DDL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, created_at TEXT NOT NULL,
  expected_rows INTEGER NOT NULL CHECK(expected_rows>0)
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  job_index INTEGER NOT NULL, payload_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
  attempts INTEGER NOT NULL DEFAULT 0, lease_token TEXT, lease_until REAL,
  worker_id TEXT, last_error TEXT, started_at REAL, completed_at REAL,
  UNIQUE(run_id, job_index)
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(run_id,status,job_index);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT NOT NULL REFERENCES runs(run_id), sample_id TEXT NOT NULL,
  job_id INTEGER NOT NULL REFERENCES jobs(job_id), ordinal INTEGER NOT NULL,
  prediction_json TEXT NOT NULL,
  PRIMARY KEY(run_id,sample_id), UNIQUE(run_id,ordinal)
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  job_id INTEGER REFERENCES jobs(job_id), timestamp REAL NOT NULL,
  kind TEXT NOT NULL, detail_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id,event_id);
"""


class Ledger:
    """Short SQLite transactions arbitrate leases; model computation never holds a DB lock."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = path
        self.clock = clock

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise IntegrityError(f"Unsupported ledger schema version: {version}")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(DDL)
            connection.execute("PRAGMA user_version=1")

    @contextmanager
    def connection(self):
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _event(self, connection, run_id: str, kind: str, detail: dict, job_id=None) -> None:
        connection.execute(
            "INSERT INTO events(run_id,job_id,timestamp,kind,detail_json) VALUES(?,?,?,?,?)",
            (run_id, job_id, self.clock(), kind, canonical_json(detail)),
        )

    def create_run(self, spec: dict, expected_rows: int, tasks: Iterable[dict]) -> str:
        run_id = digest(spec)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT spec_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing:
                if existing[0] != canonical_json(spec):
                    raise IntegrityError("Existing run has a conflicting specification")
                return run_id
            connection.execute(
                "INSERT INTO runs VALUES(?,?,?,?)",
                (run_id, canonical_json(spec), utc_now(), expected_rows),
            )
            count = 0
            for index, payload in enumerate(tasks):
                if payload["ordinal"] != count or payload["count"] != len(payload["expected"]):
                    raise IntegrityError("Task plan contains a gap or inconsistent membership")
                count += payload["count"]
                connection.execute(
                    "INSERT INTO jobs(run_id,job_index,payload_json,status) VALUES(?,?,?,'pending')",
                    (run_id, index, canonical_json(payload)),
                )
            if count != expected_rows:
                raise IntegrityError("Task plan does not cover the dataset")
            self._event(connection, run_id, "run_created", {"expected_rows": expected_rows})
        return run_id

    def claim(
        self, run_id: str, worker_id: str, lease_seconds: float, max_attempts: int
    ) -> dict | None:
        with self.transaction() as connection:
            now = self.clock()
            exhausted = connection.execute(
                "SELECT job_id FROM jobs WHERE run_id=? "
                "AND (status='pending' OR (status='running' AND lease_until<=?)) AND attempts>=?",
                (run_id, now, max_attempts),
            ).fetchall()
            for job in exhausted:
                connection.execute(
                    "UPDATE jobs SET status='failed',last_error=? WHERE job_id=?",
                    ("Retry budget exhausted", job[0]),
                )
                self._event(
                    connection, run_id, "job_failed", {"reason": "attempts_exhausted"}, job[0]
                )
            row = connection.execute(
                "SELECT * FROM jobs WHERE run_id=? AND attempts<? "
                "AND (status='pending' OR (status='running' AND lease_until<=?)) "
                "ORDER BY job_index LIMIT 1",
                (run_id, max_attempts, now),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            if row["status"] == "running":
                self._event(
                    connection,
                    run_id,
                    "lease_reclaimed",
                    {"previous_worker": row["worker_id"], "attempt": row["attempts"]},
                    row["job_id"],
                )
            connection.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,lease_token=?,lease_until=?,"
                "worker_id=?,started_at=?,last_error=NULL WHERE job_id=?",
                (token, now + lease_seconds, worker_id, now, row["job_id"]),
            )
            self._event(
                connection,
                run_id,
                "job_claimed",
                {"worker": worker_id, "attempt": row["attempts"] + 1},
                row["job_id"],
            )
            return {
                "job_id": row["job_id"],
                "job_index": row["job_index"],
                "run_id": run_id,
                "token": token,
                "attempt": row["attempts"] + 1,
                "payload": json.loads(row["payload_json"]),
            }

    def heartbeat(self, job_id: int, token: str, lease_seconds: float) -> bool:
        with self.transaction() as connection:
            now = self.clock()
            cursor = connection.execute(
                "UPDATE jobs SET lease_until=? WHERE job_id=? AND lease_token=? "
                "AND status='running' AND lease_until>?",
                (now + lease_seconds, job_id, token, now),
            )
            return cursor.rowcount == 1

    def commit(self, claim: dict, predictions: list[Prediction], *, after_insert=None) -> bool:
        """Atomically insert every sample and complete the job, fenced by the active lease."""
        with self.transaction() as connection:
            now = self.clock()
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (claim["job_id"],)
            ).fetchone()
            if (
                not job
                or job["status"] != "running"
                or job["lease_token"] != claim["token"]
                or job["lease_until"] <= now
            ):
                return False
            payload = json.loads(job["payload_json"])
            expected = payload["expected"]
            actual = [
                {"sample_id": p.sample_id, "target": p.target, "choice_count": len(p.choice_scores)}
                for p in predictions
            ]
            if actual != expected:
                raise IntegrityError("Commit does not contain the exact ordered task membership")
            for offset, prediction in enumerate(predictions):
                prediction = Prediction.model_validate(prediction)
                connection.execute(
                    "INSERT INTO results VALUES(?,?,?,?,?)",
                    (
                        job["run_id"],
                        prediction.sample_id,
                        job["job_id"],
                        payload["ordinal"] + offset,
                        canonical_json(prediction.model_dump()),
                    ),
                )
                if after_insert:
                    after_insert(offset)
            connection.execute(
                "UPDATE jobs SET status='completed',completed_at=?,lease_until=NULL WHERE job_id=?",
                (now, job["job_id"]),
            )
            self._event(
                connection,
                job["run_id"],
                "job_committed",
                {
                    "samples": len(predictions),
                    "attempt": job["attempts"],
                    "duration_seconds": now - job["started_at"],
                },
                job["job_id"],
            )
        return True

    def fail(self, claim: dict, error: str, max_attempts: int) -> bool:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (claim["job_id"],)
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_token"] != claim["token"]
                or row["lease_until"] <= self.clock()
            ):
                return False
            status = "failed" if row["attempts"] >= max_attempts else "pending"
            connection.execute(
                "UPDATE jobs SET status=?,last_error=?,lease_until=NULL WHERE job_id=?",
                (status, error[:4000], row["job_id"]),
            )
            self._event(
                connection,
                row["run_id"],
                "job_error",
                {"error": error[:4000], "next_status": status},
                row["job_id"],
            )
            return True

    def event(self, run_id: str, kind: str, detail: dict) -> None:
        with self.transaction() as connection:
            self._event(connection, run_id, kind, detail)

    def run(self, run_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")
            return {**dict(row), "spec": json.loads(row["spec_json"])}

    def counts(self, run_id: str) -> dict[str, int]:
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        with self.connection() as connection:
            connection.execute("BEGIN")
            for row in connection.execute(
                "SELECT status,count(*) AS n FROM jobs WHERE run_id=? GROUP BY status", (run_id,)
            ):
                counts[row["status"]] = row["n"]
            counts["results"] = connection.execute(
                "SELECT count(*) FROM results WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        return counts

    def results(self, run_id: str) -> list[dict]:
        with self.connection() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT prediction_json FROM results WHERE run_id=? ORDER BY ordinal", (run_id,)
                )
            ]

    def iter_results(self, run_id: str):
        with self.connection() as connection:
            for row in connection.execute(
                "SELECT prediction_json FROM results WHERE run_id=? ORDER BY ordinal", (run_id,)
            ):
                yield json.loads(row[0])

    def events(self, run_id: str) -> list[dict]:
        with self.connection() as connection:
            return [
                {**dict(row), "detail": json.loads(row["detail_json"])}
                for row in connection.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY event_id", (run_id,)
                )
            ]

    def jobs(self, run_id: str) -> list[dict]:
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT job_index,status,attempts,last_error,worker_id,started_at,completed_at FROM jobs "
                    "WHERE run_id=? ORDER BY job_index",
                    (run_id,),
                )
            ]

    def list_runs(self) -> list[dict]:
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT run_id,created_at,expected_rows FROM runs ORDER BY created_at DESC"
                )
            ]
