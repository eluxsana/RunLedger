"""Portable, self-verifying exports. DuckDB reads snapshots, never the live job ledger."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runledger.dataset import load_manifest
from runledger.ledger import Ledger
from runledger.util import (
    IntegrityError,
    atomic_json,
    canonical_json,
    digest,
    file_digest,
    percentile,
    safe_child,
    utc_now,
    wilson_interval,
)

RESULT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("choice_scores", pa.list_(pa.float64()), nullable=False),
        pa.field("token_counts", pa.list_(pa.int32()), nullable=False),
        pa.field("prediction", pa.int32(), nullable=False),
        pa.field("target", pa.int32(), nullable=False),
        pa.field("correct", pa.bool_(), nullable=False),
    ]
)


def result_digest(predictions) -> str:
    result = hashlib.sha256()
    for prediction in predictions:
        result.update((canonical_json(prediction) + "\n").encode())
    return result.hexdigest()


def query_results(path: Path) -> dict:
    with duckdb.connect() as connection:
        connection.read_parquet(str(path)).create_view("results")
        row = connection.execute(
            "SELECT count(*), coalesce(sum(correct::INTEGER),0), count(DISTINCT sample_id) FROM results"
        ).fetchone()
        by_target = connection.execute(
            "SELECT target,count(*) AS n,sum(correct::INTEGER) AS correct "
            "FROM results GROUP BY target ORDER BY target"
        ).fetchall()
    total, correct, unique = map(int, row)
    return {
        "samples": total,
        "unique_sample_ids": unique,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_wilson_95": wilson_interval(correct, total),
        "by_target": [
            {"target": int(target), "samples": int(n), "correct": int(k)}
            for target, n, k in by_target
        ],
    }


def export_run(workspace: Path, run_id: str, output: Path) -> Path:
    ledger = Ledger(workspace / "ledger.sqlite")
    run = ledger.run(run_id)
    counts = ledger.counts(run_id)
    if (
        counts["pending"]
        or counts["running"]
        or counts["failed"]
        or counts["results"] != run["expected_rows"]
    ):
        raise IntegrityError("Only complete, successful runs can become evidence exports")
    if output.exists():
        raise FileExistsError(f"Export exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output.parent))
    try:
        buffer = []
        hasher = hashlib.sha256()
        with pq.ParquetWriter(
            stage / "results.parquet", RESULT_SCHEMA, compression="zstd"
        ) as writer:
            for prediction in ledger.iter_results(run_id):
                hasher.update((canonical_json(prediction) + "\n").encode())
                buffer.append(prediction)
                if len(buffer) == 1024:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=RESULT_SCHEMA))
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=RESULT_SCHEMA))
        metrics = query_results(stage / "results.parquet")
        if (
            metrics["samples"] != run["expected_rows"]
            or metrics["unique_sample_ids"] != metrics["samples"]
        ):
            raise IntegrityError("Export has missing or duplicate results")
        jobs, events = ledger.jobs(run_id), ledger.events(run_id)
        commits = [
            event["detail"]["duration_seconds"]
            for event in events
            if event["kind"] == "job_committed"
        ]
        executions = [event["detail"] for event in events if event["kind"] == "execution_ended"]
        elapsed = sum(event["duration_seconds"] for event in executions)
        manifest = load_manifest(workspace / "datasets" / run["spec"]["dataset_id"])
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": utc_now(),
            "run_created_at": run["created_at"],
            "spec": run["spec"],
            "dataset": manifest.model_dump(),
            "counts": counts,
            "metrics": metrics,
            "result_sha256": hasher.hexdigest(),
            "result_semantics": "Ordered canonical sample predictions; excludes timing and worker identity.",
            "model_quality_evidence": run["spec"]["scorer"]["backend"] == "hf",
            "operations": {
                "jobs": len(jobs),
                "attempts": sum(job["attempts"] for job in jobs),
                "retried_jobs": sum(job["attempts"] > 1 for job in jobs),
                "event_counts": dict(Counter(event["kind"] for event in events)),
                "execution_seconds": elapsed,
                "execution_sessions": len(executions),
                "samples_per_second": metrics["samples"] / elapsed if elapsed else None,
                "job_p50_seconds": percentile(commits, 0.5),
                "job_p95_seconds": percentile(commits, 0.95),
            },
            "limits": [
                "Single-machine CPU execution; SQLite WAL requires a local filesystem.",
                "Execution can repeat; only the accepted, fenced commit is unique per sample within a run.",
                "Timing includes worker startup, model load and retries after preflight; excludes downloading and dataset verification.",
                "Job latency covers the successful attempt; failed-attempt cost is included only in execution timing.",
                "Wilson intervals summarize Bernoulli uncertainty; a convenience subset is not representative proof of full-benchmark quality.",
                "Hashes detect changes relative to this bundle; they are not a trusted signature or proof of authenticity.",
            ],
        }
        atomic_json(stage / "summary.json", summary)
        atomic_json(stage / "jobs.json", jobs)
        # Avoid leaking raw errors, local file paths, or full question content into portable evidence.
        public_events = [
            {key: event[key] for key in ("event_id", "timestamp", "kind", "job_id")}
            | {"detail": {key: value for key, value in event["detail"].items() if key != "error"}}
            for event in events
        ]
        atomic_json(stage / "events.json", public_events)
        from runledger.report import render_report

        render_report(stage)
        files = {path.name: file_digest(path) for path in sorted(stage.iterdir()) if path.is_file()}
        atomic_json(stage / "bundle.json", {"schema_version": 1, "run_id": run_id, "files": files})
        os.rename(stage, output)
        return output
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def verify_export(path: Path) -> dict:
    bundle = json.loads((path / "bundle.json").read_text())
    if bundle.get("schema_version") != 1:
        raise IntegrityError("Unsupported evidence schema")
    required = {"summary.json", "jobs.json", "events.json", "results.parquet", "index.html"}
    if set(bundle["files"]) != required:
        raise IntegrityError("Evidence bundle file list is incomplete or unsupported")
    for name, checksum in bundle["files"].items():
        if file_digest(safe_child(path, name)) != checksum:
            raise IntegrityError(f"Evidence artifact changed: {name}")
    summary = json.loads((path / "summary.json").read_text())
    if digest(summary["spec"]) != summary["run_id"] or summary["run_id"] != bundle["run_id"]:
        raise IntegrityError("Run fingerprint differs from its exported specification")
    metrics = query_results(path / "results.parquet")
    if metrics != summary["metrics"]:
        raise IntegrityError("Recomputed metrics differ from exported metrics")
    predictions = (
        row
        for batch in pq.ParquetFile(path / "results.parquet").iter_batches(batch_size=1024)
        for row in batch.to_pylist()
    )
    if result_digest(predictions) != summary["result_sha256"]:
        raise IntegrityError("Logical result digest differs from exported predictions")
    return summary


def compare_exports(
    left: Path, right: Path, *, seed: int = 17, bootstrap_samples: int = 5000
) -> dict:
    a, b = verify_export(left), verify_export(right)
    if a["dataset"]["semantic_sha256"] != b["dataset"]["semantic_sha256"]:
        raise IntegrityError("Paired comparison requires the same ordered dataset content")
    rows_a, rows_b = (
        pq.read_table(left / "results.parquet").to_pylist(),
        pq.read_table(right / "results.parquet").to_pylist(),
    )
    if [(r["sample_id"], r["target"]) for r in rows_a] != [
        (r["sample_id"], r["target"]) for r in rows_b
    ]:
        raise IntegrityError("Paired sample IDs or labels differ")
    if not 100 <= bootstrap_samples <= 100_000:
        raise ValueError("Require 100–100,000 bootstrap samples")
    differences = np.array(
        [int(y["correct"]) - int(x["correct"]) for x, y in zip(rows_a, rows_b, strict=True)]
    )
    # For binary correctness differences the multinomial is exactly equivalent to paired resampling,
    # and requires O(bootstrap_samples) memory instead of O(samples * bootstrap_samples).
    probabilities = np.array([(differences == value).mean() for value in (-1, 0, 1)])
    draws = np.random.default_rng(seed).multinomial(
        len(differences), probabilities, size=bootstrap_samples
    )
    deltas = (draws[:, 2] - draws[:, 0]) / len(differences)
    return {
        "left_run": a["run_id"],
        "right_run": b["run_id"],
        "samples": len(differences),
        "identical_result_digest": a["result_sha256"] == b["result_sha256"],
        "identical_predictions": all(
            x["prediction"] == y["prediction"] for x, y in zip(rows_a, rows_b, strict=True)
        ),
        "improved": int((differences == 1).sum()),
        "regressed": int((differences == -1).sum()),
        "accuracy_delta_right_minus_left": float(differences.mean()),
        "paired_bootstrap_95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "same_scoring_configuration": a["spec"]["scorer"] == b["spec"]["scorer"],
        "prediction_changes": sum(
            x["prediction"] != y["prediction"] for x, y in zip(rows_a, rows_b, strict=True)
        ),
        "changed_samples": [
            {
                "sample_id": x["sample_id"],
                "target": x["target"],
                "left_prediction": x["prediction"],
                "right_prediction": y["prediction"],
                "outcome": "improved" if delta == 1 else "regressed" if delta == -1 else "changed",
            }
            for x, y, delta in zip(rows_a, rows_b, differences, strict=True)
            if x["prediction"] != y["prediction"]
        ],
        "interpretation": "Paired descriptive comparison on this fixed subset; no multiple-testing correction or population guarantee.",
    }
