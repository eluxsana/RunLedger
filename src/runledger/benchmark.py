"""Fresh-process loading benchmark, with equal validation and digest work per method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from runledger.dataset import (
    from_arrow,
    ingest,
    iter_samples,
    load_manifest,
    synthetic_provenance,
    synthetic_samples,
)
from runledger.schema import Sample
from runledger.util import (
    atomic_json,
    canonical_json,
    environment_fingerprint,
    file_digest,
    percentile,
    source_fingerprint,
    utc_now,
)

METHODS = ("jsonl-materialized", "jsonl-streaming", "parquet-materialized", "parquet-streaming")


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measure(method: str, jsonl: Path, dataset_dir: Path, batch_size: int) -> dict:
    baseline = peak_rss_bytes()
    started = time.perf_counter()
    if method == "jsonl-materialized":
        with jsonl.open(encoding="utf-8") as stream:
            raw = [json.loads(line) for line in stream]
        samples = (Sample.model_validate(row) for row in raw)
    elif method == "jsonl-streaming":

        def stream_jsonl():
            with jsonl.open(encoding="utf-8") as stream:
                for line in stream:
                    yield Sample.model_validate(json.loads(line))

        samples = stream_jsonl()
    elif method == "parquet-materialized":
        manifest = load_manifest(dataset_dir)
        table = pq.read_table([str(dataset_dir / shard.path) for shard in manifest.shards])
        raw = table.to_pylist()
        samples = (from_arrow(row) for row in raw)
    elif method == "parquet-streaming":
        samples = iter_samples(dataset_dir, batch_size=batch_size)
    else:
        raise ValueError("Unknown loading method")
    hasher = hashlib.sha256()
    count = 0
    first_batch = None
    for sample in samples:
        hasher.update((canonical_json(sample.model_dump()) + "\n").encode())
        count += 1
        if count == batch_size:
            first_batch = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    return {
        "method": method,
        "samples": count,
        "semantic_sha256": hasher.hexdigest(),
        "total_seconds": elapsed,
        "first_batch_seconds": first_batch or elapsed,
        "samples_per_second": count / elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "baseline_peak_rss_bytes": baseline,
        "batch_size": batch_size,
    }


def benchmark(
    workspace: Path,
    output: Path,
    *,
    rows: int = 50000,
    words: int = 80,
    repeats: int = 5,
    batch_size: int = 256,
    seed: int = 17,
) -> dict:
    if not 100 <= rows <= 2_000_000 or not 2 <= repeats <= 20 or not 1 <= batch_size <= rows:
        raise ValueError("Require 100–2,000,000 rows, 2–20 repeats, and batch size <= rows")
    if output.exists():
        raise FileExistsError(f"Benchmark output exists: {output}")
    workspace.mkdir(parents=True, exist_ok=True)
    jsonl = workspace / f"synthetic-{rows}-{words}-{seed}.jsonl"
    # Preparing data is outside every measurement.
    with jsonl.open("w", encoding="utf-8") as stream:
        for sample in synthetic_samples(rows, seed=seed, text_words=words):
            stream.write(canonical_json(sample.model_dump()) + "\n")
    manifest = ingest(
        synthetic_samples(rows, seed=seed, text_words=words), workspace, synthetic_provenance(seed)
    )
    dataset_dir = workspace / "datasets" / manifest.dataset_id

    def trial(method):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "runledger.benchmark",
                "--worker",
                method,
                "--jsonl",
                str(jsonl),
                "--dataset",
                str(dataset_dir),
                "--batch-size",
                str(batch_size),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=900,
        )
        result = json.loads(process.stdout)
        if result["samples"] != rows or result["semantic_sha256"] != manifest.semantic_sha256:
            raise RuntimeError("Loading methods did not process identical ordered content")
        return result

    warmups = [trial(method) for method in METHODS]
    raw = []
    rng = random.Random(seed)
    for repetition in range(repeats):
        order = list(METHODS)
        rng.shuffle(order)
        for method in order:
            raw.append({"repetition": repetition, "order": len(raw), **trial(method)})
    aggregates = {}
    for method in METHODS:
        trials = [row for row in raw if row["method"] == method]
        aggregates[method] = {}
        for metric in (
            "total_seconds",
            "first_batch_seconds",
            "samples_per_second",
            "peak_rss_bytes",
        ):
            values = [row[metric] for row in trials]
            aggregates[method][metric] = {
                "median": percentile(values, 0.5),
                "min": min(values),
                "max": max(values),
            }
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "rows": rows,
        "prompt_words": words,
        "seed": seed,
        "repeats": repeats,
        "batch_size": batch_size,
        "dataset_id": manifest.dataset_id,
        "semantic_sha256": manifest.semantic_sha256,
        "jsonl_sha256": file_digest(jsonl),
        "jsonl_bytes": jsonl.stat().st_size,
        "parquet_bytes": sum(shard.bytes for shard in manifest.shards),
        "environment": environment_fingerprint("hash"),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "source_sha256": source_fingerprint(),
        "methodology": [
            "One unmeasured warm-up per method; measured order is shuffled with a fixed seed.",
            "Each trial uses a fresh Python process. OS page cache is warm or uncontrolled, never claimed cold.",
            "Timer starts after shared imports. All methods normalize, validate and hash every sample in the same order.",
            "Peak RSS is the whole process high-water mark including interpreter and libraries; import baseline is also recorded.",
            "Materialized methods retain all decoded records; JSONL streams one row and Parquet streams one batch. Workload is synthetic text, not tokenization or training throughput.",
            "Input files are prepared and checked outside timings; streaming measurements omit per-shard cryptographic verification.",
            "First-batch latency is time until batch_size validated and hashed records; this isolates the cost of full materialization.",
            "Medians and ranges describe repeated trials on one machine, not a universal speedup or scaling claim.",
        ],
        "aggregates": aggregates,
        "warmups": warmups,
        "trials": raw,
    }
    atomic_json(output, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=METHODS)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    print(canonical_json(measure(args.worker, args.jsonl, args.dataset, args.batch_size)))
