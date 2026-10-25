"""Executable failure experiment: actual worker death, durable recovery, exact result comparison."""

from __future__ import annotations

import shutil
from pathlib import Path

from runledger.dataset import ingest, synthetic_provenance, synthetic_samples
from runledger.evidence import compare_exports, export_run, verify_export
from runledger.ledger import Ledger
from runledger.runner import ExecutionOptions, create_run, execute
from runledger.schema import ScorerConfig
from runledger.util import atomic_json, utc_now


def failure_study(workspace: Path, output: Path, *, rows: int = 256) -> dict:
    if output.exists():
        raise FileExistsError(f"Study output exists: {output}")
    if workspace.exists():
        raise FileExistsError("Failure study needs a fresh workspace to prove first-attempt faults")
    source = workspace / "baseline"
    manifest = ingest(
        synthetic_samples(rows), source, synthetic_provenance(17), shard_rows=256, row_group_rows=64
    )
    output.mkdir(parents=True)
    evidence = []
    for stage in (None, "before-commit", "during-commit", "after-commit"):
        name = stage or "baseline"
        target = workspace / name
        if stage:
            shutil.copytree(source / "datasets", target / "datasets")
        run_id = create_run(target, manifest.dataset_id, ScorerConfig(), batch_size=16)
        counts = execute(
            target,
            run_id,
            ExecutionOptions(workers=2, lease_seconds=1, timeout_seconds=120, crash_stage=stage),
        )
        if counts["results"] != rows or counts["failed"]:
            raise RuntimeError(f"Incomplete failure case: {name}")
        export_run(target, run_id, output / name)
        summary = verify_export(output / name)
        events = Ledger(target / "ledger.sqlite").events(run_id)
        killed = [
            event
            for event in events
            if event["kind"] == "worker_exited" and event["detail"]["exit_code"] in (-9, 137)
        ]
        comparison = compare_exports(output / "baseline", output / name)
        if not comparison["identical_result_digest"] or (stage and not killed):
            raise RuntimeError(f"Failure evidence invariant did not hold: {name}")
        evidence.append(
            {
                "case": name,
                "run_id": run_id,
                "samples": counts["results"],
                "result_sha256": summary["result_sha256"],
                "observed_sigkill_exits": len(killed),
                "retried_jobs": summary["operations"]["retried_jobs"],
                "attempts": summary["operations"]["attempts"],
                "execution_seconds": summary["operations"]["execution_seconds"],
                "identical_to_baseline": comparison["identical_result_digest"],
            }
        )
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "rows": rows,
        "backend": "deterministic hash fixture; not a model-quality evaluation",
        "cases": evidence,
        "claim": "Worker SIGKILL at three commit boundaries produced the same ordered result digest as an uninterrupted run.",
        "limits": "Single-host process-crash experiment. Not a power-loss, disk-loss, network-partition, or distributed-cluster test.",
    }
    atomic_json(output / "study.json", result)
    return result
