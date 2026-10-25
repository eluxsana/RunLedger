from __future__ import annotations

import functools
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from runledger.dataset import ingest as ingest_data
from runledger.dataset import jsonl_samples, synthetic_provenance, synthetic_samples, verify_dataset
from runledger.evidence import compare_exports, export_run, verify_export
from runledger.ledger import Ledger
from runledger.runner import ExecutionOptions, create_run, execute
from runledger.schema import Provenance, ScorerConfig
from runledger.sources import ARC_REVISION, SMOL_MODEL, SMOL_REVISION, import_arc
from runledger.util import canonical_json, file_digest

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="RunLedger: versioned datasets, resumable CPU evaluations, inspectable evidence.",
)


def guarded(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, KeyError, OSError, RuntimeError, sqlite3.Error) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapped


def emit(value):
    typer.echo(canonical_json(value))


def workspace(ctx: typer.Context) -> Path:
    return ctx.obj["workspace"]


@app.callback()
def main(
    ctx: typer.Context,
    workspace_dir: Annotated[
        Path, typer.Option("--workspace", "-w", help="Local data/ledger/cache directory.")
    ] = Path(".runledger"),
):
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = workspace_dir.resolve()


@app.command()
@guarded
def synthetic(ctx: typer.Context, rows: int = 1024, seed: int = 17, words: int = 40):
    """Create a deterministic synthetic dataset for infrastructure tests."""
    result = ingest_data(
        synthetic_samples(rows, seed, words), workspace(ctx), synthetic_provenance(seed)
    )
    emit(
        {"dataset_id": result.dataset_id, "rows": result.row_count, "model_quality_evidence": False}
    )


@app.command("ingest")
@guarded
def ingest_command(
    ctx: typer.Context,
    input_path: Path,
    source_name: str,
    source_uri: str,
    source_license: str,
    split: str = "test",
):
    """Validate and version a JSONL dataset; require explicit provenance."""
    provenance = Provenance(
        name=source_name,
        uri=source_uri,
        revision=file_digest(input_path),
        license=source_license,
        split=split,
        description="User-supplied JSONL; revision is the input-file SHA-256.",
    )
    result = ingest_data(jsonl_samples(input_path), workspace(ctx), provenance)
    emit({"dataset_id": result.dataset_id, "rows": result.row_count})


@app.command()
@guarded
def arc(
    ctx: typer.Context,
    limit: int = 128,
    subset: str = "ARC-Easy",
    split: str = "validation",
    revision: str = ARC_REVISION,
):
    """Download a pinned public ARC subset; preserve source order and choice labels."""
    result = import_arc(workspace(ctx), subset=subset, split=split, limit=limit, revision=revision)
    emit({"dataset_id": result.dataset_id, "rows": result.row_count})


@app.command("fetch-model")
@guarded
def fetch_model_command(ctx: typer.Context, model: str = SMOL_MODEL, revision: str = SMOL_REVISION):
    """Download public safetensors; resolve the revision and hash every model asset."""
    from runledger.scorers import fetch_model

    config, path = fetch_model(model, revision, workspace(ctx))
    emit({"model_config": str(path), "model_id": config.model_id, "revision": config.revision})


@app.command()
@guarded
def verify(ctx: typer.Context, dataset_id: str):
    """Recheck schema, shard bytes, duplicate IDs, labels, and logical content."""
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("Dataset ID must be a SHA-256 digest")
    result = verify_dataset(workspace(ctx) / "datasets" / dataset_id)
    emit({"verified": True, "dataset_id": result.dataset_id, "rows": result.row_count})


@app.command()
@guarded
def run(
    ctx: typer.Context,
    dataset_id: str,
    model_config: Path | None = None,
    batch_size: int = 16,
    workers: int = 2,
    lease: float = 10.0,
    timeout: float = 600.0,
    crash_stage: str | None = None,
):
    """Create or resume the identical run. Default scorer is the hash fixture."""
    config = (
        ScorerConfig.model_validate_json(model_config.read_text())
        if model_config
        else ScorerConfig()
    )
    run_id = create_run(workspace(ctx), dataset_id, config, batch_size)
    emit({"run_id": run_id, "backend": config.backend, "status": "starting"})
    counts = execute(
        workspace(ctx),
        run_id,
        ExecutionOptions(
            workers=workers, lease_seconds=lease, timeout_seconds=timeout, crash_stage=crash_stage
        ),
    )
    emit({"run_id": run_id, **counts})
    if counts["failed"]:
        raise typer.Exit(1)


@app.command()
@guarded
def resume(
    ctx: typer.Context,
    run_id: str,
    workers: int = 2,
    lease: float = 10.0,
    timeout: float = 600.0,
    max_attempts: int = 3,
):
    """Resume pending/expired work under the original code, environment and scoring config."""
    counts = execute(
        workspace(ctx),
        run_id,
        ExecutionOptions(
            workers=workers, lease_seconds=lease, timeout_seconds=timeout, max_attempts=max_attempts
        ),
    )
    emit({"run_id": run_id, **counts})
    if counts["failed"]:
        raise typer.Exit(1)


@app.command("list")
@guarded
def list_command(ctx: typer.Context):
    """List recorded runs and their current job counts."""
    ledger = Ledger(workspace(ctx) / "ledger.sqlite")
    if not ledger.path.exists():
        emit([])
        return
    emit([{**run, **ledger.counts(run["run_id"])} for run in ledger.list_runs()])


@app.command()
@guarded
def inspect(ctx: typer.Context, run_id: str, events: bool = False):
    """Show the run receipt, job errors and optional execution events."""
    ledger = Ledger(workspace(ctx) / "ledger.sqlite")
    result = {
        "run": ledger.run(run_id),
        "counts": ledger.counts(run_id),
        "jobs": ledger.jobs(run_id),
    }
    if events:
        result["events"] = ledger.events(run_id)
    emit(result)


@app.command("export")
@guarded
def export_command(ctx: typer.Context, run_id: str, output: Path):
    """Write a complete run to a new, portable HTML/JSON/Parquet evidence bundle."""
    result = export_run(workspace(ctx), run_id, output)
    emit({"export": str(result.resolve()), "report": str((result / "index.html").resolve())})


@app.command("verify-export")
@guarded
def verify_export_command(path: Path):
    """Recompute file hashes, the result digest and DuckDB metrics."""
    summary = verify_export(path)
    emit({"verified": True, "run_id": summary["run_id"], "result_sha256": summary["result_sha256"]})


@app.command()
@guarded
def compare(left: Path, right: Path, report: Path | None = None):
    """Compare aligned samples; optionally write an offline prediction-drift report."""
    comparison = compare_exports(left, right)
    if report is not None:
        from runledger.comparison import render_comparison

        render_comparison(comparison, report)
    emit(comparison)


@app.command()
@guarded
def benchmark(
    ctx: typer.Context,
    output: Path = Path("artifacts/loading.json"),
    rows: int = 50000,
    words: int = 80,
    repeats: int = 5,
):
    """Measure materialized JSONL/Parquet versus streaming Parquet in fresh processes."""
    from runledger.benchmark import benchmark as run_benchmark

    result = run_benchmark(
        workspace(ctx) / "benchmark", output, rows=rows, words=words, repeats=repeats
    )
    emit({"output": str(output.resolve()), "aggregates": result["aggregates"]})


@app.command("failure-study")
@guarded
def failure_study_command(
    ctx: typer.Context, output: Path = Path("artifacts/failure-study"), rows: int = 256
):
    """Run baseline + three real SIGKILL experiments; verify exact result equality."""
    from runledger.study import failure_study

    emit(failure_study(workspace(ctx) / "failure-study", output, rows=rows))


@app.command()
@guarded
def demo(ctx: typer.Context, output: Path = Path("artifacts/demo"), rows: int = 384):
    """Offline CPU demo: synthetic data, two workers, a mid-commit crash and recovery."""
    manifest = ingest_data(synthetic_samples(rows), workspace(ctx), synthetic_provenance(17))
    run_id = create_run(workspace(ctx), manifest.dataset_id, ScorerConfig(), batch_size=16)
    counts = execute(
        workspace(ctx),
        run_id,
        ExecutionOptions(
            workers=2, lease_seconds=1, crash_stage="during-commit", timeout_seconds=120
        ),
    )
    if counts["failed"]:
        raise RuntimeError("Demo did not complete; inspect run events")
    export_run(workspace(ctx), run_id, output)
    emit({"run_id": run_id, "report": str((output / "index.html").resolve()), **counts})
