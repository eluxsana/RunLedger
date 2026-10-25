"""Regenerate the portfolio evidence in fresh output directories (no overwrite)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from runledger.benchmark import benchmark
from runledger.evidence import compare_exports, export_run, verify_export
from runledger.runner import ExecutionOptions, create_run, execute
from runledger.schema import ScorerConfig
from runledger.scorers import fetch_model
from runledger.sources import SMOL_MODEL, SMOL_REVISION, import_arc
from runledger.study import failure_study
from runledger.util import atomic_json, canonical_json


def real_model_study(workspace: Path, output: Path):
    baseline = workspace / "hf-baseline"
    config, config_path = fetch_model(SMOL_MODEL, SMOL_REVISION, baseline)
    manifest = import_arc(baseline, limit=128)
    evidence = []
    for name in ("sum", "sum-recovery", "mean"):
        target = baseline
        scoring = config
        crash = None
        if name == "sum-recovery":
            target = workspace / "hf-recovery"
            shutil.copytree(baseline / "datasets", target / "datasets")
            shutil.copytree(baseline / "models", target / "models")
            crash = "during-commit"
        if name == "mean":
            scoring = ScorerConfig.model_validate(config.model_dump() | {"normalization": "mean"})
        run_id = create_run(target, manifest.dataset_id, scoring, batch_size=8)
        counts = execute(
            target,
            run_id,
            ExecutionOptions(workers=1, lease_seconds=3, timeout_seconds=900, crash_stage=crash),
        )
        if counts["failed"] or counts["results"] != 128:
            raise RuntimeError(f"Real model case {name} failed: {counts}")
        export_run(target, run_id, output / name)
        summary = verify_export(output / name)
        evidence.append(
            {
                "case": name,
                "run_id": run_id,
                "metrics": summary["metrics"],
                "operations": summary["operations"],
                "result_sha256": summary["result_sha256"],
            }
        )
        print(
            canonical_json({"completed": name, "accuracy": summary["metrics"]["accuracy"]}),
            flush=True,
        )
    recovered = compare_exports(output / "sum", output / "sum-recovery")
    if not recovered["identical_result_digest"]:
        raise RuntimeError("Real-model recovered scores differ from the uninterrupted run")
    comparison = compare_exports(output / "sum", output / "mean")
    result = {
        "cases": evidence,
        "recovery_comparison": recovered,
        "normalization_comparison": comparison,
        "selection": "First 128 ARC-Easy validation rows in pinned file order; fixed before scoring.",
        "purpose": "Functional evaluation, recovery proof and scoring-policy sensitivity; not a leaderboard claim.",
    }
    atomic_json(output / "study.json", result)
    print(
        canonical_json({"real_model_study": str(output), "model_config": str(config_path)}),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".runledger/reproduction"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduction"))
    parser.add_argument(
        "--hf", action="store_true", help="Download and evaluate the public 135M model on CPU"
    )
    parser.add_argument("--rows", type=int, default=100000, help="Synthetic loading benchmark size")
    parser.add_argument("--skip-loading", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_failures:
        print("Running real process-crash study…", flush=True)
        failure_study(args.workspace / "failures", args.output / "failures")
    if args.hf:
        print("Running pinned model study…", flush=True)
        real_model_study(args.workspace, args.output / "model")
    if not args.skip_loading:
        print("Running isolated loading benchmark…", flush=True)
        benchmark(
            args.workspace / "loading", args.output / "loading.json", rows=args.rows, repeats=5
        )
    print(canonical_json({"complete": True, "evidence": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
