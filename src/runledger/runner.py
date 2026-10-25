from __future__ import annotations

import math
import multiprocessing as mp
import os
import signal
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from runledger.dataset import read_task, task_batches, verify_dataset
from runledger.ledger import Ledger
from runledger.schema import ScorerConfig
from runledger.scorers import make_scorer, model_key, verify_model
from runledger.util import IntegrityError, environment_fingerprint, source_fingerprint


@dataclass(frozen=True)
class ExecutionOptions:
    workers: int = 2
    lease_seconds: float = 10.0
    max_attempts: int = 3
    timeout_seconds: float = 600.0
    crash_stage: str | None = None
    crash_job: int = 0
    delay_seconds: float = 0.0
    max_process_failures: int = 6

    def validate(self):
        if not all(
            math.isfinite(value)
            for value in (self.lease_seconds, self.timeout_seconds, self.delay_seconds)
        ):
            raise ValueError("Lease, timeout and delay must be finite")
        if not 1 <= self.workers <= 16 or self.lease_seconds < 0.3 or self.max_attempts < 1:
            raise ValueError(
                "Require 1–16 workers, lease >= 0.3 seconds, and positive max_attempts"
            )
        if self.timeout_seconds <= 0 or self.delay_seconds < 0 or self.max_process_failures < 0:
            raise ValueError("Timeout, delay or failure budget is invalid")
        if self.crash_stage not in (None, "before-commit", "during-commit", "after-commit"):
            raise ValueError("Unsupported fault-injection stage")


def abrupt_exit() -> None:
    if hasattr(signal, "SIGKILL"):
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(137)


class Heartbeat:
    def __init__(self, ledger: Ledger, claim: dict, lease_seconds: float):
        self.ledger, self.claim, self.lease_seconds = ledger, claim, lease_seconds
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="runledger-heartbeat", daemon=True)

    def _loop(self):
        while not self.stop.wait(self.lease_seconds / 3):
            try:
                alive = self.ledger.heartbeat(
                    self.claim["job_id"], self.claim["token"], self.lease_seconds
                )
                if not alive:
                    self.lost.set()
                    return
            except Exception:
                # A commit still checks the lease atomically; no optimistic acceptance.
                self.lost.set()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=35)


def worker_main(workspace_text: str, run_id: str, options_dict: dict):
    workspace = Path(workspace_text)
    options = ExecutionOptions(**options_dict)
    ledger = Ledger(workspace / "ledger.sqlite")
    run = ledger.run(run_id)
    config = ScorerConfig.model_validate(run["spec"]["scorer"])
    worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if config.backend == "hf":
        # Each worker owns its progress output. A thread lock avoids allocating an OS
        # semaphore that a deliberately SIGKILLed model-loader process cannot unregister.
        from tqdm import tqdm

        tqdm.set_lock(threading.RLock())
    try:
        scorer = make_scorer(config, workspace)
    except Exception as exc:
        ledger.event(
            run_id, "worker_initialization_failed", {"worker": worker_id, "error": str(exc)[:4000]}
        )
        raise
    ledger.event(run_id, "worker_ready", {"worker": worker_id})
    dataset_dir = workspace / "datasets" / run["spec"]["dataset_id"]
    while True:
        claim = ledger.claim(run_id, worker_id, options.lease_seconds, options.max_attempts)
        if claim is None:
            counts = ledger.counts(run_id)
            if counts["pending"] == counts["running"] == 0:
                return
            time.sleep(min(0.1, options.lease_seconds / 4))
            continue
        inject = claim["job_index"] == options.crash_job and claim["attempt"] == 1
        try:
            with Heartbeat(ledger, claim, options.lease_seconds) as heartbeat:
                samples = read_task(dataset_dir, claim["payload"])
                if options.delay_seconds:
                    time.sleep(options.delay_seconds)
                predictions = scorer.score(samples)
                if inject and options.crash_stage == "before-commit":
                    abrupt_exit()
                if heartbeat.lost.is_set():
                    ledger.event(
                        run_id,
                        "work_discarded",
                        {"job": claim["job_index"], "reason": "lost_lease"},
                    )
                    continue
                hook = (
                    (lambda _: abrupt_exit())
                    if inject and options.crash_stage == "during-commit"
                    else None
                )
                accepted = ledger.commit(claim, predictions, after_insert=hook)
                if not accepted:
                    ledger.event(
                        run_id, "work_discarded", {"job": claim["job_index"], "reason": "fenced"}
                    )
                if inject and options.crash_stage == "after-commit":
                    abrupt_exit()
        except Exception as exc:
            ledger.fail(claim, f"{type(exc).__name__}: {exc}", options.max_attempts)


def create_run(workspace: Path, dataset_id: str, config: ScorerConfig, batch_size: int = 32) -> str:
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("Dataset ID must be a SHA-256 hexadecimal digest")
    manifest = verify_dataset(workspace / "datasets" / dataset_id)
    if config.backend == "hf":
        verify_model(config, workspace / "models" / model_key(config))
    spec = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_semantic_sha256": manifest.semantic_sha256,
        "scorer": config.model_dump(),
        "batch_size": batch_size,
        "source_sha256": source_fingerprint(),
        "environment": environment_fingerprint(config.backend),
    }
    ledger = Ledger(workspace / "ledger.sqlite")
    ledger.initialize()
    return ledger.create_run(
        spec, manifest.row_count, task_batches(workspace / "datasets" / dataset_id, batch_size)
    )


def execute(workspace: Path, run_id: str, options: ExecutionOptions | None = None) -> dict:
    options = options or ExecutionOptions()
    options.validate()
    ledger = Ledger(workspace / "ledger.sqlite")
    run = ledger.run(run_id)
    spec = run["spec"]
    if spec["source_sha256"] != source_fingerprint():
        raise IntegrityError(
            "Evaluation source changed; restore the recorded source or create a new run"
        )
    if spec["environment"] != environment_fingerprint(spec["scorer"]["backend"]):
        raise IntegrityError(
            "Runtime versions changed; restore the recorded environment or create a new run"
        )
    verify_dataset(workspace / "datasets" / spec["dataset_id"])
    counts = ledger.counts(run_id)
    if counts["pending"] == counts["running"] == 0:
        return counts
    ledger.event(run_id, "execution_started", asdict(options))
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    start = time.monotonic()
    failures = 0
    exit_reason = "completed"

    def launch():
        process = context.Process(
            target=worker_main, args=(str(workspace.resolve()), run_id, asdict(options))
        )
        process.start()
        processes.append(process)

    try:
        for _ in range(options.workers):
            launch()
        while True:
            counts = ledger.counts(run_id)
            finished = counts["pending"] == counts["running"] == 0
            for process in list(processes):
                if not process.is_alive():
                    process.join()
                    processes.remove(process)
                    ledger.event(
                        run_id, "worker_exited", {"pid": process.pid, "exit_code": process.exitcode}
                    )
                    if process.exitcode:
                        failures += 1
                    process.close()
                    if failures > options.max_process_failures:
                        raise RuntimeError("Worker failure budget exhausted; inspect run events")
                    if not finished:
                        launch()
            if finished:
                break
            if time.monotonic() - start > options.timeout_seconds:
                exit_reason = "timeout"
                raise TimeoutError(
                    "Execution timed out. Completed results are durable; resume this run later."
                )
            time.sleep(0.05)
        if counts["failed"]:
            exit_reason = "failed"
        elif counts["results"] != run["expected_rows"]:
            raise IntegrityError("Completed job counts do not cover the expected samples")
        return counts
    except BaseException:
        if exit_reason == "completed":
            exit_reason = "interrupted"
        raise
    finally:
        for process in processes:
            if exit_reason in ("completed", "failed"):
                process.join(timeout=2)
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            if process.exitcode and exit_reason == "completed":
                failures += 1
            ledger.event(
                run_id, "worker_exited", {"pid": process.pid, "exit_code": process.exitcode}
            )
            process.close()
        ledger.event(
            run_id,
            "execution_ended",
            {
                "reason": exit_reason,
                "duration_seconds": time.monotonic() - start,
                "worker_failures": failures,
            },
        )
