import multiprocessing as mp
from pathlib import Path

import pytest

from runledger.dataset import read_task, task_batches
from runledger.ledger import Ledger
from runledger.schema import ScorerConfig
from runledger.scorers import HashScorer
from runledger.util import IntegrityError


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def setup_ledger(dataset):
    workspace, manifest = dataset
    clock = Clock()
    ledger = Ledger(workspace / "ledger.sqlite", clock)
    ledger.initialize()
    run_id = ledger.create_run(
        {"dataset_id": manifest.dataset_id},
        manifest.row_count,
        task_batches(workspace / "datasets" / manifest.dataset_id, 6),
    )
    return ledger, run_id, clock


def predictions_for(dataset, claim):
    workspace, manifest = dataset
    return HashScorer(ScorerConfig()).score(
        read_task(workspace / "datasets" / manifest.dataset_id, claim["payload"])
    )


def test_stale_lease_cannot_commit_or_heartbeat(dataset):
    ledger, run_id, clock = setup_ledger(dataset)
    old = ledger.claim(run_id, "old", 10, 3)
    clock.now += 11
    new = ledger.claim(run_id, "new", 10, 3)
    assert old["job_id"] == new["job_id"] and old["token"] != new["token"]
    assert not ledger.heartbeat(old["job_id"], old["token"], 10)
    assert not ledger.commit(old, predictions_for(dataset, old))
    assert ledger.commit(new, predictions_for(dataset, new))
    assert not ledger.commit(new, predictions_for(dataset, new))
    assert ledger.counts(run_id)["results"] == 6


def test_expired_unreclaimed_worker_cannot_commit(dataset):
    ledger, run_id, clock = setup_ledger(dataset)
    claim = ledger.claim(run_id, "old", 10, 3)
    clock.now += 10
    assert not ledger.commit(claim, predictions_for(dataset, claim))
    assert ledger.counts(run_id)["results"] == 0


def test_heartbeat_extends_ownership(dataset):
    ledger, run_id, clock = setup_ledger(dataset)
    claim = ledger.claim(run_id, "one", 10, 3)
    clock.now += 7
    assert ledger.heartbeat(claim["job_id"], claim["token"], 10)
    clock.now += 7
    assert ledger.commit(claim, predictions_for(dataset, claim))


def test_partial_result_insert_rolls_back_with_job(dataset):
    ledger, run_id, _ = setup_ledger(dataset)
    claim = ledger.claim(run_id, "one", 10, 3)

    def interrupt(_):
        raise RuntimeError("Injected failure after first result INSERT")

    with pytest.raises(RuntimeError, match="Injected"):
        ledger.commit(claim, predictions_for(dataset, claim), after_insert=interrupt)
    assert ledger.counts(run_id)["results"] == 0
    assert ledger.counts(run_id)["running"] == 1
    assert ledger.commit(claim, predictions_for(dataset, claim))


def test_incomplete_or_wrong_membership_cannot_commit(dataset):
    ledger, run_id, _ = setup_ledger(dataset)
    claim = ledger.claim(run_id, "one", 10, 3)
    predictions = predictions_for(dataset, claim)
    for invalid in [predictions[:-1], list(reversed(predictions)), predictions + predictions[:1]]:
        with pytest.raises(IntegrityError, match="membership"):
            ledger.commit(claim, invalid)
    assert ledger.counts(run_id)["results"] == 0


def test_attempt_budget_expires_to_failed(dataset):
    ledger, run_id, clock = setup_ledger(dataset)
    first = ledger.claim(run_id, "one", 10, 1)
    clock.now += 11
    second = ledger.claim(run_id, "two", 10, 1)
    assert first["job_id"] != second["job_id"]
    assert ledger.counts(run_id)["failed"] == 1


def claim_process(path, run_id, queue):
    claim = Ledger(Path(path)).claim(run_id, "concurrent", 30, 3)
    queue.put(claim["job_id"] if claim else None)


@pytest.mark.integration
def test_concurrent_processes_get_distinct_jobs(dataset):
    workspace, manifest = dataset
    ledger = Ledger(workspace / "concurrent.sqlite")
    ledger.initialize()
    run_id = ledger.create_run(
        {"case": "concurrency"}, 24, task_batches(workspace / "datasets" / manifest.dataset_id, 6)
    )
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=claim_process, args=(str(ledger.path), run_id, queue)) for _ in range(4)
    ]
    for process in processes:
        process.start()
    claims = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert len(set(claims)) == 4 and None not in claims


def test_clock_is_read_after_transaction_lock(dataset, monkeypatch):
    from contextlib import contextmanager

    ledger, run_id, clock = setup_ledger(dataset)
    claim = ledger.claim(run_id, "worker", 10, 3)
    original = ledger.transaction

    @contextmanager
    def delayed_transaction():
        # Deterministically simulate a lease expiring while waiting for BEGIN IMMEDIATE.
        clock.now += 11
        with original() as connection:
            yield connection

    monkeypatch.setattr(ledger, "transaction", delayed_transaction)
    assert not ledger.commit(claim, predictions_for(dataset, claim))
    assert ledger.counts(run_id)["results"] == 0


def test_mutated_prediction_cannot_bypass_validation(dataset):
    ledger, run_id, _ = setup_ledger(dataset)
    claim = ledger.claim(run_id, "worker", 10, 3)
    predictions = predictions_for(dataset, claim)
    predictions[0].choice_scores[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ledger.commit(claim, predictions)
    assert ledger.counts(run_id)["results"] == 0


def test_wrong_number_of_scores_cannot_commit(dataset):
    from runledger.schema import Prediction

    ledger, run_id, _ = setup_ledger(dataset)
    claim = ledger.claim(run_id, "worker", 10, 3)
    predictions = predictions_for(dataset, claim)
    value = predictions[0]
    predictions[0] = Prediction(
        sample_id=value.sample_id,
        target=value.target,
        prediction=0,
        correct=value.target == 0,
        choice_scores=[0.0, -1.0, -2.0, -3.0, -4.0],
        token_counts=[1] * 5,
    )
    with pytest.raises(IntegrityError, match="membership"):
        ledger.commit(claim, predictions)


def test_lowered_retry_budget_does_not_leave_pending_jobs_stuck(dataset):
    ledger, run_id, _ = setup_ledger(dataset)
    claim = ledger.claim(run_id, "worker", 10, 3)
    assert ledger.fail(claim, "transient", 3)
    ledger.claim(run_id, "worker-two", 10, 1)
    assert ledger.jobs(run_id)[0]["status"] == "failed"


def test_future_database_version_is_rejected_before_schema_mutation(tmp_path):
    import sqlite3

    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(IntegrityError, match="Unsupported"):
        Ledger(path).initialize()
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )
