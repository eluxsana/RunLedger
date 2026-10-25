import shutil

import pytest

from runledger.ledger import Ledger
from runledger.runner import ExecutionOptions, create_run, execute
from runledger.schema import ScorerConfig
from runledger.util import IntegrityError, digest


@pytest.mark.integration
@pytest.mark.parametrize("stage", ["before-commit", "during-commit", "after-commit"])
def test_actual_killed_worker_recovers_identical_results(dataset, stage):
    workspace, manifest = dataset
    baseline = workspace / "baseline"
    faulty = workspace / "faulty"
    for root in [baseline, faulty]:
        shutil.copytree(workspace / "datasets", root / "datasets")
    run1 = create_run(baseline, manifest.dataset_id, ScorerConfig(), 4)
    run2 = create_run(faulty, manifest.dataset_id, ScorerConfig(), 4)
    assert run1 == run2
    assert execute(baseline, run1, ExecutionOptions(workers=2, lease_seconds=1.0))["results"] == 24
    counts = execute(
        faulty,
        run2,
        ExecutionOptions(workers=2, lease_seconds=1.0, crash_stage=stage, timeout_seconds=30),
    )
    assert counts == {"pending": 0, "running": 0, "completed": 8, "failed": 0, "results": 24}
    ledger = Ledger(faulty / "ledger.sqlite")
    assert digest(ledger.results(run2)) == digest(Ledger(baseline / "ledger.sqlite").results(run1))
    first = ledger.jobs(run2)[0]
    assert first["attempts"] == (1 if stage == "after-commit" else 2)
    if stage != "after-commit":
        assert any(event["kind"] == "lease_reclaimed" for event in ledger.events(run2))


@pytest.mark.integration
def test_live_heartbeat_prevents_reclaim_during_slow_work(dataset):
    workspace, manifest = dataset
    run_id = create_run(workspace, manifest.dataset_id, ScorerConfig(), 6)
    counts = execute(
        workspace, run_id, ExecutionOptions(workers=2, lease_seconds=0.6, delay_seconds=1.2)
    )
    assert counts["results"] == 24
    assert all(job["attempts"] == 1 for job in Ledger(workspace / "ledger.sqlite").jobs(run_id))


@pytest.mark.integration
def test_completed_run_is_idempotent_and_configuration_is_part_of_identity(dataset):
    workspace, manifest = dataset
    run_id = create_run(workspace, manifest.dataset_id, ScorerConfig(), 6)
    execute(workspace, run_id, ExecutionOptions(workers=1))
    ledger = Ledger(workspace / "ledger.sqlite")
    previous_events = len(ledger.events(run_id))
    execute(workspace, run_id)
    assert len(ledger.events(run_id)) == previous_events
    assert create_run(workspace, manifest.dataset_id, ScorerConfig(seed=18), 6) != run_id


def test_runtime_change_blocks_resume(dataset, monkeypatch):
    workspace, manifest = dataset
    run_id = create_run(workspace, manifest.dataset_id, ScorerConfig(), 6)
    monkeypatch.setattr("runledger.runner.source_fingerprint", lambda: "changed")
    with pytest.raises(IntegrityError, match="source changed"):
        execute(workspace, run_id)


def test_timeout_preserves_completed_work_and_resume(dataset):
    workspace, manifest = dataset
    run_id = create_run(workspace, manifest.dataset_id, ScorerConfig(), 4)
    ledger = Ledger(workspace / "ledger.sqlite")
    # Commit one real task before interrupting the supervisor.
    from runledger.dataset import read_task
    from runledger.scorers import HashScorer

    claim = ledger.claim(run_id, "preexisting", 30, 3)
    ledger.commit(
        claim,
        HashScorer(ScorerConfig()).score(
            read_task(workspace / "datasets" / manifest.dataset_id, claim["payload"])
        ),
    )
    before = ledger.results(run_id)
    with pytest.raises(TimeoutError):
        execute(
            workspace,
            run_id,
            ExecutionOptions(workers=1, timeout_seconds=0.15, lease_seconds=0.5, delay_seconds=2),
        )
    assert ledger.results(run_id)[: len(before)] == before
    counts = execute(
        workspace, run_id, ExecutionOptions(workers=2, lease_seconds=0.5, timeout_seconds=30)
    )
    assert counts["results"] == 24 and counts["failed"] == 0
    assert ledger.jobs(run_id)[0]["attempts"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"lease_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"delay_seconds": float("nan")},
    ],
)
def test_nonfinite_execution_settings_rejected(changes):
    with pytest.raises(ValueError, match="finite"):
        ExecutionOptions(**changes).validate()
