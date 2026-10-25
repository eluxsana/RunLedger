import json

import pytest
from typer.testing import CliRunner

from runledger.cli import app
from runledger.evidence import compare_exports, export_run, verify_export
from runledger.ledger import Ledger
from runledger.runner import ExecutionOptions, create_run, execute
from runledger.schema import ScorerConfig
from runledger.util import IntegrityError


def finished(dataset, tmp_path):
    workspace, manifest = dataset
    run_id = create_run(workspace, manifest.dataset_id, ScorerConfig(), 4)
    execute(workspace, run_id, ExecutionOptions(workers=1))
    output = tmp_path / "evidence"
    export_run(workspace, run_id, output)
    return workspace, run_id, output


def test_complete_export_recomputes_metrics_and_comparison(dataset, tmp_path):
    _, _, output = finished(dataset, tmp_path)
    summary = verify_export(output)
    assert summary["metrics"]["samples"] == summary["metrics"]["unique_sample_ids"] == 24
    assert not summary["model_quality_evidence"]
    comparison = compare_exports(output, output)
    assert comparison["identical_result_digest"]
    assert comparison["paired_bootstrap_95"] == [0.0, 0.0]
    assert summary["operations"]["jobs"] == 8


def test_incomplete_export_refused(dataset, tmp_path):
    workspace, manifest = dataset
    run_id = create_run(workspace, manifest.dataset_id, ScorerConfig(), 4)
    with pytest.raises(IntegrityError, match="complete"):
        export_run(workspace, run_id, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


def test_export_cannot_overwrite_and_detects_corruption(dataset, tmp_path):
    workspace, run_id, output = finished(dataset, tmp_path)
    with pytest.raises(FileExistsError):
        export_run(workspace, run_id, output)
    (output / "index.html").write_text("tampered")
    with pytest.raises(IntegrityError, match="changed"):
        verify_export(output)


def test_report_escapes_untrusted_strings(dataset, tmp_path):
    _, _, output = finished(dataset, tmp_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    payload = '</script><script>alert("xss")</script>'
    summary["dataset"]["provenance"]["name"] = payload
    summary["spec"]["scorer"]["prompt_template"] = payload + "{prompt}"
    summary_path.write_text(json.dumps(summary))
    from runledger.report import render_report

    text = render_report(output).read_text()
    assert payload not in text
    assert "&lt;/script&gt;" in text
    assert "\\u003c/script\\u003e" in text
    assert "textContent" in text and ".innerHTML" not in text


def test_compare_refuses_different_samples(dataset, tmp_path):
    _, _, output = finished(dataset, tmp_path)
    from runledger.dataset import ingest, synthetic_provenance, synthetic_samples

    other = tmp_path / "other"
    manifest = ingest(synthetic_samples(8), other, synthetic_provenance(17))
    run_id = create_run(other, manifest.dataset_id, ScorerConfig(), 4)
    execute(other, run_id, ExecutionOptions(workers=1))
    export_run(other, run_id, tmp_path / "other-export")
    with pytest.raises(IntegrityError, match="same ordered dataset"):
        compare_exports(output, tmp_path / "other-export")


def test_cli_offline_roundtrip(tmp_path):
    runner = CliRunner()
    args = ["--workspace", str(tmp_path / "work")]
    result = runner.invoke(app, args + ["synthetic", "--rows", "12"])
    assert result.exit_code == 0, result.output
    dataset_id = json.loads(result.stdout)["dataset_id"]
    assert runner.invoke(app, args + ["verify", dataset_id]).exit_code == 0
    result = runner.invoke(app, args + ["run", dataset_id, "--workers", "1"])
    assert result.exit_code == 0, result.output
    run_id = json.loads(result.stdout.splitlines()[-1])["run_id"]
    output = tmp_path / "export"
    assert runner.invoke(app, args + ["export", run_id, str(output)]).exit_code == 0
    result = runner.invoke(app, ["verify-export", str(output)])
    assert result.exit_code == 0 and json.loads(result.stdout)["verified"]
    assert runner.invoke(app, args + ["verify", "../../evil"]).exit_code == 1
    assert runner.invoke(app, args + ["synthetic", "--rows", "0"]).exit_code == 1
    assert Ledger(tmp_path / "work" / "ledger.sqlite").counts(run_id)["results"] == 12
