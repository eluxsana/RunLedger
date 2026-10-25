import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from runledger.cli import app
from runledger.comparison import render_comparison
from runledger.evidence import compare_exports

FIXTURES = Path(__file__).parents[1] / "artifacts/reproduction/model"


def test_real_scoring_change_accounts_for_every_prediction_flip():
    left, right = FIXTURES / "sum", FIXTURES / "mean"
    result = compare_exports(left, right)
    a, b = [pq.read_table(path / "results.parquet").to_pylist() for path in (left, right)]
    changed_ids = {
        x["sample_id"] for x, y in zip(a, b, strict=True) if x["prediction"] != y["prediction"]
    }
    assert {row["sample_id"] for row in result["changed_samples"]} == changed_ids
    assert result["prediction_changes"] == len(changed_ids)
    outcomes = Counter(row["outcome"] for row in result["changed_samples"])
    assert outcomes["improved"] == result["improved"] == 16
    assert outcomes["regressed"] == result["regressed"] == 24
    assert outcomes["changed"] > 0  # Wrong-to-wrong changes are invisible in aggregate accuracy.
    assert result["accuracy_delta_right_minus_left"] == -0.0625


def test_recovered_run_has_no_prediction_drift():
    result = compare_exports(FIXTURES / "sum", FIXTURES / "sum-recovery")
    assert result["changed_samples"] == []
    assert result["prediction_changes"] == 0


def test_report_escapes_ids_and_preserves_existing_output(tmp_path):
    result = compare_exports(FIXTURES / "sum", FIXTURES / "mean")
    payload = '</script><script>alert("xss")</script>'
    result["changed_samples"][0]["sample_id"] = payload
    output = tmp_path / "comparison.html"
    render_comparison(result, output)
    text = output.read_text()
    assert payload not in text
    assert "&lt;/script&gt;" in text and "\\u003c/script\\u003e" in text
    assert ".innerHTML" not in text
    with pytest.raises(FileExistsError):
        render_comparison(result, output)


def test_cli_comparison_report(tmp_path):
    output = tmp_path / "drift.html"
    result = CliRunner().invoke(
        app, ["compare", str(FIXTURES / "sum"), str(FIXTURES / "mean"), "--report", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["prediction_changes"] > 40
    assert output.is_file()
