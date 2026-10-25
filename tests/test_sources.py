import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from runledger.dataset import iter_samples
from runledger.sources import import_arc
from runledger.util import IntegrityError


def source_file(tmp_path, monkeypatch, rows):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(path))


def row(key="C"):
    return {
        "id": "a",
        "question": "What is 2+2?",
        "choices": {"label": ["B", "C", "A"], "text": ["3", "4", "5"]},
        "answerKey": key,
    }


def test_arc_maps_answer_labels_and_preserves_order(tmp_path, monkeypatch):
    source_file(tmp_path, monkeypatch, [row(), row() | {"id": "b"}])
    manifest = import_arc(tmp_path / "work", limit=1)
    sample = next(iter_samples(tmp_path / "work" / "datasets" / manifest.dataset_id))
    assert sample.target == 1 and sample.choices == ["3", "4", "5"]
    assert sample.split == "validation"
    assert sample.metadata["source_ordinal"] == 0
    assert "First 1 rows" in manifest.provenance.description
    assert manifest.provenance.license == "CC-BY-SA-4.0"


def test_arc_invalid_labels_are_errors_not_guessed_targets(tmp_path, monkeypatch):
    source_file(tmp_path, monkeypatch, [row("missing")])
    with pytest.raises(IntegrityError, match="answer key"):
        import_arc(tmp_path / "work", limit=1)
    with pytest.raises(ValueError, match="split contains"):
        import_arc(tmp_path / "work", limit=2)


def test_arc_requires_pinned_revision_before_download(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        import_arc(tmp_path, revision="main")
