import pytest
from hypothesis import given
from hypothesis import strategies as st

from runledger.dataset import (
    ingest,
    iter_samples,
    synthetic_provenance,
    synthetic_samples,
    verify_dataset,
)
from runledger.schema import Sample
from runledger.util import IntegrityError, atomic_json, digest


def test_identical_ingestion_is_idempotent(dataset):
    workspace, first = dataset
    second = ingest(
        synthetic_samples(24), workspace, synthetic_provenance(17), shard_rows=12, row_group_rows=6
    )
    assert first.dataset_id == second.dataset_id
    assert first.created_at == second.created_at
    assert len(list((workspace / "datasets").iterdir())) == 1


def test_rechunking_preserves_logical_digest_but_changes_artifact_identity(dataset):
    workspace, first = dataset
    second = ingest(
        synthetic_samples(24), workspace, synthetic_provenance(17), shard_rows=24, row_group_rows=8
    )
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.dataset_id != second.dataset_id
    assert list(iter_samples(workspace / "datasets" / first.dataset_id)) == list(
        iter_samples(workspace / "datasets" / second.dataset_id)
    )


def test_duplicate_ids_rejected_and_staging_removed(tmp_path):
    sample = next(synthetic_samples(1))
    with pytest.raises(IntegrityError, match="Duplicate"):
        ingest([sample, sample], tmp_path, synthetic_provenance(17))
    assert list((tmp_path / "datasets").iterdir()) == []


@pytest.mark.parametrize("failure", ["missing", "corrupted", "manifest", "symlink"])
def test_integrity_failures(dataset, failure):
    workspace, manifest = dataset
    root = workspace / "datasets" / manifest.dataset_id
    path = root / manifest.shards[0].path
    if failure == "missing":
        path.unlink()
    elif failure == "corrupted":
        contents = bytearray(path.read_bytes())
        contents[len(contents) // 2] ^= 1
        path.write_bytes(contents)
    elif failure == "symlink":
        outside = workspace / "outside.parquet"
        path.rename(outside)
        path.symlink_to(outside)
    else:
        value = manifest.model_dump()
        value["provenance"]["revision"] = "edited"
        atomic_json(root / "manifest.json", value)
    with pytest.raises(IntegrityError):
        verify_dataset(root)


def test_rehashed_manifest_cannot_escape_root(dataset):
    workspace, manifest = dataset
    root = workspace / "datasets" / manifest.dataset_id
    value = manifest.model_dump()
    value["shards"][0]["path"] = "../../outside.parquet"
    value["dataset_id"] = digest(
        {k: v for k, v in value.items() if k not in {"dataset_id", "created_at"}}
    )
    atomic_json(root / "manifest.json", value)
    with pytest.raises(IntegrityError, match="Unsafe"):
        verify_dataset(root)


def test_normalization_is_explicit_and_preserves_whitespace(tmp_path):
    a = Sample(sample_id="a", prompt=" cafe\u0301\r\n ", choices=["yes", "no"], target=0)
    b = Sample(sample_id="a", prompt=" café\n ", choices=["yes", "no"], target=0)
    assert a == b
    manifest = ingest([a], tmp_path, synthetic_provenance(17))
    assert next(iter_samples(tmp_path / "datasets" / manifest.dataset_id)).prompt == " café\n "


@pytest.mark.parametrize(
    "changes",
    [
        {"target": True},
        {"target": 8},
        {"choices": ["yes", "yes"]},
        {"prompt": " "},
        {"metadata": {"bad": float("nan")}},
        {"sample_id": "bad\n"},
    ],
)
def test_invalid_samples_rejected(changes):
    value = {"sample_id": "a", "prompt": "Question", "choices": ["yes", "no"], "target": 0}
    with pytest.raises(ValueError):
        Sample.model_validate({**value, **changes})


@given(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=100
    ).filter(lambda value: bool(value.strip()))
)
def test_normalization_is_idempotent(text):
    a = Sample(sample_id="a", prompt=text, choices=["yes", "no"], target=0)
    b = Sample.model_validate_json(a.model_dump_json())
    assert a == b


def test_ingestion_flushes_on_byte_budget(tmp_path, monkeypatch):
    import runledger.dataset as module

    monkeypatch.setattr(module, "MAX_SHARD_INPUT_BYTES", 1500)
    manifest = ingest(synthetic_samples(10), tmp_path, synthetic_provenance(17))
    assert len(manifest.shards) > 1
    assert verify_dataset(tmp_path / "datasets" / manifest.dataset_id).row_count == 10


def test_overlong_jsonl_line_is_rejected(tmp_path):
    from runledger.dataset import jsonl_samples

    path = tmp_path / "too-long.jsonl"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(IntegrityError, match="record limit"):
        list(jsonl_samples(path))


def test_mutated_sample_is_revalidated(tmp_path):
    sample = next(synthetic_samples(1))
    sample.choices[1] = sample.choices[0]
    with pytest.raises(ValueError, match="Duplicate answer"):
        ingest([sample], tmp_path, synthetic_provenance(17))
