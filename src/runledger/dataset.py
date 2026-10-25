from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from runledger.schema import DatasetManifest, Provenance, Sample, Shard
from runledger.util import (
    IntegrityError,
    atomic_json,
    canonical_json,
    digest,
    file_digest,
    safe_child,
    utc_now,
)

ARROW_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("choices", pa.list_(pa.string()), nullable=False),
        pa.field("target", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False),
    ]
)
MAX_SHARD_INPUT_BYTES = 16 * 1024 * 1024
MAX_SHARD_FILE_BYTES = 64 * 1024 * 1024
PREPROCESSING = {
    "version": "nfc-lf-v1",
    "unicode": "NFC",
    "newlines": "LF",
    "whitespace": "preserved",
}


def to_arrow(sample: Sample) -> dict[str, Any]:
    row = sample.model_dump(exclude={"metadata"})
    row["metadata_json"] = canonical_json(sample.metadata)
    return row


def from_arrow(row: dict[str, Any]) -> Sample:
    value = dict(row)
    value["metadata"] = json.loads(value.pop("metadata_json"))
    return Sample.model_validate(value)


def jsonl_samples(path: Path) -> Iterator[Sample]:
    with path.open("rb") as stream:
        line_number = 0
        while line := stream.readline(2 * 1024 * 1024 + 1):
            line_number += 1
            if len(line) > 2 * 1024 * 1024:
                raise IntegrityError(f"Line {line_number} exceeds the 2 MiB record limit")
            if not line.strip():
                raise IntegrityError(f"Empty JSONL record at line {line_number}")
            try:
                yield Sample.model_validate_json(line)
            except (ValueError, TypeError) as exc:
                raise IntegrityError(f"Invalid JSONL record at line {line_number}: {exc}") from exc


def ingest(
    samples: Iterable[Sample],
    workspace: Path,
    provenance: Provenance,
    *,
    shard_rows: int = 4096,
    row_group_rows: int = 256,
) -> DatasetManifest:
    if not 1 <= row_group_rows <= shard_rows <= 1_000_000:
        raise ValueError("Require 1 <= row_group_rows <= shard_rows <= 1,000,000")
    root = workspace / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ingest-", dir=root))
    index = sqlite3.connect(stage / ".ids.sqlite")
    index.execute("CREATE TABLE ids (sample_id TEXT PRIMARY KEY)")
    shards: list[Shard] = []
    buffer: list[dict[str, Any]] = []
    count = 0
    buffer_bytes = 0
    semantic = hashlib.sha256()

    def flush() -> None:
        nonlocal buffer_bytes
        if not buffer:
            return
        name = f"shard-{len(shards):05d}.parquet"
        path = stage / name
        table = pa.Table.from_pylist(buffer, schema=ARROW_SCHEMA)
        pq.write_table(
            table, path, compression="zstd", row_group_size=row_group_rows, version="2.6"
        )
        shards.append(
            Shard(
                path=name,
                sha256=file_digest(path),
                rows=len(buffer),
                bytes=path.stat().st_size,
                row_groups=pq.ParquetFile(path).num_row_groups,
            )
        )
        buffer.clear()
        buffer_bytes = 0

    try:
        for sample in samples:
            sample = Sample.model_validate(sample)
            if sample.split != provenance.split:
                raise IntegrityError("Sample split differs from the declared source split")
            try:
                index.execute("INSERT INTO ids VALUES (?)", (sample.sample_id,))
            except sqlite3.IntegrityError as exc:
                raise IntegrityError(f"Duplicate sample ID: {sample.sample_id}") from exc
            encoded = (canonical_json(sample.model_dump()) + "\n").encode()
            if buffer and buffer_bytes + len(encoded) > MAX_SHARD_INPUT_BYTES:
                flush()
            semantic.update(encoded)
            buffer_bytes += len(encoded)
            buffer.append(to_arrow(sample))
            count += 1
            if len(buffer) == shard_rows:
                flush()
        flush()
        if not count:
            raise IntegrityError("An evaluation dataset cannot be empty")
        index.close()
        (stage / ".ids.sqlite").unlink()
        fields = dict(
            provenance=provenance.model_dump(),
            preprocessing=PREPROCESSING,
            row_count=count,
            semantic_sha256=semantic.hexdigest(),
            shards=[shard.model_dump() for shard in shards],
            schema_version=1,
        )
        manifest = DatasetManifest(dataset_id=digest(fields), created_at=utc_now(), **fields)
        atomic_json(stage / "manifest.json", manifest.model_dump())
        destination = root / manifest.dataset_id
        try:
            os.rename(stage, destination)
        except OSError:
            if not destination.exists():
                raise
            # A concurrent ingestion may have published the same immutable version.
            verify_dataset(destination)
            if load_manifest(destination).identity() != manifest.identity():
                raise IntegrityError("Existing dataset does not match its identity") from None
        return load_manifest(destination)
    finally:
        index.close()
        shutil.rmtree(stage, ignore_errors=True)


def load_manifest(dataset_dir: Path) -> DatasetManifest:
    try:
        manifest = DatasetManifest.model_validate_json((dataset_dir / "manifest.json").read_text())
    except (ValueError, OSError) as exc:
        raise IntegrityError(f"Cannot read dataset manifest: {exc}") from exc
    if digest(manifest.identity()) != manifest.dataset_id:
        raise IntegrityError("Dataset manifest fingerprint does not match its contents")
    if len({shard.path for shard in manifest.shards}) != len(manifest.shards):
        raise IntegrityError("Manifest repeats a shard path")
    if sum(shard.rows for shard in manifest.shards) != manifest.row_count:
        raise IntegrityError("Manifest row counts disagree")
    if manifest.preprocessing != PREPROCESSING:
        raise IntegrityError("Unsupported preprocessing protocol")
    return manifest


def verify_dataset(dataset_dir: Path, *, deep: bool = True) -> DatasetManifest:
    manifest = load_manifest(dataset_dir)
    semantic = hashlib.sha256()
    count = 0
    # Duplicate detection stays bounded in RAM, including for large datasets.
    with tempfile.TemporaryDirectory(prefix="runledger-verify-") as temporary:
        index = sqlite3.connect(Path(temporary) / "ids.sqlite")
        try:
            index.execute("CREATE TABLE ids (sample_id TEXT PRIMARY KEY)")
            for shard in manifest.shards:
                path = safe_child(dataset_dir, shard.path)
                if (
                    not path.is_file()
                    or path.stat().st_size != shard.bytes
                    or file_digest(path) != shard.sha256
                ):
                    raise IntegrityError(f"Missing or corrupted shard: {shard.path}")
                parquet = checked_parquet(path, shard.sha256)
                if not parquet.schema_arrow.equals(ARROW_SCHEMA, check_metadata=False):
                    raise IntegrityError(f"Schema mismatch in {shard.path}")
                if (
                    parquet.metadata.num_rows != shard.rows
                    or parquet.num_row_groups != shard.row_groups
                ):
                    raise IntegrityError(f"Parquet metadata mismatch in {shard.path}")
                if deep:
                    for batch in parquet.iter_batches(batch_size=1024):
                        for row in batch.to_pylist():
                            sample = from_arrow(row)
                            if sample.split != manifest.provenance.split:
                                raise IntegrityError("Sample split differs from manifest")
                            try:
                                index.execute("INSERT INTO ids VALUES (?)", (sample.sample_id,))
                            except sqlite3.IntegrityError as exc:
                                raise IntegrityError(
                                    f"Duplicate sample ID: {sample.sample_id}"
                                ) from exc
                            semantic.update((canonical_json(sample.model_dump()) + "\n").encode())
                            count += 1
            if deep and (
                count != manifest.row_count or semantic.hexdigest() != manifest.semantic_sha256
            ):
                raise IntegrityError("Logical sample content does not match the manifest")
        finally:
            index.close()
    return manifest


def iter_samples(dataset_dir: Path, batch_size: int = 1024) -> Iterator[Sample]:
    manifest = load_manifest(dataset_dir)
    for shard in manifest.shards:
        parquet = pq.ParquetFile(safe_child(dataset_dir, shard.path))
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield from_arrow(row)


def task_batches(dataset_dir: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    if not 1 <= batch_size <= 4096:
        raise ValueError("Batch size must be between 1 and 4096")
    manifest = load_manifest(dataset_dir)
    ordinal = 0
    for shard in manifest.shards:
        parquet = checked_parquet(safe_child(dataset_dir, shard.path), shard.sha256)
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(
                group, columns=["sample_id", "target", "choices"]
            ).to_pylist()
            ids = [
                {
                    "sample_id": row["sample_id"],
                    "target": row["target"],
                    "choice_count": len(row["choices"]),
                }
                for row in rows
            ]
            for start in range(0, len(ids), batch_size):
                selected = ids[start : start + batch_size]
                yield {
                    "shard": shard.path,
                    "sha256": shard.sha256,
                    "row_group": group,
                    "start": start,
                    "count": len(selected),
                    "ordinal": ordinal,
                    "expected": selected,
                }
                ordinal += len(selected)


def checked_parquet(path: Path, expected_sha256: str) -> pq.ParquetFile:
    # Hash and decode the same bounded byte snapshot, closing the check/use race.
    with path.open("rb") as stream:
        content = stream.read(MAX_SHARD_FILE_BYTES + 1)
    if len(content) > MAX_SHARD_FILE_BYTES:
        raise IntegrityError("Shard exceeds the supported 64 MiB file limit")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise IntegrityError(f"Shard changed during execution: {path.name}")
    return pq.ParquetFile(pa.BufferReader(content))


def read_task(dataset_dir: Path, payload: dict[str, Any]) -> list[Sample]:
    path = safe_child(dataset_dir, payload["shard"])
    # Recheck at use, not just run creation; detect edits between preflight and work.
    table = checked_parquet(path, payload["sha256"]).read_row_group(payload["row_group"])
    samples = [
        from_arrow(row) for row in table.slice(payload["start"], payload["count"]).to_pylist()
    ]
    expected = [
        {
            "sample_id": sample.sample_id,
            "target": sample.target,
            "choice_count": len(sample.choices),
        }
        for sample in samples
    ]
    if expected != payload["expected"]:
        raise IntegrityError("Task membership differs from its planned sample IDs and labels")
    return samples


def synthetic_samples(count: int, seed: int = 17, text_words: int = 40) -> Iterator[Sample]:
    if count < 1 or not 1 <= text_words <= 2048:
        raise ValueError("Require positive count and 1–2048 words per synthetic prompt")
    rng = random.Random(seed)
    vocabulary = [f"term{number:04d}" for number in range(4096)]
    for i in range(count):
        yield Sample(
            sample_id=f"synthetic-{i:08d}",
            prompt=" ".join(rng.choices(vocabulary, k=text_words)),
            choices=[f"candidate {i}-{j}" for j in range(4)],
            target=rng.randrange(4),
            metadata={"kind": "synthetic-infrastructure-only", "seed": seed},
        )


def synthetic_provenance(seed: int) -> Provenance:
    return Provenance(
        name="Synthetic infrastructure workload",
        uri="runledger://synthetic-v1",
        revision=f"seed-{seed}",
        license="MIT",
        split="test",
        description="Generated fixture. Accuracy is not evidence of model quality.",
    )
