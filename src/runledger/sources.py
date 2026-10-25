"""Small, explicit adapters; no dataset-provided Python is executed."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from runledger.dataset import ingest
from runledger.schema import Provenance, Sample
from runledger.util import IntegrityError, file_digest

ARC_REPO = "allenai/ai2_arc"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
SMOL_MODEL = "HuggingFaceTB/SmolLM2-135M"
SMOL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def import_arc(
    workspace: Path,
    *,
    subset: str = "ARC-Easy",
    split: str = "validation",
    limit: int = 128,
    revision: str = ARC_REVISION,
):
    import re

    from huggingface_hub import hf_hub_download

    if subset not in ("ARC-Easy", "ARC-Challenge") or split not in ("train", "test", "validation"):
        raise ValueError("Unknown ARC subset or split")
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or limit < 1:
        raise ValueError("Require a pinned dataset commit and a positive sample limit")
    filename = f"{subset}/{split}-00000-of-00001.parquet"
    path = Path(
        hf_hub_download(
            ARC_REPO,
            filename,
            repo_type="dataset",
            revision=revision,
            token=False,
            cache_dir=workspace / "downloads",
        )
    )
    parquet = pq.ParquetFile(path)
    if limit > parquet.metadata.num_rows:
        raise ValueError(f"Requested {limit} records; split contains {parquet.metadata.num_rows}")
    source_sha = file_digest(path)

    def samples():
        emitted = 0
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if emitted == limit:
                    return
                labels, choices = row["choices"]["label"], row["choices"]["text"]
                if len(labels) != len(choices) or len(labels) != len(set(labels)):
                    raise IntegrityError("ARC labels do not map one-to-one to choices")
                if row["answerKey"] not in labels:
                    raise IntegrityError("ARC answer key is absent from the choice labels")
                yield Sample(
                    sample_id=row["id"],
                    prompt=row["question"],
                    choices=choices,
                    target=labels.index(row["answerKey"]),
                    split=split,
                    metadata={
                        "subset": subset,
                        "source_ordinal": emitted,
                        "original_choice_labels": labels,
                    },
                )
                emitted += 1

    provenance = Provenance(
        name=f"AI2 ARC / {subset}",
        uri=f"https://huggingface.co/datasets/{ARC_REPO}",
        revision=revision,
        license="CC-BY-SA-4.0",
        split=split,
        description=f"First {limit} rows in pinned source-file order; no label-based selection. "
        f"Source: {filename}; SHA-256: {source_sha}. "
        "Subset diagnostic, not a full benchmark or leaderboard-comparable score.",
    )
    return ingest(samples(), workspace, provenance)
