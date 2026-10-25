from pathlib import Path

import pytest

from runledger.dataset import ingest, synthetic_provenance, synthetic_samples


@pytest.fixture
def dataset(tmp_path: Path):
    manifest = ingest(
        synthetic_samples(24), tmp_path, synthetic_provenance(17), shard_rows=12, row_group_rows=6
    )
    return tmp_path, manifest
