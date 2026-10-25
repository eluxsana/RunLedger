from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class IntegrityError(ValueError):
    """Content, schema or provenance failed validation."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_child(root: Path, relative: str) -> Path:
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts or not relative or "\\" in relative:
        raise IntegrityError(f"Unsafe artifact path: {relative!r}")
    resolved_root = root.resolve()
    candidate = root / child
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(resolved_root):
        raise IntegrityError(f"Artifact escapes its root: {relative!r}")
    return candidate


def source_fingerprint() -> str:
    root = Path(__file__).parent
    return digest(
        {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(root.rglob("*.py"))}
    )


def environment_fingerprint(backend: str) -> dict[str, str]:
    names = ["runledger", "pyarrow", "numpy", "pydantic"]
    if backend == "hf":
        names += ["torch", "transformers", "tokenizers", "safetensors"]
    result = {
        "sqlite": sqlite3.sqlite_version,
        "python": platform.python_version(),
        "platform": platform.system(),
        "arch": platform.machine(),
    }
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def percentile(values: list[float], quantile: float) -> float | None:
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between zero and one")
    if not values:
        return None
    ordered = sorted(values)
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]
