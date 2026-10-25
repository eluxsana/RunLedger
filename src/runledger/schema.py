from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from runledger.util import canonical_json


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


class Sample(StrictModel):
    sample_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=262144)
    choices: list[str] = Field(min_length=2, max_length=32)
    target: int = Field(ge=0)
    split: str = Field(default="test", min_length=1, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "split")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if value != value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Identifiers cannot contain controls or surrounding whitespace")
        return normalize_text(value)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Prompt must contain text and no NUL bytes")
        return normalize_text(value)

    @field_validator("choices")
    @classmethod
    def normalize_choices(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 32768 or "\x00" in value for value in values):
            raise ValueError("Choices must be nonempty, bounded text without NUL bytes")
        normalized = [normalize_text(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate answer choices are ambiguous")
        return normalized

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        encoded = canonical_json(value)
        if len(encoded.encode()) > 65536:
            raise ValueError("Metadata exceeds 64 KiB")
        return json.loads(encoded)

    @model_validator(mode="after")
    def validate_target(self) -> Sample:
        if self.target >= len(self.choices):
            raise ValueError("Target is not an answer-choice index")
        return self


class Provenance(StrictModel):
    name: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    split: str = Field(min_length=1)
    description: str = ""


class Shard(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rows: int = Field(gt=0)
    bytes: int = Field(gt=0)
    row_groups: int = Field(gt=0)


class DatasetManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance: Provenance
    preprocessing: dict[str, str]
    row_count: int = Field(gt=0)
    semantic_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    shards: list[Shard] = Field(min_length=1)
    created_at: str

    def identity(self) -> dict[str, Any]:
        return self.model_dump(exclude={"dataset_id", "created_at"})


class ScorerConfig(StrictModel):
    backend: Literal["hash", "hf"] = "hash"
    model_id: str = "runledger/hash-v1"
    revision: str = "1"
    tokenizer_revision: str = "1"
    assets_sha256: dict[str, str] = Field(default_factory=dict)
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    max_length: int = Field(default=512, ge=8, le=32768)
    normalization: Literal["sum", "mean"] = "sum"
    prompt_template: str = "Question: {prompt}\nAnswer:"
    continuation_prefix: str = " "
    dtype: Literal["float32"] = "float32"
    device: Literal["cpu"] = "cpu"
    threads: int = Field(default=2, ge=1, le=32)
    choice_batch_size: int = Field(default=8, ge=1, le=128)
    protocol: Literal["continuation-loglikelihood-v1"] = "continuation-loglikelihood-v1"

    @model_validator(mode="after")
    def validate_configuration(self) -> ScorerConfig:
        import re
        import string

        fields = [
            field
            for _, field, _, _ in string.Formatter().parse(self.prompt_template)
            if field is not None
        ]
        if fields != ["prompt"] or self.prompt_template.count("{prompt}") != 1:
            raise ValueError("Template must contain exactly one plain {prompt} field")
        for name, checksum in self.assets_sha256.items():
            if (
                not name
                or ".." in Path(name).parts
                or Path(name).is_absolute()
                or "\\" in name
                or not re.fullmatch(r"[a-f0-9]{64}", checksum)
            ):
                raise ValueError("Model assets require relative paths and SHA-256 digests")
        if self.backend == "hash" and (
            self.model_id != "runledger/hash-v1"
            or self.revision != "1"
            or self.tokenizer_revision != "1"
            or self.assets_sha256
        ):
            raise ValueError("The hash fixture cannot be labeled as a learned model")
        if self.backend == "hf":
            if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
                raise ValueError("HF models require an immutable 40-character commit revision")
            if self.tokenizer_revision != self.revision or not self.assets_sha256:
                raise ValueError("Tokenizer and model must share a pinned, hashed snapshot")
        return self


class Prediction(StrictModel):
    sample_id: str
    choice_scores: list[float]
    token_counts: list[int]
    prediction: int
    target: int
    correct: bool

    @model_validator(mode="after")
    def validate_prediction(self) -> Prediction:
        n = len(self.choice_scores)
        if n < 2 or len(self.token_counts) != n:
            raise ValueError("Prediction requires one token count per answer score")
        if any(not math.isfinite(score) for score in self.choice_scores):
            raise ValueError("Scores must be finite")
        if any(count < 1 for count in self.token_counts):
            raise ValueError("Every answer requires at least one scored token")
        if not (0 <= self.prediction < n and 0 <= self.target < n):
            raise ValueError("Prediction and target must be valid indices")
        if self.prediction != max(range(n), key=self.choice_scores.__getitem__):
            raise ValueError("Prediction disagrees with the first maximum score")
        if self.correct != (self.prediction == self.target):
            raise ValueError("Correctness disagrees with prediction and target")
        return self
