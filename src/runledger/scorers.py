from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path

from runledger.schema import Prediction, Sample, ScorerConfig
from runledger.util import IntegrityError, atomic_json, digest, file_digest, safe_child


def make_prediction(sample: Sample, scores: list[float], token_counts: list[int]) -> Prediction:
    prediction = max(range(len(scores)), key=scores.__getitem__)
    return Prediction(
        sample_id=sample.sample_id,
        choice_scores=scores,
        token_counts=token_counts,
        prediction=prediction,
        target=sample.target,
        correct=prediction == sample.target,
    )


class HashScorer:
    """Deterministic infrastructure fixture. Never presented as a learned model."""

    def __init__(self, config: ScorerConfig, model_dir: Path | None = None):
        self.config = config

    def score(self, samples: list[Sample]) -> list[Prediction]:
        predictions = []
        for sample in samples:
            scores = []
            for choice in sample.choices:
                key = f"{self.config.seed}\x00{sample.prompt}\x00{choice}".encode()
                probability = (int.from_bytes(hashlib.sha256(key).digest()[:6], "big") + 1) / (
                    2**48 + 1
                )
                scores.append(math.log(probability))
            predictions.append(make_prediction(sample, scores, [1] * len(scores)))
        return predictions


def model_key(config: ScorerConfig) -> str:
    return digest({"repo": config.model_id, "revision": config.revision})


def verify_model(config: ScorerConfig, model_dir: Path) -> None:
    actual = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_dir).parts
    }
    if actual != set(config.assets_sha256):
        raise IntegrityError("Model snapshot file inventory differs from the pinned configuration")
    for name, expected in config.assets_sha256.items():
        path = safe_child(model_dir, name)
        if not path.is_file() or file_digest(path) != expected:
            raise IntegrityError(f"Pinned model artifact changed or missing: {name}")


def fetch_model(repo_id: str, revision: str, workspace: Path) -> tuple[ScorerConfig, Path]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=False)
    resolved = api.model_info(repo_id, revision=revision).sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[a-f0-9]{40}", resolved):
        raise IntegrityError("Model service did not return an immutable commit revision")
    key = digest({"repo": repo_id, "revision": resolved})
    root = workspace / "models" / key
    snapshot_download(
        repo_id,
        revision=resolved,
        local_dir=root,
        token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    assets = {
        p.relative_to(root).as_posix(): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".cache" not in p.relative_to(root).parts
    }
    if not any(name.endswith(".safetensors") for name in assets):
        raise IntegrityError("The CPU HF backend requires safetensors model weights")
    config = ScorerConfig(
        backend="hf",
        model_id=repo_id,
        revision=resolved,
        tokenizer_revision=resolved,
        assets_sha256=assets,
    )
    config_path = workspace / "model-configs" / f"{key}.json"
    atomic_json(config_path, config.model_dump())
    return config, config_path


def continuation_tokens(
    tokenizer, context: str, continuation: str, max_length: int
) -> tuple[list[int], int]:
    prefix = tokenizer.encode(context, add_special_tokens=False)
    full = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not prefix or full[: len(prefix)] != prefix:
        raise ValueError(
            "Tokenizer changed the context/answer boundary; this protocol rejects ambiguous segmentation"
        )
    count = len(full) - len(prefix)
    if count <= 0:
        raise ValueError("An answer must produce at least one continuation token")
    if len(full) > max_length:
        raise ValueError(
            f"Sequence has {len(full)} tokens, exceeding max_length={max_length}; no silent truncation"
        )
    return full, len(prefix)


def score_token_logits(
    logits, input_ids, context_lengths: list[int], lengths: list[int], normalization: str
):
    """Score only answer tokens, with the causal shift; padded and context tokens contribute zero."""
    import torch

    if normalization not in ("sum", "mean"):
        raise ValueError("Unknown score normalization")
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Logit and input tensor shapes disagree")
    if len(context_lengths) != len(lengths) or len(lengths) != input_ids.shape[0]:
        raise ValueError("Length metadata differs from batch size")
    log_probabilities = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    gathered = log_probabilities.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    output = []
    for row, (context_length, length) in enumerate(zip(context_lengths, lengths, strict=True)):
        if not 1 <= context_length < length <= input_ids.shape[1]:
            raise ValueError("Invalid context or sequence length")
        selected = gathered[row, context_length - 1 : length - 1]
        value = selected.sum() if normalization == "sum" else selected.mean()
        output.append(float(value.item()))
    return output


class HuggingFaceScorer:
    def __init__(self, config: ScorerConfig, model_dir: Path):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        verify_model(config, model_dir)
        self.config = config
        self.torch = torch
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        model_limit = getattr(self.model.config, "max_position_embeddings", config.max_length)
        if config.max_length > model_limit:
            raise ValueError(f"Configured length exceeds model limit {model_limit}")
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define a padding or EOS token")

    def score(self, samples: list[Sample]) -> list[Prediction]:
        sequences: list[list[int]] = []
        contexts: list[int] = []
        membership: list[tuple[int, int]] = []
        for sample_index, sample in enumerate(samples):
            context = self.config.prompt_template.format(prompt=sample.prompt)
            for choice_index, choice in enumerate(sample.choices):
                ids, context_length = continuation_tokens(
                    self.tokenizer,
                    context,
                    self.config.continuation_prefix + choice,
                    self.config.max_length,
                )
                sequences.append(ids)
                contexts.append(context_length)
                membership.append((sample_index, choice_index))
        scores: list[list[float]] = [[0.0] * len(sample.choices) for sample in samples]
        counts: list[list[int]] = [[0] * len(sample.choices) for sample in samples]
        batch_size = self.config.choice_batch_size
        with self.torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                width = max(map(len, batch))
                tokens = self.torch.full((len(batch), width), self.pad_id, dtype=self.torch.long)
                mask = self.torch.zeros_like(tokens)
                for i, ids in enumerate(batch):
                    tokens[i, : len(ids)] = self.torch.tensor(ids)
                    mask[i, : len(ids)] = 1
                logits = self.model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
                values = score_token_logits(
                    logits,
                    tokens,
                    contexts[start : start + len(batch)],
                    [len(ids) for ids in batch],
                    self.config.normalization,
                )
                for offset, value in enumerate(values):
                    sample_index, choice_index = membership[start + offset]
                    scores[sample_index][choice_index] = value
                    counts[sample_index][choice_index] = (
                        len(batch[offset]) - contexts[start + offset]
                    )
        return [
            make_prediction(sample, score, count)
            for sample, score, count in zip(samples, scores, counts, strict=True)
        ]


def make_scorer(config: ScorerConfig, workspace: Path):
    if config.backend == "hash":
        return HashScorer(config)
    return HuggingFaceScorer(config, workspace / "models" / model_key(config))
