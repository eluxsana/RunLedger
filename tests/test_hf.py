"""Opt-in, fully offline tests against an actual pinned local model."""

import os
from pathlib import Path

import pytest

from runledger.schema import Sample, ScorerConfig
from runledger.scorers import HuggingFaceScorer, continuation_tokens, model_key

pytestmark = pytest.mark.hf


@pytest.fixture(scope="module")
def scorer():
    config_path = os.environ.get("RUNLEDGER_HF_CONFIG")
    if not config_path:
        pytest.skip("Set RUNLEDGER_HF_CONFIG to a fetched model-config JSON for real-model checks")
    path = Path(config_path)
    config = ScorerConfig.model_validate_json(path.read_text())
    return HuggingFaceScorer(config, path.parent.parent / "models" / model_key(config))


def test_real_scores_equal_independent_masked_model_loss(scorer):
    import torch

    sample = Sample(
        sample_id="loss-check",
        prompt="Which planet is closest to the Sun?",
        choices=["Mercury", "The Earth is farther from the Sun."],
        target=0,
    )
    actual = scorer.score([sample])[0]
    for index, answer in enumerate(sample.choices):
        context = scorer.config.prompt_template.format(prompt=sample.prompt)
        ids, prefix = continuation_tokens(scorer.tokenizer, context, " " + answer, 512)
        tokens = torch.tensor([ids])
        labels = tokens.clone()
        labels[:, :prefix] = -100
        with torch.inference_mode():
            loss = scorer.model(input_ids=tokens, labels=labels, use_cache=False).loss.item()
        expected = -loss * (len(ids) - prefix)
        assert actual.choice_scores[index] == pytest.approx(expected, abs=3e-5)
        assert actual.token_counts[index] == len(ids) - prefix


def test_real_batching_and_label_invariance(scorer):
    samples = [
        Sample(
            sample_id=f"sample-{i}", prompt=prompt, choices=["yes", "no, it is not"], target=i % 2
        )
        for i, prompt in enumerate(
            ["Is water a liquid?", "Is the Moon a star in the solar system?"]
        )
    ]
    batch = scorer.score(samples)
    individual = [scorer.score([sample])[0] for sample in samples]
    for left, right in zip(batch, individual, strict=True):
        assert left.choice_scores == pytest.approx(right.choice_scores, abs=3e-5)
        assert left.prediction == right.prediction
    changed = [
        Sample.model_validate(sample.model_dump() | {"target": 1 - sample.target})
        for sample in samples
    ]
    relabeled = scorer.score(changed)
    for left, right in zip(batch, relabeled, strict=True):
        assert left.choice_scores == right.choice_scores
        assert left.prediction == right.prediction
        assert left.correct != right.correct
