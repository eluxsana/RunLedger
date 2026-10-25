import math

import pytest

from runledger.schema import Prediction, ScorerConfig
from runledger.scorers import continuation_tokens, score_token_logits


def test_causal_shift_and_continuation_mask_match_hand_calculation():
    torch = pytest.importorskip("torch")
    # Row 0 has two context tokens and two answer tokens. Row 1 has padding.
    ids = torch.tensor([[0, 1, 2, 1], [0, 2, 0, 0]])
    logits = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, math.log(2)], [0.0, math.log(3), 0.0], [99.0, 0.0, 0.0]],
            [[0.0, 0.0, math.log(4)], [99.0, 0.0, 0.0], [99.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
        ]
    )
    summed = score_token_logits(logits, ids, [2, 1], [4, 2], "sum")
    averaged = score_token_logits(logits, ids, [2, 1], [4, 2], "mean")
    assert summed == pytest.approx([math.log(2 / 4) + math.log(3 / 5), math.log(4 / 6)])
    assert averaged == pytest.approx([summed[0] / 2, summed[1]])


class CharacterTokenizer:
    def encode(self, text, **_):
        return [ord(c) for c in text]


def test_no_silent_truncation_or_empty_continuation():
    tokenizer = CharacterTokenizer()
    ids, prefix = continuation_tokens(tokenizer, "abc", " de", 8)
    assert ids == list(map(ord, "abc de")) and prefix == 3
    with pytest.raises(ValueError, match="truncation"):
        continuation_tokens(tokenizer, "abc", " too long", 8)
    with pytest.raises(ValueError, match="at least one"):
        continuation_tokens(tokenizer, "abc", "", 8)


def test_token_boundary_changes_are_rejected():
    class MergingTokenizer:
        def encode(self, text, **_):
            return [1, 2] if text == "prefix" else [1, 99, 3]

    with pytest.raises(ValueError, match="boundary"):
        continuation_tokens(MergingTokenizer(), "prefix", "answer", 8)


@pytest.mark.parametrize(
    "template",
    ["{prompt} {target}", "{prompt!r}", "{prompt:>20}", "{{prompt}}", "{prompt}{prompt}"],
)
def test_template_cannot_access_labels_or_change_protocol(template):
    with pytest.raises(ValueError):
        ScorerConfig(prompt_template=template)


def test_hf_requires_pinned_revision_and_hashed_snapshot():
    with pytest.raises(ValueError, match="immutable"):
        ScorerConfig(backend="hf", model_id="example/model", revision="main")


@pytest.mark.parametrize(
    "scores,prediction,correct",
    [([float("nan"), 0.0], 0, True), ([0.0, 1.0], 0, True), ([0.0, 1.0], 1, True)],
)
def test_invalid_predictions_cannot_reach_storage(scores, prediction, correct):
    with pytest.raises(ValueError):
        Prediction(
            sample_id="a",
            choice_scores=scores,
            token_counts=[1, 1],
            prediction=prediction,
            target=0,
            correct=correct,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": "an-actual-model"},
        {"revision": "main"},
        {"assets_sha256": {"../bad": "0" * 64}},
    ],
)
def test_hash_backend_cannot_impersonate_a_model(changes):
    with pytest.raises(ValueError):
        ScorerConfig(**changes)


def test_template_rejects_unnamed_positional_field():
    with pytest.raises(ValueError, match="exactly one"):
        ScorerConfig(prompt_template="{} {prompt}")
