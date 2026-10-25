# Scoring protocol: continuation-loglikelihood-v1

RunLedger evaluates multiple-choice questions with a causal language model. It does not generate an answer and parse free-form text. For every candidate answer it computes the model's conditional log likelihood, selects the first maximum, and compares that zero-based choice index to the target.

## Exact prompt

The default context is:

```text
Question: {prompt}
Answer:
```

A single space precedes each candidate's text. There is no system message, chat template, few-shot example, answer letter, BOS token or EOS token added by RunLedger. The pinned tokenizer encodes with `add_special_tokens=False`.

For each choice, the context and context-plus-answer are tokenized. The context token sequence must be an exact prefix of the combined tokens. If a tokenizer merges across this boundary, the job fails with an explicit error. The implementation does not silently move a token from the context to the answer or truncate an overlong sequence.

Let `c` be the number of context tokens and `n` the full sequence length. The logit at position `t-1` predicts token `t`. Only token positions `c ... n-1` are scored. Context positions and right-padding positions contribute nothing.

```text
sum:  Σ log P(answer_token_t | context, earlier_answer_tokens)
mean: sum / number_of_answer_tokens
```

The default is the sum. The mean is an explicit alternative configuration, not a hidden correction. They answer different questions: sum favors a complete answer string's joint likelihood and can penalize longer answers; mean measures average per-token log likelihood. Neither is automatically the correct metric for every benchmark.

## What is pinned

- Model and tokenizer share an immutable 40-character Hugging Face commit.
- All downloaded snapshot files have SHA-256 checksums; missing, changed or extra files reject the snapshot.
- Safetensors only; remote model code is disabled; local files only during evaluation.
- CPU, float32, eager attention, inference mode, deterministic algorithms and a fixed seed.
- Prompt, prefix, normalization, maximum length, CPU thread count and choice batch size.
- Task batch size, dataset and package source/runtime fingerprints.

The seed is not a guarantee against numerical differences across hardware or library builds. The model is loaded separately in each spawned worker.

## Tests independent of the implementation path

1. A hand-computed tiny vocabulary verifies causal shifting, context masking, padding exclusion, sum and mean aggregation.
2. An actual downloaded model's batched scores are checked against its independent built-in loss with context labels replaced by `-100`. This uses a separate cross-entropy path instead of reusing the scoring helper.
3. Batched versus individual scoring agrees within the declared numerical tolerance, with matching predictions.
4. Flipping target labels leaves scores and predictions unchanged. Only correctness changes.
5. A real-model run interrupted during its commit is compared with an uninterrupted run using the complete ordered prediction digest.

## Interpreting the included model study

The model is [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2). The dataset is [AI2 ARC at its pinned revision](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453), first 128 ARC-Easy validation rows in original file order. Selection is fixed before scoring and recorded in the manifest, including the source-file hash.

This is a functional diagnostic, not a full benchmark. RunLedger has not established protocol parity with the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Harness task versions, prompt formatting, tokenization boundary rules, normalization and few-shot policies can differ. Do not compare this subset's accuracy directly with model-card or leaderboard numbers.

The HTML shows a Wilson 95% interval and per-answer-position slices. A small, first-in-file convenience subset can have selection bias; the interval does not correct that. The paired bootstrap compares correctness on the same fixed sample IDs and labels. Its interval is descriptive, unadjusted for repeated experiments, and cannot establish general model superiority.

The displayed top-two score margin is not a calibrated probability or confidence estimate. Hash-fixture accuracy, token counts and margins have no model-quality meaning.
