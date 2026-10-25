# Third-party sources and licenses

The repository contains original application code, handwritten examples, generated synthetic fixtures, and derived measurement artifacts. It does not redistribute model weights or the ARC questions. The MIT license applies to RunLedger code and its synthetic fixtures; it does not relicense dependencies or downloaded source material.

| Source | Pinned version or revision | Purpose / license |
|---|---|---|
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | CPU functional evaluator; Apache-2.0 |
| [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) | `210d026faf9955653af8916fad021475a3f00453` | ARC-Easy validation diagnostic; CC-BY-SA-4.0 |
| [AI2 ARC paper](https://arxiv.org/abs/1803.05457) | Clark et al., 2018 | Dataset attribution: *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge* |
| [PyArrow](https://arrow.apache.org/docs/python/) | `uv.lock` | Parquet storage and record batches |
| [SQLite](https://www.sqlite.org/wal.html) | Python runtime, recorded in each run | Local transactional ledger |
| [DuckDB](https://duckdb.org/docs/current/) | `uv.lock` | Analytical queries on exported Parquet |
| [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html) | `uv.lock` | Causal-model execution and numerical checks |
| [Transformers](https://huggingface.co/docs/transformers/) | `uv.lock` | Pinned tokenizer and safetensors model loading |
| [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) | Not a runtime dependency | Established evaluator reference; no protocol-parity claim |
| [uv GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/) | Action SHAs pinned in workflow | Dependency installation and CI configuration |

The importer normalizes Unicode and line endings, maps source choice labels to zero-based indices, and selects the first requested rows in pinned file order. Those transformations are recorded. Downloads remain in the ignored local workspace. If you distribute transformed ARC question data, preserve attribution and comply with the original share-alike license. Derived evidence in this repository contains source identifiers, target/predicted indices and scores, not question/answer text.
