# Measured experiments and validation

Regenerated on 2026-10-25 UTC using macOS arm64, 10 logical CPUs and 24 GiB RAM. The evaluation runtime is recorded in every export. All seven evidence bundles match package source fingerprint `31fd9d4cae2a60050f7f94769db66a79789b16d85186012664eed78341b5f97e`.

These are measurements from one local machine. Timing ranges are observed min–max, not confidence intervals, and no claim is made about cold storage, GPU performance or a distributed cluster.

## 1. Recovery at the transaction boundary

| Case | Result rows | Attempts | Observed SIGKILL exits | Exact baseline digest? |
|---|---:|---:|---:|---|
| baseline | 256 | 16 | 0 | Yes |
| before-commit | 256 | 17 | 1 | Yes |
| during-commit | 256 | 17 | 1 | Yes |
| after-commit | 256 | 16 | 1 | Yes |

All four cases have the same ordered prediction digest:

```text
72f888880e843ba45c249317f52977ca8f658a3945c1f0fde8d2f2c03b678176
```

The workload is 256 synthetic records, 16 jobs and two worker processes. Before/during commit cases retry one job; the after-commit case retains the completed job without retry. Each fault is a real SIGKILL of a first-attempt worker. [Raw study](../artifacts/reproduction/failures/study.json), [mid-commit dashboard](../artifacts/reproduction/failures/during-commit/index.html).

This demonstrates process-crash recovery on one host. It does not test host power loss, failed storage or network partitions. The hash backend isolates orchestration from numerical model behavior.

## 2. Pinned real-model evaluation and recovery

SmolLM2-135M, CPU float32, one worker, two PyTorch threads, choice mini-batches of eight and task batches of eight questions. First 128 ARC-Easy validation samples in pinned file order; selection was fixed before scoring. See the [exact protocol](SCORING.md) and the model/dataset revisions in the reports.

| Case | Correct / total | Accuracy | Wilson 95% interval | Execution time |
|---|---:|---:|---:|---:|
| sum | 80 / 128 | 62.50% | 53.9%–70.4% | 9.57 s |
| sum-recovery | 80 / 128 | 62.50% | 53.9%–70.4% | 11.16 s |
| mean | 72 / 128 | 56.25% | 47.6%–64.5% | 8.55 s |

The sum-recovery run killed its model worker during its first commit. Its **complete ordered scores, predictions and token counts match the uninterrupted sum run byte-for-byte under canonical serialization**. No committed samples are missing or duplicated. [Model study and paired comparisons](../artifacts/reproduction/model/study.json).

Mean scoring improved 16 previously wrong answers and regressed 24 previously correct answers. The observed accuracy difference was **−6.25 percentage points**, with a paired bootstrap 95% interval of **−15.625 to +3.125 points** (5,000 resamples, seed 17). That interval includes zero. This experiment demonstrates sensitivity to the scoring policy; it does not establish that sum scoring is generally superior.

Execution timing includes worker/model startup, scoring, retries, waiting for lease recovery and worker cleanup. It excludes downloads and preflight dataset verification. The convenience subset, prompt and normalization differ from a standardized full benchmark; these are not leaderboard-comparable results.

## Prediction-drift explorer

The new `compare --report` command joins the same verified examples and shows every prediction flip, including wrong-to-wrong changes. On the sum-to-mean scoring comparison there are **45 changed predictions**: 16 improved, 24 regressed, and 5 changed between two wrong choices. The model weights and examples are unchanged; this is scoring-policy sensitivity, not a comparison of two model revisions.

```sh
runledger compare artifacts/reproduction/model/sum artifacts/reproduction/model/mean --report /tmp/drift.html
```

Open the resulting HTML file locally. Search sample IDs or filter outcomes; the report makes no network requests. [Screenshot](../artifacts/drift.png). Raw prompt text is not embedded. The bootstrap interval is descriptive for this fixed subset and includes zero.

## 3. Loading tradeoffs

100,000 synthetic records, 80 words per prompt, four choices, batch size 256 and five measured repetitions per method. Every method validates and hashes exactly the same normalized records in the same order. Each trial starts in a fresh Python process after one unmeasured warm-up per method; measurement order is shuffled with seed 17.

| Method | Peak RSS, MiB: median (range) | First 256 records, ms: median (range) | Full scan, s: median (range) |
|---|---:|---:|---:|
| jsonl-materialized | 270.5 (270.4–270.6) | 198.63 (193.29–208.55) | 1.007 (0.988–1.049) |
| jsonl-streaming | 58.5 (58.5–58.5) | 2.67 (2.52–2.71) | 0.974 (0.951–1.012) |
| parquet-materialized | 359.6 (356.9–361.6) | 317.97 (307.53–374.20) | 1.242 (1.186–1.249) |
| parquet-streaming | 72.8 (71.9–73.6) | 3.93 (3.82–4.01) | 1.200 (1.177–1.217) |

![Loading benchmark medians and ranges](../artifacts/loading.png)

On this workload, Parquet streaming used **73.1% less peak process memory** than materialized JSONL and delivered the first validated batch **50.5× sooner**. Its complete validated scan was **19.2% slower**. The streaming JSONL baseline used even less memory (58.5 MiB) and completed the scan fastest (0.974 s median). The dominant memory benefit comes from streaming rather than the file format alone.

Input sizes were 90.2 MiB for JSONL and 21.5 MiB for Parquet. RSS includes the interpreter and libraries; the raw trials also record the import-time high-water baseline. Timings start after shared imports and exclude data generation, downloads and cryptographic shard verification. End-to-end evaluation deliberately rechecks shard integrity, so these loader timings are not evaluator timings.

Both streaming implementations avoid materializing the corpus. For this full-row workload, streaming JSONL had the lowest overhead. Parquet's measured advantage was a smaller stored representation: **4.19× smaller** (21.5 versus 90.2 MiB), together with the typed/columnar representation used elsewhere in the system. No claim about projected-column query speed is made because that was not benchmarked. Including both materialized and streaming readers separates those tradeoffs.

[All raw trials, aggregates and methodology](../artifacts/reproduction/loading.json). The figure is generated from that file with Matplotlib; regenerate it with:

```sh
uv run --locked --script scripts/plot_loading.py artifacts/reproduction/loading.json artifacts/loading
```

## Validation record

- **74 tests passed**, including actual process crashes, timeout/resume, four-process claims, stale-token fencing, schema/integrity failures, report escaping, artifact verification and real-model numerical tests.
- The initial release recorded 87.74% statement coverage. Coverage has not been recomputed for the prediction-drift addition; the current test count is from an actual full local run.
- Ruff checks and formatting passed; both sdist and wheel build successfully.
- A fresh virtual environment installed core dependencies from the lock and the wheel with no editable source install and no PyTorch. From outside the repository, Python isolated mode ran the 384-record crash demo with Hugging Face offline flags and verified the exported evidence.
- All seven final exports were reverified: file checksums, run-spec fingerprints, metrics, logical result digests and source identity.
- Browser verification covered desktop layout, sample search, correctness filtering, pagination and a 390 px layout check. No page-level horizontal overflow or JavaScript console errors were observed in the tested views.
- GitHub Actions is configured for macOS/Linux and Python 3.12/3.13. The published initial-release workflow passed. The prediction-drift addition is tested locally; check the Actions tab for its current remote result. No Docker, multi-host, GPU, paid-cloud or power-loss test is claimed.

Recheck committed evidence without running a model:

```sh
uv run --no-sync python scripts/check_artifacts.py
```

## Reproduction versus identical timing

Reproduction should preserve sample identities, scoring settings and the tested invariants. Timings, process IDs, event order and generated timestamps naturally differ. Exact float digests are demonstrated for the same model/runtime/hardware experiment; they are not promised across every machine or PyTorch build. The scripts refuse existing outputs so a rerun cannot silently replace earlier measurements.
