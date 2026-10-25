# Measured experiments and validation

Recorded on 2026-09-14 UTC using macOS arm64, 10 logical CPUs and 24 GiB RAM. The evaluation runtime is recorded in every export. All seven evidence bundles match package source fingerprint `ad7f0d5b210abde4e1eb2029c8c60f38c586165070305f030f02b228494d507a`.

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
| sum | 80 / 128 | 62.50% | 53.9%–70.4% | 12.04 s |
| sum-recovery | 80 / 128 | 62.50% | 53.9%–70.4% | 14.57 s |
| mean | 72 / 128 | 56.25% | 47.6%–64.5% | 12.14 s |

The sum-recovery run killed its model worker during its first commit. Its **complete ordered scores, predictions and token counts match the uninterrupted sum run byte-for-byte under canonical serialization**. No committed samples are missing or duplicated. [Model study and paired comparisons](../artifacts/reproduction/model/study.json).

Mean scoring improved 16 previously wrong answers and regressed 24 previously correct answers. The observed accuracy difference was **−6.25 percentage points**, with a paired bootstrap 95% interval of **−15.625 to +3.125 points** (5,000 resamples, seed 17). That interval includes zero. This experiment demonstrates sensitivity to the scoring policy; it does not establish that sum scoring is generally superior.

Execution timing includes worker/model startup, scoring, retries, waiting for lease recovery and worker cleanup. It excludes downloads and preflight dataset verification. The convenience subset, prompt and normalization differ from a standardized full benchmark; these are not leaderboard-comparable results.

## 3. Loading tradeoffs

100,000 synthetic records, 80 words per prompt, four choices, batch size 256 and five measured repetitions per method. Every method validates and hashes exactly the same normalized records in the same order. Each trial starts in a fresh Python process after one unmeasured warm-up per method; measurement order is shuffled with seed 17.

| Method | Peak RSS, MiB: median (range) | First 256 records, ms: median (range) | Full scan, s: median (range) |
|---|---:|---:|---:|
| jsonl-materialized | 270.3 (270.3–270.4) | 203.48 (202.18–217.04) | 1.058 (1.029–1.076) |
| jsonl-streaming | 58.4 (58.3–58.5) | 2.77 (2.70–3.11) | 1.007 (0.966–1.049) |
| parquet-materialized | 360.2 (358.6–361.3) | 312.65 (297.26–323.59) | 1.198 (1.194–1.236) |
| parquet-streaming | 71.7 (71.5–72.7) | 3.99 (3.89–4.24) | 1.205 (1.154–1.220) |

![Loading benchmark medians and ranges](../artifacts/loading.png)

On this workload, Parquet streaming used **73.5% less peak process memory** than materialized JSONL and delivered the first validated batch **51.0× sooner**. Its complete validated scan was **13.9% slower**. The streaming JSONL baseline used even less memory (58.4 MiB) and completed the scan fastest (1.007 s median). The dominant memory benefit comes from streaming rather than the file format alone.

Input sizes were 90.2 MiB for JSONL and 21.5 MiB for Parquet. RSS includes the interpreter and libraries; the raw trials also record the import-time high-water baseline. Timings start after shared imports and exclude data generation, downloads and cryptographic shard verification. End-to-end evaluation deliberately rechecks shard integrity, so these loader timings are not evaluator timings.

Both streaming implementations avoid materializing the corpus. For this full-row workload, streaming JSONL had the lowest overhead. Parquet's measured advantage was a smaller stored representation: **4.19× smaller** (21.5 versus 90.2 MiB), together with the typed/columnar representation used elsewhere in the system. No claim about projected-column query speed is made because that was not benchmarked. Including both materialized and streaming readers separates those tradeoffs.

[All raw trials, aggregates and methodology](../artifacts/reproduction/loading.json). The figure is generated from that file with Matplotlib; regenerate it with:

```sh
uv run --locked --script scripts/plot_loading.py artifacts/reproduction/loading.json artifacts/loading
```

## Validation record

- **70 tests passed**, including actual process crashes, timeout/resume, four-process claims, stale-token fencing, schema/integrity failures, report escaping, artifact verification and real-model numerical tests.
- **87.74% statement coverage** (1159/1321 statements), with multiprocessing/subprocess coverage enabled. Killed processes may not flush coverage. Coverage is supporting evidence, not a proof of correctness.
- Ruff checks and formatting passed; both sdist and wheel build successfully.
- A fresh virtual environment installed core dependencies from the lock and the wheel with no editable source install and no PyTorch. From outside the repository, Python isolated mode ran the 384-record crash demo with Hugging Face offline flags and verified the exported evidence.
- All seven final exports were reverified: file checksums, run-spec fingerprints, metrics, logical result digests and source identity.
- Browser verification covered desktop layout, sample search, correctness filtering, pagination and a 390 px layout check. No page-level horizontal overflow or JavaScript console errors were observed in the tested views.
- GitHub Actions is configured for macOS/Linux and Python 3.12/3.13. **Remote CI has not been executed** because this repository has not been published. No Docker, multi-host, GPU, paid-cloud or power-loss test is claimed.

Recheck committed evidence without running a model:

```sh
uv run --no-sync python scripts/check_artifacts.py
```

## Reproduction versus identical timing

Reproduction should preserve sample identities, scoring settings and the tested invariants. Timings, process IDs, event order and generated timestamps naturally differ. Exact float digests are demonstrated for the same model/runtime/hardware experiment; they are not promised across every machine or PyTorch build. The scripts refuse existing outputs so a rerun cannot silently replace earlier measurements.
