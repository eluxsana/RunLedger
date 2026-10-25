# Walkthrough: understand it, demonstrate it, defend it

## The five-minute demonstration

1. Open the real-model HTML report in `artifacts/reproduction/model/sum/index.html`. Show the model revision, dataset selection, result digest and full run specification. Explain why an accuracy number without those details is difficult to reproduce.
2. Open `artifacts/reproduction/model/study.json`. The uninterrupted and recovered sum-scoring runs have identical ordered result digests. The recovery case killed a real worker during the result transaction.
3. Open `artifacts/reproduction/failures/study.json`. Point to the before/during/after commit cases. Before and during commit require another attempt; after commit does not.
4. Run a fresh offline demo: `uv run --no-sync runledger -w .runledger/live-demo demo --output artifacts/live-demo`. Open its HTML and filter an individual result. Explain why the hash fixture is appropriate for recovery tests but irrelevant to model quality.
5. Show `docs/EXPERIMENTS.md`. Streaming lowers peak RSS and first-batch latency on this synthetic workload, while total scan throughput is a different measurement. Discuss the tradeoff instead of presenting only the largest speedup.

## Reading order

| File | Question to answer |
|---|---|
| `schema.py` | Which malformed inputs are rejected, and why does revalidating a frozen model matter when it contains a mutable list? |
| `dataset.py` | How do logical identity, physical artifact identity and provenance differ? What is actually bounded in memory? |
| `ledger.py` | Where is the linearization point of a claim? How does a fencing token stop an old worker? |
| `runner.py` | Why does model computation happen outside the write transaction? Who sends heartbeats? |
| `scorers.py` | Which logit predicts the first answer token? Which tokens are masked? |
| `evidence.py` | Why does the exporter refuse a partial run? Which fields belong in a result digest? |
| `tests/test_runner.py` | What would remain durable if the process dies at each tested boundary? |

## Questions an interviewer could ask

**Why not just write results to a JSONL file?**

Appending alone does not couple results to durable job completion. A worker can append, die before acknowledging, and append the same samples on retry. Deduplication after the fact can conceal mismatched scores. Here the result rows and completion are one transaction, and uniqueness violations are errors.

**Does this guarantee exactly-once execution?**

No. It guarantees unique accepted results per sample within a run, with an atomic commit per job. Computation may happen twice. External side effects in a custom scorer would need their own idempotency design.

**Why are both a lease and a token needed?**

A lease allows recovery when a worker stops responding. The token distinguishes the newly authorized attempt from the expired one. Without fencing, the old worker could return after a new worker has already claimed the job.

**Why check the clock after obtaining the database lock?**

A worker can wait for a lock while its lease expires. Checking against the time before that wait could accept stale ownership. The transaction rereads ownership and time after acquiring the lock.

**Why SQLite instead of Redis, Kafka, or Kubernetes?**

The scope is one host and short durable transactions. SQLite keeps the invariant easy to inspect and test without operating extra services. It is a deliberate local coordinator. A multi-host system would need server-side coordination and different failure assumptions; adding infrastructure names alone would not establish that correctness.

**Why Parquet if JSONL can scan faster here?**

Storage size, first-batch availability and peak memory are separate from a full sequential scan's speed. This benchmark includes Python validation and hashing, which can dominate either format. Parquet streaming avoids retaining the entire decoded corpus. The report includes materialized and streaming readers for both formats and total scan time so the tradeoff is visible.

**Why use a small, older model?**

It makes the entire functional experiment reproducible on a laptop. The project demonstrates evaluator correctness and recovery, not model research. Its contribution is not a higher ARC score or a newly trained model.

**What does the sum-versus-mean experiment prove?**

Scoring policy can change predictions even when the weights and data do not change. The small subset's observed difference is not a universal ranking of those policies. The paired interval includes uncertainty and does not correct selection bias.

**What are the biggest remaining bottlenecks?**

Repeated shard hashing/row-group decoding, one SQLite writer, model duplication per worker, and full-vocabulary CPU logits. Job/event metadata and paired comparison are currently materialized in memory. Measure each cost before replacing components.

## Hands-on exercises to make the project yours

These are learning extensions; the delivered implementation and evidence do not depend on completing them.

1. **Trace one sample.** Find an ID in the report, locate its Parquet row and SQLite job, explain the target mapping, and recompute the prediction's argmax by hand.
2. **Reproduce a stale worker.** Read the lease test, then modify its fake clock values and predict whether each heartbeat and commit will be accepted before running it.
3. **Change one scoring setting.** Use a new normalization or prompt configuration, create a new run and explain exactly why its ID changes. Do not alter the checked-in evidence bundle.
4. **Vary workload size.** Benchmark 10,000 and 200,000 records in fresh output files, plot memory versus rows, and test whether the streaming advantage persists. Do not generalize from one machine without new evidence.
5. **Optimize one measured bottleneck.** Add a bounded cache of verified shard snapshots or an isolated optimization for selecting answer logits. Keep the same integrity and numerical tests, and report any accuracy/timing tradeoff.

If you discuss this project in an interview, be candid about assistance used in building it. Be able to run it, explain the invariants, and implement a small change yourself. The value is demonstrated understanding, not the appearance of a large codebase.

## Portfolio wording

A defensible project heading is **RunLedger — Reproducible LLM Evaluation Infrastructure**.

Suggested description, after personally reviewing the included evidence:

> Built a versioned LLM evaluation system with Parquet datasets, SQLite lease fencing, pinned Hugging Face scoring and auditable reports; verified exact result recovery after worker crashes at three commit boundaries.

For an ML infrastructure application, the strongest discussion is data/version integrity, transactional recovery and honest performance measurement. Avoid “production scale,” “distributed exactly once,” “SOTA,” or an unmeasured GPU/training speedup. Add a benchmark metric only with the precise workload and baseline from the experiments document.
