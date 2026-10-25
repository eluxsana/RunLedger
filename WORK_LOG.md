# Implementation record

## Delivered scope

RunLedger is an installable local Python package with a reproducible CPU demo, content-addressed datasets, transactional evaluation recovery, a pinned real-model scorer, verified evidence exports, an interactive report, and repeated loading experiments.

No private employer code or data was used. The initial delivery was local and unpublished. Public GitHub publication of this personal project was separately authorized on October 25, 2026. No employer systems, paid compute or private datasets are required.

## Completed verification

- Strict schema, normalization, duplicate-ID, corruption and safe-path tests.
- Atomic commits with active-token fencing, heartbeat extension, expired ownership rejection and fresh clock checks after lock acquisition.
- Exact sample, label and answer-count membership validation; mutable model contents are revalidated.
- Real SIGKILL before, during and after commit; timeout/resume; concurrent process claims.
- Pinned HF model/tokenizer assets and offline CPU evaluation; numerical comparison with independent masked model loss.
- Label and batching invariance tests, scoring-policy comparison, real-model crash recovery.
- Complete-only evidence exports; DuckDB metric recomputation; result and file checksums; HTML escaping.
- Four-way fresh-process loading benchmark: materialized and streaming readers for JSONL and Parquet.
- 70 passing tests; Ruff and formatting checks; wheel and sdist build; isolated clean wheel-install demo with no PyTorch.
- Desktop browser checks of layout, filtering and pagination; a narrow viewport check.

## Design choices

SQLite WAL coordinates one host; DuckDB reads immutable Parquet exports. Execution may repeat, but accepted sample results are unique within a run and committed atomically per job. The report distinguishes model evaluation from the deterministic infrastructure fixture. Every claim is tied to the receipts under `artifacts/reproduction`.

A progress-bar multiprocessing semaphore observed after a killed HF worker was replaced with a per-worker thread lock. Later real-model fault runs completed without that resource-tracker warning. Process handles are explicitly closed after joining.

## Remaining boundaries

Remote CI is configured but not executed. There is no multi-host, GPU, disk-loss, power-loss or production-scale claim. Detailed limitations and extension decisions are in `docs/ARCHITECTURE.md`; recorded results and methods are in `docs/EXPERIMENTS.md`.
