# Development

Use Python 3.12/3.13, `uv sync --locked`, and the checks in the README. Install `--extra hf` for real-model numerical tests. The core suite must remain runnable without downloading models or datasets.

Correctness changes should include a test at the relevant failure boundary. New scorers must keep label information out of score computation, return finite scores for every answer, specify tokenization and normalization, and record their identity in the run specification. Do not add silent truncation, partial-run accuracy exports, unpinned model loading, or duplicate-suppressing result inserts.

The `artifacts/reproduction` directory contains measured evidence, not golden timing thresholds. Never hand-edit a score or benchmark result. Regenerate into a fresh directory, compare the receipts, and update the documented interpretation. Wall time and process IDs are expected to vary. Reference experiments belong on local disk with no private employer data.

Run `uv build` and test the resulting wheel from outside the source directory before release. Package publication is a separate maintainer decision; contributing code does not automatically publish a package or model artifact.
