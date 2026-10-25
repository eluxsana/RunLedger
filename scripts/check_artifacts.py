"""Recompute the checked-in evidence, including the stated loading aggregates."""

import json
import math
from pathlib import Path

from runledger.evidence import compare_exports, verify_export
from runledger.util import percentile, source_fingerprint


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "artifacts" / "reproduction"
    bundles = sorted(evidence.glob("*/*/bundle.json"))
    if len(bundles) != 7:
        raise RuntimeError("Expected four crash cases and three model cases")
    for bundle in bundles:
        summary = verify_export(bundle.parent)
        if summary["spec"]["source_sha256"] != source_fingerprint():
            raise RuntimeError(
                f"Evidence was produced by a different source revision: {bundle.parent}"
            )
    for study_name, pair in [
        ("failures", ("baseline", "before-commit")),
        ("failures", ("baseline", "during-commit")),
        ("failures", ("baseline", "after-commit")),
        ("model", ("sum", "sum-recovery")),
    ]:
        result = compare_exports(evidence / study_name / pair[0], evidence / study_name / pair[1])
        if not result["identical_result_digest"]:
            raise RuntimeError("Recovery result digest differs from baseline")
    recorded = json.loads((evidence / "model" / "study.json").read_text())
    observed = compare_exports(evidence / "model" / "sum", evidence / "model" / "mean")
    if recorded["normalization_comparison"] != observed:
        raise RuntimeError("Scoring comparison does not match its raw predictions")
    loading = json.loads((evidence / "loading.json").read_text())
    if loading["source_sha256"] != source_fingerprint():
        raise RuntimeError("Loading benchmark source changed")
    for method, aggregates in loading["aggregates"].items():
        trials = [trial for trial in loading["trials"] if trial["method"] == method]
        if len(trials) != loading["repeats"]:
            raise RuntimeError("Missing loading trials")
        for trial in trials:
            if (
                trial["samples"] != loading["rows"]
                or trial["semantic_sha256"] != loading["semantic_sha256"]
            ):
                raise RuntimeError("Loading methods processed different content")
        for metric, stats in aggregates.items():
            values = [trial[metric] for trial in trials]
            expected = {"median": percentile(values, 0.5), "min": min(values), "max": max(values)}
            if not all(
                math.isclose(stats[key], value, rel_tol=1e-12) for key, value in expected.items()
            ):
                raise RuntimeError(f"Incorrect aggregate: {method}/{metric}")
    print(
        json.dumps(
            {
                "verified_bundles": len(bundles),
                "verified_loading_trials": len(loading["trials"]),
                "all_recovery_digests_match": True,
                "source_sha256": source_fingerprint(),
            }
        )
    )


if __name__ == "__main__":
    main()
