from runledger.benchmark import METHODS, benchmark


def test_loading_methods_process_identical_content_in_fresh_processes(tmp_path):
    result = benchmark(
        tmp_path / "work",
        tmp_path / "measurements.json",
        rows=100,
        repeats=2,
        words=10,
        batch_size=16,
    )
    assert len(result["trials"]) == 8
    assert set(result["aggregates"]) == set(METHODS)
    for trial in result["trials"]:
        assert trial["samples"] == 100
        assert trial["semantic_sha256"] == result["semantic_sha256"]
        assert 0 < trial["first_batch_seconds"] <= trial["total_seconds"]
        assert trial["peak_rss_bytes"] >= trial["baseline_peak_rss_bytes"] > 0
