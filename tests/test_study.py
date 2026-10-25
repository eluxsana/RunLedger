from runledger.study import failure_study


def test_real_failure_study_produces_auditable_equal_digests(tmp_path):
    result = failure_study(tmp_path / "work", tmp_path / "artifacts", rows=32)
    assert len(result["cases"]) == 4
    assert all(case["identical_to_baseline"] for case in result["cases"])
    assert len({case["result_sha256"] for case in result["cases"]}) == 1
    assert [case["observed_sigkill_exits"] for case in result["cases"]] == [0, 1, 1, 1]
    assert [case["retried_jobs"] for case in result["cases"]] == [0, 1, 1, 0]
