"""Quality reports count unique JUnit identities and fail on errors or skips."""

import pytest


def test_junit_counts_unique_ids_and_rejects_failed_or_skipped_runs(tmp_path):
    from seed202_mqar_quality import junit_stats

    first, second = [tmp_path / name for name in ("first.xml", "second.xml")]
    first.write_text(
        '<testsuites><testsuite><testcase classname="a" name="one"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    second.write_text(
        '<testsuites><testsuite><testcase classname="a" name="one"/>'
        '<testcase classname="a" name="two"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    assert junit_stats([first, second])["unique_passed"] == 2
    for tag in ("failure", "error", "skipped"):
        second.write_text(
            '<testsuites><testsuite><testcase classname="a" name="two">'
            f"<{tag}/></testcase></testsuite></testsuites>",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="AFFECTED_TESTS_NOT_ALL_PASSED"):
            junit_stats([first, second])
