"""Bounded public future-case denial matrix; no science, credentials or allocation."""

import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("seed", [303, 404])
@pytest.mark.parametrize("family", ["classic", "deltanet"])
@pytest.mark.parametrize("task", ["sorting", "mqar-k64"])
def test_all_future_matrix_plans_denied_before_evidence(monkeypatch, tmp_path, seed, family, task):
    candidate = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    monkeypatch.syspath_prepend(str(candidate / "payload/source/src"))
    from seed202_pilot.seal import singleton_plan

    case_id = f"{task}-{family}-seed{seed}"
    with pytest.raises(ValueError, match="UNIFIED_REQUESTED_CASE_ONLY"):
        singleton_plan(case_id, candidate, tmp_path / "journal", [])
    assert not (tmp_path / "journal").exists()
