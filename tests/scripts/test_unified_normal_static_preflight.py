"""Tiny real Win32 fixtures; synthetic plan builder, never launch authority."""

import contextvars
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from seed202_pilot import seal

    root = Path(os.environ["UNIFIED_LIFETIME_FIXTURES"]) / (tmp_path.name + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    candidate = root / "candidate"
    candidate.mkdir()
    source = candidate / "source.py"
    source.write_bytes(b"synthetic immutable source\n")
    plan = {
        "schema_version": 2,
        "case_id": "SYNTHETIC",
        "candidate_root": str(candidate),
        "journal_directory": str(root / "journals"),
        "predecessors": [],
    }
    plan_path, review = root / "plan.json", root / "review.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    review.write_text('{"scope":"SYNTHETIC_NOT_AUTHORITY"}', encoding="utf-8")
    calls = []

    def builder(*_args):
        calls.append("immutable_reconstruction")
        return plan

    monkeypatch.setattr(seal, "singleton_plan", builder)
    return candidate, source, plan, plan_path, review, calls


def test_preflight_reconstructs_before_live_only(fixture):
    from seed202_pilot.seal import _plan
    from unified_normal_live import retained_lease
    from unified_normal_preflight import static_preflight

    candidate, source, plan, path, review, calls = fixture
    with retained_lease([source, path, review]) as held:
        with static_preflight(candidate, held, path, review):
            assert calls == ["immutable_reconstruction"]
            calls.append("LIVE_SYNTHETIC_BOUNDARY")
            _plan(plan)
            _plan(plan)
            assert calls == ["immutable_reconstruction", "LIVE_SYNTHETIC_BOUNDARY"]
            with pytest.raises(PermissionError):
                source.write_bytes(b"changed")
    _plan(plan)
    assert calls[-1] == "immutable_reconstruction"


def test_expired_saved_proof_still_rejects_unchanged_window():
    import time

    from seed202_pilot.seal import _quota

    proof = json.loads(Path(os.environ["EXPIRED_PROVIDER_PROOF"]).read_bytes())
    with pytest.raises(ValueError, match="^FRESH_ABSENCE_WINDOW$"):
        _quota(proof, 30600, time.time(), fresh=True)


@pytest.mark.parametrize("exit_error", [False, True])
def test_copied_scope_expires_on_exit(fixture, exit_error):
    from unified_normal_live import retained_lease
    from unified_normal_preflight import authenticated_plan, static_preflight

    candidate, source, plan, path, review, _ = fixture
    with retained_lease([source, path, review]) as held:
        try:
            with static_preflight(candidate, held, path, review):
                copied = contextvars.copy_context()
                if exit_error:
                    raise RuntimeError("synthetic exit")
        except RuntimeError:
            assert exit_error
        with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_OWNERSHIP$"):
            copied.run(authenticated_plan, plan)


def test_other_thread_cannot_borrow_scope(fixture):
    from unified_normal_live import retained_lease
    from unified_normal_preflight import authenticated_plan, static_preflight

    candidate, source, plan, path, review, _ = fixture
    with retained_lease([source, path, review]) as held:
        with static_preflight(candidate, held, path, review):
            copied = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(copied.run, authenticated_plan, plan)
                with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_OWNERSHIP$"):
                    future.result(timeout=10)


def test_changed_plan_and_unlocked_root_addition_reject(fixture):
    from seed202_pilot.seal import _plan
    from unified_normal_live import retained_lease
    from unified_normal_preflight import static_preflight

    candidate, source, plan, path, review, _ = fixture
    with retained_lease([source, path, review]) as held:
        with static_preflight(candidate, held, path, review):
            with pytest.raises(ValueError, match="^EXACT_PLAN_PINS$"):
                _plan({**plan, "case_id": "WRONG"})
            with pytest.raises(ValueError, match="^EXACT_PLAN_PINS$"):
                _plan({**plan, "candidate_root": str(candidate.parent)})
            added = candidate / "unlocked.py"
            added.write_bytes(b"not authenticated")
            try:
                with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_CLOSURE$"):
                    _plan(plan)
            finally:
                added.unlink()


def test_release_inside_active_scope_rejects(fixture):
    from unified_normal_live import retained_lease
    from unified_normal_preflight import authenticated_plan, static_preflight

    candidate, source, plan, path, review, _ = fixture
    lease = retained_lease([source, path, review])
    held = lease.__enter__()
    with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_OWNERSHIP$"):
        with static_preflight(candidate, held, path, review):
            copied = contextvars.copy_context()
            lease.__exit__(None, None, None)
            source.write_bytes(b"modified after release")
            with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_OWNERSHIP$"):
                copied.run(authenticated_plan, plan)


@pytest.mark.parametrize("missing", ["review", "source"])
def test_uncovered_review_and_stale_lease_reject(fixture, missing):
    from unified_normal_live import retained_lease
    from unified_normal_preflight import static_preflight

    candidate, source, _, path, review, _ = fixture
    files = [source, path] if missing == "review" else [path, review]
    with retained_lease(files) as incomplete:
        with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_COVERAGE$"):
            with static_preflight(candidate, incomplete, path, review):
                pytest.fail("unlocked review accepted")
    with pytest.raises(ValueError, match="^STATIC_PREFLIGHT_OWNERSHIP$"):
        with static_preflight(candidate, incomplete, path, review):
            pytest.fail("released lease accepted")
