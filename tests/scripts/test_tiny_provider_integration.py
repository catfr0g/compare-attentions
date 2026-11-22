"""Finite Tiny HOST routing; no CUDA, credentials, or provider calls."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))


def test_tiny_independent_finite_specs_and_prefix():
    from normal_remaining_scope import CASE_IDS as old
    from normal_tiny_scope import CASE_IDS, case_rows, require_prefix

    assert CASE_IDS == (
        "TinyClassic202",
        "TinyDeltanet202",
        "TinyClassic303",
        "TinyDeltanet303",
        "TinyClassic404",
        "TinyDeltanet404",
    )
    assert not set(CASE_IDS) & set(old)
    assert require_prefix(CASE_IDS[0], []) == 0
    assert require_prefix(CASE_IDS[1], [CASE_IDS[0]]) == 1
    for row in case_rows():
        assert row["task"] == "tiny_shakespeare_char"
        assert row["global_batch"] == 2 * row["local_batch"] == 64
        assert row["test_split_permitted"] is False
        assert row["requested_flops"] == 5_000_000_000_000_000
    with pytest.raises(ValueError, match="^TINY_ACCEPTED_PREFIX$"):
        require_prefix(CASE_IDS[0], [old[-1]])
    with pytest.raises(ValueError, match="^TINY_ACCEPTED_PREFIX$"):
        require_prefix(CASE_IDS[1], [])


def test_tiny_profile_is_separate_and_generation_is_finite():
    from normal_tiny_scope import AUTHORIZATION_SHA256
    from unified_normal_generation import intent_value
    from unified_normal_profiles import TINY, is_remaining_profile, selected

    assert not is_remaining_profile(TINY)
    path = Path(
        "E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1/authorization01.json"
    )
    value = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "ab1297e03da54b06b71cafeb3928d401",
        "case_id": "TinyClassic202",
        "source_identity_sha256": selected(TINY)["source"],
        "authorization": {"path": str(path), "sha256": AUTHORIZATION_SHA256},
    }
    if path.exists():
        assert intent_value(value, TINY) == value
    value["case_id"] = "sorting-classic-seed303"
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
        intent_value(value, TINY)


def test_actual_cloud_candidate_has_tiny_data_and_honest_global_contract():
    import json

    root = Path(
        "E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1/provider04/first-classic202-candidate03"
    )
    if not root.parent.exists():
        pytest.skip("private engineering artifact root absent")
    assert (root / "candidate.json").exists()
    plan = json.loads((root / "PLAN-UNAPPROVED.json").read_bytes())
    assert plan["seeds_authorized"] == [202, 303, 404]
    assert plan["seeds_forbidden_pending_user_decision"] == []
    assert plan["host_acceptance"]["task_all_validation_rows"] == 109
    assert plan["host_acceptance"]["fresh_clearml_training_and_inference"] is False
    from unified_normal_contract import descriptor

    _, request = descriptor(root)
    assert Path(request["run"]["config"]["data"]).is_relative_to(root.parent)
    assert request["predecessors"] == []


def test_actual_emitted_kernel_extracts_tiny_then_stops_at_real_review_gate(tmp_path):
    import hashlib
    import runpy
    import sys
    import time

    root = Path(
        "E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1/provider04/first-classic202-candidate03"
    )
    if not root.exists():
        pytest.skip("private emitted candidate absent")
    kernel = root / "payload/deploy/TinyClassic202/kernel/kernel.py"
    namespace = runpy.run_path(str(kernel), run_name="tiny_emitted_no_main")
    main = namespace["normal_main"]
    scope = main.__globals__
    scope["inspect_hardware"] = lambda _driver: None  # stdlib facade, NOT CUDA proof
    scope["driver_directory"] = lambda: tmp_path
    archives = {
        "seed202-pilot.zip.bin": root / "payload/worker/seed202-pilot.zip.bin",
        "scientific.zip.bin": root / "payload/inherited/sorting-science.zip.bin",
    }
    scope["resolve_mount"] = lambda _root, _owner, _slug, name: archives[name]
    observed = {}

    def trace(frame, event, _arg):
        if frame.f_code is main.__code__ and event == "exception":
            observed.update(frame.f_locals)
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(FileNotFoundError) as failure:
            main(tmp_path / "mounts", tmp_path / "working", tmp_path / "packages", time.monotonic())
    finally:
        sys.settrace(previous)
    assert Path(failure.value.filename).name == "independent-review.json"
    data = observed["data"]
    assert data == observed["release"] / "data/tiny"
    assert (data / "input.txt").is_file()
    assert (data / "val_labels.npy").is_file()
    assert (
        hashlib.sha256((data / "input.txt").read_bytes()).hexdigest()
        == "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    )
    assert "setup" not in observed  # no runtime bootstrap, model, or forward


def test_actual_emitted_contract_authorizes_exact_six():
    import runpy

    from normal_tiny_scope import CASE_IDS

    root = Path(
        "E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1/provider04/first-classic202-candidate03"
    )
    if not root.exists():
        pytest.skip("private emitted candidate absent")
    namespace = runpy.run_path(
        str(root / "payload/source/src/seed202_pilot/contract.py"), run_name="tiny_emitted_contract"
    )
    assert tuple(case.case_id for case in namespace["cases"]()) == CASE_IDS
    for case_id in CASE_IDS:
        assert namespace["require_authorized_case"](case_id).task == "tiny_shakespeare_char"
    with pytest.raises(ValueError, match="^UNAUTHORIZED_PILOT_CASE$"):
        namespace["require_authorized_case"]("sorting-classic-seed202")
