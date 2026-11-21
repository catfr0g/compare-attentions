"""Bounded one-case fresh retry registration, never execution authority."""

import unified_normal_profiles as profiles


def test_fresh_restart_has_distinct_finite_profile():
    name = "SORTING_DELTA303_FRESH_RESTART_V1"
    assert name in profiles.PROFILES, "fresh retry must not consume the failed source profile"
    assert profiles.selected(name)["source"] != profiles.selected(profiles.REMAINING)["source"]
    assert profiles.selected(name)["review_policy"] == "EXTERNAL_TYPED_NORMAL_SOURCE_AND_PLAN"


def actual_generation():
    import json
    from pathlib import Path

    path = Path(
        "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/delta303-fresh-restart11/source03/generation.json"
    )
    return json.loads(path.read_bytes())


def test_actual_new_intent_and_raw_plan():
    from normal_fresh_restart_scope import CASE, PLAN
    from unified_normal_generation import intent_value
    from unified_normal_remaining import require_plan_pin

    generation = actual_generation()
    assert intent_value(generation, profiles.FRESH_RESTART) is generation
    require_plan_pin(CASE, PLAN, profiles.FRESH_RESTART)
    assert profiles.is_remaining_profile(profiles.REMAINING)
    assert profiles.is_remaining_profile(profiles.FRESH_RESTART)
    assert not profiles.is_remaining_profile("arbitrary-source")


def test_unauthorized_case_nonce_source_resume_rejected():
    import copy

    import pytest
    from normal_remaining_scope import CASE_IDS
    from unified_normal_generation import intent_value

    original = actual_generation()
    changes = [{"case_id": case} for case in CASE_IDS if case != original["case_id"]]
    changes += [
        {"nonce": "7e447ce213be444888c670a50aa3e5c8"},
        {"nonce": "e" * 32},
        {"source_identity_sha256": profiles.selected(profiles.REMAINING)["source"]},
        {"resume": "step_50232.pt"},
        {"restart": {**original["restart"], "sha256": "0" * 64}},
    ]
    for change in changes:
        candidate = {**copy.deepcopy(original), **change}
        with pytest.raises(ValueError):
            intent_value(candidate, profiles.FRESH_RESTART)


def test_old_consumed_plan_and_other_case_denied():
    import pytest
    from normal_fresh_restart_scope import CASE, PLAN
    from unified_normal_remaining import PLAN_PINS, require_plan_pin

    with pytest.raises(ValueError, match="FRESH_RESTART_FROZEN_PLAN"):
        require_plan_pin(CASE, PLAN_PINS[CASE], profiles.FRESH_RESTART)
    with pytest.raises(ValueError, match="FRESH_RESTART_FROZEN_PLAN"):
        require_plan_pin("sorting-classic-seed303", PLAN, profiles.FRESH_RESTART)
    require_plan_pin(
        "sorting-classic-seed303", PLAN_PINS["sorting-classic-seed303"], profiles.REMAINING
    )


def test_actual_unbound_candidate_uses_new_source_full_budget_no_resume():
    import json
    from pathlib import Path

    from normal_fresh_restart_scope import CASE, NONCE, PLAN
    from unified_normal_authority import binding
    from unified_normal_contract import descriptor

    root = Path(
        "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/delta303-fresh-restart11"
    )
    candidate = root / "candidate06"
    _path, request = descriptor(candidate)
    run = request["run"]
    singleton = json.loads((root / "singleton06.json").read_bytes())
    assert run["id"] == CASE and run["seed"] == 303 and run["engineering_only"] is False
    assert run["updates"] == 92082 and run["actual_flops"] == 5_000_002_837_413_888
    assert run["global_batch"] == 64 and run["world_size"] == 2 and run["local_batch"] == 32
    assert run["plan_sha256"] == PLAN
    assert len(run["checkpoint_steps"]) == len(run["validation_steps"]) == 11
    assert run["checkpoint_steps"][0] == 8372
    assert run["config"].get("resume") is None
    assert run["config"].get("warmstart") is None
    assert singleton["nonce"] == NONCE
    assert singleton["deployment"]["mode"] == "FRESH_TWO_STAGE"
    assert [row["case_id"] for row in request["predecessors"]] == ["sorting-classic-seed303"]
    assert request["predecessors"][0]["replay"]["profile"] == profiles.REMAINING
    assert binding(candidate)["normal_authority_bound"] is False
    assert (candidate / "payload/unified/normal-plan.json").read_bytes() == (
        root / "source03/normal-plan.json"
    ).read_bytes()
    assert not any("sorting-delta303-terminal10" in path for path in request["files"])
    kernel = (candidate / "payload/deploy" / CASE / "kernel/kernel.py").read_text()
    assert "MODE_MQAR_CLASSIC_STEP980" not in kernel
    assert "--resume" not in kernel
    worker = (candidate / "payload/normal-package/src/normal_training_worker.py").read_text()
    assert (
        'result = train(TrainConfig.model_validate(case["config"]), tracker, distributed=True)'
        in worker
    )


def test_actual_observer_release_only_launch_members_changed():
    import zipfile
    from pathlib import Path

    import pytest
    from build_normal_training_lifecycle import build_observer_release
    from unified_normal_profiles import authenticate_package

    root = Path("E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1")
    old = root / "launch-ready02/normal-package"
    new = root / "delta303-fresh-restart11/source03/normal-package"
    identity = authenticate_package(new, profiles.FRESH_RESTART)
    assert (
        identity["wheel_sha256"]
        == "a1e1db146ac0fef3583edbc6bee1d41ed1e83dffe85702c09667d34c96047e90"
    )
    changed = {
        p.relative_to(new).as_posix()
        for p in new.rglob("*")
        if p.is_file()
        and (
            not (old / p.relative_to(new)).exists()
            or p.read_bytes() != (old / p.relative_to(new)).read_bytes()
        )
    }
    assert changed == {"setup.py", "normal_torchrun_entry.py", "normal-source-identity.json"}
    with zipfile.ZipFile(new / "compare_attention-0.1.0-py3-none-any.whl") as wheel:
        assert len(wheel.namelist()) == 75
    with pytest.raises(ValueError, match="NORMAL_SOURCE_PARENT_OR_CREATE_ONLY"):
        build_observer_release(old, Path(__file__).resolve().parents[2], new)
