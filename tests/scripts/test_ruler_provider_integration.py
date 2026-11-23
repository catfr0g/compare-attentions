"""Independent RULER HOST routes; no SDK or scientific execution."""

import pytest


def test_ruler_mount_uses_only_explicit_authenticated_nonce():
    from unified_normal_host import declared_data_domains

    case = {"task": "ruler_variable_tracking", "engineering_only": False}
    nonce = "ab1297e03da54b06b71cafeb3928d401"
    assert declared_data_domains(case, nonce) == (f"/kaggle/temp/normal-{nonce}/data/ruler",)
    for invalid in (None, "", "A" * 32, "../" + nonce, 123):
        with pytest.raises(ValueError, match="^UNIFIED_HOST_RULER_NONCE$"):
            declared_data_domains(case, invalid)


def test_host_rejects_unknown_task_instead_of_mqar_fallback():
    from unified_normal_host import declared_data_domains

    with pytest.raises(ValueError, match="^UNIFIED_HOST_TASK$"):
        declared_data_domains({"task": "unknown", "engineering_only": False})


def test_ruler_finite_authority_rejects_historical_or_malformed_prefix():
    from normal_remaining_scope import CASE_IDS as old
    from normal_ruler_scope import CASE_IDS, case_rows, require_prefix

    assert CASE_IDS == (
        "RulerClassic202",
        "RulerDeltanet202",
        "RulerClassic303",
        "RulerDeltanet303",
        "RulerClassic404",
        "RulerDeltanet404",
    )
    for index, case_id in enumerate(CASE_IDS):
        assert require_prefix(case_id, CASE_IDS[:index]) == index
        assert case_rows()[index]["task"] == "ruler_variable_tracking"
        assert case_rows()[index]["primary_metric"] == "generated_chain_exact_match"
    for prefix in (
        old,
        [CASE_IDS[0], CASE_IDS[0]],
        [CASE_IDS[1], CASE_IDS[0]],
        [],
        ["TinyClassic202"],
    ):
        with pytest.raises(ValueError, match="^RULER_ACCEPTED_PREFIX$"):
            require_prefix(CASE_IDS[2], prefix)
    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
        require_prefix("ruler-classic-seed303", [])


def test_v5_reader_pins_helper_and_replays_authentic_v4(tmp_path):
    import hashlib
    import subprocess
    from pathlib import Path

    from unified_normal_host_view import reader_identity

    current = reader_identity()
    assert current["release"] == "normal-host-readonly-ruler-v5"
    assert "normal_training_tracking.py" in current["files"]
    root = tmp_path / "candidate/payload/source/src"
    root.mkdir(parents=True)
    names = (
        "unified_normal_host.py",
        "unified_normal_host_view.py",
        "unified_normal_ordinary.py",
        "unified_normal_publications.py",
    )
    for name in names:
        child = subprocess.run(
            ["git", "show", f"fbe038a0ab05ab65deb297da61118efb3ee313df:scripts/{name}"],
            cwd=Path(__file__).parents[2],
            capture_output=True,
            check=True,
        )
        (root / name).write_bytes(child.stdout)
    recorded = {
        "release": "normal-host-readonly-tiny-domain-v4",
        "purpose": "posttraining-readonly-observer-client",
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    }
    assert reader_identity(candidate=root.parents[2], recorded=recorded) == recorded
    (root / names[0]).write_bytes(b"TAMPERED")
    with pytest.raises(ValueError, match="UNIFIED_LEGACY_READER_BYTES"):
        reader_identity(candidate=root.parents[2], recorded=recorded)


def test_ruler_profile_authenticates_real_source_and_permission():
    from pathlib import Path

    from normal_ruler_scope import AUTHORIZATION_SHA256
    from normal_training_plan import digest
    from unified_normal_generation import intent_value
    from unified_normal_profiles import RULER, authenticate_package, is_remaining_profile, selected

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    package = root / "implementation05/source03/normal-package"
    assert not is_remaining_profile(RULER)
    assert selected(RULER)["source"] == digest(package / "normal-source-identity.json")
    assert selected(RULER)["archive"] == digest(package.parent / "ruler-normal-source.zip")
    authenticate_package(package, RULER)
    value = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "a6ab17b20eb1491d854e0c0e2c1ee8b1",
        "case_id": "RulerClassic202",
        "source_identity_sha256": selected(RULER)["source"],
        "authorization": {
            "path": str(root / "USER-AUTHORIZATION01.json"),
            "sha256": AUTHORIZATION_SHA256,
        },
    }
    assert intent_value(value, RULER) == value
    value["case_id"] = "TinyClassic202"
    with pytest.raises(ValueError, match="UNIFIED_GENERATION_SCOPE"):
        intent_value(value, RULER)


def test_callback_full_grid_includes_flat_retained_and_nested_final(tmp_path, monkeypatch):
    import hashlib

    import normal_training_native as native
    from unified_normal_ordinary import validate_snapshot_fulls

    import compare_attention.train.checkpoint as checkpoint

    steps = [*range(10, 101, 10), 107]
    files = {}
    for index, step in enumerate(steps):
        name = f"revision{index}/step_{step:08d}.pt"
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text(str(step))
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    duplicate = "final/out/checkpoints/step_00000107.pt"
    path = tmp_path / duplicate
    path.parent.mkdir(parents=True)
    path.write_text("107")
    files[duplicate] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    visited = []
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda p: {"step": int(p.read_text())})
    monkeypatch.setattr(native, "validate_full", lambda value, case, step: visited.append(step))
    validate_snapshot_fulls(tmp_path, {"checkpoint_steps": steps}, files)
    assert visited == steps
    files["other/step_00000099.pt"] = files[duplicate]
    with pytest.raises(ValueError, match="UNIFIED_ORDINARY_FULL_GRID"):
        validate_snapshot_fulls(tmp_path, {"checkpoint_steps": steps}, files)
    del files["other/step_00000099.pt"]
    del files["revision0/step_00000010.pt"]
    with pytest.raises(ValueError, match="UNIFIED_ORDINARY_FULL_COVERAGE"):
        validate_snapshot_fulls(tmp_path, {"checkpoint_steps": steps}, files)


def test_finite_ruler_request_rejects_changed_science():
    import copy
    import json
    from pathlib import Path

    from unified_normal_ruler import validate_request

    path = Path(
        "E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1/"
        "implementation05/source01/derived-case.json"
    )
    case = json.loads(path.read_bytes())["case"]
    validate_request({"run": case, "predecessors": []})
    for section, field, changed in (
        ("schedule", "train_flops", 1),
        ("optim", "learning_rate", 0.1),
        ("model", "n_embd", 256),
        ("dataset", "source_commit", "0" * 40),
        ("dataset", "n_val", 999),
    ):
        bad = copy.deepcopy(case)
        bad["config"][section][field] = changed
        with pytest.raises(ValueError, match="RULER_FROZEN_SCIENTIFIC_CONFIG"):
            validate_request({"run": bad, "predecessors": []})
