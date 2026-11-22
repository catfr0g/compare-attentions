"""Tiny HOST routing trusts the saved deployment, never downloaded configuration."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
NONCE = "687f71e7a0c4479e821a549284351a73"


def host():
    spec = importlib.util.spec_from_file_location(
        "tiny_host_under_test", SCRIPTS / "unified_normal_host.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_mount_requires_explicit_trusted_nonce():
    case = {"task": "tiny_shakespeare_char", "engineering_only": False}
    assert host().declared_data_domains(case, NONCE) == (f"/kaggle/temp/normal-{NONCE}/data/tiny",)
    for nonce in (None, "", "a" * 31, "A" * 32, "../" + "a" * 32, 123):
        with pytest.raises(ValueError, match="^UNIFIED_HOST_TINY_NONCE$"):
            host().declared_data_domains(case, nonce)


def test_legacy_mounts_remain_unchanged():
    module = host()
    assert module.declared_data_domains(
        {"engineering_only": True, "config": {"data": "synthetic"}}
    ) == ("synthetic",)
    assert module.declared_data_domains({"engineering_only": False, "task": "sorting"}) == (
        "/kaggle/temp/sorting-science-fcff1b5daf574b70/data/sorting",
    )
    assert module.declared_data_domains({"engineering_only": False, "task": "kv_retrieval"}) == (
        "/kaggle/input/datasets/zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4",
        "/kaggle/input/ca-kv-1c660aa5f54d88ea9e0c1ce4",
    )


def test_previous_v3_reader_is_authenticated_against_its_own_files(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "tiny_view_under_test", SCRIPTS / "unified_normal_host_view.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import hashlib

    root = tmp_path / "candidate/payload/source/src"
    root.mkdir(parents=True)
    names = (
        "unified_normal_host.py",
        "unified_normal_host_view.py",
        "unified_normal_ordinary.py",
        "unified_normal_publications.py",
    )
    for name in names:
        (root / name).write_text("historical reader fixture", encoding="utf-8")
    recorded = {
        "release": "normal-host-readonly-nonblocking-tracking-v3",
        "purpose": "posttraining-readonly-observer-client",
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    }
    assert module.reader_identity(candidate=root.parents[2], recorded=recorded) == recorded
    (root / names[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="^UNIFIED_LEGACY_READER_BYTES$"):
        module.reader_identity(candidate=root.parents[2], recorded=recorded)


def test_saved_tiny_producer_in_isolated_emitted_reader(tmp_path):
    """Opt-in real checkpoint inspection; no model allocation, forward or services."""
    base = os.environ.get("TEST_TINY_SAVED_BASE")
    if base is None:
        pytest.skip("requires immutable completed Tiny producer")
    emitted = tmp_path / "reader"
    emitted.mkdir()
    for name in (
        "unified_normal_host.py",
        "unified_normal_host_view.py",
        "unified_normal_ordinary.py",
        "unified_normal_publications.py",
    ):
        shutil.copyfile(SCRIPTS / name, emitted / name)
    probe = tmp_path / "probe.py"
    probe.write_text(
        """import copy, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import unified_normal_host as host
assert Path(host.__file__).parent == Path(sys.argv[1])
base = Path(sys.argv[2])
payload = base / "candidate/payload"
nonce = sys.argv[3]
with host.sources(payload / "normal-package"):
    from normal_training_plan import derive
    from normal_training_native import verify_native, validate_full
    from compare_attention.train.checkpoint import load_checkpoint
    case = derive(payload / "normal-package", payload / "unified/normal-plan.json",
                  "TinyClassic202", payload / "configs/TinyClassic202.yaml",
                  base.parent / "data/tiny")
    root = base / "closure11/collected" / ("normal-" + nonce) / "out"
    domain, = host.declared_data_domains(case, nonce)
    case = {**case, "config": {**case["config"], "data": domain}}
    proof = verify_native(root, case)
    assert proof == host.read(root.parent / "native-proof.json")
    for changed in (
        {**case, "config": {**case["config"], "data": "/arbitrary/downloaded/path"}},
        {**case, "config": {**case["config"], "data": host.declared_data_domains(case, "0"*32)[0]}},
        {**case, "config": {**case["config"], "optim": {**case["config"]["optim"], "lr": 1.0}}},
    ):
        try:
            verify_native(root, changed)
        except ValueError as error:
            assert error.args == ("NORMAL_NATIVE_CONFIG",)
        else:
            raise AssertionError("changed config accepted")
    full = load_checkpoint(root / "checkpoints/step_00019281.pt")
    name = "wte.weight"
    full["model"][name] = full["model"][name].clone()
    full["model"][name].view(-1)[0] += 1
    try:
        validate_full(full, case, step=case["updates"])
    except ValueError as error:
        assert error.args == ("DDP checkpoint integrity checksum mismatch",)
    else:
        raise AssertionError("changed weights accepted")
    print(json.dumps({"native": proof, "reader": str(Path(host.__file__).resolve()),
                      "negative_controls": 4}))
""",
        encoding="utf-8",
    )
    child = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-I",
            "-B",
            str(probe),
            str(emitted),
            base,
            NONCE,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["native"]["updates"] == 19281
    assert result["negative_controls"] == 4
    assert Path(result["reader"]).parent == emitted
