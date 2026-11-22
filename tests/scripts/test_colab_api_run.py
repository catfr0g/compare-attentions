"""Controller tests run generated stdlib code locally, never on Google."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def mod():
    return importlib.import_module("colab_api_run")


def test_default_plan_never_reads_private_files(tmp_path, capsys):
    assert (
        mod().main(
            [
                "setup",
                "--runtime",
                "runtimes/test",
                "--run-name",
                "run-1",
                "--journal",
                str(tmp_path / "absent.jsonl"),
                "--export-dir",
                str(tmp_path / "missing"),
                "--env-file",
                str(tmp_path / "SECRET"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["phase"] == "plan"
    assert not list(tmp_path.iterdir())


def test_generated_supervisor_completes_real_child(tmp_path):
    remote = importlib.import_module("colab_api_run_remote")
    state = tmp_path / "state.json"
    command = [sys.executable, "-c", 'print("actual-child");raise SystemExit(7)']
    code = remote.supervisor_code(
        command, str(state), str(tmp_path / "log"), "", {"phase": "setup", "sha256": "a" * 64}
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert json.loads(state.read_text())["returncode"] == 7
    assert json.loads(state.read_text())["state"] == "completed"
    assert (tmp_path / "log").read_text().strip() == "actual-child"


def test_remote_train_blocks_before_spawn_without_verified_setup(tmp_path, monkeypatch):
    remote = importlib.import_module("colab_api_run_remote")
    monkeypatch.setattr(remote, "ROOT", str(tmp_path) + "/")
    binding = {"runtime": "runtimes/test", "run_name": "run-1", "sha256": "a" * 64}
    root = tmp_path / "run-1"
    root.mkdir()
    (root / "identity.json").write_text(json.dumps(binding))
    (root / "inputs.json").write_text(json.dumps({"verified": True, "sha256": "a" * 64}))
    (root / "setup").mkdir()
    (root / "setup/state.json").write_text(
        json.dumps(dict(binding, state="completed", returncode=7))
    )
    result = subprocess.run(
        [sys.executable, "-c", remote.launch_code(binding, "train")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "SETUP_NOT_VERIFIED" in result.stderr
    assert not (root / "train").exists()


def test_status_reads_real_numeric_metrics_and_no_arbitrary_strings(tmp_path, monkeypatch):
    remote = importlib.import_module("colab_api_run_remote")
    monkeypatch.setattr(remote, "ROOT", str(tmp_path / "jobs") + "/")
    monkeypatch.setattr(remote, "RESULTS", str(tmp_path / "results") + "/")
    binding = {"runtime": "runtimes/test", "run_name": "run-1", "sha256": "a" * 64}
    root = tmp_path / "jobs/run-1"
    root.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(binding))
    resultdir = tmp_path / "results/run-1"
    resultdir.mkdir(parents=True)
    (resultdir / "metrics.jsonl").write_text(
        json.dumps(
            {"type": "step", "flops": 1234, "loss": 2.5, "url": "SECRET", "train_loss": "SECRET"}
        )
        + "\n"
    )
    proof = {
        "verified": True,
        "revision": "b" * 40,
        "step": 125,
        "identity": {"run": "run-1", "source_sha256": mod().SOURCE, "data_fingerprint": mod().DATA},
    }
    resultdir.with_name(resultdir.name + ".hf-checkpoint-receipt.json").write_text(
        json.dumps(proof)
    )
    result = subprocess.run(
        [sys.executable, "-c", remote.status_code(binding)],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout.removeprefix(remote.MARKER))
    assert value["metrics"] == {"flops": 1234, "loss": 2.5}
    assert value["hf_checkpoint"] == {"verified": True, "revision": "b" * 40, "step": 125}
    assert "SECRET" not in result.stdout
    assert mod().clean_status(value, binding) == value
    value["metrics"]["loss"] = "SECRET"
    with pytest.raises(ValueError, match="STATUS_INVALID"):
        mod().clean_status(value, binding)


def test_bootstrap_failure_does_not_run_child(tmp_path):
    remote = importlib.import_module("colab_api_run_remote")
    sentinel = tmp_path / "must-not-exist"
    command = [sys.executable, "-c", f'open({str(sentinel)!r}, "w").close()']
    code = remote.supervisor_code(
        command, str(tmp_path / "state.json"), str(tmp_path / "log"), "raise SystemExit(9)", {}
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert not sentinel.exists()
    assert json.loads((tmp_path / "state.json").read_text())["failure"] == "BOOTSTRAP_FAILED"


def test_state_publication_retries_only_atomic_replace_not_child(tmp_path):
    remote = importlib.import_module("colab_api_run_remote")
    state = tmp_path / "state.json"
    sentinel = tmp_path / "child"
    command = [sys.executable, "-c", f'open({str(sentinel)!r}, "w").write("once")']
    code = remote.supervisor_code(command, str(state), str(tmp_path / "log"), "", {})
    fault = """import os
original_replace = os.replace
replace_calls = 0
def transient_replace(source, target):
    global replace_calls
    replace_calls += 1
    if replace_calls == 1:
        raise PermissionError('synthetic file sharing lock')
    return original_replace(source, target)
os.replace = transient_replace
"""
    result = subprocess.run(
        [sys.executable, "-c", fault + code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert sentinel.read_text() == "once"
    assert json.loads(state.read_text())["state"] == "completed"


def test_invalid_phase_cannot_generate_arbitrary_remote_action():
    remote = importlib.import_module("colab_api_run_remote")
    with pytest.raises(ValueError, match="INVALID_PHASE"):
        remote.launch_code({"run_name": "x"}, "arbitrary")


def test_exclusive_journal_refuses_reexecution(tmp_path):
    path = tmp_path / "job.jsonl"
    with mod().Journal(path, {"runtime": "runtimes/test"}) as journal:
        journal.record({"phase": "pending"})
    with pytest.raises(ValueError, match="JOURNAL_EXISTS"):
        with mod().Journal(path, {}):
            pass
