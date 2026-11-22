"""Detached controller tests run only harmless real CPU subprocesses."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def module():
    assert importlib.util.find_spec("colab_job"), "job helper missing"
    import colab_job  # pylint: disable=import-outside-toplevel

    return colab_job


def test_atomic_state_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    job = module()
    target = tmp_path / "state.json"
    target.write_text('{"state": "running"}')
    original = Path.replace
    attempts = []

    def replace(path, destination):
        attempts.append(path.read_text())
        if len(attempts) == 1:
            assert json.loads(target.read_text())["state"] == "running"
            raise PermissionError("synthetic Windows reader lock")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    job.atomic(target, {"state": "completed"})
    assert len(attempts) == 2
    assert json.loads(target.read_text()) == {"state": "completed"}


def test_real_detached_cpu_bootstrap_and_duplicate_barrier(tmp_path):
    job = module()
    control = tmp_path / "control"
    control.mkdir()
    (control / "bootstrap.py").write_text("print('harmless CPU fixture')\n")
    result = job.start("bootstrap", control)
    assert result["state"] == "requested"
    with pytest.raises(FileExistsError):
        job.start("bootstrap", control)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = job.status(control)["bootstrap"]
        if status["state"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["state"] == "completed" and status["exit_code"] == 0
    assert "harmless CPU fixture" in (control / "bootstrap.log").read_text()
    assert not (control / "run.requested.json").exists()
    assert set(status) == {"state", "exit_code", "pid", "phase"}


def test_run_requires_successful_setup_and_fixed_delta_command(tmp_path):
    job = module()
    with pytest.raises(ValueError, match="PREREQUISITE"):
        job.start("run", tmp_path)
    command = job.command("run", tmp_path)
    assert "--family" in command and command[command.index("--family") + 1] == "deltanet"
    assert "--resume-from" not in command and "--run" in command
    assert "--env-file" in command and "HF_TOKEN" not in " ".join(command)


def test_status_never_returns_raw_log_or_unknown_fields(tmp_path):
    job = module()
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "state": "running",
                "pid": 22,
                "exit_code": None,
                "phase": "run",
                "stdout": "SECRET",
                "url": "signed-url",
            }
        )
    )
    result = job.status(tmp_path)
    assert "SECRET" not in json.dumps(result) and "signed-url" not in json.dumps(result)
    assert result["run"]["pid"] == 22
