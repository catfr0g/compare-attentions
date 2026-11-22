"""Execute real generated launch programs with local-only durability probes."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
remote = importlib.import_module("colab_api_run_remote")


@pytest.mark.parametrize("phase", ["setup", "train"])
def test_pending_state_is_flushed_and_fsynced_before_supervisor_spawn(
    tmp_path, monkeypatch, capsys, phase
):
    monkeypatch.setattr(remote, "ROOT", str(tmp_path / "jobs") + "/")
    monkeypatch.setattr(remote, "RESULTS", str(tmp_path / "results") + "/")
    binding = {"runtime": "runtimes/test", "run_name": "durability", "sha256": "a" * 64}
    root = tmp_path / "jobs/durability"
    root.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(binding))
    (root / "inputs.json").write_text(json.dumps({"verified": True, "sha256": "a" * 64}))
    if phase == "train":
        (root / "setup").mkdir()
        (root / "setup/state.json").write_text(
            json.dumps(dict(binding, phase="setup", state="completed", returncode=0))
        )
    state_path = root / phase / "state.json"
    expected = dict(binding, phase=phase, state="spawn_unknown")
    events = []
    real_fsync = os.fsync

    def record_fsync(descriptor):
        assert os.path.samestat(os.fstat(descriptor), state_path.stat())
        # A separate reader must see the complete JSON while the writer is open:
        # fsync alone cannot flush Python's userspace text buffer.
        assert json.loads(state_path.read_text()) == expected
        real_fsync(descriptor)
        events.append(("fsync", json.loads(state_path.read_text())))

    def record_spawn(command, **_kwargs):
        events.append(("Popen", json.loads(state_path.read_text())))
        assert events == [("fsync", expected), ("Popen", expected)]
        assert command == [sys.executable, "-B", str(root / phase / "supervisor.py")]
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    exec(remote.launch_code(binding, phase), {})  # pylint: disable=exec-used
    assert events == [("fsync", expected), ("Popen", expected)]
    assert json.loads(capsys.readouterr().out.removeprefix(remote.MARKER)) == {
        "phase": phase,
        "state": "spawned",
        "pid": 12345,
    }
