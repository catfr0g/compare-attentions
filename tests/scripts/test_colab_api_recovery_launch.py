"""Execute real generated recovery launchers; replace only spawn/transport boundaries."""

# Repeat literal identity in isolated tests; synthetic peers expose only the API used.
# pylint: disable=duplicate-code

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
recovery = importlib.import_module("colab_api_recovery")
remote = importlib.import_module("colab_api_run_remote")
controller = importlib.import_module("colab_api_run")


@pytest.fixture(name="job")
def job_fixture(tmp_path, monkeypatch):
    for key in ("ROOT", "RESULTS"):
        monkeypatch.setattr(remote, key, str(tmp_path / key.lower()) + "/")
    for key in ("PACKAGE", "VENV", "PRIVATE", "ARCHIVE"):
        monkeypatch.setattr(remote, key, str(tmp_path / key.lower()))
    spec = recovery.RecoverySpec("runtimes/new-runtime", "new-recovery-run")
    binding = {
        "runtime": spec.runtime,
        "run_name": spec.run_name,
        "sha256": recovery.ARCHIVE_SHA,
        "recovery_sha256": spec.sha256(),
    }
    root = Path(remote.ROOT) / spec.run_name
    root.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(binding))
    (root / "inputs.json").write_text(json.dumps({"verified": True, "sha256": binding["sha256"]}))
    (root / "setup").mkdir()
    (root / "setup/state.json").write_text(
        json.dumps(dict(binding, state="completed", returncode=0))
    )
    return binding, root


def test_restore_stage_durable_before_spawn_and_never_repeated(job, monkeypatch):
    binding, root = job
    calls = []
    synced = []
    original = os.fsync

    def sync(fd):
        synced.append(fd)
        original(fd)

    def spawn(command, **_kwargs):
        state = json.loads((root / "restore/state.json").read_text())
        assert state == dict(binding, phase="restore", state="spawn_unknown")
        assert synced
        helper = root / "restore/recovery.py"
        assert helper.is_file()
        program = (root / "restore/supervisor.py").read_text()
        assert "restore/recovery.py" in program.replace("\\\\", "/")
        compile(helper.read_text(), str(helper), "exec")
        calls.append(command)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    code = remote.launch_code(binding, "restore")
    exec(code, {})  # pylint: disable=exec-used
    with pytest.raises(FileExistsError):
        exec(code, {})  # pylint: disable=exec-used
    assert len(calls) == 1
    assert not (root / "train").exists()


def test_resume_requires_explicit_reviewed_proof_before_auth(job, tmp_path, monkeypatch):
    binding, _root = job
    setup = tmp_path / "setup.jsonl"
    setup.write_text(json.dumps(binding) + "\n")

    def forbidden(*_args, **_kwargs):
        pytest.fail("missing proof reached auth")

    monkeypatch.setattr(controller.auth, "access_token", forbidden)
    assert (
        controller.main(
            [
                "resume",
                "--execute",
                "--runtime",
                binding["runtime"],
                "--run-name",
                binding["run_name"],
                "--setup-journal",
                str(setup),
                "--journal",
                str(tmp_path / "resume.jsonl"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "resume.jsonl").exists()


def test_resume_rechecks_guard_and_uses_shared_train_barrier(job, monkeypatch):
    binding, root = job
    (root / "restore").mkdir()
    (root / "restore/state.json").write_text(
        json.dumps(dict(binding, state="completed", returncode=0))
    )
    contexts = []

    def guard(context, helper_sha, proof_sha):
        contexts.append(context)
        assert helper_sha == "a" * 64 and proof_sha == "b" * 64
        return "guard_checked = True\n"

    monkeypatch.setattr(recovery, "resume_guard", guard)
    calls = []

    def spawn(command, **_kwargs):
        state = json.loads((root / "train/state.json").read_text())
        assert state == dict(binding, phase="resume", state="spawn_unknown")
        program = (root / "train/supervisor.py").read_text()
        assert "--resume-from" in program
        assert "step_00046500.pt" in program
        assert "--train-flops" not in program
        assert "guard_checked" in program  # Rechecked in supervisor at mutation boundary.
        calls.append(command)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    code = remote.launch_code(binding, "resume", helper_sha="a" * 64, proof_sha="b" * 64)
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used
    assert namespace["guard_checked"] is True
    assert (
        contexts[0]["spec"]
        == recovery.RecoverySpec(binding["runtime"], binding["run_name"]).value()
    )
    assert not Path(remote.RESULTS + binding["run_name"]).exists()
    with pytest.raises(FileExistsError):
        exec(code, {})  # pylint: disable=exec-used
    with pytest.raises(ValueError, match="EXPLICIT_RESUME"):
        remote.launch_code(binding, "train")
    assert len(calls) == 1


def test_remote_resume_requires_successful_restore_before_guard(job, monkeypatch):
    binding, root = job
    monkeypatch.setattr(
        recovery, "resume_guard", lambda *_a: "raise AssertionError('guard too early')\n"
    )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        exec(remote.launch_code(binding, "resume", helper_sha="a" * 64, proof_sha="b" * 64), {})  # pylint: disable=exec-used
    assert not (root / "train").exists()


@pytest.mark.parametrize("command", ["train", "restore", "resume"])
def test_recovery_flag_is_setup_only_before_any_io(command, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid recovery action reached I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(controller.auth, "access_token", forbidden)
    assert (
        controller.main(
            [
                command,
                "--recovery",
                "--runtime",
                "runtimes/new-runtime",
                "--run-name",
                "new-recovery-run",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--restore-proof-sha256", "BAD"],
        ["--recovery-helper-sha256", "a" * 63],
        ["--restore-proof-sha256", "b" * 64],
    ],
)
def test_resume_plan_rejects_partial_or_malformed_approval(flags, monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("plan did I/O"))
    assert (
        controller.main(
            [
                "resume",
                "--runtime",
                "runtimes/new-runtime",
                "--run-name",
                "new-recovery-run",
                *flags,
            ]
        )
        == 2
    )


def test_controller_threads_reviewed_hashes_to_remote_resume(job, tmp_path, monkeypatch):
    binding, _root = job
    setup = tmp_path / "setup.jsonl"
    session = binding["runtime"] + "/sessions/job-" + "a" * 32
    setup.write_text(
        json.dumps(binding)
        + "\n"
        + json.dumps({"phase": "create_session_pending", "session": session})
        + "\n"
    )
    monkeypatch.setattr(controller.auth, "access_token", lambda *_a, **_kw: "synthetic")
    monkeypatch.setattr(controller, "MCP", lambda _token: SimpleNamespace(initialize=lambda: None))
    calls = []

    class Peer:  # pylint: disable=too-few-public-methods
        def __init__(self, *_a, **_kw):
            pass

        def code(self, code):
            calls.append(code)
            if len(calls) == 1:
                return {"phase": "resume", "state": "spawned", "pid": 123}
            return {
                "sha256": binding["sha256"],
                "stages": {
                    "setup": {"state": "completed", "returncode": 0},
                    "restore": {"state": "completed", "returncode": 0},
                    "train": {"state": "spawn_unknown"},
                },
                "metrics": {},
                "hf_checkpoint": None,
                "recovery": {
                    "spec_sha256": binding["recovery_sha256"],
                    "inherited_updates": 46500,
                    "new_updates": 0,
                    "proof_sha256": "b" * 64,
                    "helper_sha256": "a" * 64,
                },
            }

    monkeypatch.setattr(controller, "Execution", Peer)
    # Generation is real. Execution alone is synthetic, never a cloud call.
    assert (
        controller.main(
            [
                "resume",
                "--execute",
                "--runtime",
                binding["runtime"],
                "--run-name",
                binding["run_name"],
                "--setup-journal",
                str(setup),
                "--journal",
                str(tmp_path / "resume.jsonl"),
                "--recovery-helper-sha256",
                "a" * 64,
                "--restore-proof-sha256",
                "b" * 64,
            ]
        )
        == 0
    )
    assert len(calls) == 2
    assert "RESTORE_PROOF_CHANGED" in calls[0]
    assert repr("b" * 64) in calls[0]


def test_recovery_status_separates_replayed_prefix_from_new_updates(job):
    binding, _root = job
    result = Path(remote.RESULTS + binding["run_name"])
    result.mkdir(parents=True)
    (result / "metrics.jsonl").write_text(
        json.dumps({"type": "step", "flops": 46502 * 211577470976, "loss": 1.0}) + "\n"
    )
    namespace = {}
    exec(remote.status_code(binding), namespace)  # pylint: disable=exec-used
    answer = namespace["answer"]
    assert set(answer["stages"]) == {"setup", "restore", "train"}
    assert answer["recovery"]["inherited_updates"] == 46500
    assert answer["recovery"]["new_updates"] == 2
    assert controller.clean_status(answer, binding) == answer
    answer["recovery"]["new_updates"] = -1
    with pytest.raises(ValueError, match="STATUS_INVALID"):
        controller.clean_status(answer, binding)
