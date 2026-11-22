"""Five launch acceptance contracts migrated to the implemented no-Drive APIs.

Only transport peers/private inputs are synthetic; controller and generated guards
are real. No missing-module facade, cloud calls, dependencies or training.
"""

import argparse
import ast
import importlib
import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
controller = importlib.import_module("colab_api_run")
remote = importlib.import_module("colab_api_run_remote")


def test_plan_never_reads_credentials_or_calls_remote(monkeypatch, capsys, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("plan attempted credentials, local inputs or account access")

    for owner, name in (
        (controller.auth, "access_token"),
        (controller, "private_bytes"),
        (controller, "export_identity"),
        (controller, "load_binding"),
        (controller, "MCP"),
        (controller.google, "get"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for command in ("setup", "train", "status", "logsummary"):
        assert (
            controller.main(
                [
                    command,
                    "--runtime",
                    "runtimes/demo",
                    "--run-name",
                    "launch-fixture",
                    "--journal",
                    str(tmp_path / "journal"),
                    "--env-file",
                    str(tmp_path / "absent.env"),
                    "--token-file",
                    str(tmp_path / "absent-token.json"),
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out) == {
            "phase": "plan",
            "command": command,
            "runtime": "runtimes/demo",
            "run_name": "launch-fixture",
            "storage": "hf",
            "training_started": False,
        }
    assert not list(tmp_path.iterdir())


def test_hardware_guard_precedes_marker_and_private_file_read(monkeypatch, tmp_path, capsys):
    preflight = importlib.import_module("colab_runner").PREFLIGHT

    binding = {"runtime": "runtimes/demo", "run_name": "launch-fixture", "sha256": "a" * 64}
    monkeypatch.setattr(remote, "ROOT", str(tmp_path / "jobs") + "/")
    code = remote.marker_code(binding, "b" * 32)
    assert code.startswith(preflight)
    assert "kaggle-env.env" not in code and "drive.mount" not in code
    queries = []

    def wrong_gpu(arguments, **_kwargs):
        queries.append(arguments)
        return "NVIDIA A100, 8.0"

    def forbidden(*_args, **_kwargs):
        pytest.fail("hardware rejection must precede private reads and transport")

    class LocalExecution:  # pylint: disable=too-few-public-methods
        def code(self, program):
            exec(compile(program, "<actual-marker>", "exec"), {})  # pylint: disable=exec-used

    monkeypatch.setattr(controller, "private_bytes", forbidden)
    monkeypatch.setattr(controller.google, "get", forbidden)
    args = argparse.Namespace(runtime=binding["runtime"], env_file=tmp_path / "unread.env")
    path = tmp_path / "journal"
    with controller.Journal(path, binding) as journal, monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 13, 0))
        patch.setattr("subprocess.check_output", wrong_gpu)
        with pytest.raises(RuntimeError, match="Requires exactly one T4"):
            controller.transfer(
                args, binding, tmp_path / "unread.zip", LocalExecution(), "synthetic-token", journal
            )
    assert queries == [["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]]
    assert not (tmp_path / "jobs").exists()
    assert remote.MARKER not in capsys.readouterr().out
    assert [json.loads(line)["phase"] for line in path.read_text().splitlines()] == [
        "prepared",
        "transfer_prepared",
    ]


def test_remote_reply_rejects_foreign_id_error_and_uncontrolled_fields(tmp_path):
    changes = [
        {},
        {"executionError": "SECRET"},
        {"session": "foreign"},
        {"executionId": "foreign"},
        {"stdout": "SECRET"},
        {"outputTruncated": True},
        {"stdout": remote.MARKER + "{}\n" + remote.MARKER + "{}"},
        {"stdout": remote.MARKER + "[]"},
        {"stdout": remote.MARKER + "invalid"},
        {"stdout": None},
        {"stdout": "x" * 32769},
    ]
    for index, change in enumerate(changes):

        class Peer:  # pylint: disable=too-few-public-methods
            def call(self, name, arguments, reply_change=change):
                if name == "create_session":
                    return {"name": arguments["parent"] + "/sessions/" + arguments["sessionId"]}
                assert name == "execute_code"
                return {
                    "session": arguments["session"],
                    "executionId": arguments["executionId"],
                    "stdout": remote.MARKER + '{"ok": true}',
                    **reply_change,
                }

        path = tmp_path / str(index)
        with controller.Journal(path, {}) as journal:
            execution = controller.Execution(Peer(), "runtimes/demo", journal)
            if not change:
                assert execution.code("fixed harmless program") == {"ok": True}
            else:
                with pytest.raises(
                    ValueError, match="EXECUTION_(UNKNOWN_OR_FAILED|INVALID_RESULT)"
                ):
                    execution.code("fixed harmless program")
        assert "SECRET" not in path.read_text()


def test_ambiguous_start_journal_blocks_replay(tmp_path, monkeypatch):
    binding = {"runtime": "runtimes/demo", "run_name": "launch-fixture", "sha256": "a" * 64}
    setup = tmp_path / "setup.jsonl"
    session = binding["runtime"] + "/sessions/job-" + "a" * 32
    setup.write_text(
        json.dumps(binding)
        + "\n"
        + json.dumps({"phase": "create_session_pending", "session": session})
        + "\n"
    )
    path = tmp_path / "launch.jsonl"
    calls = []

    class Peer:
        def __init__(self, _token):
            pass

        def initialize(self):
            assert json.loads(path.read_text().splitlines()[-1])["phase"] == "initialize_pending"

        def call(self, name, arguments):
            calls.append(name)
            if name == "list_sessions":
                return {"sessions": [{"name": session}]}
            entry = json.loads(path.read_text().splitlines()[-1])
            assert entry == {"phase": "execute_pending", "execution_id": arguments["executionId"]}
            assert str(uuid.UUID(entry["execution_id"])) == entry["execution_id"]
            raise OSError("SECRET signed-url")

    monkeypatch.setattr(controller.auth, "access_token", lambda *_a, **_k: "synthetic-token")
    monkeypatch.setattr(controller, "MCP", Peer)
    args = argparse.Namespace(
        command="train",
        runtime=binding["runtime"],
        run_name=binding["run_name"],
        execute=True,
        journal=path,
        setup_journal=setup,
        token_file=None,
        client_file=None,
    )
    with pytest.raises(OSError, match="SECRET"):
        controller.run(args)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="JOURNAL_EXISTS"):
        controller.run(args)
    assert calls == ["list_sessions", "execute_code"]
    assert path.read_bytes() == original and b"SECRET" not in original

    # Another host journal cannot bypass the actual remote exclusive spawn barrier.
    monkeypatch.setattr(remote, "ROOT", str(tmp_path / "jobs") + "/")
    root = tmp_path / "jobs" / binding["run_name"]
    root.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(binding))
    (root / "inputs.json").write_text(json.dumps({"verified": True, "sha256": binding["sha256"]}))
    spawns = []

    def ambiguous_spawn(*_args, **_kwargs):
        assert json.loads((root / "setup/state.json").read_text())["state"] == "spawn_unknown"
        spawns.append("spawn")
        raise OSError("synthetic ambiguous spawn")

    monkeypatch.setattr("subprocess.Popen", ambiguous_spawn)
    program = compile(remote.launch_code(binding, "setup"), "<actual-launch>", "exec")
    with pytest.raises(OSError, match="ambiguous spawn"):
        exec(program, {})  # pylint: disable=exec-used
    with pytest.raises(FileExistsError):
        exec(program, {})  # pylint: disable=exec-used
    assert spawns == ["spawn"]


def test_bootstrap_uses_verified_notebook_not_a_new_extractor(tmp_path, monkeypatch):
    exporter = importlib.import_module("export_colab")
    run_name = "deltanet-full-seed42-run01"
    sha = "a" * 64
    programs = []

    class Peer:
        def __init__(self, _token):
            pass

        def initialize(self):
            pass

        def call(self, name, arguments):
            if name == "create_session":
                return {"name": arguments["parent"] + "/sessions/" + arguments["sessionId"]}
            programs.append(arguments["code"])
            value = (
                {"phase": "setup", "state": "spawned", "pid": 123}
                if len(programs) == 1
                else {
                    "sha256": sha,
                    "stages": {"setup": {"state": "running"}, "train": {"state": "not_started"}},
                    "metrics": {},
                    "hf" + "_checkpoint": None,
                }
            )
            return {
                "session": arguments["session"],
                "executionId": arguments["executionId"],
                "stdout": remote.MARKER + json.dumps(value),
            }

    monkeypatch.setattr(controller.auth, "access_token", lambda *_a, **_k: "synthetic-token")
    monkeypatch.setattr(controller, "MCP", Peer)
    # Only external input/transport boundaries are replaced, not code generation/parsing.
    monkeypatch.setattr(controller, "export_identity", lambda _p: (tmp_path / "archive.zip", sha))
    monkeypatch.setattr(controller, "transfer", lambda *_a: None)
    args = argparse.Namespace(
        command="setup",
        runtime="runtimes/demo",
        run_name=run_name,
        execute=True,
        journal=tmp_path / "journal",
        export_dir=tmp_path,
        env_file=tmp_path / "unread.env",
        token_file=None,
        client_file=None,
    )
    assert controller.run(args)["stages"]["setup"]["state"] == "running"
    outer = ast.parse(programs[0])
    supervisor = next(
        ast.literal_eval(node.args[0])
        for node in ast.walk(outer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    )
    bootstrap = next(
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(supervisor))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "bootstrap" for target in node.targets
        )
    )
    assert bootstrap == exporter.bootstrap_code(sha)
    assert any(
        "".join(cell.get("source", [])) == bootstrap for cell in exporter.notebook(sha)["cells"]
    )
    assert "drive.mount" not in bootstrap and "HF_TOKEN=" not in bootstrap
    compile(bootstrap, "<bootstrap>", "exec")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"corrupted archive")
    work = tmp_path / "must-not-extract"
    code = exporter.bootstrap_code(sha, archive_path=str(archive), work_dir=str(work))
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 13, 0))
        patch.setattr("subprocess.check_output", lambda *_a, **_k: "Tesla T4, 7.5")
        with pytest.raises(ValueError, match="Archive SHA256 mismatch"):
            exec(compile(code, "<bootstrap>", "exec"), {})  # pylint: disable=exec-used
    assert not work.exists()
