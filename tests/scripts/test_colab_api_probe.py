"""Fixed probe tests use fixtures, not actual GPU claims."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module():
    assert importlib.util.find_spec("colab_api_probe") is not None, "probe not implemented"
    import colab_api_probe  # pylint: disable=import-outside-toplevel

    return colab_api_probe


def test_plan_requires_no_auth_or_cloud(monkeypatch, capsys):
    probe = module()
    monkeypatch.setattr(probe, "run_probe", lambda *_: pytest.fail("remote call"))
    assert probe.main(["--runtime", "runtimes/demo"]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "plan"


def test_fixed_code_begins_with_original_guard():
    probe = module()
    from colab_runner import PREFLIGHT  # pylint: disable=import-outside-toplevel

    assert probe.CODE.startswith(PREFLIGHT)
    assert "kaggle-env.env" in probe.CODE
    assert "colab-package.zip" in probe.CODE
    assert ".is_file()" in probe.CODE
    for forbidden in (
        "pip",
        "torch",
        "load_dotenv",
        "mount(",
        "write_text",
        "extractall",
        "train(",
    ):
        assert forbidden not in probe.CODE.replace("ismount(", "")


def payload(probe, **changes):
    return {
        "python": "3.13.9",
        "gpu": "Tesla T4",
        "guard_passed": True,
        "drive_mounted": True,
        "mydrive_present": True,
        "archive_exists": True,
        "archive_sha256": probe.ARCHIVE_SHA,
        "archive_matches": True,
        "env_file_exists": True,
        **changes,
    }


@pytest.mark.parametrize("mode", ["success", "ambiguous", "foreign", "cleanup-failure"])
def test_execution_journal_order_cleanup_and_no_repeat(tmp_path, monkeypatch, mode):
    probe = module()
    import argparse  # pylint: disable=import-outside-toplevel

    args = argparse.Namespace(
        runtime="runtimes/demo", journal=tmp_path / "probe.jsonl", client_file=None, token_file=None
    )
    seen = []

    class Client:
        def __init__(self, token):
            assert token == "SECRET"

        def initialize(self):
            assert args.journal.exists()

        def call(self, name, arguments):
            entries = [json.loads(line) for line in args.journal.read_text().splitlines()]
            assert entries[-1]["phase"] == name + "_pending"
            assert "SECRET" not in args.journal.read_text()
            seen.append(name)
            if name == "create_session":
                return {"name": "foreign" if mode == "foreign" else entries[0]["session"]}
            if name == "execute_code":
                if mode == "ambiguous":
                    raise probe.ProbeError("MCP_CALL_AMBIGUOUS")
                assert arguments["code"] == probe.CODE
                return {
                    "session": entries[0]["session"],
                    "executionId": entries[0]["execution_id"],
                    "stdout": probe.MARKER + json.dumps(payload(probe)),
                    "outputTruncated": False,
                }
            if name == "list_sessions":
                return {
                    "sessions": [{"name": entries[0]["session"]}]
                    if mode == "cleanup-failure"
                    else []
                }
            return {}

    monkeypatch.setattr(probe, "MCP", Client)
    monkeypatch.setattr(
        probe.auth,
        "access_token",
        lambda *_, required_scope: (
            "SECRET" if required_scope == probe.auth.MANAGE else pytest.fail("scope")
        ),
    )
    if mode == "success":
        assert probe.run_probe(args)["phase"] == "complete"
    else:
        code = {
            "ambiguous": "MCP_CALL_AMBIGUOUS",
            "foreign": "REMOTE_SESSION_MISMATCH",
            "cleanup-failure": "SESSION_CLEANUP_REQUIRED",
        }[mode]
        with pytest.raises(probe.ProbeError, match="^" + code + "$"):
            probe.run_probe(args)
    expected = (
        ["create_session"]
        if mode == "foreign"
        else ["create_session", "execute_code", "delete_session", "list_sessions"]
    )
    assert seen == expected
    with pytest.raises(probe.ProbeError, match="JOURNAL_EXISTS"):
        probe.run_probe(args)
    assert seen == expected


@pytest.mark.parametrize(
    "changes",
    [
        {
            "drive_mounted": False,
            "mydrive_present": False,
            "archive_exists": False,
            "archive_sha256": None,
            "archive_matches": False,
            "env_file_exists": False,
        },
        {"env_file_exists": False},
        {"archive_matches": False, "archive_sha256": "0" * 64},
    ],
)
def test_missing_persistence_is_blocked_not_complete(changes):
    probe = module()
    data = payload(probe, **changes)
    result = {
        "session": "runtimes/demo/sessions/test",
        "executionId": "uuid",
        "stdout": probe.MARKER + json.dumps(data),
    }
    assert probe.parse_result(result, result["session"], "uuid")["phase"] == "blocked"


@pytest.mark.parametrize(
    "change",
    [
        {"outputTruncated": True},
        {"executionError": {"value": "SECRET"}},
        {"session": "foreign"},
        {"executionId": "foreign"},
        {"stdout": "SECRET"},
    ],
)
def test_invalid_remote_result_never_leaks_raw_strings(change):
    probe = module()
    result = {
        "session": "runtimes/demo/sessions/test",
        "executionId": "uuid",
        "stdout": probe.MARKER + json.dumps(payload(probe)),
        **change,
    }
    with pytest.raises(probe.ProbeError) as error:
        probe.parse_result(result, "runtimes/demo/sessions/test", "uuid")
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "version,gpu",
    [((3, 12), "Tesla T4, 7.5"), ((3, 13), "A100, 8.0"), ((3, 13), "Tesla T4, 7.5\nTesla T4, 7.5")],
)
def test_actual_generated_guard_stops_before_storage(version, gpu):
    import builtins  # pylint: disable=import-outside-toplevel
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    probe = module()
    imported = []

    def fake_import(name, *_args, **_kwargs):
        imported.append(name)
        if name == "sys":
            return SimpleNamespace(version_info=version, version="3.13.9 fixture")
        if name == "subprocess":
            return SimpleNamespace(
                check_output=lambda *a, **k: gpu, CalledProcessError=RuntimeError
            )
        raise AssertionError("storage import reached before guard passed: " + name)

    with pytest.raises(RuntimeError):
        exec(  # pylint: disable=exec-used
            compile(probe.CODE, "<fixture-only>", "exec"),
            {"__builtins__": {**vars(builtins), "__import__": fake_import}},
        )  # pylint: disable=exec-used
    assert imported == ["sys", "subprocess"]


@pytest.mark.parametrize(
    "change",
    [
        {"guard_passed": 1},
        {"python": "SECRET"},
        {"gpu": "SECRET"},
        {"extra": "SECRET"},
        {"archive_sha256": "SECRET"},
        {"guard_passed": False},
    ],
)
def test_marker_schema_is_strict(change):
    probe = module()
    with pytest.raises(probe.ProbeError, match="^REMOTE_INVALID_RESULT$"):
        probe.parse_result(
            {
                "session": "s",
                "executionId": "e",
                "stdout": probe.MARKER + json.dumps(payload(probe, **change)),
            },
            "s",
            "e",
        )


def test_invalid_runtime_precedes_auth(monkeypatch, capsys):
    probe = module()
    monkeypatch.setattr(probe.auth, "access_token", lambda *a, **k: pytest.fail("auth reached"))
    assert probe.main(["--runtime", "runtimes/a/../b", "--execute"]) == 2
    assert "INVALID_RUNTIME" in capsys.readouterr().out


@pytest.mark.parametrize("mounted", [False, True])
def test_generated_storage_code_reads_only_archive_not_env(mounted, capsys):
    import builtins  # pylint: disable=import-outside-toplevel
    import hashlib  # pylint: disable=import-outside-toplevel
    import io  # pylint: disable=import-outside-toplevel
    from pathlib import PurePosixPath  # pylint: disable=import-outside-toplevel
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    probe = module()
    opened = []

    class FixturePath(PurePosixPath):
        def is_dir(self):
            assert mounted
            return True

        def is_file(self):
            assert mounted
            return True

        def open(self, mode):
            assert self.name == "colab-package.zip" and mode == "rb"
            opened.append(self.name)
            return io.BytesIO(b"fixture-archive-only")

    real_import = builtins.__import__

    def fixture_import(name, *args, **kwargs):
        if name == "sys":
            return SimpleNamespace(version_info=(3, 13), version="3.13.9 fixture")
        if name == "subprocess":
            return SimpleNamespace(
                check_output=lambda *a, **k: "Tesla T4, 7.5", CalledProcessError=RuntimeError
            )
        if name == "os":
            return SimpleNamespace(path=SimpleNamespace(ismount=lambda _: mounted))
        if name == "pathlib":
            return SimpleNamespace(Path=FixturePath)
        return real_import(name, *args, **kwargs)

    exec(  # pylint: disable=exec-used
        compile(probe.CODE, "<fixture-only>", "exec"),
        {"__builtins__": {**vars(builtins), "__import__": fixture_import}},
    )  # pylint: disable=exec-used
    result = probe.parse_result(
        {"session": "s", "executionId": "e", "stdout": capsys.readouterr().out}, "s", "e"
    )
    assert result["phase"] == "blocked"
    assert opened == (["colab-package.zip"] if mounted else [])
    assert result["readiness"]["archive_sha256"] == (
        hashlib.sha256(b"fixture-archive-only").hexdigest() if mounted else None
    )


def test_main_sanitizes_unexpected_local_exception(monkeypatch, capsys):
    probe = module()

    def fail(_args):
        raise OSError("SECRET /private/path")

    monkeypatch.setattr(probe, "run_probe", fail)
    assert probe.main(["--runtime", "runtimes/demo", "--execute"]) == 2
    assert "SECRET" not in capsys.readouterr().out
