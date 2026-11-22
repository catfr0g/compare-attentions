"""Real CLI phase-2 defaults and fake-auth scope boundaries."""

# Independent subprocess fixtures repeat the public CLI argument contract.
# pylint: disable=import-outside-toplevel,duplicate-code
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def test_real_create_plan_without_auth_or_writes(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/colab_api.py"),
            "--token-file",
            str(tmp_path / "absent.json"),
            "--journal",
            str(tmp_path / "j.json"),
            "create",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["execute"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments,error",
    [
        (["--management", "runtimes"], "INVALID_ARGUMENT"),
        (["--management", "auth-login"], "SEPARATE_MANAGEMENT_TOKEN_REQUIRED"),
        (["--execute", "runtimes"], "INVALID_ARGUMENT"),
        (["--journal", "j.json", "get", "runtimes/id"], "INVALID_ARGUMENT"),
        (["create"], "JOURNAL_REQUIRED"),
        (["status", "runtimes/foreign"], "INVALID_ARGUMENT"),
    ],
)
def test_invalid_management_flags_fail_before_credentials(monkeypatch, capsys, arguments, error):
    import colab_api as cli

    monkeypatch.setattr(
        cli.auth, "access_token", lambda *_a, **_kw: pytest.fail("credentials read")
    )
    monkeypatch.setattr(cli.auth, "login", lambda *_a, **_kw: pytest.fail("login called"))
    monkeypatch.setattr(sys, "argv", ["colab_api.py", *arguments])
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["error"] == error


@pytest.mark.parametrize("foreign", [False, True])
def test_operation_cli_exact_name_and_safe_done(tmp_path, monkeypatch, capsys, foreign):
    import colab_api as cli

    monkeypatch.setattr(cli.auth, "access_token", lambda *_a: "FAKE")
    monkeypatch.setattr(
        cli.client,
        "operation",
        lambda name, _t: {
            "name": "operations/foreign" if foreign else name,
            "done": True,
            "metadata": {"x": "SECRET"},
            "response": {"connectionInfo": {"token": "SECRET"}},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["colab_api.py", "--token-file", str(tmp_path / "fake"), "operation", "operations/one"],
    )
    assert cli.main() == (2 if foreign else 0)
    text = capsys.readouterr().out
    assert "SECRET" not in text
    if foreign:
        assert json.loads(text)["error"] == "INVALID_OPERATION"
    else:
        assert json.loads(text) == {"name": "operations/one", "done": True}


def test_status_wait_dispatch_and_poll_flags_scope(tmp_path, monkeypatch, capsys):
    import uuid

    import colab_api as cli
    import colab_api_journal as journal

    path = tmp_path / "j.json"
    journal.save(
        path,
        {
            "schema": 1,
            "runtimeId": "one-t4",
            "requestId": str(uuid.uuid4()),
            "name": "runtimes/one-t4",
            "runtimeSpec": cli.client.SPEC,
            "state": "ready",
        },
        initial=True,
    )
    monkeypatch.setattr(cli.auth, "access_token", lambda *_a: "FAKE")

    def reconcile(p, token, **kw):
        assert p == path and token == "FAKE"
        assert kw == {"wait_timeout": 4.0, "poll_interval": 1.0}
        return {"state": "ready"}

    monkeypatch.setattr(cli.runtime, "reconcile", reconcile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "colab_api.py",
            "--journal",
            str(path),
            "--wait-timeout",
            "4",
            "--poll-interval",
            "1",
            "status",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["state"] == "ready"
    for args, error in [
        (["--wait-timeout", "1", "runtimes"], "INVALID_ARGUMENT"),
        (["--journal", str(path), "--poll-interval", "0", "status"], "INVALID_POLL_BOUNDS"),
    ]:
        monkeypatch.setattr(sys, "argv", ["colab_api.py", *args])
        assert cli.main() == 2
        assert json.loads(capsys.readouterr().out)["error"] == error


def test_management_create_requires_actual_full_scope(tmp_path, monkeypatch, capsys):
    import colab_api as cli
    import colab_api_auth as auth

    token = tmp_path / "readonly.json"
    auth.save_token(token, {"type": "authorized_user", "scopes": [auth.READONLY]})
    monkeypatch.setattr(cli.client, "manage", lambda *_a, **_kw: pytest.fail("write called"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "colab_api.py",
            "--token-file",
            str(token),
            "--journal",
            str(tmp_path / "j.json"),
            "--execute",
            "create",
        ],
    )
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["error"] == "SCOPE_REQUIRED"
    assert not (tmp_path / "j.json").exists()
