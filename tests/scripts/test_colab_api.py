"""Read-only host CLI protocol checks; no user credentials."""
# Test-first imports defer loading production modules until each behavior runs.
# pylint: disable=import-outside-toplevel

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_missing_auth_is_actionable_json(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/colab_api.py"),
            "--token-file",
            str(tmp_path / "absent.json"),
            "auth-status",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["error"] == "AUTH_REQUIRED"
    assert "auth-login" in payload["action"]
    assert "Traceback" not in result.stderr


def test_help_is_real_subprocess():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/colab_api.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "t4-eligible" in result.stdout


def test_cli_dispatch_safe_readonly_paths(tmp_path, monkeypatch, capsys):
    import importlib

    sys.path.insert(0, str(ROOT / "scripts"))
    cli = importlib.import_module("colab_api")
    monkeypatch.setattr(cli.auth, "access_token", lambda *_a: "PRIVATE_ACCESS")
    urls = []

    def get(url, token):
        assert token == "PRIVATE_ACCESS"
        urls.append(url)
        if url.endswith("runtimespecs"):
            return {
                "runtimeSpecs": [
                    {
                        "key": {
                            "variant": "VARIANT_GPU",
                            "accelerator": "T4",
                            "shape": "SHAPE_STANDARD",
                        },
                        "eligible": True,
                    }
                ]
            }
        raw = {"name": "runtimes/id", "runtimeSpec": {}, "connectionInfo": {"url": "PRIVATE_URL"}}
        return {"runtimes": [raw]} if url.endswith("/runtimes") else raw

    monkeypatch.setattr(cli.client, "get", get)
    for command in (["specs"], ["t4-eligible"], ["runtimes"], ["get", "runtimes/id"]):
        monkeypatch.setattr(
            sys, "argv", ["colab_api.py", "--token-file", str(tmp_path / "t.json"), *command]
        )
        assert cli.main() == 0
        text = capsys.readouterr().out
        assert "PRIVATE" not in text
        json.loads(text)
    assert urls == [
        cli.client.BASE + "runtimespecs",
        cli.client.BASE + "runtimespecs",
        cli.client.BASE + "runtimes",
        cli.client.BASE + "runtimes/id",
    ]
