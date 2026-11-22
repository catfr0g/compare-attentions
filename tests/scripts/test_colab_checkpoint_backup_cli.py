"""Synthetic service child and CLI policy; never open real .env files."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_child_reads_credentials_only_itself_and_online_before_sdk(tmp_path, monkeypatch, capsys):
    module = importlib.import_module("colab_checkpoint_backup")
    service = importlib.import_module("colab_services")
    credentials = {key: "synthetic-private-value" for key in service.KEYS}
    credentials["HF_REPO"] = "synthetic/repo"
    monkeypatch.setattr(service, "load_credentials", lambda path: credentials)
    monkeypatch.setattr(module.os, "environ", dict(os.environ))
    env_file = tmp_path / "private.env"
    env_file.write_text("not read by test")
    package, result = tmp_path / "package", tmp_path / "result"
    events = []

    def factory(*, token):
        assert token == credentials["HF_TOKEN"]
        assert "HF_HUB_OFFLINE" not in os.environ
        assert not any(key.startswith("CLEARML_") for key in os.environ)
        assert os.environ["HF_TOKEN"] == token
        assert Path(os.environ["HF_HOME"]).is_dir()
        events.append("sdk")
        return SimpleNamespace()

    def probe(_package, _result, **options):
        assert events == ["sdk"]
        assert options["repo"] == credentials["HF_REPO"]
        assert _package == package and _result == result
        events.append("probe")

    monkeypatch.setattr(module, "probe", probe)
    module.service_phase(
        env_file, package, result, "probe", api_factory=factory, download=lambda **_: None
    )
    assert events == ["sdk", "probe"]
    assert "synthetic" not in str(capsys.readouterr())


def test_cli_paths_and_masked_exception(tmp_path, monkeypatch, capsys):
    module = importlib.import_module("colab_checkpoint_backup")
    calls = []
    monkeypatch.setattr(
        module, "service_phase", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    for phase in ("probe", "publish"):
        assert (
            module.main(
                [
                    phase,
                    "--env-file",
                    str(tmp_path / "p.env"),
                    "--package-dir",
                    str(tmp_path / "package"),
                    "--result-dir",
                    str(tmp_path / "result"),
                ]
            )
            == 0
        )
    assert [call[0][3] for call in calls] == ["probe", "publish"]

    def fail(*_args, **_kwargs):
        logging.error("synthetic-secret")
        print("synthetic-secret", file=sys.stderr)
        raise RuntimeError("synthetic-secret")

    monkeypatch.setattr(module, "service_phase", fail)
    assert (
        module.main(
            [
                "probe",
                "--env-file",
                str(tmp_path / "p.env"),
                "--package-dir",
                str(tmp_path / "package"),
                "--result-dir",
                str(tmp_path / "result"),
            ]
        )
        == 1
    )
    assert "synthetic-secret" not in str(capsys.readouterr())


def test_sdk_signatures_are_pinned_and_export_import_is_stdlib(tmp_path):

    assert importlib.metadata.version("huggingface-hub") == "2.3.0"
    assert {"parent_commit", "revision", "operations"} <= inspect.signature(
        HfApi.create_commit
    ).parameters.keys()
    assert "local_dir" in inspect.signature(hf_hub_download).parameters
    script = Path(__file__).resolve().parents[2] / "scripts"
    command = [
        sys.executable,
        "-S",
        "-c",
        "import sys; sys.path.insert(0, " + repr(str(script)) + "); "
        "import colab_checkpoint_backup; "
        "assert not any(n in sys.modules for n in "
        "('torch','huggingface_hub','dotenv','clearml')); print('stdlib-only')",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=tmp_path)
    assert result.stdout.strip() == "stdlib-only"
