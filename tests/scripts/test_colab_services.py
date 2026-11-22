"""Private synthetic credentials only; service boundaries never contact a network."""

# pylint: disable=import-outside-toplevel
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def services():
    """Import only after asserting the feature exists, for a meaningful RED."""
    assert importlib.util.find_spec("colab_services"), "private credential helper is missing"
    return importlib.import_module("colab_services")


def test_missing_credentials_fail_without_echoing_private_values(tmp_path):
    core = services()
    private = tmp_path / "private.env"
    with pytest.raises(ValueError, match="private env file"):
        core.load_credentials(private)
    private.write_text("HF_TOKEN=DO_NOT_ECHO\n")
    with pytest.raises(ValueError, match="CLEARML_API_ACCESS_KEY") as error:
        core.load_credentials(private)
    assert "DO_NOT_ECHO" not in str(error.value)


def private_file(tmp_path):
    core = services()
    path = tmp_path / "private.env"
    values = {key: "synthetic-private-value" for key in core.KEYS}
    values.update({key: "https://example.invalid" for key in core.KEYS if key.endswith("HOST")})
    values["HF_REPO"] = "owner/private-model"
    path.write_text("\n".join(f'{key}="{value}"' for key, value in values.items()))
    return path, values


@pytest.mark.parametrize(
    "bad", ["owner/repo/extra", " owner/repo", "owner/repo.git", "owner/re..po"]
)
def test_bad_repo_refused_literally_without_echo(tmp_path, bad):
    core = services()
    path, _ = private_file(tmp_path)
    with path.open("a") as stream:
        stream.write(f'\nHF_REPO="{bad}"\n')
    with pytest.raises(ValueError, match="HF_REPO") as error:
        core.load_credentials(path)
    assert bad not in str(error.value)


@pytest.mark.parametrize(
    "bad",
    [
        "https://@example.invalid",
        "https://example.invalid:bad",
        "file:///private",
        "https://user:secret@example.invalid",
    ],
)
def test_invalid_service_url_refused_without_echo(tmp_path, bad):
    core = services()
    path, _ = private_file(tmp_path)
    with path.open("a") as stream:
        stream.write(f'\nCLEARML_API_HOST="{bad}"\n')
    with pytest.raises(ValueError, match="CLEARML_API_HOST") as error:
        core.load_credentials(path)
    assert bad not in str(error.value)


def test_phase_environment_and_train_import_happen_in_that_order(monkeypatch, tmp_path):
    core = services()
    path, values = private_file(tmp_path)
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    monkeypatch.setenv("HF_TOKEN", "inherited-secret")
    env = core.service_environment(core.load_credentials(path), "train")
    assert env["CLEARML_API_SECRET_KEY"] == values["CLEARML_API_SECRET_KEY"]
    assert "HF_TOKEN" not in env and "HF_REPO" not in env
    assert "CLEARML_OFFLINE_MODE" not in env
    uploaded = core.service_environment(values, "upload")
    assert uploaded["HF_TOKEN"] == values["HF_TOKEN"]
    assert "HF_HUB_OFFLINE" not in uploaded
    assert not any(key.startswith("CLEARML_API_") for key in uploaded)
    calls = []

    def run_module(name, run_name, *, alter_sys):
        import os

        assert alter_sys is True

        assert os.environ["CLEARML_API_SECRET_KEY"] == values["CLEARML_API_SECRET_KEY"]
        assert "HF_TOKEN" not in os.environ and "HF_REPO" not in os.environ
        calls.append((name, run_name, sys.argv[:]))

    monkeypatch.setattr(core.runpy, "run_module", run_module)
    monkeypatch.setattr(core, "clearml_stream_redaction", lambda _: core.contextlib.nullcontext())
    monkeypatch.setattr(core, "preflight", lambda: None, raising=False)
    monkeypatch.setattr(core.os, "environ", dict(core.os.environ))
    core.train(path, ["--config", "frozen.yaml", "--tracker", "clearml"])
    assert calls[0][:2] == ("compare_attention.cli.train", "__main__")
    assert calls[0][2][-2:] == ["--tracker", "clearml"]


@pytest.mark.parametrize("failure", [None, "hash", "listing", "extra", "download", "public"])
def test_upload_readback_pinned_head_and_failure_no_receipt(monkeypatch, tmp_path, failure):  # pylint: disable=too-many-locals
    import hashlib
    import json
    from types import SimpleNamespace

    core = services()
    path, _ = private_file(tmp_path)
    result = tmp_path / "completed-run"
    result.mkdir()
    for name in ("model.pt", "test-evaluation.json", "colab-native-gate.json"):
        (result / name).write_bytes(b"synthetic result")
    inventory = {p.name: core.checksum(p) for p in result.iterdir()}
    (result / "colab-provenance.json").write_text(
        json.dumps(
            {
                "package_manifest_sha256": "package-hash",
                "result_sha256": inventory,
                "tracking": "clearml",
                "hf_upload_requested": True,
            }
        )
    )
    monkeypatch.setattr(
        core,
        "checksum",
        lambda p: (
            "package-hash"
            if p.name == "manifest.json"
            else hashlib.sha256(p.read_bytes()).hexdigest()
        ),
    )
    monkeypatch.setattr(core.os, "environ", dict(core.os.environ))
    events = []
    prefix = "runs/completed-run/"

    class Api:
        def __init__(self, *, token):
            assert token == "synthetic-private-value"

        def model_info(self, repo_id, **kwargs):
            assert repo_id == "owner/private-model"
            events.append(("head", kwargs))
            return SimpleNamespace(sha="a" * 40, private=failure != "public")

        def list_repo_files(self, repo_id, **kwargs):
            assert repo_id == "owner/private-model" and kwargs["revision"] == "a" * 40
            events.append(("list", kwargs))
            names = [prefix + p.name for p in result.iterdir()]
            if failure == "extra":
                names.append(prefix + "unexpected-file")
            return [] if failure == "listing" else names

    def upload_fn(directory, repo_id):
        assert directory == result and repo_id == "owner/private-model"
        events.append(("upload", {}))

    def download(**kwargs):
        assert kwargs["revision"] == "a" * 40 and kwargs["force_download"]
        events.append(("download", kwargs))
        if failure == "download":
            raise RuntimeError("synthetic-private-value")
        if failure == "hash":
            bad = tmp_path / "bad.bin"
            bad.write_bytes(b"mismatch")
            return bad
        return result / Path(kwargs["filename"]).name

    receipt = result.parent / (result.name + ".hf-upload-receipt.json")
    if failure:
        with pytest.raises(RuntimeError, match="HF upload/readback failed") as error:
            core.upload(
                path,
                result,
                tmp_path / "package",
                api_factory=Api,
                download=download,
                upload_fn=upload_fn,
            )
        assert "synthetic-private-value" not in str(error.value)
        assert not receipt.exists()
        if failure == "public":
            assert not any(event[0] == "upload" for event in events)
    else:
        core.upload(
            path,
            result,
            tmp_path / "package",
            api_factory=Api,
            download=download,
            upload_fn=upload_fn,
        )
        report = json.loads(receipt.read_text())
        assert report["verified"] and report["revision"] == "a" * 40
        assert set(report["sha256"]) == {p.name for p in result.iterdir()}
        assert [event[0] for event in events][:4] == ["head", "upload", "head", "list"]
        assert "synthetic-private-value" not in receipt.read_text()


def test_train_redacts_chunked_output_and_refuses_offline_tracker(monkeypatch, tmp_path, capsys):
    core = services()
    path, _ = private_file(tmp_path)
    monkeypatch.setattr(core.os, "environ", dict(core.os.environ))
    monkeypatch.setattr(core, "preflight", lambda: None, raising=False)

    def fake_train(*_args, **_kwargs):
        sys.stdout.write("synthetic-pri")
        sys.stdout.flush()
        sys.stdout.write("vate-value\nmetric=1\n")
        raise RuntimeError("synthetic-private-value")

    monkeypatch.setattr(core.runpy, "run_module", fake_train)
    monkeypatch.setattr(core, "clearml_stream_redaction", lambda _: core.contextlib.nullcontext())
    assert core.main(["train", "--env-file", str(path), "--", "--tracker", "clearml"]) == 1
    output = capsys.readouterr()
    assert "synthetic-private-value" not in output.out + output.err
    assert "metric=1" in output.out and "offline fallback" in output.err
    with pytest.raises(ValueError, match="clearml"):
        core.train(path, ["--tracker", "disabled"])


def test_train_requires_verified_recovery_before_cli_import(monkeypatch, tmp_path):
    core = services()
    path, _ = private_file(tmp_path)
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core.os, "environ", dict(core.os.environ))
    monkeypatch.setattr(core, "clearml_stream_redaction", lambda _: core.contextlib.nullcontext())
    monkeypatch.setattr(core.runpy, "run_module", lambda *a, **k: pytest.fail("updates started"))
    with pytest.raises((ValueError, FileNotFoundError)):
        core.train(
            path,
            ["--tracker", "clearml", "--result-dir", str(tmp_path / "new")],
            package_dir=tmp_path,
            hf_checkpoints=True,
        )


@pytest.mark.parametrize("missing", range(7))
def test_each_required_key_refused_before_cli_import(monkeypatch, tmp_path, missing):
    core = services()
    path, values = private_file(tmp_path)
    key = core.KEYS[missing]
    path.write_text("\n".join(f'{name}="{value}"' for name, value in values.items() if name != key))
    monkeypatch.setattr(core, "preflight", lambda: None, raising=False)
    monkeypatch.setattr(core.runpy, "run_module", lambda *a, **k: pytest.fail("train imported"))
    with pytest.raises(ValueError, match=key):
        core.train(path, ["--tracker", "clearml"])
