"""Exercise the installed native CLI through the service, without training or network."""

# pylint: disable=import-outside-toplevel
import importlib
import sys
from pathlib import Path

import pytest
from test_colab_services import private_file, services
from test_colab_streams import sdk_streams


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251", None])
def test_redacted_output_encoding_is_explicit_read_only_metadata(encoding):
    """Tyro can inspect encoding without bypassing redaction through a raw stream."""
    from types import SimpleNamespace

    target = SimpleNamespace(encoding=encoding, buffer=object(), errors="strict")
    output = services().RedactedOutput(target, ["synthetic-secret"])
    assert output.encoding == encoding
    target.encoding = "ascii"
    assert output.encoding == "ascii"
    with pytest.raises(AttributeError):
        output.encoding = "utf-16"
    for name in ("buffer", "errors", "fileno"):
        with pytest.raises(AttributeError):
            getattr(output, name)


@pytest.mark.parametrize("arguments, code", [(["--help"], 0), (["--unknown-option"], 2)])
def test_native_cli_namespace_exit_and_cleanup(monkeypatch, tmp_path, capsys, arguments, code):  # pylint: disable=too-many-locals
    """Real runpy/Pydantic/tyro must resolve annotations in the temporary __main__."""
    import tyro

    from compare_attention.train import loop

    core = services()
    path, _ = private_file(tmp_path)
    sdk = sdk_streams()
    original_write = sdk.PrintPatchLogger.write
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core.os, "environ", dict(core.os.environ))
    monkeypatch.setattr(sys, "argv", ["service-parent"])
    # Match python -m colab_services, rather than pytest's accidental globals.
    monkeypatch.setitem(sys.modules, "__main__", core)
    monkeypatch.setitem(sys.modules, "clearml.backend_interface.logger", sdk)
    monkeypatch.setattr(loop, "train", lambda *_a, **_k: pytest.fail("training started"))
    tracker = importlib.import_module("compare_attention.train.tracker")
    monkeypatch.setattr(tracker, "select", lambda *_a, **_k: pytest.fail("Task init started"))
    data = importlib.import_module("compare_attention.data.store")
    monkeypatch.setattr(
        data, "prepare_kv_retrieval", lambda *_a, **_k: pytest.fail("dataset preparation started")
    )
    actual_cli = tyro.cli
    observed = []

    def checked_cli(model, *args, **kwargs):
        namespace = sys.modules["__main__"]
        observed.append(
            namespace is not core
            and getattr(namespace, "TrainArgs", None) is model
            and Path(sys.argv[0]).name == "train.py"
        )
        return actual_cli(model, *args, **kwargs)

    monkeypatch.setattr(tyro, "cli", checked_cli)
    streams = sys.stdout, sys.stderr
    cli_arguments = ["--tracker", "clearml", *arguments]
    with pytest.raises(SystemExit) as error:
        core.train(path, cli_arguments)
    assert error.value.code == code
    assert observed == [True]
    assert sys.modules["__main__"] is core
    assert sys.argv == ["service-parent"]
    assert (sys.stdout, sys.stderr) == streams
    assert sdk.PrintPatchLogger.write is original_write
    output = capsys.readouterr()
    assert "synthetic-private-value" not in output.out + output.err
    if code == 0:
        assert "--result-dir" in output.out and "--tracker" in output.out


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_full_frozen_delta_reaches_first_cli_training_boundary(
    monkeypatch, tmp_path, capsys, failure, resume
):  # pylint: disable=too-many-locals,too-many-statements
    """Parse canonical full Delta argv; stop before any dataset, Task or model work."""
    import zipfile
    from types import SimpleNamespace

    import dotenv
    import tyro
    import yaml

    from compare_attention.train import loop

    core = services()
    runner = importlib.import_module("colab_runner")
    package_core = importlib.import_module("colab_package")
    bundle = Path(
        "E:/HermesArtifacts/compare-attention/fp16-k64-small/"
        "272c2da7a9ef619711e4e92a1a98400ac0ed64b08f481d13752093d450820975/"
        "release/release.bundle"
    )
    assert package_core.checksum(bundle) == package_core.RELEASE_SHA256
    config_name = package_core.CONFIGS["deltanet"]
    with zipfile.ZipFile(bundle) as archive:
        config_bytes = archive.read(config_name)
    package = tmp_path / "package"
    config_path = package / "release" / config_name
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(config_bytes)
    frozen = yaml.safe_load(config_bytes)
    result = tmp_path / "result"
    snapshot = tmp_path / "recovery-input/step_00046500.pt" if resume else None
    arguments = runner.train_command(package, tmp_path / "venv", result, "deltanet", snapshot)[3:]
    path, _ = private_file(tmp_path)
    sdk = sdk_streams()
    original_write = sdk.PrintPatchLogger.write
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core.os, "environ", {})
    original_argv = ["service-parent"]
    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setitem(sys.modules, "__main__", core)
    monkeypatch.setitem(sys.modules, "clearml.backend_interface.logger", sdk)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda **_k: "")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "train", lambda *_a, **_k: pytest.fail("training started"))
    tracker = importlib.import_module("compare_attention.train.tracker")
    monkeypatch.setattr(tracker, "select", lambda *_a, **_k: pytest.fail("Task init started"))
    data = importlib.import_module("compare_attention.data.store")
    monkeypatch.setattr(
        data, "prepare_kv_retrieval", lambda *_a, **_k: pytest.fail("dataset preparation started")
    )
    captured, messages = [], []
    logger = SimpleNamespace(_console=lambda text, **_k: messages.append(text))
    actual_cli = tyro.cli

    def boundary(validated):
        captured.append(validated.model_dump(mode="json"))
        sdk.StdStreamPatch.patch_std_streams(logger, load_config_defaults=False)
        sys.stdout.write("metric=1\nsynthetic-pri")
        sys.stdout.flush()
        sys.stdout.write("vate-value\n")
        sys.stderr.write("tail=synthetic-pri")
        sys.stderr.flush()
        sys.stderr.write("vate-value")
        if failure:
            raise RuntimeError("synthetic boundary failure")
        raise SystemExit(0)

    def checked_cli(model, *args, **kwargs):
        namespace = sys.modules["__main__"]
        assert namespace is not core and getattr(namespace, "TrainArgs", None) is model
        assert Path(sys.argv[0]).name == "train.py"
        validated = actual_cli(model, *args, **kwargs)
        namespace._ensure_dataset = boundary  # pylint: disable=protected-access
        return validated

    monkeypatch.setattr(tyro, "cli", checked_cli)
    streams = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError if failure else SystemExit) as error:
        core.train(path, arguments)
    if failure:
        assert str(error.value) == "synthetic boundary failure"
    else:
        assert error.value.code == 0
    assert captured == [
        {
            **frozen,
            "config": str(config_path),
            "data": str(package / "data"),
            "result_dir": str(result),
            "tracker": "clearml",
            "resume_from": str(snapshot) if resume else None,
            "hf_repo": None,
        }
    ]
    assert sys.argv is original_argv
    assert sys.modules["__main__"] is core
    assert (sys.stdout, sys.stderr) == streams
    assert sdk.PrintPatchLogger.write is original_write
    output = capsys.readouterr()
    for text in (output.out + output.err, "\n".join(messages)):
        assert "synthetic-pri" not in text
        assert "metric=1" in text and "tail=[REDACTED]" in text
    assert not (package / "data").exists()
    assert not (result / "model.pt").exists()
