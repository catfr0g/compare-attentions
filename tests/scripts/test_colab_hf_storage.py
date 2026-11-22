"""HF-only launcher contracts; no credentials or cloud access."""

# Test-local imports and synthetic fixture objects isolate all service activity.
# pylint: disable=import-outside-toplevel

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_exported_checkpoint_helpers_import_without_host_paths(tmp_path):
    import shutil
    import subprocess

    from colab_package import TOOLS

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    exported = tmp_path / "tools"
    exported.mkdir()
    for name in TOOLS:
        shutil.copyfile(scripts / name, exported / name)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys; sys.dont_write_bytecode=True; sys.path.insert(0,sys.argv[1]); "
            "import colab_checkpoint_backup; import colab_checkpoint_store",
            str(exported),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_interface_is_preflight_first_and_literal_paths():
    import export_colab
    from colab_runner import PREFLIGHT

    code = export_colab.bootstrap_code("a" * 64, env_file="/content/private/kaggle-env.env")
    assert code.startswith(PREFLIGHT)
    assert "ENV_FILE = Path('/content/private/kaggle-env.env')" in code
    assert "drive.mount" not in code and "google.colab" not in code
    assert code.index("actual != EXPECTED_SHA256") < code.index("from colab_package import")
    compile(code, "<bootstrap>", "exec")


@pytest.mark.parametrize("sha", ["bad", "a" * 63, "a'\nraise RuntimeError('injection')#"])
def test_bootstrap_refuses_non_sha_literals(sha):
    import export_colab

    with pytest.raises(ValueError, match="SHA256"):
        export_colab.bootstrap_code(sha)


def test_bootstrap_paths_are_independent_literal_values():
    import ast

    import export_colab

    code = export_colab.bootstrap_code(
        "b" * 64,
        archive_path="/content/private/kaggle-env.env",
        env_file="/content/quotes'\\file.env",
    )
    tree = ast.parse(code)
    bindings = {
        node.targets[0].id: node.value.args[0].value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "Path"
    }
    assert bindings["ARCHIVE_PATH"] == "/content/private/kaggle-env.env"
    assert bindings["ENV_FILE"] == "/content/quotes'\\file.env"


def test_hf_hook_is_opt_in_and_enters_after_clearml_env_before_trainer(monkeypatch, tmp_path):
    import colab_services as core

    events = []
    values = {key: "synthetic" for key in core.KEYS}
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "load_credentials", lambda _: values)
    monkeypatch.setattr(core.os, "environ", {})
    monkeypatch.setattr(core, "clearml_stream_redaction", lambda _: contextlib.nullcontext())

    @contextlib.contextmanager
    def hook(env_file, package_dir):
        assert core.os.environ["CLEARML_API_SECRET_KEY"] == "synthetic"
        assert "HF_TOKEN" not in core.os.environ
        assert env_file == tmp_path / "private.env" and package_dir == tmp_path
        events.append("hook")
        yield
        events.append("restored")

    monkeypatch.setattr(core, "require_checkpoint_proof", lambda *a: events.append("proof"))
    monkeypatch.setitem(
        sys.modules, "colab_checkpoint_backup", SimpleNamespace(checkpoint_backups=hook)
    )
    monkeypatch.setattr(core.runpy, "run_module", lambda *a, **k: events.append("trainer"))
    core.train(
        tmp_path / "private.env",
        ["--tracker", "clearml"],
        package_dir=tmp_path,
        hf_checkpoints=True,
    )
    assert events == ["proof", "hook", "trainer", "restored"]
    events.clear()
    core.train(tmp_path / "private.env", ["--tracker", "clearml"])
    assert events == ["trainer"]


def test_hf_local_storage_never_checks_drive_mount(monkeypatch, tmp_path):
    import colab_runner as core

    monkeypatch.setattr(core, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(core, "DRIVE_ROOT", tmp_path / "drive/MyDrive")
    monkeypatch.setattr(core.os.path, "ismount", lambda _: pytest.fail("Drive mount checked"))
    assert (
        core.persistent_result(tmp_path / "results/new", tmp_path / "package")
        == tmp_path / "results/new"
    )
    with pytest.raises(ValueError, match="explicit"):
        core.persistent_result(tmp_path / "drive/MyDrive/new", tmp_path / "package")


@pytest.mark.parametrize("parent", ["package", "venv", "result"])
def test_private_env_excludes_executable_and_output_inputs(tmp_path, parent):
    import colab_runner as core

    roots = {name: tmp_path / name for name in ("package", "venv", "result")}
    roots[parent].mkdir()
    private = roots[parent] / "keys.env"
    private.write_text("synthetic")
    with pytest.raises(ValueError, match="outside"):
        core.private_env_path(private, roots["package"], roots["result"], venv=roots["venv"])


@pytest.mark.parametrize("corruption", [None, "identity", "verified", "revision"])
def test_real_hook_requires_bound_proof_and_restores_on_failure(monkeypatch, tmp_path, corruption):
    import json

    import colab_checkpoint_backup as backup
    import colab_services as core

    from compare_attention.train.backup import BackupTracker

    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps({"source_sha256": "a" * 64, "data_fingerprint": "b" * 64})
    )
    result = tmp_path / "new-result"
    proof = {"verified": True, "revision": "a" * 40, "identity": backup.identity(package, result)}
    if corruption:
        proof[corruption] = False
    backup.receipt_path(result).write_text(json.dumps(proof))
    original = BackupTracker.save_checkpoint
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "load_credentials", lambda _: {key: "synthetic" for key in core.KEYS})
    monkeypatch.setattr(core.os, "environ", {})
    monkeypatch.setattr(core, "clearml_stream_redaction", lambda _: contextlib.nullcontext())

    def trainer(*_args, **_kwargs):
        assert not corruption, "trainer imported with invalid proof"
        assert BackupTracker.save_checkpoint is not original
        raise RuntimeError("synthetic trainer failure")

    monkeypatch.setattr(core.runpy, "run_module", trainer)
    with pytest.raises(ValueError if corruption else RuntimeError):
        core.train(
            tmp_path / "private.env",
            ["--tracker", "clearml", "--result-dir", str(result)],
            package_dir=package,
            hf_checkpoints=True,
        )
    assert BackupTracker.save_checkpoint is original
    assert not result.exists()


def test_service_parser_keeps_hf_wrapper_flags_out_of_original_cli(monkeypatch, tmp_path):
    import colab_services as core

    calls = []
    monkeypatch.setattr(core, "train", lambda *a, **k: calls.append((a, k)))
    assert (
        core.main(
            [
                "train",
                "--env-file",
                str(tmp_path / "private.env"),
                "--package-dir",
                str(tmp_path),
                "--hf-checkpoints",
                "--",
                "--tracker",
                "clearml",
            ]
        )
        == 0
    )
    assert calls[0][0][1] == ["--tracker", "clearml"]
    assert calls[0][1] == {"package_dir": tmp_path, "hf_checkpoints": True}
