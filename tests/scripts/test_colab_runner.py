"""Offline policy tests: never use the host GPU, network, installer or credentials."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def runner():
    if importlib.util.find_spec("colab_runner") is None:
        pytest.fail("explicit Colab runner is missing")
    return importlib.import_module("colab_runner")


@pytest.mark.parametrize("gpu", [None, "NVIDIA GeForce RTX 5070, 12.0", "Tesla T4, 8.0"])
def test_gpu_rejection_precedes_installer_and_credentials(monkeypatch, tmp_path, gpu):
    core = runner()
    calls = []
    monkeypatch.setattr(sys, "version_info", (3, 13, 15))

    def query(*_args, **_kwargs):
        if gpu is None:
            raise FileNotFoundError("no nvidia-smi")
        return gpu

    monkeypatch.setattr(subprocess, "check_output", query)
    monkeypatch.setattr(core, "verify_directory", lambda _: pytest.fail("package accessed"))
    monkeypatch.setattr(
        core, "clean_environment", lambda: pytest.fail("setup environment accessed")
    )
    monkeypatch.setattr(core, "execute", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(RuntimeError, match="T4"):
        core.setup(tmp_path, tmp_path / "venv")
    assert not calls


def test_pinned_plan_and_training_only_operational_overrides(tmp_path):
    core = runner()
    package = tmp_path / "package"
    venv = Path("/content/isolated")
    plan = core.installation_plan(package, venv, "/usr/bin/python3.13")
    assert plan[0] == ["/usr/bin/python3.13", "-m", "venv", str(venv), "--without-pip"]
    assert all(
        command[:6]
        == ["/usr/bin/python3.13", "-m", "pip", "--python", str(venv / "bin/python"), "install"]
        for command in plan[1:]
    )
    assert "torch==2.11.0+cu130" in plan[1]
    assert "https://download.pytorch.org/whl/cu130" in plan[1]
    assert "triton==3.6.0" in plan[2]
    assert "--no-deps" in plan[-1] and "--no-deps" in plan[-2]
    assert not any("--upgrade" in command for command in plan)
    for family in ("classic", "deltanet"):
        command = core.train_command(package, venv, tmp_path / "result", family)
        assert command[1:3] == ["-m", "compare_attention.cli.train"]
        assert command[-2:] == ["--tracker", "clearml"]
        assert family in command[4]
        flags = {item for item in command if item.startswith("--")}
        assert flags == {"--config", "--data", "--result-dir", "--tracker"}


def test_environment_discards_optional_credentials(monkeypatch):
    core = runner()
    for name in (
        "HF_TOKEN",
        "HF_REPO",
        "CLEARML_API_SECRET_KEY",
        "KAGGLE_KEY",
        "PYTHONPATH",
        "PYTHONOPTIMIZE",
        "PYTHONINSPECT",
    ):
        monkeypatch.setenv(name, "SECRET_SENTINEL")
    env = core.clean_environment()
    assert "SECRET_SENTINEL" not in json.dumps(env)
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_inspect_cli_is_offline_and_default_is_dryrun(monkeypatch, tmp_path, capsys):
    core = runner()
    manifest = {"files": {"one": {}}, "source_sha256": "source", "data_fingerprint": "data"}
    monkeypatch.setattr(core, "verify_directory", lambda _: manifest)
    monkeypatch.setattr(core, "preflight", lambda: pytest.fail("GPU inspection attempted"))
    monkeypatch.setattr(core, "execute", lambda *a, **k: pytest.fail("child attempted"))
    for flags in ([], ["--inspect"], ["--verify-only"]):
        core.main(["--package-dir", str(tmp_path), *flags])
        report = json.loads(capsys.readouterr().out)
        assert report["dry_run"] and not report["training_started"]


def test_training_gate_snapshot_then_public_test_eval(monkeypatch, tmp_path):
    core = runner()
    manifest = {"kaggle_reference": {"python": "3.13.16"}}
    calls = []
    result = tmp_path / "result"
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "colab-package.sha256").write_text("hash")
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "verify_directory", lambda _: manifest)
    monkeypatch.setattr(core, "checksum", lambda _: "hash")
    monkeypatch.setattr(core, "persistent_result", lambda *_a, **_k: result)

    def execute(command, env, *_args, **_kwargs):
        calls.append(command)
        assert "HF_TOKEN" not in env and "CLEARML_API_SECRET_KEY" not in env
        if "train" in command:
            result.mkdir()
        if "upload" in command:
            assert (result / "colab-provenance.json").is_file()
        return SimpleNamespace(stdout=json.dumps({"gpu": "Tesla T4", "python": "3.13.15"}))

    monkeypatch.setattr(core, "execute", execute)
    private = tmp_path.parent / "private.env"
    private.write_text("synthetic")
    core.run(tmp_path, venv, result, env_file=private)
    assert calls[0][-1] == core.PROBE
    assert "check" in calls[1]
    assert "probe" in calls[2]
    assert calls[2][-4:] == ["--result-dir", str(result), "--package-dir", str(tmp_path)]
    assert "train" in calls[3]
    assert "--hf-checkpoints" in calls[3] and "--package-dir" in calls[3]
    assert calls[4][1].endswith("colab_verify.py")
    assert "compare_attention.cli.evaluate" in calls[5]
    assert calls[5][calls[5].index("--split") + 1] == "test"
    assert "upload" in calls[6]
    report = json.loads((result / "colab-provenance.json").read_text())
    assert report["matched_runtime"] is False
    assert report["bitwise_initialization_claimed"] is False
    assert report["tracking"] == "clearml" and report["hf_upload_requested"] is True


def test_native_full_gate_and_snapshot_share_exact_expected_config(monkeypatch, tmp_path):
    if importlib.util.find_spec("colab_verify") is None:
        pytest.fail("native result gate entrypoint is missing")
    core = importlib.import_module("colab_verify")
    calls = []
    frozen = {
        "model": {"attention": "deltanet"},
        "schedule": {"train_flops": 10**16},
        "optim": {"learning_rate": 0.001},
    }
    manifest = {"configs": {"deltanet": frozen}, "dataset_spec": {"num_kv_pairs": 64}}
    monkeypatch.setattr(core, "verify_directory", lambda _: manifest)

    def gate(_directory, *, expected_config, expected_gpu):
        calls.append(("gate", expected_config, expected_gpu))
        return {"verified": True}

    monkeypatch.setitem(
        sys.modules,
        "compare_attention.exploration_gate",
        SimpleNamespace(verify_exploration_run=gate),
    )
    monkeypatch.setattr(
        core.runpy,
        "run_path",
        lambda _path: {
            "verify_snapshot": lambda _directory, expected: calls.append(("snapshot", expected))
        },
    )
    core.verify(tmp_path, tmp_path, "deltanet", "Tesla T4")
    assert [call[0] for call in calls] == ["gate", "snapshot"]
    assert (
        calls[0][1]
        == calls[1][1]
        == {**frozen, "data": str(tmp_path / "data"), "dataset": {"num_kv_pairs": 64}}
    )
    assert json.loads((tmp_path / "colab-native-gate.json").read_text())["snapshot_verified"]


def test_persistent_fresh_and_resume_guards(monkeypatch, tmp_path):
    core = runner()
    content = tmp_path / "content"
    content.mkdir()
    drive = content / "drive/MyDrive"
    drive.mkdir(parents=True)
    monkeypatch.setattr(core, "LOCAL_ROOT", content)
    monkeypatch.setattr(core, "DRIVE_ROOT", drive)
    monkeypatch.setattr(core.os.path, "ismount", lambda _: False)
    package = content / "inputs"
    fresh = content / "results/new-result"
    assert core.persistent_result(fresh, package) == fresh
    fresh.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        core.persistent_result(fresh, package)
    old = content / "checkpoint.pt"
    old.write_text("explicit snapshot")
    assert core.persistent_result(content / "new-resume", package, old) == content / "new-resume"
    with pytest.raises(FileExistsError):
        core.persistent_result(fresh, package, old)
    with pytest.raises(ValueError, match="resume"):
        core.persistent_result(content / "new-resume", package, content / "missing.pt")
    with pytest.raises(ValueError, match="result-dir"):
        core.persistent_result(tmp_path / "outside", package)
    with pytest.raises(ValueError, match="not mounted"):
        core.persistent_result(drive / "another", package, storage="drive")
    monkeypatch.setattr(core.os.path, "ismount", lambda _: True)
    assert core.persistent_result(drive / "another", package, storage="drive") == drive / "another"


def test_public_gate_failure_blocks_evaluation(monkeypatch, tmp_path):
    core = runner()
    result = tmp_path / "result"
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "colab-package.sha256").write_text("hash")
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "verify_directory", lambda _: {"kaggle_reference": {}})
    monkeypatch.setattr(core, "persistent_result", lambda *_a, **_k: result)
    monkeypatch.setattr(core, "checksum", lambda _: "hash")
    calls = []

    def execute(command, *_args, **_kwargs):
        calls.append(command)
        if len(command) > 1 and command[1].endswith("colab_verify.py"):
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout='{"gpu":"Tesla T4"}')

    monkeypatch.setattr(core, "execute", execute)
    with pytest.raises(subprocess.CalledProcessError):
        private = tmp_path.parent / "private.env"
        private.write_text("synthetic")
        core.run(tmp_path, venv, result, env_file=private)
    assert not any("compare_attention.cli.evaluate" in c for c in calls)
    assert not any("upload" in c for c in calls)


def test_remote_recovery_failure_blocks_model_updates(monkeypatch, tmp_path):
    core = runner()
    result = tmp_path / "result"
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "colab-package.sha256").write_text("hash")
    private = tmp_path.parent / "private.env"
    private.write_text("synthetic")
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "verify_directory", lambda _: {"kaggle_reference": {}})
    monkeypatch.setattr(core, "persistent_result", lambda *_a, **_k: result)
    monkeypatch.setattr(core, "checksum", lambda _: "hash")
    calls = []

    def execute(command, *_args, **_kwargs):
        calls.append(command)
        if "probe" in command:
            assert not result.exists()
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout='{"gpu":"Tesla T4"}')

    monkeypatch.setattr(core, "execute", execute)
    with pytest.raises(subprocess.CalledProcessError):
        core.run(tmp_path, venv, result, env_file=private)
    assert calls[0][-1] == core.PROBE
    assert "probe" in calls[-1]
    assert not any("train" in command for command in calls)
    assert not result.exists()


@pytest.mark.parametrize("destination", ["package", "venv", "private", "relative"])
def test_result_paths_exclude_inputs_venv_and_private(monkeypatch, tmp_path, destination):
    core = runner()
    monkeypatch.setattr(core, "LOCAL_ROOT", tmp_path)
    package = tmp_path / "package"
    venv = tmp_path / "venv"
    private = tmp_path / "private/keys.env"
    paths = {
        "package": package / "result",
        "venv": venv / "result",
        "private": private.parent / "result",
        "relative": Path("relative-result"),
    }
    with pytest.raises(ValueError, match="result-dir"):
        core.persistent_result(paths[destination], package, venv=venv, env_file=private)


def test_private_env_symlink_into_package_rejected(tmp_path):
    core = runner()
    package = tmp_path / "package"
    package.mkdir()
    internal = package / "private.env"
    internal.write_text("synthetic")
    alias = tmp_path / "external.env"
    try:
        alias.symlink_to(internal)
    except OSError:
        pytest.skip("Windows symlinks require developer mode")
    with pytest.raises(ValueError, match="outside"):
        core.private_env_path(alias, package, tmp_path / "result")


def test_local_no_drive_run_reaches_required_recovery_before_train(monkeypatch, tmp_path):
    core = runner()
    package = tmp_path / "package"
    package.mkdir()
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "colab-package.sha256").write_text("hash")
    private = tmp_path / "private/keys.env"
    private.parent.mkdir()
    private.write_text("synthetic")
    result = tmp_path / "results/new-run"
    monkeypatch.setattr(core, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(core.os.path, "ismount", lambda _: False)
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "verify_directory", lambda _: {"kaggle_reference": {}})
    monkeypatch.setattr(core, "checksum", lambda _: "hash")
    calls = []

    def execute(command, *_args, **_kwargs):
        calls.append(command)
        if "train" in command:
            result.mkdir()
        return SimpleNamespace(stdout='{"gpu":"Tesla T4"}')

    monkeypatch.setattr(core, "execute", execute)
    core.run(package, venv, result, env_file=private)
    assert "probe" in calls[2] and "train" in calls[3]
    assert (result / "colab-provenance.json").is_file()


def test_missing_private_env_precedes_probe_and_training(monkeypatch, tmp_path):
    core = runner()
    monkeypatch.setattr(core, "preflight", lambda: None)
    monkeypatch.setattr(core, "verify_directory", lambda _: {})
    monkeypatch.setattr(core, "persistent_result", lambda *a, **k: tmp_path / "result")
    monkeypatch.setattr(core, "execute", lambda *a, **k: pytest.fail("child attempted"))
    with pytest.raises(ValueError, match="env-file"):
        core.run(tmp_path, tmp_path / "venv", tmp_path / "result")


def test_setup_executes_only_isolated_pinned_commands(monkeypatch, tmp_path):
    core = runner()
    package = tmp_path / "inputs"
    venv = tmp_path / "content" / "venv"
    calls = []
    monkeypatch.setattr(core, "LOCAL_ROOT", tmp_path / "content", raising=False)
    monkeypatch.setattr(sys, "version_info", (3, 13, 15))
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: "Tesla T4, 7.5")
    monkeypatch.setattr(core, "verify_directory", lambda _: {})
    monkeypatch.setattr(core, "checksum", lambda _: "verified-sha")

    def execute(command, *_args, **_kwargs):
        calls.append(command)
        if command[1:3] == ["-m", "venv"]:
            venv.mkdir(parents=True)
        return SimpleNamespace(stdout='{"gpu":"Tesla T4"}')

    monkeypatch.setattr(core, "execute", execute)
    core.setup(package, venv)
    assert calls[0] == [sys.executable, "-m", "venv", str(venv), "--without-pip"]
    assert len(calls) == 6
    for command in calls[1:-1]:
        assert command[:6] == [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(venv / "bin/python"),
            "install",
        ]
        assert command.count("--python") == 1
    assert [command[6:] for command in calls[1:-1]] == [
        ["torch==2.11.0+cu130", "--index-url", "https://download.pytorch.org/whl/cu130"],
        ["triton==3.6.0", "--no-deps"],
        ["--no-deps", "-r", str(package / "release/requirements.txt")],
        ["--no-deps", str(package / "release" / core.WHEEL)],
    ]
    assert calls[-1] == [str(venv / "bin/python"), "-c", core.PROBE]
    assert "sys.prefix != sys.base_prefix" in core.PROBE
    assert (venv / "colab-package.sha256").read_text() == "verified-sha"
    with pytest.raises(FileExistsError):
        core.setup(package, venv)


def test_wrong_python_fails_before_hardware_and_install(monkeypatch, tmp_path):
    core = runner()
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: pytest.fail("SMI attempted"))
    monkeypatch.setattr(core, "execute", lambda *_a, **_k: pytest.fail("install attempted"))
    with pytest.raises(RuntimeError, match="3.13"):
        core.setup(tmp_path, tmp_path / "venv")
