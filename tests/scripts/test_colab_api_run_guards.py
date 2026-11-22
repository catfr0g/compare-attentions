"""Actual generated job guards and synthetic export identity contracts."""

# Independent function/subprocess entrypoint tests intentionally repeat CLI literals.
# pylint: disable=duplicate-code

import hashlib
import importlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
controller = importlib.import_module("colab_api_run")
remote = importlib.import_module("colab_api_run_remote")


def test_detached_setup_is_exclusive_and_actual_child_exit_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", str(tmp_path / "jobs") + "/")
    monkeypatch.setattr(remote, "PACKAGE", str(tmp_path / "package"))
    binding = {"runtime": "runtimes/test", "run_name": "run-1", "sha256": "a" * 64}
    root = tmp_path / "jobs/run-1"
    root.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(binding))
    (root / "inputs.json").write_text(json.dumps({"verified": True, "sha256": "a" * 64}))
    tools = tmp_path / "package/tools"
    tools.mkdir(parents=True)
    sentinel = tmp_path / "child-invocations"
    # This is a real local benign subprocess, not a simulated scientific training result.
    (tools / "colab_runner.py").write_text(
        f'with open({str(sentinel)!r}, "a") as stream: stream.write("one\\n")\n'
    )
    code = remote.launch_code(binding, "setup")
    first = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    assert json.loads(first.stdout.removeprefix(remote.MARKER))["state"] == "spawned"
    state_path = root / "setup/state.json"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = json.loads(state_path.read_text())
        if state.get("state") == "completed":
            break
        time.sleep(0.02)
    assert state["state"] == "completed"
    assert state["returncode"] == 0
    second = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert second.returncode != 0
    assert sentinel.read_text() == "one\n"


def test_mcp_ambiguous_execution_never_retried(tmp_path):
    calls = []

    class Fake:  # pylint: disable=too-few-public-methods
        def call(self, name, arguments):
            calls.append(name)
            if name == "create_session":
                return {"name": arguments["parent"] + "/sessions/" + arguments["sessionId"]}
            raise ValueError("https://private.google.com/?token=SECRET")

    path = tmp_path / "journal"
    with controller.Journal(path, {}) as journal:
        execution = controller.Execution(Fake(), "runtimes/test", journal)
        with pytest.raises(ValueError):
            execution.code("fixed nonsecret program")
    assert calls == ["create_session", "execute_code"]
    assert "SECRET" not in path.read_text()
    assert "execute_pending" in path.read_text()


def test_hardware_guard_precedes_remote_input_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", str(tmp_path) + "/")
    monkeypatch.setattr(remote, "PREFLIGHT", 'raise RuntimeError("hardware-sentinel")\n')
    code = remote.marker_code({"run_name": "guarded"}, "nonce")
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "hardware-sentinel" in result.stderr
    assert not list(tmp_path.iterdir())


def test_reuse_session_does_not_create_another_kernel(tmp_path):
    session = "runtimes/test/sessions/job-" + "a" * 32
    calls = []

    class Fake:  # pylint: disable=too-few-public-methods
        def call(self, name, arguments):
            calls.append((name, arguments))
            return {"sessions": [{"name": session}]}

    with controller.Journal(tmp_path / "journal", {}) as journal:
        execution = controller.Execution(Fake(), "runtimes/test", journal, session=session)
        assert execution.session == session
    assert [item[0] for item in calls] == ["list_sessions"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, AttributeError])
def test_cli_suppresses_arbitrary_exception_secrets(tmp_path, monkeypatch, capsys, error_type):
    def fail(_args):
        raise error_type("https://private.google.com/?signed=SECRET")

    monkeypatch.setattr(controller, "run", fail)
    assert (
        controller.main(
            [
                "status",
                "--runtime",
                "runtimes/test",
                "--run-name",
                "run-1",
                "--journal",
                str(tmp_path / "journal"),
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "SECRET" not in output
    assert json.loads(output)["error"] == "CONTROLLER_STOPPED_RECONCILE_JOURNAL"


def test_private_env_exact_allowlist_without_hash_receipts(tmp_path):
    keys = importlib.import_module("colab_services").KEYS
    path = tmp_path / "synthetic.env"
    raw = "".join(key + "=" + json.dumps("synthetic-" + key) + "\n" for key in keys).encode()
    path.write_bytes(raw)
    assert controller.private_bytes(path) == raw
    path.write_bytes(raw + b'UNEXPECTED="must reject"\n')
    with pytest.raises(ValueError, match="INVALID_PRIVATE_INPUT"):
        controller.private_bytes(path)


def test_transfer_records_owned_staging_id_before_first_remote_write(tmp_path):
    entries = []

    class Recorder:  # pylint: disable=too-few-public-methods
        def record(self, value):
            entries.append(value)

    class Ambiguous:  # pylint: disable=too-few-public-methods
        def code(self, _code):
            raise TimeoutError("synthetic ambiguous marker")

    args = SimpleNamespace(runtime="runtimes/test", env_file=tmp_path / "not-read")
    binding = {"runtime": args.runtime, "run_name": "run-1", "sha256": "a" * 64}
    with pytest.raises(TimeoutError):
        controller.transfer(
            args, binding, tmp_path / "not-read", Ambiguous(), "synthetic", Recorder()
        )
    assert entries[0]["phase"] == "transfer_prepared"
    assert len(entries[0]["nonce"]) == 32
    assert entries[0]["staging_archive"] == "compare-archive-" + entries[0]["nonce"] + ".zip"


def synthetic_export(tmp_path):
    package = importlib.import_module("colab_package")
    exporter = importlib.import_module("export_colab")
    spec = {
        "vocab_size": 8192,
        "input_seq_len": 512,
        "num_kv_pairs": 64,
        "power_a": 0.01,
        "n_train": 100000,
        "n_val": 1000,
        "n_test": 1000,
        "seed": 1337,
    }
    schedule = {
        "train_flops": 10**16,
        "axis": "train_flops",
        "batch_size": 32,
        "seed": 42,
        "dtype": "float16",
        "compile": True,
        "val_every_steps": 25,
        "log_every_steps": 25,
        "checkpoint_every_steps": 125,
        "checkpoint_keep_last": 2,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.001,
        "early_stopping_min_steps": 2000,
        "amp_max_scale": 128.0,
    }
    configs = {
        family: {
            "model": {
                "vocab_size": 8192,
                "block_size": 512,
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
                "dropout": 0.0,
                "bias": True,
                "attention": family,
            },
            "schedule": schedule,
            "optim": {
                "learning_rate": 0.001,
                "weight_decay": 0.1,
                "lr_schedule": "constant",
                "warmup_fraction": 0.05,
                "min_lr_ratio": 0.1,
            },
        }
        for family in package.CONFIGS
    }
    manifest = {
        "source_sha256": controller.SOURCE,
        "data_fingerprint": controller.DATA,
        "schema_version": 3,
        "storage": "hf",
        "periodic_full_checkpoints": True,
        "tracking": "clearml",
        "hf_upload": True,
        "default_family": "deltanet",
        "expected_updates": {"deltanet": 47265},
        "configs": configs,
        "dataset_spec": spec,
    }
    archive = tmp_path / "colab-package.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (tmp_path / "export-report.json").write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "archive_bytes": archive.stat().st_size,
                "source_sha256": controller.SOURCE,
                "training_started": False,
            }
        )
    )
    (tmp_path / "launch_colab.ipynb").write_text(json.dumps(exporter.notebook(sha)))
    return archive, sha


def test_actual_export_report_archive_manifest_notebook_binding(tmp_path):
    archive, sha = synthetic_export(tmp_path)
    assert controller.export_identity(tmp_path) == (archive, sha)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="EXPORT_REPORT_MISMATCH"):
        controller.export_identity(tmp_path)


def test_cli_actual_subprocess_plan_has_zero_io(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/colab_api_run.py",
            "setup",
            "--runtime",
            "runtimes/test",
            "--run-name",
            "run-1",
            "--journal",
            str(tmp_path / "journal"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["phase"] == "plan"
    assert not list(tmp_path.iterdir())
