"""Серия с fake runner: один release, безопасное возобновление и gate."""

# Exercise the artifact-verification boundary independently of cloud orchestration.
# pylint: disable=protected-access

import ast
import hashlib
import json
import re
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from compare_attention import kaggle_series
from compare_attention.kaggle_artifacts import file_manifest
from tests.compare_attention.test_exploration_gate import artifact, persist
from tests.compare_attention.test_kaggle_artifacts import make_inputs


@pytest.mark.parametrize("provenance", ["missing", "null"])
@pytest.mark.parametrize("mutation", ["none", "lr", "declared_constant"])
def test_frozen_scheduler_cannot_downgrade_to_legacy_result(
    tmp_path, monkeypatch, provenance, mutation
):
    """Receipt integrity cannot replace mandatory scheduled provenance/LR validation."""
    from tests.compare_attention.test_lr_gate import (  # pylint: disable=import-outside-toplevel
        test_real_scheduled_cpu_artifact_passes_raw_gate_and_receipt_seam as genuine_run,
    )

    genuine_run(tmp_path, monkeypatch, "completed")
    output = tmp_path / "download"
    directory = output / "scheduled"
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    result = rows[-1]
    frozen = json.loads(json.dumps(result["config"]))
    if provenance == "missing":
        del result["provenance"]
    else:
        result["provenance"] = None
    if mutation == "lr":
        next(row for row in rows if row["type"] == "learning_rate")["learning_rate"] = 999.0
    if mutation == "declared_constant":
        result["config"]["optim"]["lr_schedule"] = "constant"
    metrics.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    receipt_path = output / "result-manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"]["scheduled/metrics.jsonl"] = {
        "bytes": metrics.stat().st_size,
        "sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    state = {
        "owner": "owner",
        "data_id": "owner/tiny",
        "identity": {
            "fingerprint": "f" * 64,
            "configs": {"tiny": frozen},
            "data": {"spec": frozen["dataset"]},
        },
        "runs": {"tiny": {"slug": "scheduled"}},
    }
    with pytest.raises(ValueError):
        kaggle_series._verify_result(output, state, "tiny")  # pylint: disable=protected-access


def early_output(tmp_path):
    """Real strict checkpoint and receipt, independent of the runner fake."""
    output = tmp_path / "output"
    output.mkdir()
    rows = artifact(output / "pilot", early=True)
    config = rows[-1]["config"]
    kv = dict(config["dataset"])
    kv["dataset_seed"] = kv.pop("seed")
    del kv["vocab_size"]
    state = {
        "owner": "user",
        "data_id": "user/data",
        "runs": {"pilot.yaml": {"slug": "pilot"}},
        "identity": {
            "fingerprint": "release",
            "configs": {
                "pilot.yaml": {
                    "model": config["model"],
                    "schedule": config["schedule"],
                    "optim": config["optim"],
                    "kv": kv,
                }
            },
            "data": {"spec": config["dataset"]},
        },
    }
    config["data"] = "/kaggle/input/datasets/user/data"
    persist(output / "pilot", rows)
    receipt = {
        "identity": {
            "series": "release",
            "config": "pilot.yaml",
            "id": "user/pilot",
            "slug": "pilot",
        },
        "files": file_manifest(output, list((output / "pilot").iterdir())),
    }
    (output / "result-manifest.json").write_text(json.dumps(receipt))
    return output, state, rows


def test_real_early_result_requires_gate_and_keeps_checksum_receipt(tmp_path):
    output, state, rows = early_output(tmp_path)
    receipt = kaggle_series._verify_result(output, state, "pilot.yaml")
    assert receipt["stop_reason"] == "early_stopping"
    assert receipt["steps"] == rows[-1]["steps"]
    assert receipt["train_flops"] == rows[-1]["train_flops"]
    (output / "pilot/model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        kaggle_series._verify_result(output, state, "pilot.yaml")


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "record_disabled",
        "wrong_config",
        "wrong_data",
        "compiler",
        "unbound_identity",
        "forged_reason",
    ],
)
def test_invalid_early_result_cannot_enter_journal(tmp_path, change):

    output, state, rows = early_output(tmp_path)
    state = deepcopy(state)
    experiment = state["identity"]["configs"]["pilot.yaml"]
    if change == "disabled":
        experiment["schedule"]["early_stopping_patience"] = 0
    elif change == "record_disabled":
        rows[-1]["config"]["schedule"]["early_stopping_patience"] = 0
    elif change == "wrong_config":
        experiment["schedule"]["batch_size"] = 2
    elif change == "wrong_data":
        state["identity"]["data"]["spec"]["seed"] += 1
    elif change == "compiler":
        rows[-1]["provenance"]["compiler"]["successful_training_backend_invocations"] = 0
    elif change == "forged_reason":
        rows[-1]["stop_reason"] = "budget_exhausted"
    persist(output / "pilot", rows)
    receipt = json.loads((output / "result-manifest.json").read_text())
    receipt["files"] = file_manifest(output, list((output / "pilot").iterdir()))
    if change == "unbound_identity":
        del receipt["files"]["pilot/run_identity.json"]
    (output / "result-manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        kaggle_series._verify_result(output, state, "pilot.yaml")


def test_journal_early_terminal_readback_never_repushes_or_selects_next(tmp_path, monkeypatch):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    manifest = kaggle_series.stage_series(
        tmp_path, configs, data, tmp_path / "staged", "user", runner
    )
    kaggle_series.publish_series(manifest, runner)
    state = json.loads(manifest.read_text())
    names = list(state["runs"])
    calls = []

    def verified_early(output, _state, name):
        # Only the verification seam is mocked here; independent tests use real artifacts.
        calls.append(name)
        receipt = json.loads((output / "result-manifest.json").read_text())
        return {
            **receipt,
            "stop_reason": "early_stopping",
            "steps": 4,
            "train_flops": 8,
            "quality": {"val_loss": 1.9},
        }

    monkeypatch.setattr(kaggle_series, "_verify_result", verified_early)
    kaggle_series.run_selected(manifest, names[:1], run=runner, poll_seconds=0)
    kaggle_series.run_selected(manifest, names[:1], run=runner, poll_seconds=0)
    state = json.loads(manifest.read_text())
    entry = state["runs"][names[0]]
    assert entry["state"] == "early_stopped"
    assert entry["steps"] == 4 and entry["train_flops"] == 8
    assert entry["stop_reason"] == "early_stopping"
    assert entry["quality"] == {"val_loss": 1.9}
    assert len(calls) == 2
    assert sum(command[1:3] == ["kernels", "push"] for command in runner.commands) == 1
    assert state["runs"][names[1]]["state"] == "staged"

    def rejected_readback(*_args):
        raise ValueError("corrupt terminal evidence")

    monkeypatch.setattr(kaggle_series, "_verify_result", rejected_readback)
    with pytest.raises(ValueError, match="corrupt terminal"):
        kaggle_series.run_selected(manifest, names, run=runner, poll_seconds=0)
    assert sum(command[1:3] == ["kernels", "push"] for command in runner.commands) == 1


class Runner:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.commands = []
        self.fail_download = False

    def __call__(self, command):
        self.commands.append(command)
        text = "ok"
        if command[:3] == ["uv", "build", "--wheel"]:
            dist = Path(command[command.index("-o") + 1])
            dist.mkdir(parents=True)
            with zipfile.ZipFile(dist / "project.whl", "w") as archive:
                archive.writestr("compare_attention/__init__.py", "")
        elif command[:2] == ["uv", "export"]:
            Path(command[command.index("-o") + 1]).write_text("pyyaml==6.0.3\n", encoding="utf-8")
        elif command[1:3] == ["datasets", "status"]:
            text = "ready"
        elif command[1:3] == ["kernels", "push"]:
            text = "Kernel version 7 successfully pushed. Please check progress at https://example.invalid"
        elif command[1:3] == ["kernels", "status"]:
            text = "complete"
        elif command[1:3] == ["kernels", "output"]:
            if self.fail_download:
                return subprocess.CompletedProcess(command, 1, "", "synthetic failure")
            output = Path(command[command.index("-p") + 1])
            slug = command[3].split("/")[1]
            result = output / slug
            result.mkdir(parents=True, exist_ok=True)
            (result / "model.pt").write_bytes(b"checkpoint")
            (result / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "type": "result",
                        "steps": 1,
                        "train_flops": 12,
                        "quality": {"train_loss": 1.0, "val_loss": 1.0, "val_accuracy": 0.5},
                    }
                )
                + "\n"
            )
            push = next(cmd for cmd in reversed(self.commands) if cmd[1:3] == ["kernels", "push"])
            kernel = Path(push[push.index("-p") + 1]) / "kernel.py"
            tree = ast.parse(kernel.read_text())
            identity = next(
                ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign) and node.targets[0].id == "_IDENTITY"
            )

            receipt = {"identity": identity, "files": file_manifest(output, list(result.iterdir()))}
            (output / "result-manifest.json").write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 0, text, "")


def test_stage_once_all_configs_private_inputs(tmp_path, monkeypatch):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    manifest = kaggle_series.stage_series(
        tmp_path, configs, data, tmp_path / "staged", "user", runner
    )
    again = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    assert manifest == again
    assert sum(cmd[:2] == ["uv", "build"] for cmd in runner.commands) == 1
    assert sum(cmd[:2] == ["uv", "export"] for cmd in runner.commands) == 1
    with zipfile.ZipFile(manifest.parent / "release/release.bundle") as archive:
        assert "experiments/classic/seed_42.yaml" in archive.namelist()
        assert "experiments/deltanet/seed_42.yaml" in archive.namelist()
        assert not any(".env" in name for name in archive.namelist())
    state = json.loads(manifest.read_text())
    assert state["release_id"] != state["data_id"]
    kaggle_series.publish_series(manifest, runner)
    kaggle_series.publish_series(manifest, runner)
    creates = [cmd for cmd in runner.commands if "dataset_create_new" in " ".join(cmd)]
    assert len(creates) == 2
    assert all("public=False" in " ".join(cmd) for cmd in creates)
    assert not any(cmd[1:3] == ["datasets", "version"] for cmd in runner.commands)


def test_gate_selection_resume_download_without_repush(tmp_path, monkeypatch):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    manifest = kaggle_series.stage_series(
        tmp_path, configs, data, tmp_path / "staged", "user", runner
    )
    kaggle_series.publish_series(manifest, runner)
    names = list(json.loads(manifest.read_text())["runs"])
    runner.fail_download = True
    with pytest.raises(RuntimeError, match="output"):
        kaggle_series.run_selected(manifest, names[:1], run=runner, poll_seconds=0)
    state = json.loads(manifest.read_text())
    assert state["runs"][names[0]]["state"] == "completed"
    assert state["runs"][names[0]]["version"] == 7
    assert all(
        cmd[3].endswith("/7")
        for cmd in runner.commands
        if cmd[1:3] in (["kernels", "status"], ["kernels", "output"])
    )
    assert all("--force" in cmd for cmd in runner.commands if cmd[1:3] == ["kernels", "output"])
    assert state["runs"][names[1]]["state"] == "staged"
    runner.fail_download = False
    kaggle_series.run_selected(manifest, names[:1], run=runner, poll_seconds=0)
    assert sum(cmd[1:3] == ["kernels", "push"] for cmd in runner.commands) == 1
    kaggle_series.run_selected(manifest, names[1:], run=runner, poll_seconds=0)
    assert sum(cmd[:2] == ["uv", "build"] for cmd in runner.commands) == 1
    assert sum(cmd[1:3] == ["kernels", "push"] for cmd in runner.commands) == 2
    state = json.loads(manifest.read_text())
    assert all(row["state"] == "verified" for row in state["runs"].values())
    assert {row["phase"] for row in state["timings"]} >= {"build", "export", "stage_data"}
    for entry in state["runs"].values():
        code = (manifest.parent / "runs" / entry["slug"] / "kernel.py").read_text()
        ast.parse(code, feature_version=(3, 13))
        assert "prepare_kv_retrieval" not in code
        assert "public" not in code
        assert "phase" in code


@pytest.mark.parametrize(
    "accelerator, required",
    [
        ("NvidiaTeslaT4", {"name": "Tesla T4", "compute_capability": [7, 5]}),
        ("NvidiaL4", {"name": "NVIDIA L4", "compute_capability": [8, 9]}),
        (
            "NvidiaTeslaA100",
            {"name": "NVIDIA A100-SXM4-40GB", "compute_capability": [8, 0]},
        ),
    ],
)
def test_selected_accelerator_reaches_rendered_runtime_gate(tmp_path, accelerator, required):
    configs, data = make_inputs(tmp_path)
    runner = Runner()
    path = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    kaggle_series.publish_series(path, runner)
    state = json.loads(path.read_text())
    name = next(iter(state["runs"]))
    kaggle_series.run_selected(path, [name], accelerator=accelerator, run=runner, poll_seconds=0)
    push = next(cmd for cmd in runner.commands if cmd[1:3] == ["kernels", "push"])
    assert push[push.index("--accelerator") + 1] == accelerator
    tree = ast.parse((path.parent / "runs" / state["runs"][name]["slug"] / "kernel.py").read_text())
    rendered = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_REQUIRED_GPU"
    )
    assert rendered == required


@pytest.mark.parametrize("accelerator", ["UnknownGPU", "nvidiateslat4", "NvidiaTeslaP100", "None"])
def test_unknown_accelerator_rejected_before_submission(tmp_path, accelerator):
    configs, data = make_inputs(tmp_path)
    runner = Runner()
    path = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    kaggle_series.publish_series(path, runner)
    before = path.read_bytes()
    name = next(iter(json.loads(before)["runs"]))
    runner.commands.clear()
    with pytest.raises(ValueError, match="unsupported.*accelerator"):
        kaggle_series.run_selected(path, [name], accelerator=accelerator, run=runner)
    assert not runner.commands
    assert path.read_bytes() == before
    assert not (path.parent / "runs").exists()


def test_incomplete_stage_is_rebuilt_and_owner_is_not_silently_reused(tmp_path, monkeypatch):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    path = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    state = json.loads(path.read_text())
    del state["ready"]
    del state["artifacts"]
    path.write_text(json.dumps(state))
    kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    with pytest.raises(ValueError, match="owner"):
        kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "other", runner)


def test_kernel_slug_is_immutable_per_series(tmp_path, monkeypatch):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    first = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    (tmp_path / "uv.lock").write_text("changed lock")
    second = kaggle_series.stage_series(
        tmp_path, configs, data, tmp_path / "staged", "user", runner
    )
    assert first != second
    first_slugs = {v["slug"] for v in json.loads(first.read_text())["runs"].values()}
    second_slugs = {v["slug"] for v in json.loads(second.read_text())["runs"].values()}
    assert not first_slugs & second_slugs


def test_truncated_slug_prefix_never_creates_double_hyphen(tmp_path, monkeypatch):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    smokes = []
    for family, config in zip(("classic", "deltanet"), configs, strict=True):
        path = tmp_path / "experiments" / "flops_v2_smoke" / f"{family}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(config.read_bytes())
        smokes.append(path)
    manifest = kaggle_series.stage_series(
        tmp_path, smokes, data, tmp_path / "staged", "user", Runner()
    )
    slugs = [run["slug"] for run in json.loads(manifest.read_text())["runs"].values()]
    assert len(set(slugs)) == 2
    assert all(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) for slug in slugs)


@pytest.mark.parametrize("failure", ["push", "status", "output", "corrupt"])
def test_failure_is_failfast_and_never_submits_second(tmp_path, monkeypatch, failure):
    configs, data = make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = Runner()
    path = kaggle_series.stage_series(tmp_path, configs, data, tmp_path / "staged", "user", runner)
    kaggle_series.publish_series(path, runner)

    def failing(command):
        result = runner(command)
        if command[1:3] == ["kernels", failure]:
            return subprocess.CompletedProcess(command, 1, "", "synthetic failure")
        if failure == "corrupt" and command[1:3] == ["kernels", "output"]:
            output = Path(command[command.index("-p") + 1])
            next(output.rglob("model.pt")).write_bytes(b"corrupted")
        return result

    names = list(json.loads(path.read_text())["runs"])
    with pytest.raises((RuntimeError, ValueError)):
        kaggle_series.run_selected(path, names, run=failing, poll_seconds=0)
    assert sum(cmd[1:3] == ["kernels", "push"] for cmd in runner.commands) == 1
    assert json.loads(path.read_text())["runs"][names[1]]["state"] == "staged"
