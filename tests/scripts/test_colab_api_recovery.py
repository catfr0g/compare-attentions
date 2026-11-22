"""Auxiliary recovery contracts; no account, GPU or service credentials."""

# Keep ML imports inside the one native integration test; plans remain stdlib-only.
# pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements,duplicate-code

import dataclasses
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
controller = importlib.import_module("colab_api_run")


def test_immutable_recovery_spec_and_zero_io_plan(monkeypatch, capsys):
    recovery = importlib.import_module("colab_api_recovery")
    spec = recovery.RecoverySpec("runtimes/new-runtime", "new-recovery-run")
    assert spec.value()["origin_run"] == "deltanet-k64-seed42-colab07b"
    assert spec.value()["step"] == 46500
    assert spec.value()["bytes"] == 27788535
    assert spec.value()["target_updates"] == 47265
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.run_name = "changed"

    def forbidden(*_args, **_kwargs):
        pytest.fail("plan performed I/O")

    monkeypatch.setattr(controller.auth, "access_token", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    for command in ("restore", "resume"):
        assert (
            controller.main([command, "--runtime", spec.runtime, "--run-name", spec.run_name]) == 0
        )
        plan = json.loads(capsys.readouterr().out)
        assert plan["training_started"] is False
        assert plan["recovery_sha256"] == spec.sha256()
    with pytest.raises(ValueError, match="RECOVERY_IDENTITY"):
        recovery.RecoverySpec("runtimes/new-runtime", "deltanet-k64-seed42-colab07b")


def test_recovery_binding_preserved_and_fresh_train_blocked_before_auth(tmp_path, monkeypatch):
    recovery = importlib.import_module("colab_api_recovery")
    spec = recovery.RecoverySpec("runtimes/new-runtime", "new-recovery-run")
    binding = {
        "runtime": spec.runtime,
        "run_name": spec.run_name,
        "sha256": recovery.ARCHIVE_SHA,
        "recovery_sha256": spec.sha256(),
    }
    journal = tmp_path / "setup.jsonl"
    journal.write_text(json.dumps(binding) + "\n")
    assert controller.load_binding(journal, spec.runtime, spec.run_name) == binding

    def forbidden(*_args, **_kwargs):
        pytest.fail("recovery fresh train reached auth")

    monkeypatch.setattr(controller.auth, "access_token", forbidden)
    assert (
        controller.main(
            [
                "train",
                "--execute",
                "--runtime",
                spec.runtime,
                "--run-name",
                spec.run_name,
                "--setup-journal",
                str(journal),
                "--journal",
                str(tmp_path / "train.jsonl"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "train.jsonl").exists()


def test_native_semantic_restore_uses_original_validators_without_updates(tmp_path, monkeypatch):
    import torch

    from compare_attention.data.store import prepare_kv_retrieval
    from compare_attention.models.gpt2 import GPT2
    from compare_attention.train import loop
    from compare_attention.train.checkpoint import load_checkpoint
    from compare_attention.train.flops import estimate_step
    from tests.compare_attention.data.factories import config as data_config
    from tests.compare_attention.models.factories import gpt2_config
    from tests.compare_attention.train.factories import train_config
    from tests.compare_attention.train.test_resume_equivalence import (
        _assert_nested_equal,
        _InterruptedBackup,
        _next_random_values,
        _RandomBackup,
        _SnapshotInterrupt,
    )

    recovery = importlib.import_module("colab_api_recovery")
    namespace = {}
    exec(recovery.NATIVE_SOURCE, namespace)  # pylint: disable=exec-used
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    data = tmp_path / "data"
    prepare_kv_retrieval(
        data_config(vocab_size=32, input_seq_len=16, num_kv_pairs=2, n_train=8, n_val=3), data
    )
    config = train_config(
        data,
        model=gpt2_config(
            vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.2
        ),
        val_every_steps=2,
        checkpoint_every_steps=2,
        checkpoint_keep_last=3,
    )
    cost = estimate_step(config).total
    config = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"train_flops": cost * 6})}
    )
    import random

    import numpy as np

    random.seed(123)
    np.random.seed(123)
    full = tmp_path / "full"
    loop.train(config, _RandomBackup(full, False))
    expected_rng = _next_random_values(False)
    expected = load_checkpoint(full / "checkpoints/latest.json")
    random.seed(123)
    np.random.seed(123)
    lost = tmp_path / "lost"
    with pytest.raises(_SnapshotInterrupt):
        loop.train(config, _InterruptedBackup(lost, False))
    snapshot = lost / "checkpoints/step_00000002.pt"
    saved = load_checkpoint(snapshot)
    (lost / "metrics.jsonl").unlink()
    (lost / "run_identity.json").unlink()
    pin = {
        "step": 2,
        "flops": 2 * cost,
        "cursor": 4,
        "step_flops": cost,
        "budget": 6 * cost,
        "target_updates": 6,
    }

    def forbidden(*_args, **_kwargs):
        pytest.fail("semantic preflight attempted a forward/update/tracker")

    with monkeypatch.context() as patch:
        patch.setattr(GPT2, "forward", forbidden)
        patch.setattr(torch.optim.AdamW, "step", forbidden)
        patch.setattr(loop, "train", forbidden)
        measured = namespace["validate_native"](config, snapshot, pin)
    assert measured["step"] == 2
    assert measured["remaining_updates"] == 4
    assert measured["identity"] == saved["identity"]
    assert measured["history_counts"] == {"losses": 2, "validations": 1}
    # Genuine native validator rejects corruptions; rehashing a forged fixture
    # deliberately bypasses only the outer checksum, never compatibility checks.
    import copy

    from compare_attention.train import checkpoint

    corruptions = {
        "config": lambda p: p["config"]["optim"].update(learning_rate=0.4),
        "runtime": lambda p: p["identity"]["runtime"].update(python="wrong"),
        "initialization": lambda p: p["identity"].update(initial_parameters_sha256="f" * 64),
        "accounting": lambda p: p["identity"]["accounting"].update(total=1),
        "model": lambda p: p["model"].pop(next(iter(p["model"]))),
        "optimizer": lambda p: p.update(optimizer_class="wrong"),
        "precision": lambda p: p.update(precision={}),
        "rng": lambda p: p["rng"].update(torch=torch.zeros(1, dtype=torch.uint8)),
        "cursor": lambda p: p.update(cursor=3),
        "history": lambda p: p.update(validation_history=[]),
        "integrity": lambda p: p.update(integrity_sha256="0" * 64),
    }
    for name, corrupt in corruptions.items():
        candidate = copy.deepcopy(saved)
        corrupt(candidate)
        if name != "integrity":
            candidate["integrity_sha256"] = checkpoint._digest(candidate)  # pylint: disable=protected-access
        bad = tmp_path / (name + ".pt")
        torch.save(candidate, bad)
        with pytest.raises((ValueError, RuntimeError)):
            namespace["validate_native"](config, bad, pin)
    # Exercise the generated frozen-parser wrapper too, not only its validator.
    import yaml
    from colab_package import CONFIGS

    config_path = tmp_path / "release" / CONFIGS["deltanet"]
    config_path.parent.mkdir(parents=True)
    raw = config.model_dump(mode="json", exclude={"dataset"})
    config_path.write_text(yaml.safe_dump(raw))
    (tmp_path / "restore").mkdir()
    native_context = {
        "paths": {"package": str(tmp_path), "snapshot": str(snapshot), "root": str(tmp_path)},
        "spec": pin,
    }
    with monkeypatch.context() as patch:
        patch.setattr(GPT2, "forward", forbidden)
        patch.setattr(torch.optim.AdamW, "step", forbidden)
        exec(recovery.native_program(native_context), {})  # pylint: disable=exec-used
    semantic = json.loads((tmp_path / "restore/semantic.json").read_text())
    assert semantic["integrity_sha256"] == saved["integrity_sha256"]
    assert semantic["remaining_updates"] == 4
    # Genuine native replay from the snapshot, with both original raw files absent.
    resumed = tmp_path / "resumed"
    updates = []
    original_step = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        updates.append(1)
        return original_step(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", counted_step)
        loop.train(config, _RandomBackup(resumed, False), resume_from=snapshot)
    assert len(updates) == 4  # Native replay emits two inherited rows, not updates.
    actual_rng = _next_random_values(False)
    actual = load_checkpoint(resumed / "checkpoints/latest.json")
    for key in (
        "model",
        "optimizer",
        "precision",
        "rng",
        "cursor",
        "losses",
        "validation_history",
        "early_stopping",
        "step",
        "flops",
    ):
        _assert_nested_equal(expected[key], actual[key], key)
    _assert_nested_equal(expected_rng, actual_rng)
    rows = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [r["loss"] for r in rows if r["type"] == "step"] == expected["losses"]
    assert [r for r in rows if r["type"] == "validation"] == expected["validation_history"]
    assert not (lost / "metrics.jsonl").exists()
    assert not (lost / "run_identity.json").exists()


def test_generated_restore_calls_pinned_service_and_publishes_bound_proof(tmp_path, monkeypatch):
    import hashlib
    import subprocess

    recovery = importlib.import_module("colab_api_recovery")
    spec = recovery.RecoverySpec("runtimes/new-runtime", "new-recovery-run")
    binding = {
        "runtime": spec.runtime,
        "run_name": spec.run_name,
        "sha256": recovery.ARCHIVE_SHA,
        "recovery_sha256": spec.sha256(),
    }
    paths = {
        key: str(tmp_path / key)
        for key in ("package", "venv", "private", "archive", "result", "root")
    }
    paths["root"] = str(tmp_path / "jobs" / spec.run_name)
    context = recovery.context(binding, **paths)
    helper = recovery.restore_source(context)
    compile(helper, "<restore-helper>", "exec")
    # Exercise the actual generated program with synthetic file bytes and transport
    # boundary only. Semantic validator is exercised genuinely in the native test.
    payload = b"synthetic opaque snapshot"
    value = context["spec"]
    value["snapshot_sha256"] = hashlib.sha256(payload).hexdigest()
    value["bytes"] = len(payload)
    # Test-only regeneration, never exposed by CLI/spec.
    helper = recovery.restore_source(context)
    root = Path(paths["root"])
    (root / "restore").mkdir(parents=True)
    helper_path = root / "restore/recovery.py"
    helper_path.write_text(helper, encoding="utf-8", newline="\n")
    package = Path(paths["package"])
    (package / "tools").mkdir(parents=True)
    archive = Path(paths["archive"])
    archive.write_bytes(b"synthetic archive")
    (package / "manifest.json").write_bytes(b"{}")
    calls = []
    runner = importlib.import_module("colab_runner")
    package_module = importlib.import_module("colab_package")
    monkeypatch.setattr(runner, "preflight", lambda: None)
    monkeypatch.setattr(package_module, "verify_directory", lambda _p: {})
    # Pins are changed only in a test-only context, not RecoverySpec.
    value["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    value["manifest_sha256"] = hashlib.sha256(b"{}").hexdigest()
    helper = recovery.restore_source(context)
    helper_path.write_text(helper, encoding="utf-8", newline="\n")

    def transport(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        assert "HF_TOKEN" not in kwargs["env"]
        if "restore" in command:
            assert command[1] == str(package / "tools/colab_checkpoint_backup.py")
            assert command[command.index("--revision") + 1] == spec.value()["revision"]
            target = Path(command[command.index("--target") + 1])
            target.mkdir()
            row = {
                "step": value["step"],
                "path": "step_00046500.pt",
                "sha256": value["snapshot_sha256"],
            }
            (target / row["path"]).write_bytes(payload)
            (target / "latest.json").write_text(json.dumps(row))
            identity = {
                "schema_version": 1,
                "run": value["origin_run"],
                **{key: value[key] for key in ("source_sha256", "data_fingerprint")},
                "package_manifest_sha256": value["manifest_sha256"],
            }
            (target / "backup.json").write_text(
                json.dumps({"identity": identity, "nonce": "a" * 32, "owned": [row]})
            )
        else:
            assert command[1:3] == ["-B", "-c"]
            assert recovery.NATIVE_SOURCE in command[3]
            output = root / "restore/semantic.json"
            output.write_text(json.dumps({"remaining_updates": 765, "step": 46500}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", transport)
    exec(compile(helper, str(helper_path), "exec"), {"__file__": str(helper_path)})  # pylint: disable=exec-used
    proof = json.loads((root / "restore/proof.json").read_text())
    assert len(calls) == 2
    assert proof["verified"] is True
    assert proof["binding"] == binding
    assert proof["spec"] == value
    assert proof["helper_sha256"] == hashlib.sha256(helper.encode()).hexdigest()
    assert not Path(paths["result"]).exists()
    approved = hashlib.sha256((root / "restore/proof.json").read_bytes()).hexdigest()
    guard = recovery.resume_guard(context, proof["helper_sha256"], approved)
    exec(compile(guard, "<resume-guard>", "exec"), {})  # pylint: disable=exec-used
    (root / "recovery-input/step_00046500.pt").write_bytes(b"tampered")
    with pytest.raises((RuntimeError, ValueError), match="SNAPSHOT"):
        exec(compile(guard, "<resume-guard>", "exec"), {})  # pylint: disable=exec-used
    assert len(calls) == 2  # Guard never invokes transport or training.
