"""Real Gloo protocol and accepted-wheel differential worker (CPU only)."""

# ruff: noqa: E402
import hashlib
import importlib.util
import json
import os
import random
import sys
import zipfile
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.distributed import DistributedRun, distributed_run
from tests.compare_attention.train.ddp_worker import (
    CpuScalerPrecision,
    InterruptedBackup,
    RandomBackup,
    draws,
)
from tests.compare_attention.train.test_distributed import assert_equal

WHEEL = Path(
    "E:/HermesArtifacts/compare-attention/ddp-cloud-benchmark-v1/"
    "prepared-finite-opt-v1-final/package/compare_attention-0.1.0-py3-none-any.whl"
)
WHEEL_SHA = "7095c368ea2d6a9cec13a97bde7d8ab6722d47a93df4d6791dcdb5ae5ae575e1"
DISTRIBUTED_SHA = "c4be71c833421e875e73c90751b75b04b12d905b8a952d45e06f2e35274ab697"


def reference(root):
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest() == WHEEL_SHA
    with zipfile.ZipFile(WHEEL) as archive:
        source = archive.read("compare_attention/train/distributed.py")
    assert hashlib.sha256(source).hexdigest() == DISTRIBUTED_SHA
    path = root / f"accepted-distributed-{os.environ.get('RANK', 'single')}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("accepted_distributed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DistributedRun.call


def protocol(run, old_call, red=False):
    candidate = DistributedRun.call
    original_objects = torch.distributed.all_gather_object
    objects = 0

    def count(*args, **kwargs):
        nonlocal objects
        objects += 1
        return original_objects(*args, **kwargs)

    torch.distributed.all_gather_object = count
    try:
        # Genuine warm collective calls before measuring the successful path.
        for _ in range(3):
            candidate(run, lambda: None, phase="forward")
        objects = 0
        selected = old_call if red else candidate
        for phase in ("forward", "scaled_optimizer_step", "optimizer_step", "log_loss"):
            selected(run, lambda: 17, phase=phase)
        assert objects == 0, "warm successful hot phases serialized objects"
        rows = []
        for baseline in (True, False):
            call = old_call if baseline else candidate
            errors = []
            for phase_a, phase_b in (
                ("forward", "optimizer_step"),
                ("forward", "unknown-phase"),
                ("unknown-α", "unknown-β"),
            ):
                try:
                    call(run, lambda: None, phase=phase_a if run.rank == 0 else phase_b)
                except RuntimeError as error:
                    assert "phase order mismatch" in str(error)
                    errors.append(str(error))
                else:
                    raise AssertionError("mismatched phases continued")
            for phase in ("forward", "scaled_optimizer_step", "unknown-雪", None):

                def fail():
                    if run.rank == 1:
                        raise ValueError("ошибка 雪 🧪\nsecond line")
                    return 9

                try:
                    call(run, fail, phase=phase)
                except RuntimeError as error:
                    assert "ValueError: ошибка 雪 🧪\nsecond line" in str(error).replace(
                        "\\n", "\n"
                    )
                    errors.append(str(error))
                else:
                    raise AssertionError("failed peer continued")

            def unnamed():
                return 11

            assert call(run, unnamed) == 11
            assert call(run, unnamed, phase="unknown-雪") == 11
            assert call(run, unnamed, primary=True, phase="log_loss") == (
                11 if run.rank == 0 else None
            )
            # Compare exact rank-ordered diagnostics before the next case.
            peer_errors = run.gather(errors)
            assert peer_errors[0] == peer_errors[1]
            rows.append(errors)
        assert rows[0] == rows[1]
        return {
            "warm_hot_object_calls": 0,
            "diagnostic_cases": len(rows[0]),
            "exact_old_errors": True,
        }
    finally:
        torch.distributed.all_gather_object = original_objects


def trajectory(root, config, run, old_call):
    candidate = DistributedRun.call
    gradients = []
    original_step = torch.optim.AdamW.step
    original_precision = loop.Precision

    def record(optimizer, *args, **kwargs):
        gradients.append(
            [
                p.grad.detach().clone() if p.grad is not None else None
                for group in optimizer.param_groups
                for p in group["params"]
            ]
        )
        return original_step(optimizer, *args, **kwargs)

    class CpuPolicy(CpuScalerPrecision):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    try:
        torch.optim.AdamW.step = record
        loop.Precision = CpuPolicy
        for name in (
            "old-full",
            "old-partial",
            "old-resumed",
            "new-full",
            "new-partial",
            "new-resumed",
        ):
            DistributedRun.call = old_call if name.startswith("old") else candidate
            seed = 999 if name.endswith("resumed") else 123
            random.seed(seed + run.rank)
            np.random.seed(seed + run.rank)
            torch.manual_seed(seed + run.rank)
            gradients.clear()
            prefix, kind = name.split("-")
            sink = (InterruptedBackup if kind == "partial" else RandomBackup)(root / name)
            try:
                result = loop.train(
                    config,
                    sink,
                    distributed=True,
                    resume_from=root / f"{prefix}-partial/checkpoints/latest.json"
                    if kind == "resumed"
                    else None,
                )
            except RuntimeError as error:
                assert kind == "partial" and "intentional durable interruption" in str(error)
                assert len(gradients) == 2
            else:
                assert kind != "partial" and result.steps == 7
                assert result.provenance["precision"]["optimizer_steps"] == 7
                assert len(gradients) == (5 if kind == "resumed" else 7)
                torch.save(draws(), root / f"{name}-next-{run.rank}.pt")
            torch.save(gradients, root / f"{name}-gradients-{run.rank}.pt")
        fields = (
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
            "distributed",
        )
        expected = load_checkpoint(root / "old-full/checkpoints/latest.json")
        for name in ("old-resumed", "new-full", "new-resumed"):
            actual = load_checkpoint(root / name / "checkpoints/latest.json")
            for field in fields:
                assert_equal(expected[field], actual[field])
            assert_equal(
                torch.load(root / f"old-full-next-{run.rank}.pt", weights_only=True),
                torch.load(root / f"{name}-next-{run.rank}.pt", weights_only=True),
            )

        def grads(name):
            return torch.load(root / f"{name}-gradients-{run.rank}.pt", weights_only=True)

        assert_equal(grads("old-full"), grads("new-full"))
        for prefix in ("old", "new"):
            assert_equal(grads("old-full"), grads(f"{prefix}-partial") + grads(f"{prefix}-resumed"))
        return {
            "updates": 7,
            "equal_fields": list(fields),
            "gradients_bitwise": True,
            "next_rng_bitwise": True,
        }
    finally:
        DistributedRun.call = candidate
        torch.optim.AdamW.step = original_step
        loop.Precision = original_precision


def failures(root, config, run, old_call):
    candidate = DistributedRun.call
    original_loss = loop._loss  # pylint: disable=protected-access
    original_step = torch.optim.AdamW.step
    original_precision = loop.Precision
    counts = {"attempts": 0, "updates": 0}

    class CpuPolicy(CpuScalerPrecision):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    mode = None

    class Sink(BackupTracker):
        def log_loss(self, loss, flops):
            assert run.rank == 0, "nonprimary writer invoked"
            super().log_loss(loss, flops)
            if mode == "writer":
                raise OSError("writer 雪 failed")

    def loss(*args, **kwargs):
        if mode == "forward" and run.rank == 1:
            raise ValueError("forward 雪 failed")
        return original_loss(*args, **kwargs)

    def step(optimizer, *args, **kwargs):
        counts["attempts"] += 1
        if mode in ("optimizer", "scaled_optimizer") and run.rank == 1:
            raise ValueError(f"{mode} 雪 failed")
        return original_step(optimizer, *args, **kwargs)

    def updated(*_args):
        counts["updates"] += 1

    handle = register_optimizer_step_post_hook(updated)
    proofs = []
    try:
        loop._loss = loss  # pylint: disable=protected-access
        torch.optim.AdamW.step = step
        loop.Precision = CpuPolicy
        plain = config.model_copy(
            update={"schedule": config.schedule.model_copy(update={"dtype": "float32"})}
        )
        for mode in ("forward", "optimizer", "scaled_optimizer", "writer", "callback"):
            errors = []
            for baseline in (True, False):
                DistributedRun.call = old_call if baseline else candidate
                counts.update(attempts=0, updates=0)
                try:
                    if mode == "callback":

                        def callback():
                            if run.rank == 1:
                                raise RuntimeError("callback 雪 failed")

                        run.call(callback, phase="log_loss")
                        raise AssertionError("callback failure continued")
                    loop.train(
                        config if mode == "scaled_optimizer" else plain,
                        Sink(root / f"{mode}-{baseline}"),
                        distributed=True,
                    )
                except RuntimeError as error:
                    message = str(error)
                    assert f"{mode} 雪 failed" in message
                    peers = run.gather(message)
                    assert peers[0] == peers[1]
                    errors.append(message)
                else:
                    raise AssertionError("failed training continued")
                expected = (
                    0
                    if mode in ("forward", "callback")
                    else (0 if mode in ("optimizer", "scaled_optimizer") and run.rank == 1 else 1)
                )
                assert counts["updates"] == expected
                assert counts["attempts"] == (0 if mode in ("forward", "callback") else 1)
            assert errors[0] == errors[1]
            proofs.append({"mode": mode, "rank": run.rank, **counts, "error": errors[1]})
        return proofs
    finally:
        handle.remove()
        DistributedRun.call = candidate
        loop._loss = original_loss  # pylint: disable=protected-access
        torch.optim.AdamW.step = original_step
        loop.Precision = original_precision


def single_control(root, config, old_call):
    """The isolated method change must not enter single-device execution."""
    candidate = DistributedRun.call
    original_precision = loop.Precision

    class CpuPolicy(CpuScalerPrecision):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    results = []
    calls = 0

    try:
        loop.Precision = CpuPolicy
        for name, selected in (("single-old", old_call), ("single-new", candidate)):

            def tracked(*args, _selected=selected, **kwargs):
                nonlocal calls
                calls += 1
                return _selected(*args, **kwargs)

            DistributedRun.call = tracked
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            result = loop.train(config, RandomBackup(root / name))
            assert result.steps == 7
            results.append((load_checkpoint(root / name / "checkpoints/latest.json"), draws()))
        fields = (
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
        )
        for field in fields:
            assert_equal(results[0][0][field], results[1][0][field])
        assert_equal(results[0][1], results[1][1])
        assert calls == 0
        return {
            "updates_each": 7,
            "distributed_calls": calls,
            "equal_fields": list(fields),
            "next_rng_bitwise": True,
        }
    finally:
        DistributedRun.call = candidate
        loop.Precision = original_precision


def main():
    from compare_attention.train.config import TrainConfig

    root = Path(sys.argv[1])
    old_call = reference(root)
    config = TrainConfig.model_validate_json((root / "config.json").read_text())
    with distributed_run(True) as run:
        proof = {"protocol": protocol(run, old_call, red=len(sys.argv) > 2)}
        proof["trajectory"] = trajectory(root, config, run, old_call)
        proof["failures"] = failures(root, config, run, old_call)
    assert not torch.distributed.is_initialized()
    proof.update(group_destroyed=True, accepted_wheel_sha256=WHEEL_SHA)
    (root / f"phase-proof-{os.environ['RANK']}.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
