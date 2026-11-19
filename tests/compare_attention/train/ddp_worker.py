"""CPU-only torchrun engineering worker; never imported by production."""

# Mask must be checked before third-party imports, including NumPy and Torch.
# ruff: noqa: E402

import faulthandler
import os
import random
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"

import numpy as np
import torch

from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import TrainConfig
from compare_attention.train.distributed import distributed_run
from compare_attention.train.lifecycle import primary_failure
from compare_attention.train.loop import train
from compare_attention.train.precision import Precision


class AfterUpdateBackup(BackupTracker):
    def __init__(self, directory, root):
        super().__init__(directory)
        self.root, self.steps = root, 0

    def start(self, config):
        super().start(config)
        torch.manual_seed(9999)  # Rank-zero tracker-induced reseed must be undone.

    def log_loss(self, loss, flops):
        self.steps += 1
        assert [(self.root / f"updated-{rank}.txt").read_text() for rank in range(2)] == [
            str(self.steps),
            str(self.steps),
        ]
        super().log_loss(loss, flops)


def draws():
    return {"python": random.random(), "numpy": float(np.random.random()), "torch": torch.rand(4)}


class RandomBackup(BackupTracker):
    def log_loss(self, loss, flops):
        draws()
        super().log_loss(loss, flops)

    def log_validation(self, loss, accuracy, flops):
        draws()
        super().log_validation(loss, accuracy, flops)


class InterruptedBackup(RandomBackup):
    def save_checkpoint(self, payload, step, keep_last):
        super().save_checkpoint(payload, step, keep_last)
        if step == 2:
            raise RuntimeError("intentional durable interruption")


class CpuScalerPrecision(Precision):
    """Exercise real scaler/unscale coordination on CPU, not a CUDA FP16 claim."""

    def __init__(self):
        self.scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2)
        self.max_scale = 128.0
        self.optimizer_steps = 0
        self.skipped_optimizer_steps = 0

    def autocast(self):
        from contextlib import nullcontext

        return nullcontext()


def resume_proof(root, config, rank):
    for name, seed in (("full", 123), ("partial", 123), ("resumed", 999)):
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        sink = (InterruptedBackup if name == "partial" else RandomBackup)(root / name)
        try:
            result = train(
                config,
                sink,
                distributed=True,
                resume_from=root / "partial/checkpoints/latest.json" if name == "resumed" else None,
            )
        except RuntimeError as error:
            assert name == "partial"
            failure = primary_failure(error, "training")
            assert failure["phase"] == "save_checkpoint"
            assert failure["exception_class"] == "RuntimeError"
            assert any(
                frame["basename"] == "ddp_worker.py" and frame["function"] == "save_checkpoint"
                for frame in failure["frames"]
            )
            assert "intentional durable interruption" not in str(error)
        else:
            assert name != "partial"
            torch.save(draws(), root / f"{name}-next-{rank}.pt")
            (root / f"{name}-result-{rank}.json").write_text(result.model_dump_json())


def main():
    root = Path(sys.argv[1])
    config = TrainConfig.model_validate_json((root / "config.json").read_text())
    rank = int(os.environ["RANK"])
    debug = (root / f"stack-{rank}.txt").open("w")
    faulthandler.enable(debug)
    faulthandler.dump_traceback_later(30, file=debug)
    if len(sys.argv) > 2 and sys.argv[2].startswith("stopping:"):
        from tests.compare_attention.train.ddp_guards import stopping

        stopping(root, config, sys.argv[2].split(":")[1])
        return
    if len(sys.argv) > 2 and sys.argv[2] == "phase":
        with distributed_run(True) as run:
            try:
                run.call(lambda: None, phase=f"different-{rank}")
            except RuntimeError as error:
                assert "phase order mismatch" in str(error)
            else:
                raise AssertionError("inconsistent collective phases accepted")
        return
    if len(sys.argv) > 2 and sys.argv[2].startswith("failure:"):
        from tests.compare_attention.train.ddp_guards import failures

        failures(root, config, sys.argv[2].split(":")[1])
        return
    if len(sys.argv) > 2 and sys.argv[2] == "guards":
        from tests.compare_attention.train.ddp_guards import resume_guards

        resume_guards(root, config)
        return
    if len(sys.argv) > 2 and sys.argv[2] == "scaler":
        precision = CpuScalerPrecision()
        with precision.activate():
            result = train(config, BackupTracker(root / "scaled"), distributed=True)
        assert precision.optimizer_steps == result.steps == 4
        assert precision.skipped_optimizer_steps == 0
        assert precision.scaler.get_scale() == 128.0
        return
    if len(sys.argv) > 2 and sys.argv[2] == "cli":
        from compare_attention.cli.train import main as cli_main

        code = cli_main(["--config", str(root / "config.yaml"), "--distributed"])
        (root / f"cli-exit-{rank}.txt").write_text(str(code))
        return
    if len(sys.argv) > 2 and sys.argv[2] == "resume":
        with distributed_run(True):
            resume_proof(root, config, rank)
        return
    from torch.optim.optimizer import register_optimizer_step_post_hook

    from compare_attention.train import loop

    traces, updates = [], 0
    original = loop._step  # pylint: disable=protected-access

    def record(model, optimizer, inputs, labels):
        traces.append({"inputs": inputs.tolist(), "labels": labels.tolist()})
        return original(model, optimizer, inputs, labels)

    def updated(_optimizer, _args, _kwargs):
        nonlocal updates
        updates += 1
        (root / f"updated-{rank}.txt").write_text(str(updates))

    handle = register_optimizer_step_post_hook(updated)
    loop._step = record  # pylint: disable=protected-access
    try:
        result = train(config, AfterUpdateBackup(root / "distributed", root), distributed=True)
    finally:
        handle.remove()
        loop._step = original  # pylint: disable=protected-access
    import json

    (root / f"trace-{rank}.json").write_text(json.dumps(traces))
    (root / f"rank-{rank}.json").write_text(result.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
