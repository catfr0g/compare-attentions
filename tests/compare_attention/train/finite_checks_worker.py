"""Real CPU/Gloo AMP-policy differential and synchronized-abort worker."""

# ruff: noqa: E402
import json
import os
import random
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.config import TrainConfig
from compare_attention.train.distributed import distributed_run
from tests.compare_attention.train import finite_reference
from tests.compare_attention.train.ddp_guards import assert_phase_failure
from tests.compare_attention.train.ddp_worker import (
    CpuScalerPrecision,
    InterruptedBackup,
    RandomBackup,
    draws,
)
from tests.compare_attention.train.test_distributed import assert_equal


class CpuPolicy(CpuScalerPrecision):
    """Real CPU GradScaler and FP32 math, never claimed as CUDA FP16 proof."""

    def __init__(self, *_args, **_kwargs):
        super().__init__()


class LegacyCpuPolicy(CpuPolicy):
    backward_step = finite_reference.Precision.backward_step


def trajectories(root, config, rank):
    for name, policy_class in (
        ("legacy", LegacyCpuPolicy),
        ("candidate", CpuPolicy),
        ("partial", CpuPolicy),
        ("resumed", CpuPolicy),
    ):
        seed = 999 if name == "resumed" else 123
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        loop.Precision = policy_class
        sink = (InterruptedBackup if name == "partial" else RandomBackup)(root / name)
        try:
            result = loop.train(
                config,
                sink,
                distributed=True,
                resume_from=root / "partial/checkpoints/latest.json" if name == "resumed" else None,
            )
        except RuntimeError as error:
            assert name == "partial"
            assert_phase_failure(
                error,
                "save_checkpoint",
                "RuntimeError",
                "ddp_worker.py",
                "save_checkpoint",
                "intentional durable interruption",
            )
        else:
            assert name != "partial"
            assert result.steps == 7
            assert result.provenance["precision"]["optimizer_steps"] == 7
            torch.save(draws(), root / f"{name}-next-{rank}.pt")
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
    reference = load_checkpoint(root / "legacy/checkpoints/latest.json")
    for name in ("candidate", "resumed"):
        actual = load_checkpoint(root / name / "checkpoints/latest.json")
        for field in fields:
            assert_equal(reference[field], actual[field])
        assert_equal(
            torch.load(root / f"legacy-next-{rank}.pt", weights_only=True),
            torch.load(root / f"{name}-next-{rank}.pt", weights_only=True),
        )
    return {"updates": 7, "equal_fields": list(fields), "next_rng_equal": True}


def failures(root, config, run):
    proofs = []
    original_model, original_loss = loop.GPT2, loop._loss  # pylint: disable=protected-access
    updates = 0

    def updated(*_args, **_kwargs):
        nonlocal updates
        updates += 1

    handle = register_optimizer_step_post_hook(updated)
    try:
        for name in ("scaled-gradient", "scaled-loss", "unscaled-gradient"):
            policy_rows = []
            for baseline in (True, False):
                policy = (LegacyCpuPolicy if baseline else CpuPolicy)()
                if name == "scaled-gradient":
                    original_unscale = policy.scaler.unscale_

                    def unscale(optimizer, original=original_unscale):
                        original(optimizer)
                        if run.rank == 1:
                            # Poison only rank one's last gradient AFTER DDP reduction.
                            optimizer.param_groups[-1]["params"][-1].grad.fill_(float("nan"))

                    policy.scaler.unscale_ = unscale
                if name == "unscaled-gradient":

                    def model(model_config):
                        instance = original_model(model_config)
                        if run.rank == 1:
                            list(instance.parameters())[-1].register_hook(
                                lambda gradient: gradient * float("nan")
                            )
                        return instance

                    loop.GPT2 = model
                if name == "scaled-loss":

                    def loss(*args, **kwargs):
                        value = original_loss(*args, **kwargs)
                        return value + float("nan") if run.rank == 1 else value

                    loop._loss = loss  # pylint: disable=protected-access
                loop.Precision = lambda *_args, _policy=policy, **_kwargs: _policy
                case_config = (
                    config
                    if name != "unscaled-gradient"
                    else config.model_copy(
                        update={"schedule": config.schedule.model_copy(update={"dtype": "float32"})}
                    )
                )
                path = root / f"{name}-{'legacy' if baseline else 'candidate'}"
                try:
                    loop.train(case_config, BackupTracker(path), distributed=True)
                except RuntimeError as error:
                    message = str(error)
                    contract = message
                    if name == "scaled-gradient":
                        failure = assert_phase_failure(
                            error,
                            "scaled_optimizer_step",
                            "RuntimeError",
                            "distributed.py",
                            "require",
                            "nonfinite scaled gradients; optimizer update aborted",
                        )
                        policy_file = "finite_reference.py" if baseline else "precision.py"
                        assert any(
                            frame["basename"] == policy_file
                            and frame["function"] == "backward_step"
                            for frame in failure["frames"]
                        )
                        # Distinct implementations have distinct traceback frames. Compare
                        # the exact public failure identity, after checking each origin.
                        contract = {
                            key: failure[key] for key in ("schema", "phase", "exception_class")
                        }
                    else:
                        assert "nonfinite" in message
                    assert len(set(run.gather(message))) == 1
                    policy_rows.append(contract)
                else:
                    raise AssertionError("nonfinite rank accepted")
                assert updates == 0
                assert policy.optimizer_steps == policy.skipped_optimizer_steps == 0
                assert policy.scaler.get_scale() == 128.0
                assert not (path / "metrics.jsonl").exists()
                loop.GPT2, loop._loss = original_model, original_loss
            assert policy_rows[0] == policy_rows[1]
            proofs.append({"mode": name, "optimizer_steps": updates, "error": policy_rows[1]})
    finally:
        handle.remove()
        loop.GPT2, loop._loss = original_model, original_loss
    return proofs


def main():
    root = Path(sys.argv[1])
    config = TrainConfig.model_validate_json((root / "config.json").read_text())
    original_precision = loop.Precision
    try:
        with distributed_run(True) as run:
            proof = trajectories(root, config, run.rank)
            proof["failures"] = failures(root, config, run)
    finally:
        loop.Precision = original_precision
    assert not torch.distributed.is_initialized()
    proof["group_destroyed"] = True
    (root / f"finite-proof-{os.environ['RANK']}.json").write_text(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
