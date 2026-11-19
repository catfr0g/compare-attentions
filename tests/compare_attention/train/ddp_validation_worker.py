"""Real CPU/Gloo validation consumer tests with v5 scalar transport fixtures."""

# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"

import torch
import torch.distributed as dist

from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import TrainConfig
from compare_attention.train.distributed import distributed_run
from tests.compare_attention.train.ddp_guards import assert_phase_failure
from tests.compare_attention.train.ddp_worker import resume_proof
from tests.compare_attention.train.test_validation_observation import V5_LOSSES


def main():
    root, mode = Path(sys.argv[1]), sys.argv[2]
    config = TrainConfig.model_validate_json((root / "config.json").read_text())
    rank = int(os.environ["RANK"])
    original = loop._quality  # pylint: disable=protected-access
    evaluations = []

    def quality(model, inputs, labels, batch, losses):
        # Execute the real complete-split evaluator, including weighting/mode changes.
        observed = original(model, inputs, labels, batch, losses)
        assert inputs.size(0) == config.dataset.n_val
        assert batch == config.schedule.batch_size
        evaluations.append({"step": len(losses), "inputs": inputs.tolist()})
        if mode == "validation_failure" and rank == 1:
            raise ValueError("rank one validation failed")
        loss = V5_LOSSES[0 if len(losses) == 2 else 1][rank]
        return observed.model_copy(update={"val_loss": loss})

    loop._quality = quality  # pylint: disable=protected-access

    class CheckedBackup(BackupTracker):
        def log_validation(self, loss, accuracy, flops):
            if mode == "writer_failure":
                raise ValueError("validation writer failed")
            super().log_validation(loss, accuracy, flops)

        def save_checkpoint(self, payload, step, keep_last):
            # Reaching this callback proves the unchanged full shared-state gate passed.
            assert payload["validation_history"][-1]["loss"] == V5_LOSSES[0 if step == 2 else 1][0]
            super().save_checkpoint(payload, step, keep_last)

    if mode == "resume":
        with distributed_run(True):
            resume_proof(root, config, rank)
    else:
        try:
            result = loop.train(config, CheckedBackup(root / "result"), distributed=True)
        except RuntimeError as error:
            expected = {
                "validation_failure": "rank one validation failed",
                "writer_failure": "validation writer failed",
            }
            assert mode in expected
            assert_phase_failure(
                error,
                "validation" if mode == "validation_failure" else "log_validation",
                "ValueError",
                "ddp_validation_worker.py",
                "quality" if mode == "validation_failure" else "log_validation",
                expected[mode],
            )
            assert not dist.is_initialized()
            (root / f"failure-{rank}.txt").write_text(str(error))
        else:
            assert mode == "stop"
            assert result.steps == 4 and result.stop_reason == "early_stopping"
            (root / f"result-{rank}.json").write_text(result.model_dump_json())
    (root / f"evaluations-{rank}.json").write_text(json.dumps(evaluations))
    torch.set_num_threads(1)


if __name__ == "__main__":
    main()
