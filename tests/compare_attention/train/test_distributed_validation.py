"""Real two-process validation agreement, stopping, recovery and failure guards."""

import json

import pytest
import torch

from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.flops import estimate_step
from tests.compare_attention.train.test_distributed import assert_equal, fixture, launch
from tests.compare_attention.train.test_validation_observation import V5_LOSSES

WORKER = "tests.compare_attention.train.ddp_validation_worker"


def configure(root, mode):
    config = fixture(root, dropout=0.2)
    schedule = {
        "checkpoint_every_steps": 2,
        "checkpoint_keep_last": 2,
        "train_flops": estimate_step(config).total * 7,
        "val_every_steps": 2,
    }
    if mode == "stop":
        # Exact v5 observations straddle this strict improvement threshold:
        # rank zero stops at step 4, rank one would continue without agreement.
        improvements = [V5_LOSSES[0][r] - V5_LOSSES[1][r] for r in range(2)]
        schedule.update(
            early_stopping_patience=1,
            early_stopping_min_steps=0,
            early_stopping_min_delta=sum(improvements) / 2,
        )
        assert improvements[0] < schedule["early_stopping_min_delta"] < improvements[1]
    config = config.model_copy(update={"schedule": config.schedule.model_copy(update=schedule)})
    (root / "config.json").write_text(config.model_dump_json())
    return config


def assert_full_split_evaluations(root):
    observations = [
        json.loads((root / f"evaluations-{rank}.json").read_text()) for rank in range(2)
    ]
    assert observations[0] == observations[1]
    assert observations[0]
    assert all(len(row["inputs"]) == 3 for row in observations[0])


def test_real_gloo_v5_disagreement_stops_before_rank_decisions_diverge(tmp_path):
    configure(tmp_path, "stop")
    launch(tmp_path, str(tmp_path), "stop", module=WORKER)
    results = [json.loads((tmp_path / f"result-{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["early_stopping"] == results[1]["early_stopping"]
    assert results[0]["quality"] == results[1]["quality"]
    assert results[0]["early_stopping"]["best_loss"] == V5_LOSSES[0][0]
    checkpoint = load_checkpoint(tmp_path / "result/checkpoints/latest.json")
    assert checkpoint["step"] == 4
    assert [row["loss"] for row in checkpoint["validation_history"]] == [x[0] for x in V5_LOSSES]
    metrics = [
        json.loads(line) for line in (tmp_path / "result/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["loss"] for row in metrics if row["type"] == "validation"] == [
        x[0] for x in V5_LOSSES
    ]
    assert_full_split_evaluations(tmp_path)


def test_real_gloo_v5_observation_checkpoint_resume_is_bitwise(tmp_path):
    configure(tmp_path, "resume")
    launch(tmp_path, str(tmp_path), "resume", module=WORKER)
    full = load_checkpoint(tmp_path / "full/checkpoints/latest.json")
    resumed = load_checkpoint(tmp_path / "resumed/checkpoints/latest.json")
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
        "distributed",
    ):
        assert_equal(full[key], resumed[key])
    assert full["step"] == 7
    assert [row["loss"] for row in full["validation_history"]] == [
        V5_LOSSES[0][0],
        V5_LOSSES[1][0],
        V5_LOSSES[1][0],
        V5_LOSSES[1][0],
    ]
    for rank in range(2):
        assert_equal(
            torch.load(tmp_path / f"full-next-{rank}.pt", weights_only=True),
            torch.load(tmp_path / f"resumed-next-{rank}.pt", weights_only=True),
        )
    assert_full_split_evaluations(tmp_path)


@pytest.mark.parametrize("mode", ["validation_failure", "writer_failure"])
def test_validation_or_writer_failure_remains_collective_before_checkpoint(tmp_path, mode):
    configure(tmp_path, mode)
    launch(tmp_path, str(tmp_path), mode, module=WORKER)
    assert [(tmp_path / f"failure-{rank}.txt").is_file() for rank in range(2)] == [True, True]
    assert not (tmp_path / "result/checkpoints/latest.json").exists()
    metrics = [
        json.loads(line) for line in (tmp_path / "result/metrics.jsonl").read_text().splitlines()
    ]
    assert not [row for row in metrics if row["type"] == "validation"]
