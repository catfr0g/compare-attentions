"""Real trainer LR updates, persisted FLOPs metrics and strict checkpoint validation."""

# Exercise the actual optimizer seam and complete independent trainer artifacts together.
# pylint: disable=protected-access,too-many-locals

import copy
import json
import logging
from typing import Literal

import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train import checkpoint as checkpoint_core
from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.config import Optim
from compare_attention.train.flops import estimate_step
from compare_attention.train.lr_schedule import next_learning_rate
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config
from tests.compare_attention.train.test_resume_equivalence import _assert_nested_equal


def tiny_config(
    tmp_path, *, policy: Literal["constant", "warmup_cosine"] = "warmup_cosine", batch=2, steps=6
):
    data = tmp_path / "data"
    prepare_kv_retrieval(
        data_config(vocab_size=32, input_seq_len=16, num_kv_pairs=2, n_train=32, n_val=3), data
    )
    config = train_config(
        data,
        model=gpt2_config(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16),
        batch_size=batch,
        val_every_steps=1,
        log_every_steps=1,
        checkpoint_every_steps=1,
        checkpoint_keep_last=steps,
    )
    return config.model_copy(
        update={
            "optim": Optim(lr_schedule=policy, warmup_fraction=0.5),
            "schedule": config.schedule.model_copy(
                update={"train_flops": estimate_step(config).total * steps}
            ),
        }
    )


def metric_rows(folder):
    return [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]


def test_real_updates_backup_snapshots_and_readable_log_use_actual_lr(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = tiny_config(tmp_path)
    cost = estimate_step(config).total
    expected = [
        next_learning_rate(config.optim, config.schedule.train_flops, i * cost, cost)
        for i in range(6)
    ]
    observed = []
    original_step = torch.optim.AdamW.step

    def recording_step(optimizer, *args, **kwargs):
        observed.append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", recording_step)
    output = tmp_path / "output"
    handler = logging.FileHandler(tmp_path / "train.log", encoding="utf-8")
    loop.logger.addHandler(handler)
    old_level = loop.logger.level
    loop.logger.setLevel(logging.INFO)
    try:
        result = loop.train(config, BackupTracker(output))
    finally:
        loop.logger.removeHandler(handler)
        loop.logger.setLevel(old_level)
        handler.close()
    assert observed == expected
    assert result.steps == 6
    assert result.train_flops == 6 * cost
    lr_rows = [row for row in metric_rows(output) if row["type"] == "learning_rate"]
    assert lr_rows == [
        {"type": "learning_rate", "learning_rate": lr, "flops": (i + 1) * cost}
        for i, lr in enumerate(expected)
    ]
    for i, lr in enumerate(expected, 1):
        state = load_checkpoint(output / f"checkpoints/step_{i:08d}.pt")
        assert state["optimizer"]["param_groups"][0]["lr"] == lr
        assert state["config"]["optim"] == config.optim.model_dump(mode="json")
    text = (tmp_path / "train.log").read_text()
    fresh = GPT2(config.model)
    optimizer = torch.optim.AdamW(
        fresh.parameters(), lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay
    )
    before = copy.deepcopy(optimizer.state_dict())
    state = load_checkpoint(output / "checkpoints/step_00000002.pt")
    checkpoint_core._validate_optimizer(state, optimizer)
    _assert_nested_equal(before, optimizer.state_dict())
    for option, value in (("lr", 0.123), ("weight_decay", 0.456), ("eps", 0.01)):
        bad = copy.deepcopy(state)
        bad["optimizer"]["param_groups"][0][option] = value
        with pytest.raises(ValueError, match="optimizer settings"):
            checkpoint_core._validate_optimizer(bad, optimizer)
        _assert_nested_equal(before, optimizer.state_dict())
    for i, lr in enumerate(expected, 1):
        line = next(line for line in text.splitlines() if f"train step={i} " in line)
        assert f"lr={lr:.3g} " in line


def test_scheduled_lr_is_readable_every_update_even_with_sparse_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = tiny_config(tmp_path)
    config = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"log_every_steps": 10})}
    )
    handler = logging.FileHandler(tmp_path / "train.log", encoding="utf-8")
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(logging.INFO)
    try:
        loop.train(config, BackupTracker(tmp_path / "output"))
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()
    text = (tmp_path / "train.log").read_text()
    for row in metric_rows(tmp_path / "output"):
        if row["type"] == "learning_rate":
            assert f"learning_rate flops={row['flops']} lr={row['learning_rate']:.17g}" in text


def test_constant_default_matches_legacy_manual_adamw_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = tiny_config(tmp_path, policy="constant", steps=4)
    output = tmp_path / "output"
    loop.train(config, BackupTracker(output))
    state = load_checkpoint(output / "checkpoints/latest.json")
    loop._seed(config.schedule.seed)
    model = GPT2(config.model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay
    )
    inputs, labels = loop._load(config.data, "train")
    cursor = 0
    for _ in range(4):
        _, cursor = loop._tracked_step(
            model, optimizer, inputs, labels, config.schedule.batch_size, cursor
        )
    _assert_nested_equal(dict(model.state_dict()), state["model"])
    _assert_nested_equal(optimizer.state_dict(), state["optimizer"])
    assert state["losses"]
    assert len(state["losses"]) == 4
    assert not [row for row in metric_rows(output) if row["type"] == "learning_rate"]


def test_validation_and_checkpoint_cadence_do_not_advance_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = tiny_config(tmp_path)
    sparse = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"val_every_steps": 3, "checkpoint_every_steps": 3}
            )
        }
    )
    first, second = tmp_path / "dense", tmp_path / "sparse"
    loop.train(config, BackupTracker(first))
    loop.train(sparse, BackupTracker(second))
    for field in ("model", "optimizer", "flops", "step", "losses"):
        _assert_nested_equal(
            load_checkpoint(first / "checkpoints/latest.json")[field],
            load_checkpoint(second / "checkpoints/latest.json")[field],
        )
    assert [r for r in metric_rows(first) if r["type"] == "learning_rate"] == [
        r for r in metric_rows(second) if r["type"] == "learning_rate"
    ]


def test_b8_b32_checkpoint_lr_matches_at_same_compute_not_step_index(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    b8 = tiny_config(tmp_path / "b8", batch=8, steps=8)
    b32 = tiny_config(tmp_path / "b32", batch=32, steps=2)
    assert b8.schedule.train_flops == b32.schedule.train_flops
    outputs = [tmp_path / "out8", tmp_path / "out32"]
    for config, output in zip((b8, b32), outputs, strict=True):
        loop.train(config, BackupTracker(output))
    by_flops = []
    for output in outputs:
        snapshots = [load_checkpoint(p) for p in (output / "checkpoints").glob("step_*.pt")]
        by_flops.append(
            {s["flops"]: (s["step"], s["optimizer"]["param_groups"][0]["lr"]) for s in snapshots}
        )
    for flops, (step, lr) in by_flops[1].items():
        assert by_flops[0][flops][1] == lr
        assert by_flops[0][flops][0] == 4 * step


@pytest.mark.parametrize(
    "field,value", [("lr_schedule", "constant"), ("warmup_fraction", 0.2), ("min_lr_ratio", 0.0)]
)
def test_resume_rejects_changed_scheduler_config(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = tiny_config(tmp_path)
    output = tmp_path / "output"
    loop.train(config, BackupTracker(output))
    changed = config.model_copy(update={"optim": config.optim.model_copy(update={field: value})})
    with pytest.raises(ValueError, match="config mismatch"):
        loop.train(
            changed,
            BackupTracker(tmp_path / "resumed"),
            resume_from=output / "checkpoints/step_00000002.pt",
        )
