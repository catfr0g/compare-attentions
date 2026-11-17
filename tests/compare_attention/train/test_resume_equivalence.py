"""Real six-update restart equivalence, including optimizer, scaler and every RNG."""

import json
import random

import numpy as np
import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.config import Optim
from compare_attention.train.flops import estimate_step
from compare_attention.train.lr_schedule import next_learning_rate
from compare_attention.train.precision import Precision
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config


def _next_random_values(cuda):
    """Draw from global streams, not independently constructed generators."""
    values = {
        "python": [random.random() for _ in range(4)],
        "numpy": np.random.random(4).tolist(),
        "torch": torch.rand(4),
    }
    if cuda:
        values["cuda"] = [
            torch.rand(4, device=f"cuda:{index}").cpu()
            for index in range(torch.cuda.device_count())
        ]
    return values


def _assert_nested_equal(expected, actual, path="payload"):
    """Compare every nested value and tensor bitwise, with useful failure paths."""
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert actual.layout == expected.layout, path
        assert torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        ), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            _assert_nested_equal(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_nested_equal(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path


class _FastGrowthPrecision(Precision):
    """Exercise real CUDA GradScaler growth boundaries without 2,000 updates."""

    def __init__(self, device, max_scale=None):
        super().__init__(device, max_scale=max_scale)
        self.scaler.set_growth_interval(2)


class _RandomBackup(BackupTracker):
    """Replay has observable RNG side effects on all streams used by this run."""

    def __init__(self, directory, cuda):
        super().__init__(directory)
        self.cuda = cuda
        self.loss_events = []
        self.validation_events = []

    def log_loss(self, loss, flops):
        self.loss_events.append((loss, flops))
        _next_random_values(self.cuda)
        super().log_loss(loss, flops)

    def log_learning_rate(self, learning_rate, flops):
        _next_random_values(self.cuda)
        super().log_learning_rate(learning_rate, flops)

    def log_validation(self, loss, accuracy, flops):
        self.validation_events.append((loss, accuracy, flops))
        _next_random_values(self.cuda)
        super().log_validation(loss, accuracy, flops)


class _SnapshotInterrupt(RuntimeError):
    """Intentional interruption after a durable successful-update boundary."""


class _InterruptedBackup(_RandomBackup):
    def save_checkpoint(self, payload, step, keep_last):
        super().save_checkpoint(payload, step, keep_last)
        if step == 2:
            raise _SnapshotInterrupt("durable step-2 interruption")


@pytest.mark.parametrize("policy", ["constant", "warmup_cosine"])
@pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "cuda-fp16"])
def test_full_restart_matches_optimizer_scaler_history_and_all_rng(
    tmp_path, monkeypatch, cuda, policy
):
    # Compare three real trajectories and their independent persisted/RNG states together.
    # pylint: disable=too-many-locals
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA restart requires a GPU")
    device = torch.device("cuda" if cuda else "cpu")
    monkeypatch.setattr(loop, "_device", lambda: device)
    if cuda:
        monkeypatch.setattr(loop, "Precision", _FastGrowthPrecision)
    data = tmp_path / "data"
    prepare_kv_retrieval(
        data_config(vocab_size=32, input_seq_len=16, num_kv_pairs=2, n_train=8, n_val=3), data
    )
    model = gpt2_config(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.2)
    config = train_config(
        data,
        model=model,
        dtype="float16" if cuda else "float32",
        val_every_steps=2,
        checkpoint_every_steps=2,
        checkpoint_keep_last=3,
        amp_max_scale=128.0 if cuda else None,
    )
    config = config.model_copy(
        update={
            "optim": Optim(lr_schedule=policy, warmup_fraction=0.5),
            "schedule": config.schedule.model_copy(
                update={"train_flops": estimate_step(config).total * 6}
            ),
        }
    )

    random.seed(123)
    np.random.seed(123)
    expected_dir = tmp_path / "uninterrupted"
    expected = loop.train(config, _RandomBackup(expected_dir, cuda))
    expected_next = _next_random_values(cuda)
    expected_state = load_checkpoint(expected_dir / "checkpoints/latest.json")

    random.seed(123)
    np.random.seed(123)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(_SnapshotInterrupt, match="durable step-2"):
        loop.train(config, _InterruptedBackup(interrupted_dir, cuda))
    snapshot = interrupted_dir / "checkpoints/latest.json"
    assert load_checkpoint(snapshot)["step"] == 2
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    actual_dir = tmp_path / "resumed"
    actual = loop.train(config, _RandomBackup(actual_dir, cuda), resume_from=snapshot)
    actual_next = _next_random_values(cuda)
    actual_state = load_checkpoint(actual_dir / "checkpoints/latest.json")

    assert actual.steps == expected.steps == 6
    assert actual.train_flops == expected.train_flops
    for field in (
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
        _assert_nested_equal(expected_state[field], actual_state[field], field)
    _assert_nested_equal(expected_next, actual_next, "next_rng")
    expected_rows = [
        json.loads(line) for line in (expected_dir / "metrics.jsonl").read_text().splitlines()
    ]
    actual_rows = [
        json.loads(line) for line in (actual_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r for r in actual_rows if r["type"] != "result"] == [
        r for r in expected_rows if r["type"] != "result"
    ]
    learning_rates = [row for row in actual_rows if row["type"] == "learning_rate"]
    if policy == "warmup_cosine":
        cost = estimate_step(config).total
        assert learning_rates == [
            {
                "type": "learning_rate",
                "learning_rate": next_learning_rate(
                    config.optim, config.schedule.train_flops, step * cost, cost
                ),
                "flops": (step + 1) * cost,
            }
            for step in range(6)
        ]
    else:
        assert not learning_rates
    if cuda:
        assert actual_state["precision"]["scaler"]["scale"] == 128.0
        assert actual_state["precision"]["scaler"]["growth_interval"] == 2
        assert actual_state["precision"]["optimizer_steps"] == 6
        assert actual_state["precision"]["skipped_optimizer_steps"] == 0
