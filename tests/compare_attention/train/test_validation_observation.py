"""V5 transport replay: agree on one full-split observation before consumers."""

from types import SimpleNamespace

import pytest
import torch

from compare_attention.train import distributed, distributed_checkpoint, loop
from compare_attention.train.checkpoint import _digest
from compare_attention.train.result import Quality

# Genuine v5 capture-0001 validation_history losses, independently audited locally.
V5_LOSSES = (
    (float.fromhex("0x1.b4a03fc03fc04p+0"), float.fromhex("0x1.b4a0448844884p+0")),
    (float.fromhex("0x1.9da179b979b98p+0"), float.fromhex("0x1.9da1782178218p+0")),
)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("losses", V5_LOSSES)
def test_full_split_observation_is_rank_zero_before_tracker(monkeypatch, rank, losses):
    run = distributed.DistributedRun(rank, 2, rank, torch.device("cpu"))
    monkeypatch.setattr(distributed, "current", lambda: run)
    monkeypatch.setattr(loop, "current", lambda: run)
    monkeypatch.setattr(run, "gather", lambda value: [value, value])
    monkeypatch.setattr(
        distributed.dist, "all_gather", lambda outputs, value: [out.copy_(value) for out in outputs]
    )
    primary = Quality(train_loss=2.0, val_loss=losses[0], val_accuracy=0.25)
    local = primary.model_copy(update={"val_loss": losses[rank]})
    monkeypatch.setattr(loop, "_quality", lambda *_args: local)
    broadcasts = []

    def broadcast(values, src, device):
        broadcasts.append((src, device))
        assert values == [local]
        values[0] = primary

    monkeypatch.setattr(distributed.dist, "broadcast_object_list", broadcast)
    observed = []
    tracker = SimpleNamespace(log_validation=lambda *args: observed.append(args))
    config = SimpleNamespace(schedule=SimpleNamespace(batch_size=32))
    quality = loop._evaluate(  # pylint: disable=protected-access
        config, None, (None, None, torch.zeros(257, 1), None), [2.0], tracker, 100
    )
    assert quality == primary
    assert observed == [(primary.val_loss, primary.val_accuracy, 100)]
    assert broadcasts == [(0, torch.device("cpu"))]


def test_single_device_observation_is_untouched_without_collectives(monkeypatch):
    monkeypatch.setattr(distributed, "current", lambda: None)
    monkeypatch.setattr(loop, "current", lambda: None)
    quality = Quality(train_loss=2, val_loss=V5_LOSSES[1][1], val_accuracy=0.5)
    monkeypatch.setattr(loop, "_quality", lambda *_args: quality)

    def forbidden(*_args, **_kwargs):
        pytest.fail("single-device validation must not enter a collective")

    monkeypatch.setattr(distributed.dist, "broadcast_object_list", forbidden)
    monkeypatch.setattr(distributed.dist, "all_gather_object", forbidden)
    monkeypatch.setattr(distributed.dist, "all_gather", forbidden)
    config = SimpleNamespace(schedule=SimpleNamespace(batch_size=32))
    tracker = SimpleNamespace(log_validation=lambda *_args: None)
    actual = loop._evaluate(  # pylint: disable=protected-access
        config, None, (None, None, torch.zeros(257, 1), None), [2.0], tracker, 100
    )
    assert actual is quality


@pytest.mark.parametrize(
    "field", ["model", "optimizer", "precision", "validation_history", "early_stopping"]
)
def test_shared_checkpoint_gate_still_rejects_each_divergent_field(monkeypatch, field):
    shared = {
        "model": {"weight": torch.ones(1)},
        "optimizer": {"state": {}},
        "precision": None,
        "step": 3,
        "cursor": 96,
        "flops": 67721232384,
        "losses": [2.0, 1.8, 1.7],
        "early_stopping": {"best_loss": V5_LOSSES[1][0]},
        "validation_history": [{"loss": V5_LOSSES[1][0]}],
    }
    other = dict(shared)
    other[field] = {"deliberately": "divergent"}
    hashes = [_digest(shared), _digest(other)]
    calls = []

    def gather(value):
        assert not calls and value == hashes[0]
        calls.append(value)
        return hashes

    run = SimpleNamespace(gather=gather)
    monkeypatch.setattr(distributed_checkpoint, "current", lambda: run)
    with pytest.raises(RuntimeError, match="shared model/optimizer/scaler/progress diverged"):
        distributed_checkpoint.capture(shared, None)
    assert calls == [hashes[0]]
