"""Tight cross-platform policy tolerance does not weaken other optimizer options."""

import math
from copy import deepcopy

import pytest
import torch

from compare_attention.train.checkpoint_optimizer import _validate_optimizer
from compare_attention.train.config import Optim
from compare_attention.train.lr_schedule import next_learning_rate


def case():
    optim = Optim(learning_rate=0.001, lr_schedule="warmup_cosine")
    config = {"optim": optim.model_dump(mode="json"), "schedule": {"train_flops": 40}}
    policy_lr = next_learning_rate(optim, 40, 0, 10)
    parameter = torch.nn.Parameter(torch.ones(2))
    trained = torch.optim.AdamW([parameter], lr=policy_lr, weight_decay=0.1)
    parameter.square().sum().backward()
    trained.step()
    payload = {
        "config": config,
        "identity": {"config": config, "accounting": {"total": 10}},
        "step": 1,
        "flops": 10,
        "optimizer": deepcopy(trained.state_dict()),
    }
    fresh = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(2))], lr=optim.learning_rate, weight_decay=0.1
    )
    return payload, fresh, policy_lr


def test_scheduled_checkpoint_accepts_last_ulp_without_mutating_live_optimizer():
    payload, fresh, expected = case()
    payload["optimizer"]["param_groups"][0]["lr"] = math.nextafter(expected, math.inf)
    before = deepcopy(fresh.state_dict())
    _validate_optimizer(payload, fresh)
    assert fresh.state_dict() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 0.02),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("lr", True),
        ("weight_decay", math.nextafter(0.1, math.inf)),
    ],
)
def test_optimizer_policy_tolerance_rejects_wrong_lr_and_other_setting_changes(field, value):
    payload, fresh, _ = case()
    payload["optimizer"]["param_groups"][0][field] = value
    with pytest.raises(ValueError, match="optimizer settings"):
        _validate_optimizer(payload, fresh)


def test_scheduled_checkpoint_rejects_meaningful_relative_lr_error():
    payload, fresh, expected = case()
    payload["optimizer"]["param_groups"][0]["lr"] = expected * (1 + 1e-6)
    with pytest.raises(ValueError, match="optimizer settings"):
        _validate_optimizer(payload, fresh)
