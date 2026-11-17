"""FLOPs-axis LR policy: validated opt-in configuration and stateless boundaries."""

import pytest
from pydantic import ValidationError

from compare_attention.train.config import Optim
from compare_attention.train.lr_schedule import next_learning_rate


@pytest.mark.parametrize(
    "spent,cost,warmup,minimum,expected",
    [
        (0, 1, 0.2, 0.1, 0.5),
        (1, 1, 0.2, 0.1, 1.0),
        (5, 1, 0.2, 0.1, 0.55),
        (9, 1, 0.2, 0.1, 0.1),
        (9, 4, 0.2, 0.1, 0.1),
        (0, 5, 0.0, 0.1, 0.55),
        (0, 1, 0.0, 1.0, 1.0),
        (9, 1, 0.2, 0.0, 0.0),
    ],
)
def test_end_of_next_update_policy(spent, cost, warmup, minimum, expected):
    optim = Optim(
        learning_rate=1.0, lr_schedule="warmup_cosine", warmup_fraction=warmup, min_lr_ratio=minimum
    )
    assert next_learning_rate(optim, 10, spent, cost) == pytest.approx(expected)
    assert next_learning_rate(Optim(), 10, spent, cost) == 3e-4


def test_same_compute_different_batch_update_indices():
    optim = Optim(lr_schedule="warmup_cosine")
    assert next_learning_rate(optim, 1000, 24, 8) == next_learning_rate(optim, 1000, 0, 32)
    assert next_learning_rate(optim, 1000, 760, 8) == next_learning_rate(optim, 1000, 736, 32)


def test_schedule_defaults_preserve_constant_policy():
    optim = Optim()
    assert optim.lr_schedule == "constant"
    assert optim.warmup_fraction == 0.05
    assert optim.min_lr_ratio == 0.1


@pytest.mark.parametrize(
    "field,value",
    [("lr_schedule", "linear"), ("lr_schedule", True)]
    + [
        (field, value)
        for field in ("warmup_fraction", "min_lr_ratio")
        for value in (True, False, float("nan"), float("inf"), -0.1)
    ]
    + [("warmup_fraction", 1.0), ("min_lr_ratio", 1.1)],
)
def test_reject_invalid_schedule_configuration(field, value):
    with pytest.raises(ValidationError):
        Optim(**{field: value})
