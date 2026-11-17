"""Opt-in training controls preserve fixed-budget baseline defaults."""

import pytest
from pydantic import ValidationError

from compare_attention.train.config import Schedule


def test_training_controls_are_disabled_for_existing_experiments():
    schedule = Schedule(train_flops=1)
    assert schedule.early_stopping_patience == 0
    assert schedule.early_stopping_min_delta == 0.001
    assert schedule.early_stopping_min_steps == 2000
    assert schedule.checkpoint_every_steps == 0
    assert schedule.checkpoint_keep_last == 2
    assert schedule.amp_max_scale is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("early_stopping_patience", -1),
        ("early_stopping_min_steps", -1),
        ("early_stopping_min_delta", -0.001),
        ("early_stopping_min_delta", float("nan")),
        ("early_stopping_min_delta", float("inf")),
        ("checkpoint_every_steps", -1),
        ("checkpoint_keep_last", 0),
        ("amp_max_scale", 64),
        ("amp_max_scale", float("nan")),
        ("amp_max_scale", float("inf")),
    ],
)
def test_invalid_training_controls_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        Schedule.model_validate({"train_flops": 1, field: value})
