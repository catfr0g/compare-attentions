"""Validation-event early stopping, including warmup and resumable state."""

import copy
import json

import pytest

from compare_attention.train.stopping import EarlyStopping


def test_defaults_disable_stopping_and_track_best() -> None:
    stopping = EarlyStopping()
    assert (stopping.patience, stopping.min_delta, stopping.min_steps) == (0, 0.001, 2000)
    assert not stopping.observe(2.0, 1)
    assert not stopping.observe(1.0, 2000)
    for step in range(2001, 2010):
        assert not stopping.observe(3.0, step)
    assert (stopping.best_loss, stopping.best_step, stopping.last_step) == (1.0, 2000, 2009)


def test_decimal_equality_at_min_delta_is_not_an_improvement() -> None:
    stopping = EarlyStopping(patience=1, min_delta=0.001, min_steps=0)
    assert not stopping.observe(1.0, 0)
    assert stopping.observe(0.999, 1)
    assert stopping.best_loss == 1.0


def test_strict_improvement_resets_patience_at_exact_boundary() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.125, min_steps=0)
    assert not stopping.observe(1.0, 0)
    assert not stopping.observe(1.0, 1)
    assert stopping.bad_validations == 1
    assert stopping.observe(0.875, 2)  # Exactly min_delta is NOT an improvement.
    assert (stopping.best_loss, stopping.best_step, stopping.bad_validations) == (1.0, 0, 2)
    assert not stopping.observe(0.75, 3)
    assert (stopping.best_loss, stopping.best_step, stopping.bad_validations) == (0.75, 3, 0)


@pytest.mark.parametrize("first_eligible_loss, expected_bad", [(1.5, 1), (0.5, 0)])
def test_warmup_tracks_global_best_without_accumulating_bad_events(
    first_eligible_loss, expected_bad
) -> None:
    stopping = EarlyStopping(patience=2, min_steps=10)
    assert not stopping.observe(1.0, 1)
    for step in (2, 3, 9):
        assert not stopping.observe(2.0, step)
        assert stopping.bad_validations == 0
    assert not stopping.observe(first_eligible_loss, 10)
    assert stopping.bad_validations == expected_bad
    assert stopping.best_loss == min(1.0, first_eligible_loss)
    assert stopping.observe(3.0, 11) == (expected_bad == 1)
    assert stopping.observe(3.0, 12)


def test_first_observation_at_eligibility_is_baseline_not_bad_event() -> None:
    stopping = EarlyStopping(patience=1, min_steps=10)
    assert not stopping.observe(1.0, 10)
    assert stopping.bad_validations == 0
    assert stopping.best_step == 10
    assert stopping.observe(1.0, 11)


@pytest.mark.parametrize("field", ["patience", "min_steps"])
@pytest.mark.parametrize("value", [-1, 1.5, True, float("nan"), float("inf"), "1"])
def test_invalid_integer_settings_are_rejected(field, value) -> None:
    with pytest.raises(ValueError):
        EarlyStopping(**{field: value})


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_delta_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=value)


@pytest.mark.parametrize(
    "loss, step",
    [
        (float("nan"), 2),
        (float("inf"), 2),
        (-float("inf"), 2),
        (True, 2),
        ("1.0", 2),
        (1.0, -1),
        (1.0, True),
        (1.0, 1.5),
        (1.0, 0),
        (1.0, 1),
    ],
)
def test_invalid_observations_are_rejected_without_mutation(loss, step) -> None:
    stopping = EarlyStopping()
    stopping.observe(2.0, 1)
    with pytest.raises(ValueError):
        stopping.observe(loss, step)
    assert (stopping.best_loss, stopping.best_step, stopping.last_step) == (2.0, 1, 1)
    assert stopping.bad_validations == 0


@pytest.mark.parametrize(
    "prefix",
    [[], [(1.0, 0)], [(1.0, 0), (2.0, 9)], [(1.0, 0), (2.0, 10)], [(1.0, 0), (2.0, 10), (2.0, 11)]],
)
def test_json_state_restore_preserves_future_decisions(prefix) -> None:
    original = EarlyStopping(patience=2, min_steps=10)
    restored = EarlyStopping(patience=2, min_steps=10)
    for loss, step in prefix:
        original.observe(loss, step)
    state = original.state_dict()
    assert set(state) == {"best_loss", "best_step", "bad_validations", "last_step", "settings"}
    assert state["settings"] == {"patience": 2, "min_delta": 0.001, "min_steps": 10}
    restored.load_state_dict(json.loads(json.dumps(state, allow_nan=False)))
    assert restored.state_dict() == original.state_dict()
    state["settings"]["patience"] = 100
    assert original.patience == restored.patience == 2
    for loss, step in [(3.0, 12), (0.5, 13), (0.5, 14), (0.5, 15)]:
        assert restored.observe(loss, step) == original.observe(loss, step)
        assert restored.state_dict() == original.state_dict()


@pytest.mark.parametrize(
    "patch",
    [
        {"best_loss": float("nan")},
        {"best_loss": float("inf")},
        {"best_loss": True},
        {"best_loss": None},
        {"best_step": None},
        {"last_step": None},
        {"best_step": -1},
        {"best_step": True},
        {"best_step": 11},
        {"last_step": -1},
        {"last_step": True},
        {"last_step": 1.5},
        {"bad_validations": -1},
        {"bad_validations": True},
        {"bad_validations": 1.5},
        {"bad_validations": 0},
        {"bad_validations": 2},
        {"unknown": 0},
        {"settings": {"patience": 3, "min_delta": 0.001, "min_steps": 10}},
        {"settings": {"patience": True, "min_delta": 0.001, "min_steps": 10}},
        {"settings": {"patience": 2, "min_delta": float("nan"), "min_steps": 10}},
        {"settings": {}},
        {"settings": None},
    ],
)
def test_invalid_state_is_rejected_atomically(patch) -> None:
    stopping = EarlyStopping(patience=2, min_steps=10)
    stopping.observe(1.0, 1)
    stopping.observe(2.0, 10)
    original = stopping.state_dict()
    invalid = copy.deepcopy(original)
    invalid.update(patch)
    with pytest.raises(ValueError):
        stopping.load_state_dict(invalid)
    assert stopping.state_dict() == original


@pytest.mark.parametrize("state", [None, {}, {"best_loss": 1.0}])
def test_invalid_state_schema_is_rejected(state) -> None:
    with pytest.raises(ValueError):
        EarlyStopping().load_state_dict(state)


def test_inconsistent_empty_or_warmup_state_is_rejected() -> None:
    stopping = EarlyStopping(patience=2, min_steps=10)
    empty = stopping.state_dict()
    empty["bad_validations"] = 1
    with pytest.raises(ValueError):
        stopping.load_state_dict(empty)
    stopping.observe(1.0, 1)
    stopping.observe(2.0, 9)
    warmup = stopping.state_dict()
    warmup["bad_validations"] = 1
    with pytest.raises(ValueError):
        stopping.load_state_dict(warmup)
