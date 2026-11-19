"""Negative controls for the active Gloo helpers' sanitized failure assertions."""

import copy

import pytest

from compare_attention.train.lifecycle import safe_failure
from tests.compare_attention.train.ddp_guards import assert_phase_failure

CANARY = "private-test-canary-never-on-wire"


def original_failure():
    raise ValueError(CANARY)


def report():
    try:
        original_failure()
    except ValueError as error:
        return safe_failure(error, "load_train")
    raise AssertionError("fixture did not fail")


def envelope(value):
    error = RuntimeError(f"DDP phase failed: {[None, value]}")
    error._training_primary_failure = value  # pylint: disable=protected-access
    return error


def check(error):
    return assert_phase_failure(
        error,
        "load_train",
        "ValueError",
        "test_ddp_failure_contract.py",
        "original_failure",
        CANARY,
    )


def test_accepts_safe_original_facts_without_private_message():
    value = report()
    assert CANARY not in str(value)
    assert check(envelope(value)) == value


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "phase",
        "exception_class",
        "basename",
        "function",
        "line_zero",
        "line_bool",
        "frames_empty",
        "private_path",
        "raw_message",
        "extra_frame_field",
    ],
)
def test_rejects_wrong_or_leaky_original_facts(mutation):
    value = copy.deepcopy(report())
    if mutation in {"schema", "phase", "exception_class"}:
        value[mutation] = "unrelated"
    elif mutation in {"basename", "function"}:
        value["frames"][-1][mutation] = "unrelated.py" if mutation == "basename" else "unrelated"
    elif mutation in {"line_zero", "line_bool"}:
        value["frames"][-1]["line"] = 0 if mutation == "line_zero" else True
    elif mutation == "frames_empty":
        value["frames"] = []
    elif mutation == "private_path":
        value["frames"][-1]["basename"] = "C:/private/worker.py"
    elif mutation == "raw_message":
        value["message"] = CANARY
    else:
        value["frames"][-1]["locals"] = {"private": "hidden"}
    with pytest.raises(AssertionError):
        check(envelope(value))


@pytest.mark.parametrize("kind", ["unrelated", "wrong_type", "empty", "one_rank", "detached"])
def test_rejects_unrelated_runtime_error_or_noncollective_envelope(kind):
    value = report()
    error = envelope(value)
    if kind == "unrelated":
        error = RuntimeError("unrelated failure")
    elif kind == "wrong_type":
        error = ValueError(str(error))
    elif kind == "empty":
        error = RuntimeError("DDP phase failed: [None, None]")
    elif kind == "one_rank":
        error = RuntimeError(f"DDP phase failed: {[value]}")
    else:
        error._training_primary_failure = {**value, "phase": "unrelated"}  # pylint: disable=protected-access
    with pytest.raises(AssertionError):
        check(error)
