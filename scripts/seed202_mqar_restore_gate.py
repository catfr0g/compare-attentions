"""Cloud-only read observer after native resume replay and RNG restoration."""

from __future__ import annotations

import contextlib
import dis
import re
import sys
from pathlib import Path

FIELDS = (
    "model",
    "optimizer",
    "rng",
    "precision",
    "step",
    "cursor",
    "flops",
    "losses",
    "elapsed",
    "early_stopping",
    "validation_history",
    "config",
    "model_structure",
    "optimizer_parameters",
    "optimizer_class",
)


def normal_return(frame):
    """A profile return event also occurs on exception unwind; distinguish bytecode."""
    return dis.opname[frame.f_code.co_code[frame.f_lasti]].startswith("RETURN")


def compare_snapshot(expected, actual, *, digest=None):
    """Compare complete typed fields, including every tensor and every RNG byte."""
    if digest is None:
        from compare_attention.train.checkpoint import _digest

        digest = _digest
    for name in FIELDS:
        if digest({name: actual[name]}) != digest({name: expected[name]}):
            raise ValueError("RESTORE_EQUALITY_" + name.upper())


@contextlib.contextmanager
def observe_restore(payload, directory, rank):
    """Observe the unchanged native method returning; fail before update981.

    The profile callback reads native state and invokes the native capture helper.
    No trainer, restore method, seed, optimizer or RNG function is replaced.
    This context is forbidden on the host; metadata tests use observe_return.
    """
    import torch
    from quality_recovery import create_json

    from compare_attention.train.checkpoint import capture_state, restore_state
    from compare_attention.train.control import TrainingControl
    from compare_attention.train.distributed import local_call
    from compare_attention.train.distributed_checkpoint import single_view
    from compare_attention.train.loop import _step
    from compare_attention.train.precision import current_precision

    if not torch.cuda.is_available() or sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError("EXCLUSIVE_TARGET_CUDA_RESTORE_OBSERVER_REQUIRED")
    if TrainingControl._resume.__globals__["restore_state"] is not restore_state:
        raise ValueError("AUTHENTIC_NATIVE_RESTORE_ALIAS_REQUIRED")
    expected = single_view(payload, rank)

    def capture(control):
        return capture_state(
            control.model,
            control.optimizer,
            control.config,
            step=len(control.losses),
            cursor=control.cursor,
            flops=control.flops,
            losses=control.losses,
            elapsed=control.elapsed,
            early_stopping=control.stopper.state_dict(),
            identity=control.identity,
            precision=current_precision(),
            validation_history=control.validations,
        )

    with observe_return(
        expected,
        directory,
        rank,
        resume=TrainingControl._resume,
        restore=restore_state,
        step=_step,
        capture=capture,
        collective=local_call,
        compare=compare_snapshot,
        write=create_json,
    ):
        yield


@contextlib.contextmanager
def observe_return(
    expected, directory, rank, *, resume, restore, step, capture, collective, compare, write
):
    """Profile the passed code identities; production binds only native functions."""
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError("EXCLUSIVE_RESTORE_OBSERVER_REQUIRED")
    resume_code = resume.__code__
    observed = False
    emitted = False

    def failure(error, phase, traceback=None):
        nonlocal emitted
        if emitted:
            return
        traceback = error.__traceback__ if traceback is None else traceback
        while traceback is not None and traceback.tb_next is not None:
            traceback = traceback.tb_next
        if traceback is None:
            return
        frame = traceback.tb_frame

        def token(value):
            return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:80]

        class_name = type(error).__name__
        if class_name not in {"ValueError", "RuntimeError", "OSError", "TypeError", "KeyError"}:
            class_name = "Exception"
        try:
            write(
                directory / f"native-resume-failure-rank-{rank}.json",
                {
                    "scope": "SANITIZED_ORIGINAL_EXCEPTION_BEFORE_DISTRIBUTED_WRAPPER",
                    "phase": phase,
                    "rank": rank,
                    "exception_class": class_name,
                    "source_basename": token(Path(frame.f_code.co_filename).name),
                    "source_function": token(frame.f_code.co_name),
                    "source_line": traceback.tb_lineno,
                    "retry": False,
                },
            )
            emitted = True
        except OSError:
            pass  # Preserve the original exception if diagnostic persistence itself fails.

    def trace(frame, event, argument):
        if frame.f_code in {resume_code, restore.__code__}:
            if event == "exception":
                failure(argument[1], "native_restore_before_981", argument[2])
            return trace
        return None

    def profile(frame, event, _argument):
        nonlocal observed
        if frame.f_code is step.__code__ and event == "call" and not observed:
            try:
                raise ValueError("RESTORE_NOT_VERIFIED_BEFORE_OPTIMIZER")
            except ValueError as error:
                failure(error, "native_step_before_restore_gate")
                raise
        if frame.f_code is not resume_code or event != "return" or not normal_return(frame):
            return
        if observed:
            raise ValueError("RESTORE_REPEATED")
        control = frame.f_locals["self"]

        def equality():
            actual = capture(control)
            try:
                compare(expected, actual)
            except Exception as error:
                failure(error, "native_restore_equality_before_981")
                raise
            write(
                directory / f"restore-equality-rank-{rank}.json",
                {
                    "scope": "ACTUAL_TARGET_NATIVE_RESTORE_EQUALITY_BEFORE_981",
                    "rank": rank,
                    "step": 980,
                    "next_step": 981,
                    "fields": list(FIELDS),
                    "native_restore_unchanged": True,
                    "rng_checked_after_history_replay": True,
                },
            )

        collective(equality, "mqar_restore_equality_before_981")
        observed = True
        sys.setprofile(None)
        sys.settrace(None)

    sys.settrace(trace)
    sys.setprofile(profile)
    try:
        yield
        if not observed:
            raise ValueError("NATIVE_RESTORE_RETURN_NOT_OBSERVED")
    finally:
        sys.setprofile(None)
        sys.settrace(None)
