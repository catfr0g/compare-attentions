"""Persist only safe original failure facts before service/DDP error conversion."""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import PurePosixPath

_ACTIVE = ContextVar("training_failure_observer", default=None)


def _identifier(value):
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.<>]*", value) else "unknown"


def safe_failure(error, phase):
    """Do not read exception text, locals, source lines, credentials or full paths."""
    frames = []
    trace = error.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        basename = PurePosixPath(code.co_filename.replace("\\", "/")).name
        if re.fullmatch(r"[A-Za-z_0-9.-]+\.py", basename) is None:
            basename = "<generated>"
        frames.append(
            {"basename": basename, "function": _identifier(code.co_name), "line": trace.tb_lineno}
        )
        trace = trace.tb_next
    return {
        "schema": "training-lifecycle-failure-v1",
        "phase": _identifier(phase),
        "exception_class": _identifier(type(error).__name__),
        "frames": frames,
    }


def primary_failure(error, phase):
    """Return the observed innermost cause even after a service sanitizes it."""
    state = _ACTIVE.get()
    if state is not None and state.get("primary"):
        return state["primary"]
    observed = getattr(error, "_training_primary_failure", None)
    if isinstance(observed, dict) and observed.get("schema") == "training-lifecycle-failure-v1":
        # Do not accept arbitrary added fields/messages even on an SDK exception.
        return {
            "schema": "training-lifecycle-failure-v1",
            "phase": _identifier(str(observed.get("phase", "unknown"))),
            "exception_class": _identifier(str(observed.get("exception_class", "unknown"))),
            "frames": [
                {
                    "basename": row["basename"]
                    if re.fullmatch(r"[A-Za-z_0-9.-]+\.py", row["basename"])
                    else "<generated>",
                    "function": _identifier(row["function"]),
                    "line": row["line"],
                }
                for row in observed.get("frames", [])
                if isinstance(row, dict)
                and type(row.get("line")) is int
                and isinstance(row.get("basename"), str)
                and isinstance(row.get("function"), str)
            ],
        }
    return safe_failure(error, phase)


@contextmanager
def observe_failures(tracker, rank=None):
    """Bind one run's sink; never consume a random stream or replace its exception."""
    state = {"tracker": tracker, "recorded": False, "rank": rank}
    token = _ACTIVE.set(state)
    try:
        with failure_phase("training"):
            yield
    except Exception as error:
        try:
            error._training_primary_failure = primary_failure(error, "training")  # pylint: disable=protected-access
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        raise
    finally:
        _ACTIVE.reset(token)


@contextmanager
def failure_phase(name):
    """Write the innermost original failure once, before an outer wrapper can mask it."""
    try:
        yield
    except Exception as error:
        state = _ACTIVE.get()
        if state is not None and not state["recorded"]:
            state["recorded"] = True
            report = primary_failure(error, name)
            state["primary"] = report
            try:
                sink = getattr(state["tracker"], "record_failure", None)
                if callable(sink):
                    report = dict(report)
                    if state["rank"] is not None:
                        report["rank"] = state["rank"]
                    sink(report)
            except Exception:  # pylint: disable=broad-exception-caught
                # Diagnostic I/O must not obscure the original exception.
                pass
        raise
