"""Finite credential-free failures persist even with child streams suppressed."""

import json
import os
from pathlib import Path

CLASSES = {
    "ValueError",
    "RuntimeError",
    "FileNotFoundError",
    "FileExistsError",
    "TypeError",
    "KeyError",
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "CalledProcessError",
    "TimeoutExpired",
    "OSError",
}
PHASES = {
    "entry",
    "hardware",
    "setup",
    "worker_runtime",
    "qualification",
    "evaluation",
    "publication",
    "readback",
    "composite",
    "recovery_start",
    "recovery_attach",
    "recovery_complete",
}


FILES = {
    "mqar_final_tail_cli.py",
    "mqar_final_tail_native.py",
    "mqar_final_tail_package.py",
    "mqar_final_tail_metrics.py",
    "mqar_final_tail_runner.py",
    "mqar_final_tail_tracking.py",
    "mqar_final_tail_publish.py",
    "mqar_final_tail_service_cli.py",
    "mqar_final_tail_runtime.py",
    "mqar_final_tail_composite.py",
    "mqar_final_tail_errors.py",
    "evaluate.py",
    "evaluation.py",
    "evaluation_checkpoint.py",
}
FUNCTIONS = {
    "main",
    "run",
    "guard",
    "runtime_guard",
    "decision",
    "audit",
    "inspect_full",
    "check",
    "require",
    "execute_once",
    "start",
    "finish",
    "publish_final",
    "complete_tracking",
    "read_unsampled_events",
    "verify_unsampled_events",
    "operation",
    "verify_report",
    "load_evaluation_model",
    "evaluate_checkpoint",
    "evaluate_model",
    "composite",
}


def public(error, phase):
    frames = []
    trace = error.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        frames.append(
            {
                "file": Path(code.co_filename).name
                if Path(code.co_filename).name in FILES
                else "REDACTED",
                "function": code.co_name if code.co_name in FUNCTIONS else "REDACTED",
                "line": trace.tb_lineno,
            }
        )
        trace = trace.tb_next
    return {
        "phase": phase if phase in PHASES else "entry",
        "exception_class": type(error).__name__ if type(error).__name__ in CLASSES else "OTHER",
        "category": "RUNTIME_OR_CONTRACT_FAILURE",
        "returncode": getattr(error, "returncode", None)
        if type(getattr(error, "returncode", None)) is int
        else None,
        "frames": frames[-12:],
    }


def persist(path, error, phase):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(public(error, phase), stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
