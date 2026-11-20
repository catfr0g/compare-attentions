"""Finite stdlib diagnostics for NORMAL launch boundaries; never child stream capture."""

import argparse
import builtins
import json
import os
import subprocess
from collections import deque
from pathlib import Path

_NORMAL_CLASSES = frozenset(
    {
        "ValueError",
        "RuntimeError",
        "TypeError",
        "ImportError",
        "ModuleNotFoundError",
        "FileNotFoundError",
        "PermissionError",
        "OSError",
        "CalledProcessError",
        "TimeoutExpired",
        "AssertionError",
        "SystemExit",
        "KeyboardInterrupt",
    }
)
_NORMAL_TYPES = tuple(
    getattr(builtins, name)
    for name in _NORMAL_CLASSES
    if name not in {"CalledProcessError", "TimeoutExpired"}
) + (subprocess.CalledProcessError, subprocess.TimeoutExpired)
_NORMAL_PHASES = frozenset({"worker", "finish", "task", "accept", "setup"})
_NORMAL_STAGES = ("pre-worker", "worker", "finish", "task", "setup", "bootstrap", "supervisor")
_NORMAL_FILES = frozenset(
    {
        "kernel.py",
        "setup.py",
        "worker.py",
        "normal_training_launch.py",
        "normal_training_worker.py",
        "normal_training_plan.py",
        "normal_training_native.py",
        "normal_training_acceptance.py",
        "normal_launch_diagnostics.py",
        "normal_driver_environment.py",
        "unified_normal_task.py",
        "normal-task.py",
        "runtime.py",
        "services.py",
        "quality_inherit.py",
        "quality_services.py",
        "quality_recovery.py",
        "training_completion.py",
        "lifecycle.py",
        "loop.py",
        "tracker.py",
        "distributed.py",
        "backup.py",
    }
)
_NORMAL_FUNCTIONS = frozenset(
    {
        "entry",
        "<module>",
        "main",
        "worker",
        "finish",
        "accept",
        "derive",
        "require",
        "execution_authority",
        "guard",
        "hardware",
        "credentials",
        "load_credentials",
        "leaf",
        "run",
        "bounded_run",
        "normal_main",
        "normal_entry",
        "normal_context",
        "capture",
        "supervise",
        "train",
        "_train",
        "primary_failure",
        "failure_phase",
        "record_failure",
        "verify_native",
        "training_readback",
        "verify_native_service",
        "validated_driver_environment",
    }
)
_NORMAL_CODES = frozenset(
    {
        "NORMAL_SETUP_SOURCE_PATH",
        "NORMAL_SETUP_SOURCE_BYTES",
        "NORMAL_SETUP_NEW_SOURCE_BINDING",
        "NORMAL_TARGET_MASK_FORBIDDEN",
        "NORMAL_NO_PRODUCTION_BOOTSTRAP",
        "NORMAL_CPU_ENV_BEFORE_CHILD",
        "NORMAL_EXPLICIT_SELECTED_UV_PYTHON",
        "NORMAL_WHEEL_DUPLICATE",
        "NORMAL_WHEEL_UNSAFE",
        "NORMAL_OWNED_CHILD_TIMEOUT_STOP",
        "NORMAL_CPU_ENGINEERING_ENV",
        "NORMAL_DISTINCT_NAMESPACE",
        "NORMAL_DDP_RANKS",
        "NORMAL_EXACT_TARGET_RUNTIME",
        "VISIBILITY_MASK_FORBIDDEN",
        "PYTHON_TARGET_MISMATCH",
        "LINUX_REQUIRED",
        "PINNED_TORCH_RUNTIME_REQUIRED",
        "PINNED_RUNTIME_PACKAGE_REQUIRED",
        "HARDWARE_EXACT_TWO_T4_REQUIRED",
        "HARDWARE_T4_SM75_REQUIRED",
        "NORMAL_VALIDATED_DRIVER_REQUIRED",
        "NORMAL_RETURN_ENVELOPE",
        "NORMAL_RETURN_RECEIPT",
        "NORMAL_FRESH_UNSAMPLED_DRIFT",
        "NORMAL_NATIVE_SERVICE_VALUES",
    }
)
_NORMAL_PRIMARY_PHASES = frozenset(
    {
        "unknown",
        "training",
        "finish",
        "identity_write",
        "clearml_start",
        "clearml_loss",
        "clearml_learning_rate",
        "clearml_validation",
        "clearml_finish",
        "clearml_flush",
        "clearml_close",
        "clearml_readback",
        "clearml_raw_receipt_write",
        "clearml_validate_receipt",
        "validation",
        "checkpoint",
        "gradient",
        "backward",
        "optimizer_step",
        "resume_destination",
        "resume_preflight",
    }
) | frozenset(_NORMAL_STAGES)
_NORMAL_KEYS = frozenset(
    {
        "schema",
        "source",
        "case",
        "phase",
        "stage",
        "primary_phase",
        "rank",
        "exception_class",
        "category",
        "returncode",
        "frames",
    }
)


class NormalArgumentParser(argparse.ArgumentParser):
    """Usage errors may contain private argument values; retain only exit code two."""

    def error(self, message):
        del message
        raise SystemExit(2)


def _normal_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("NORMAL_DIAGNOSTIC_JSON")
        value[key] = item
    return value


def _normal_constant(_value):
    raise ValueError("NORMAL_DIAGNOSTIC_JSON")


def _normal_exit_status(error):
    code = vars(SystemExit)["code"].__get__(error) if issubclass(type(error), SystemExit) else 1
    return 0 if code is None else int.__int__(code) if issubclass(type(code), int) else 1


def _normal_attribute(error, name):
    try:
        return getattr(error, name, None)
    except BaseException:  # pylint: disable=broad-exception-caught
        return None


def _normal_frames(error):
    frames = deque(maxlen=12)
    trace = BaseException.__traceback__.__get__(error)
    while trace is not None:
        code = trace.tb_frame.f_code
        basename = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        if basename in _NORMAL_FILES and code.co_name in _NORMAL_FUNCTIONS:
            frames.append({"basename": basename, "function": code.co_name, "line": trace.tb_lineno})
        trace = trace.tb_next
    return list(frames)


def _normal_frame_valid(row):
    return (
        type(row) is dict
        and set(row) == {"basename", "function", "line"}
        and type(row["basename"]) is str
        and row["basename"] in _NORMAL_FILES
        and type(row["function"]) is str
        and row["function"] in _NORMAL_FUNCTIONS
        and type(row["line"]) is int
        and 0 < row["line"] <= 100000
    )


class NormalDiagnostics:
    """One fresh launch scope; bounded create-only rank/stage records, no raw messages."""

    def __init__(self, root, phase, source, case):
        self.root = Path(root)
        self.phase = phase
        self.source = source
        self.case = case
        if phase not in _NORMAL_PHASES:
            raise ValueError("NORMAL_DIAGNOSTIC_PHASE")

    def environment(self):
        return {
            "NORMAL_DIAGNOSTIC_ROOT": str(self.root),
            "NORMAL_DIAGNOSTIC_PHASE": self.phase,
            "NORMAL_DIAGNOSTIC_SOURCE": self.source,
            "NORMAL_DIAGNOSTIC_CASE": self.case,
        }

    def report(self, error, stage, rank=None):
        name = next((cls.__name__ for cls in _NORMAL_TYPES if type(error) is cls), "unknown")
        category = "UNCLASSIFIED"
        if type(error) in (ValueError, RuntimeError) and len(error.args) == 1:
            value = error.args[0]
            if type(value) is str and value in _NORMAL_CODES:
                category = value
        code = error.returncode if type(error) is subprocess.CalledProcessError else None
        if type(error) is SystemExit:
            code = error.code
        report = {
            "schema": "normal-launch-primary-v1",
            "source": self.source,
            "case": self.case,
            "phase": self.phase,
            "stage": stage,
            "primary_phase": stage,
            "rank": rank,
            "exception_class": name if name in _NORMAL_CLASSES else "unknown",
            "category": category,
            "returncode": code if type(code) is int and -255 <= code <= 255 else None,
            "frames": _normal_frames(error),
        }
        original = _normal_attribute(error, "_training_primary_failure")
        if (
            type(original) is dict
            and set(original) == {"schema", "phase", "exception_class", "frames"}
            and original["schema"] == "training-lifecycle-failure-v1"
        ):
            phase, name, frames = original["phase"], original["exception_class"], original["frames"]
            if (
                type(phase) is str
                and phase in _NORMAL_PRIMARY_PHASES
                and type(name) is str
                and name in _NORMAL_CLASSES
                and type(frames) is list
                and len(frames) <= 32
            ):
                report.update(
                    primary_phase=phase,
                    exception_class=name,
                    category="UNCLASSIFIED",
                    frames=[row for row in frames if _normal_frame_valid(row)][-12:],
                )
        return report

    def eligible(self, stage, rank):
        """Relational launch scope, distinct from the original lifecycle phase."""
        if stage == "supervisor":
            return rank is None
        if self.phase == "setup":
            return stage == "bootstrap" and rank is None
        if stage == "setup":
            return self.phase in ("worker", "finish", "accept") and rank is None
        if self.phase == "task":
            return stage == "task" and rank is None
        if self.phase == "worker":
            return stage in ("pre-worker", "worker")
        return stage in ("pre-worker", self.phase) and rank is None

    def valid(self, value, stage, rank):
        if (
            type(stage) is not str
            or stage not in _NORMAL_STAGES
            or (rank is not None and (type(rank) is not int or rank not in (0, 1)))
            or not self.eligible(stage, rank)
        ):
            return False
        if type(value) is not dict or set(value) != _NORMAL_KEYS:
            return False
        if value["rank"] is not None and type(value["rank"]) is not int:
            return False
        if (
            value["schema"],
            value["source"],
            value["case"],
            value["phase"],
            value["stage"],
            value["rank"],
        ) != ("normal-launch-primary-v1", self.source, self.case, self.phase, stage, rank):
            return False
        if (
            type(value["exception_class"]) is not str
            or type(value["category"]) is not str
            or type(value["primary_phase"]) is not str
            or value["primary_phase"] not in _NORMAL_PRIMARY_PHASES
            or value["exception_class"] not in _NORMAL_CLASSES | {"unknown"}
            or value["category"] not in _NORMAL_CODES | {"UNCLASSIFIED"}
        ):
            return False
        if value["returncode"] is not None and (
            type(value["returncode"]) is not int or not -255 <= value["returncode"] <= 255
        ):
            return False
        frames = value["frames"]
        return type(frames) is list and len(frames) <= 12 and all(map(_normal_frame_valid, frames))

    def path(self, stage, rank):
        if stage not in _NORMAL_STAGES or rank not in (None, 0, 1):
            raise ValueError("NORMAL_DIAGNOSTIC_SCOPE")
        return self.root / f"{self.phase}-{stage}-{'parent' if rank is None else rank}.json"

    def persist(self, value):
        """Best effort; failure must never mask or replace the primary exception."""
        try:
            stage, rank = value["stage"], value["rank"]
            if not self.valid(value, stage, rank):
                return
            raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
            if len(raw) > 8192 or self.root.is_symlink() or not self.root.is_dir():
                return
            with self.path(stage, rank).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def read(self, stage, rank):
        try:
            path = self.path(stage, rank)
            if path.is_symlink() or self.root.is_symlink():
                return None
            with path.open("rb") as stream:
                raw = stream.read(8193)
            if len(raw) > 8192:
                return None
            value = json.loads(
                raw, object_pairs_hook=_normal_object, parse_constant=_normal_constant
            )
            return value if self.valid(value, stage, rank) else None
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def capture(self, operation, stage, rank=None):
        try:
            return operation()
        except BaseException as error:
            try:
                if _normal_exit_status(error) == 0:
                    raise
                original = _normal_attribute(error, "_normal_primary")
                if (
                    type(original) is dict
                    and type(original.get("phase")) is str
                    and original["phase"] in _NORMAL_PHASES
                ):
                    context = NormalDiagnostics(
                        self.root, original["phase"], self.source, self.case
                    )
                    if context.valid(original, original.get("stage"), original.get("rank")):
                        raise
                primary = self.report(error, stage, rank)
                self.persist(primary)
                error._normal_primary = primary  # pylint: disable=protected-access
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
            raise

    def mark(self, stage, callsite):
        """Finite phase intent is flushed before the selected child or bootstrap call."""
        try:
            if stage not in _NORMAL_STAGES or callsite not in ("bounded_run", "setup"):
                return
            value = {
                "schema": "normal-launch-stage-v1",
                "source": self.source,
                "case": self.case,
                "phase": self.phase,
                "stage": stage,
                "callsite": callsite,
            }
            if self.root.is_symlink() or self.root.is_junction():
                return
            with (self.root / (self.phase + "-entered-" + stage + ".json")).open("x") as stream:
                json.dump(value, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def supervise(self, operation, stage="setup"):
        self.mark(stage, "bounded_run")
        try:
            return operation()
        except Exception as error:
            try:
                primary = None
                # Setup precedes its descendants; rank precedence applies only
                # within the actually supervised phase and intent.
                scopes = [(stage, None)]
                if stage == "setup" and self.phase in ("worker", "finish", "accept"):
                    scopes += [
                        (child, rank)
                        for rank in (0, None, 1)
                        for child in (self.phase, "pre-worker")
                    ]
                for child, rank in scopes:
                    primary = self.read(child, rank)
                    if primary is not None:
                        break
                if primary is None:
                    primary = self.report(error, "supervisor")
                    self.persist(primary)
                error._normal_primary = primary  # pylint: disable=protected-access
            except BaseException:  # pylint: disable=broad-exception-caught
                pass
            raise


def normal_context():
    """Read only the explicit launch channel, never discover private inputs."""
    if "NORMAL_DIAGNOSTIC_ROOT" not in os.environ:
        return None
    try:
        source = os.environ["NORMAL_DIAGNOSTIC_SOURCE"]
        if len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
            return None
        return NormalDiagnostics(
            os.environ["NORMAL_DIAGNOSTIC_ROOT"],
            os.environ["NORMAL_DIAGNOSTIC_PHASE"],
            source,
            os.environ["NORMAL_DIAGNOSTIC_CASE"],
        )
    except (KeyError, ValueError, OSError):
        return None


def normal_entry(operation, stage):
    context = normal_context()
    if context is None:
        return operation()
    rank = os.environ.get("RANK")
    rank = int(rank) if rank in ("0", "1") else None
    return context.capture(operation, stage, rank)


def normal_cli(operation, stage):
    """Entry protection includes dependency imports and parser exits, without raw stderr."""
    try:
        return normal_entry(operation, stage)
    except BaseException as error:
        try:
            if _normal_exit_status(error) == 0:
                return 0
            context = normal_context()
            if context is not None:
                primary = _normal_attribute(error, "_normal_primary")
                if primary is not None and context.valid(
                    primary, primary.get("stage"), primary.get("rank")
                ):
                    print(json.dumps(primary, sort_keys=True), flush=True)
        except BaseException:  # pylint: disable=broad-exception-caught
            pass
        return _normal_exit_status(error)
