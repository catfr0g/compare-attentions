"""R1/R2/R3 regressions: synthetic CPU diagnostics, never scientific execution."""

import json
import os
import subprocess
from pathlib import Path

import normal_launch_diagnostics as diag
import pytest

from tests.scripts.test_normal_launch_diagnostics import UV, tiny_kernel


@pytest.mark.parametrize("hostile", ["traceback", "returncode"])
def test_hostile_exception_identity_in_isolated_stdlib_child(tmp_path, hostile):
    child = r"""
import json, sys
from pathlib import Path
import normal_launch_diagnostics as diag
class PrivateError(RuntimeError):
    @property
    def __traceback__(self):
        raise OSError("synthetic_private_attribute")
class CalledProcessError(RuntimeError):
    @property
    def returncode(self):
        raise OSError("synthetic_private_attribute")
error = (PrivateError if sys.argv[2] == "traceback" else CalledProcessError)("original")
context = diag.NormalDiagnostics(Path(sys.argv[1]), "worker", "a" * 64, "case")
def operation():
    raise error
same = False
try:
    context.capture(operation, "pre-worker")
except BaseException as observed:
    same = observed is error
    observed.__context__ = None
primary = getattr(error, "_normal_primary", None)
Path(sys.argv[1], "receipt.json").write_text(json.dumps({"same": same, "primary": primary}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(diag.__file__).parent)
    result = subprocess.run(
        UV + ["-c", child, str(tmp_path), hostile],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 0
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    assert receipt["same"] is True
    assert receipt["primary"]["exception_class"] == "unknown"
    assert receipt["primary"]["stage"] == "pre-worker"
    assert receipt["primary"]["returncode"] is None
    assert "synthetic_private_attribute" not in json.dumps(receipt)


@pytest.mark.parametrize("boundary", ["capture", "supervise"])
@pytest.mark.parametrize("failure", ["report", "persist", "path", "fsync"])
def test_entire_observation_failure_never_masks_original(tmp_path, monkeypatch, boundary, failure):
    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "case")
    error = ValueError("synthetic_private_primary")

    def broken(*_args, **_kwargs):
        raise OSError("synthetic_private_observation")

    if failure == "fsync":
        monkeypatch.setattr(diag.os, "fsync", broken)
    else:
        monkeypatch.setattr(context, failure, broken)

    def operation():
        raise error

    observed_same = False
    try:
        if boundary == "capture":
            context.capture(operation, "pre-worker")
        else:
            context.supervise(operation)
    except BaseException as observed:
        observed_same = observed is error
        observed.__context__ = None
    assert observed_same


@pytest.mark.parametrize(
    "phase,stage,rank",
    [
        ("worker", "task", 0),
        ("task", "worker", 0),
        ("finish", "worker", 1),
        ("worker", "setup", 0),
        ("task", "task", 0),
        ("finish", "finish", 1),
    ],
)
def test_impossible_scope_rejected_in_memory_and_serialized(tmp_path, phase, stage, rank):
    context = diag.NormalDiagnostics(tmp_path, phase, "a" * 64, "mqar-k64-deltanet-seed202")
    value = context.report(TypeError(), stage, rank)
    assert context.valid(value, stage, rank) is False
    context.persist(value)
    assert not context.path(stage, rank).exists()
    context.path(stage, rank).write_text(json.dumps(value))
    assert context.read(stage, rank) is None


def test_genuine_setup_failure_precedes_unreached_worker_rank_zero(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch, setup_failure=True)
    original = kernel.extract_archive

    def extracted(*args):
        root = original(*args)
        if root.name == "sorting-science-fcff1b5daf574b70":
            import process_control

            real = process_control.bounded_run

            def poisoned(command, env, cwd, timeout):
                context = diag.NormalDiagnostics(
                    env["NORMAL_DIAGNOSTIC_ROOT"],
                    env["NORMAL_DIAGNOSTIC_PHASE"],
                    env["NORMAL_DIAGNOSTIC_SOURCE"],
                    env["NORMAL_DIAGNOSTIC_CASE"],
                )
                context.persist(context.report(TypeError(), "worker", 0))
                return real(command, env, cwd, timeout)

            monkeypatch.setattr(process_control, "bounded_run", poisoned)
        return root

    monkeypatch.setattr(kernel, "extract_archive", extracted)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    primary = kernel.public_error(caught.value, "entry")["primary"]
    assert primary["stage"] == "setup" and primary["rank"] is None
    assert primary["exception_class"] == "ValueError"
    assert primary["category"] == "NORMAL_SETUP_NEW_SOURCE_BINDING"


@pytest.mark.parametrize(
    "prefix",
    [
        '"exception_class":"synthetic_private_duplicate",',
        '"returncode":NaN,',
        '"returncode":Infinity,',
        '"returncode":-Infinity,',
        '"frames":[],',
    ],
)
def test_duplicate_and_nonfinite_json_never_selected(tmp_path, prefix):
    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    value = context.report(ValueError(), "pre-worker", 0)
    assert context.valid(value, "pre-worker", 0)
    context.path("pre-worker", 0).write_text("{" + prefix + json.dumps(value)[1:])
    assert context.read("pre-worker", 0) is None

    def fail():
        raise subprocess.CalledProcessError(7, ["synthetic_private_command"])

    with pytest.raises(subprocess.CalledProcessError) as caught:
        context.supervise(fail)
    assert caught.value._normal_primary["stage"] == "supervisor"
    assert caught.value._normal_primary["returncode"] == 7


@pytest.mark.parametrize("first", ["1", "NaN", "Infinity", "-Infinity"])
def test_nested_duplicate_members_are_rejected(tmp_path, first):
    context = diag.NormalDiagnostics(tmp_path, "worker", "a" * 64, "mqar-k64-deltanet-seed202")
    value = context.report(ValueError(), "pre-worker", 0)
    value["frames"] = [{"basename": "worker.py", "function": "main", "line": 1}]
    assert context.valid(value, "pre-worker", 0)
    raw = json.dumps(value).replace('"line": 1', '"line":' + first + ',"line":1')
    context.path("pre-worker", 0).write_text(raw)
    assert context.read("pre-worker", 0) is None


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("none", 0),
        ("zero", 0),
        ("false", 0),
        ("true", 1),
        ("two", 2),
        ("seven", 7),
        ("int-zero", 0),
        ("int-seven", 7),
        ("opaque", 1),
    ],
)
def test_systemexit_family_safe_status_and_real_persistence(tmp_path, cell, expected):
    """Real UV child, original identity and both capture/CLI persisted consumers."""
    child = r"""
import json, os, sys
from pathlib import Path
import normal_launch_diagnostics as diag
root = Path(sys.argv[1])
hooks = []
def hostile(name):
    hooks.append(name)
    raise OSError("synthetic_private_attribute")
class PrivateExit(SystemExit):
    @property
    def code(self): return hostile("property-code")
    @property
    def __traceback__(self): return hostile("property-traceback")
    def __getattribute__(self, name):
        if name in ("code", "__traceback__", "__class__"):
            return hostile("getattribute")
        return super().__getattribute__(name)
class PrivateInt(int):
    def __int__(self): return hostile("int")
    def __eq__(self, other): return hostile("eq")
    def __bool__(self): return hostile("bool")
    def __str__(self): return hostile("str")
    def __repr__(self): return hostile("repr")
class PrivateCode:
    def __eq__(self, other): return hostile("eq")
    def __bool__(self): return hostile("bool")
    def __str__(self): return hostile("str")
    def __repr__(self): return hostile("repr")
code = {"none": None, "zero": 0, "false": False, "true": True, "two": 2,
        "seven": 7, "int-zero": PrivateInt(0), "int-seven": PrivateInt(7),
        "opaque": PrivateCode()}[sys.argv[2]]
error = PrivateExit(code)
context = diag.NormalDiagnostics(root, "worker", "a" * 64, "case")
def operation(): raise error
same = False
try:
    context.capture(operation, "pre-worker")
except BaseException as observed:
    same = observed is error
    observed.__context__ = None
capture = context.read("pre-worker", None)
attached = object.__getattribute__(error, "__dict__").get("_normal_primary")
os.environ.update(context.environment())
class CountedOutput:
    def __init__(self, stream): self.stream, self.bytes = stream, 0
    def write(self, text):
        self.bytes += len(text)
        return self.stream.write(text)
    def flush(self): return self.stream.flush()
output = CountedOutput(sys.stdout)
sys.stdout = output
status = diag.normal_cli(operation, "pre-worker")
sys.stdout = output.stream
primary = context.read("pre-worker", None)
receipt = {"same": same, "status": status, "status_type": type(status) is int,
           "capture": capture, "attached": attached, "primary": primary,
           "hooks": hooks, "published_bytes": output.bytes}
(root / "receipt.json").write_text(json.dumps(receipt))
raise SystemExit(status)
"""
    assert diag.__file__ is not None
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(diag.__file__).parent)
    result = subprocess.run(
        UV + ["-c", child, str(tmp_path), cell],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    assert receipt["same"] is True
    assert receipt["hooks"] == []
    assert receipt["status_type"] is True
    if expected == 0:
        assert receipt["published_bytes"] == 0
        assert receipt["capture"] is None
        assert receipt["attached"] is None
        assert receipt["primary"] is None
        assert not (tmp_path / "worker-pre-worker-parent.json").exists()
    else:
        assert receipt["capture"] == receipt["attached"] == receipt["primary"]
        assert receipt["primary"]["exception_class"] == "unknown"
        assert receipt["primary"]["returncode"] is None
        assert receipt["primary"]["stage"] == "pre-worker"
    assert receipt["status"] == expected
    assert result.returncode == expected
    text = json.dumps(receipt)
    assert "PrivateExit" not in text and "synthetic_private_attribute" not in text
