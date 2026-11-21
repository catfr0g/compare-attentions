"""Real selected UV/torchrun/DEVNULL boundary, stdlib children only."""

import json
import subprocess

import pytest
from test_normal_launch_diagnostics import SECRET, tiny_kernel


def test_hard_rank_exit_survives_selected_kernel(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch, hard_exit=True)
    working = tmp_path / "working"
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", working, tmp_path / "extracted")
    public = kernel.public_error(caught.value, "entry")
    facts = public.get("elastic_failure")
    assert facts is not None, "real torchrun rank exit lost at DEVNULL boundary"
    assert facts["source"] == kernel.BINDING["normal_source_sha256"]
    assert facts["case"] == kernel.BINDING["case_id"]
    assert facts["phase"] == "worker"
    assert any(r["global_rank"] == 0 and r["exitcode"] == 23 for r in facts["failures"])
    assert all(r["signal"] is None and r["pid"] > 0 for r in facts["failures"])
    assert facts["python_trace_recovered"] is False
    assert caught.value.returncode == 1
    assert SECRET not in json.dumps(public)
    sink = working / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    assert json.loads((sink / "worker-elastic.json").read_bytes()) == facts
    assert SECRET not in b"".join(p.read_bytes() for p in sink.glob("*.json")).decode()


def test_python_rank_failure_keeps_primary_and_elastic_facts(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    public = kernel.public_error(caught.value, "entry")
    assert public["primary"]["rank"] == 0
    assert public["primary"]["exception_class"] == "ValueError"
    assert public["elastic_failure"]["failures"][0]["exitcode"] == 1
    assert public["elastic_failure"]["python_trace_recovered"] is False
    assert SECRET not in json.dumps(public)


def test_pre_main_rank_import_failure_is_finite(tmp_path, monkeypatch, capsys):
    kernel = tiny_kernel(tmp_path, monkeypatch, pre_main_error=True)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    public = kernel.public_error(caught.value, "entry")
    primary = public["primary"]
    assert primary["stage"] == "pre-worker" and primary["rank"] == 0
    assert primary["exception_class"] == "ModuleNotFoundError"
    assert any(
        row["basename"] == "normal_training_worker.py" and row["function"] == "<module>"
        for row in primary["frames"]
    )
    assert public["elastic_failure"]["failures"][0]["exitcode"] == 1
    console = capsys.readouterr()
    assert SECRET not in console.out + console.err + json.dumps(public)


def test_successful_selected_worker_has_no_elastic_failure(tmp_path, monkeypatch):
    kernel = tiny_kernel(tmp_path, monkeypatch, failing_phase="success")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        kernel.main(tmp_path / "mount", tmp_path / "working", tmp_path / "extracted")
    # Both worker ranks returned zero: only the subsequent intentionally absent
    # synthetic task dependency fails. This is not scientific task execution.
    public = kernel.public_error(caught.value, "entry")
    assert public["primary"]["phase"] == "task"
    assert "elastic_failure" not in public
    sink = tmp_path / "working" / ("normal-" + kernel.BINDING["nonce"]) / ".launch-diagnostics"
    assert (sink / "task-entered-task.json").is_file()
    assert not (sink / "worker-elastic.json").exists()


def fixture_failure(tmp_path, code=-9):
    """Synthetic negative signal projection ONLY; not Windows SIGKILL execution."""
    import normal_launch_diagnostics as diagnostics
    import normal_torchrun_entry as observer
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, ProcessFailure

    context = diagnostics.NormalDiagnostics(
        tmp_path, "worker", "a" * 64, "sorting-deltanet-seed303"
    )
    failure = ProcessFailure(0, 123, code, "<N/A>")
    failure.message = SECRET
    return observer, context, ChildFailedError(SECRET, {0: failure})


def test_synthetic_negative_signal_projection(tmp_path):
    observer, context, error = fixture_failure(tmp_path)
    value = observer.project(error, context)
    assert value["failures"][0]["signal"] == 9
    assert value["failures"][0]["error_file_reported"] is False
    assert SECRET not in json.dumps(value)
    observer.persist(value, context)
    assert observer.read(context) == value


@pytest.mark.parametrize(
    "change",
    [
        {"source": "b" * 64},
        {"case": "sorting-classic-seed303"},
        {"phase": "finish"},
        {"first_global_rank": True},
        {"python_trace_recovered": True},
        {"message": SECRET},
        {"failures": []},
    ],
)
def test_tampered_facts_rejected(tmp_path, change):
    observer, context, error = fixture_failure(tmp_path)
    value = observer.project(error, context)
    value.update(change)
    (tmp_path / observer.NAME).write_text(json.dumps(value))
    assert observer.read(context) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_rank", 2),
        ("local_rank", 1),
        ("signal", True),
        ("exitcode", 0),
        ("pid", False),
        ("timestamp", SECRET),
    ],
)
def test_tampered_rank_fields_rejected(tmp_path, field, value):
    observer, context, error = fixture_failure(tmp_path)
    record = observer.project(error, context)
    record["failures"][0][field] = value
    assert not observer.valid(record, context)


@pytest.mark.parametrize("raw", [b'{"source":"x","source":"y"}', b" " * 4097, b'{"x":NaN}'])
def test_malformed_bounded_json_rejected(tmp_path, raw):
    observer, context, _error = fixture_failure(tmp_path)
    (tmp_path / observer.NAME).write_bytes(raw)
    assert observer.read(context) is None


@pytest.mark.parametrize("code", [None, 0, 3])
def test_successful_exit_has_no_false_facts(tmp_path, monkeypatch, code):
    observer, context, _error = fixture_failure(tmp_path)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_ROOT", str(tmp_path))
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_PHASE", context.phase)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_SOURCE", context.source)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_CASE", context.case)
    error = SystemExit(code)

    def operation():
        raise error

    with pytest.raises(SystemExit) as caught:
        observer.invoke(operation)
    assert caught.value is error
    assert observer.invoke(lambda: 0) == 0
    assert not (tmp_path / observer.NAME).exists()


def test_persistence_failure_preserves_original(tmp_path, monkeypatch):
    observer, context, error = fixture_failure(tmp_path, 23)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_ROOT", str(tmp_path / "missing"))
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_PHASE", context.phase)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_SOURCE", context.source)
    monkeypatch.setenv("NORMAL_DIAGNOSTIC_CASE", context.case)

    def operation():
        raise error

    with pytest.raises(type(error)) as caught:
        observer.invoke(operation)
    assert caught.value is error
    assert not (tmp_path / "missing").exists()


def test_observe_keeps_outer_identity_and_rejects_wrong_phase(tmp_path):
    observer, context, error = fixture_failure(tmp_path)
    observer.persist(observer.project(error, context), context)
    outer = subprocess.CalledProcessError(1, [SECRET])

    def operation():
        raise outer

    with pytest.raises(subprocess.CalledProcessError) as caught:
        observer.observe(context, operation)
    assert caught.value is outer
    assert observer.public(outer, context.source, context.case) == observer.read(context)
    context.phase = "finish"
    other = subprocess.CalledProcessError(1, [SECRET])

    def fail_other():
        raise other

    with pytest.raises(subprocess.CalledProcessError):
        observer.observe(context, fail_other)
    assert "_normal_elastic" not in vars(other)
