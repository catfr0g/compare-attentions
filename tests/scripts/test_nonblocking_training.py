"""Auxiliary service failures must not reject native completed science."""

import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import normal_training_acceptance as acceptance
import pytest

ROOT = Path(
    "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/mqar-delta303-terminal23"
)
RESULT = ROOT / "rawoutputs/normal-007d90ea4622493a8fe41959713a816a"


def captured(monkeypatch, raw):
    """Replace only transport; retain actual native/history/config validators."""
    import quality_inherit
    from training_completion import TrainingServiceTracker

    services = SimpleNamespace(
        service_env=lambda *_: contextlib.nullcontext(), safe=lambda _, action: action()
    )
    monkeypatch.setattr(quality_inherit, "leaf", lambda _: services)
    monkeypatch.setattr(TrainingServiceTracker, "_production_readback", staticmethod(lambda _: raw))


def test_captured_twenty_late_resource_rows_do_not_block(monkeypatch):
    fresh = json.loads((ROOT / "clearml-training-readback.quarantine.json").read_bytes())
    cached = json.loads((RESULT / "out/training-unsampled-readback.json").read_bytes())
    assert len(fresh["events"]) - len(cached["events"]) == 20
    before = copy.deepcopy(fresh)
    captured(monkeypatch, fresh)
    case = json.loads((RESULT / "training-task.json").read_bytes())["case"]
    observed = acceptance.training_readback(RESULT, case, {})
    assert observed["raw"] == before == fresh
    assert observed["receipt"]["scientific_event_count"] == 11849
    assert observed["receipt"]["resource_event_count"] == 1800
    assert observed["diagnostic"]["status"] == "verified"


@pytest.mark.parametrize("fault", ["getter", "scalar", "config", "unknown", "pages"])
def test_tracking_discrepancies_are_truthful_warnings(monkeypatch, fault):
    from training_completion import TrainingServiceTracker

    raw = json.loads((ROOT / "clearml-training-readback.quarantine.json").read_bytes())
    if fault == "scalar":
        raw["events"][0]["value"] += 1
    elif fault == "config":
        raw["parameters"]["optim"]["lr"] = "not-the-native-config"
    elif fault == "unknown":
        raw["events"][0]["metric"] = "application-extra"
    elif fault == "pages":
        raw["pages"][0]["events"][0]["value"] += 1
    captured(monkeypatch, raw)
    if fault == "getter":

        def unavailable(_):
            raise RuntimeError("PRIVATE_MESSAGE_MUST_NOT_ESCAPE")

        monkeypatch.setattr(TrainingServiceTracker, "_production_readback", unavailable)
    case = json.loads((RESULT / "training-task.json").read_bytes())["case"]
    with pytest.warns(RuntimeWarning, match="auxiliary tracking"):
        observed = acceptance.training_readback(RESULT, case, {})
    assert observed["diagnostic"]["status"] == "warning"
    assert observed["diagnostic"]["authority"] == "AUXILIARY_ONLY"
    assert observed["receipt"] is None
    if fault != "getter":
        assert observed["raw"] == raw
    assert "PRIVATE_MESSAGE" not in json.dumps(observed)


@pytest.mark.parametrize("phase", ["start", "log_loss", "flush", "close", "readback", "receipt"])
def test_native_backup_identity_survive_service_failure(phase):
    from training_completion import CompletionTracker, TrainingServiceTracker, _FlushTask

    events = []

    def call(name):
        events.append(name)
        if name == phase:
            raise RuntimeError("PRIVATE_MESSAGE_MUST_NOT_ESCAPE")

    task = SimpleNamespace(
        id="synthetic", flush=lambda **_: call("flush"), close=lambda: call("close")
    )
    service = TrainingServiceTracker({}, "synthetic", updates=1, validations=1)
    service._proxy = _FlushTask(task, lambda _: (call("readback"), {})[1])
    service.delegate = SimpleNamespace(finish=lambda *_: service._proxy.close())
    service._validate_receipt = lambda *_: call("receipt")
    service.start = lambda _: call("start")
    service.log_loss = lambda *_: call("log_loss")
    backup = SimpleNamespace(
        record_scalar_readback=lambda _: None,
        start=lambda _: events.append("native_start"),
        log_loss=lambda *_: events.append("native_loss"),
        finish=lambda *_: events.append("native_finish"),
        record_identity=lambda _: events.append("native_identity"),
    )
    tracker = CompletionTracker(backup, service)
    result = SimpleNamespace(provenance={"status": "completed"})
    with pytest.warns(RuntimeWarning, match="auxiliary tracking"):
        tracker.start({})
        tracker.log_loss(1, 1)
        tracker.finish(result, None)
    assert events.index("native_finish") < events.index("flush")
    assert events[-1] == "native_identity"
    assert tracker.diagnostics
    assert all(row["diagnostic"]["status"] == "warning" for row in tracker.diagnostics)
    assert "PRIVATE_MESSAGE" not in json.dumps(tracker.diagnostics)


def test_native_backup_failure_remains_fatal():
    from training_completion import CompletionTracker

    events = []

    def invalid(*_):
        raise ValueError("NATIVE_INVALID")

    backup = SimpleNamespace(record_scalar_readback=None, finish=invalid)
    service = SimpleNamespace(finish=lambda *_: events.append("service"))
    tracker = CompletionTracker(backup, service)
    with pytest.raises(ValueError, match="^NATIVE_INVALID$"):
        tracker.finish(None, None)
    assert events == []


@pytest.mark.parametrize("native_invalid", [False, True])
def test_actual_host_consumer_does_not_gate_on_saved_tracking(
    monkeypatch, tmp_path, native_invalid
):
    import normal_training_native
    import unified_normal_host as host
    import unified_normal_publications

    case = {"engineering_only": True}
    (tmp_path / "native-proof.json").write_text('{"valid":true,"final_sha256":"synthetic"}')
    (tmp_path / "training-fresh-readback.json").write_text('{"different":"tracking"}')
    monkeypatch.setattr(acceptance, "verify_sources", lambda *_: {})

    def native(*_):
        if native_invalid:
            raise ValueError("NATIVE_INVALID")
        return {"valid": True, "final_sha256": "synthetic"}

    monkeypatch.setattr(normal_training_native, "verify_native", native)
    monkeypatch.setattr(acceptance, "training_readback", lambda *_: {"receipt": {}, "raw": {}})
    monkeypatch.setattr(host, "task_quality", lambda *_: {})
    monkeypatch.setattr(host, "inference_readback", lambda *_: {})
    publications = []
    monkeypatch.setattr(
        unified_normal_publications,
        "publication_readback",
        lambda *_: publications.append(1) or {"publications": []},
    )
    if native_invalid:
        with pytest.raises(ValueError, match="^NATIVE_INVALID$"):
            host.consume(tmp_path, tmp_path, case, {}, tmp_path, snapshot=tmp_path, accept=True)
        assert not publications
    else:
        result = host.consume(
            tmp_path, tmp_path, case, {}, tmp_path, snapshot=tmp_path, accept=True
        )
        assert result["status"] == "ACCEPTED_UNIFIED_NORMAL"
        assert result["host_reader"]["release"] == "normal-host-readonly-nonblocking-tracking-v3"


@pytest.mark.parametrize("invalid", [None, "budget", "config", "data", "weights", "quality"])
def test_actual_finisher_validates_science_before_hf(monkeypatch, tmp_path, invalid):
    import normal_training_worker as worker
    import quality_recovery
    import quality_services
    import unified_normal_host

    events = []
    root = tmp_path / "result"
    root.mkdir()
    args = SimpleNamespace(
        output=root, package=tmp_path, data=tmp_path, prefix="normal-training/synthetic"
    )
    case = {"updates": 1, "checkpoint_steps": [], "engineering_only": False}
    monkeypatch.setattr(acceptance, "verify_sources", lambda *_: events.append("source"))

    def native(*_):
        events.append("native")
        if invalid in {"budget", "config", "data", "weights"}:
            raise ValueError("NATIVE_" + invalid.upper())
        return {}

    def quality(*_):
        events.append("quality")
        if invalid == "quality":
            raise ValueError("NATIVE_QUALITY")
        return {}

    monkeypatch.setattr(worker, "verify_native", native)
    monkeypatch.setattr(unified_normal_host, "task_quality", quality)
    monkeypatch.setattr(
        acceptance,
        "training_readback",
        lambda *_: {
            "receipt": None,
            "raw": None,
            "diagnostic": {"status": "warning", "authority": "AUXILIARY_ONLY"},
        },
    )
    monkeypatch.setattr(quality_services, "HFTransport", lambda _: object())

    class Publisher:
        def __init__(self, *_):
            pass

        def complete(self, _):
            return []

        def publish(self, *_):
            events.append("hf_publish")
            return {"state": "verified"}

    monkeypatch.setattr(quality_recovery, "RecoveryPublisher", Publisher)
    if invalid:
        with pytest.raises(ValueError, match="^NATIVE_" + invalid.upper() + "$"):
            worker.finish(args, case, {})
        assert "hf_publish" not in events
    else:
        worker.finish(args, case, {})
        assert events == ["source", "native", "quality", "hf_publish"]


def test_legacy_reader_identity_uses_its_frozen_bytes(tmp_path):
    import hashlib

    from unified_normal_host_view import reader_identity

    names = (
        "unified_normal_host.py",
        "unified_normal_host_view.py",
        "unified_normal_ordinary.py",
        "unified_normal_publications.py",
    )
    root = tmp_path / "payload/source/src"
    root.mkdir(parents=True)
    pins = {}
    for name in names:
        (root / name).write_bytes(b"historical frozen reader")
        pins[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    record = {
        "release": "normal-host-readonly-equal-iteration-v2",
        "purpose": "posttraining-readonly-observer-client",
        "files": pins,
    }
    assert reader_identity(candidate=tmp_path, recorded=record) == record
    (root / names[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="UNIFIED_LEGACY_READER_BYTES"):
        reader_identity(candidate=tmp_path, recorded=record)


def test_task_tracking_failure_preserves_cached_report(monkeypatch, tmp_path):
    import quality_inherit
    import unified_normal_task as task

    def unavailable(*_, **__):
        raise RuntimeError("PRIVATE_MESSAGE_MUST_NOT_ESCAPE")

    monkeypatch.setattr(quality_inherit, "leaf", lambda _: SimpleNamespace(credentials=unavailable))
    report = tmp_path / "task-val.json"
    report.write_text('{"genuine-cached-report":"synthetic-only"}')
    before = report.read_bytes()
    args = SimpleNamespace(output=tmp_path, private=tmp_path / "not-read.env")
    with pytest.warns(RuntimeWarning, match="auxiliary tracking"):
        observed = task.publish_tracking(args, {}, {})
    assert observed["diagnostic"]["status"] == "warning"
    assert report.read_bytes() == before
    assert "PRIVATE_MESSAGE" not in json.dumps(observed)


def test_auxiliary_failure_does_not_poison_later_native_primary():
    from training_completion import CompletionTracker, TrainingServiceTracker, _FlushTask

    from compare_attention.train.lifecycle import observe_failures

    reports = []

    def down(**_):
        raise RuntimeError("auxiliary down")

    def native_invalid(_):
        raise ValueError("NATIVE_IDENTITY_INVALID")

    service = TrainingServiceTracker({}, "synthetic", updates=1, validations=1)
    service._proxy = _FlushTask(SimpleNamespace(flush=down, id="synthetic"), None)
    service.delegate = SimpleNamespace(finish=lambda *_: service._proxy.close())
    backup = SimpleNamespace(
        record_scalar_readback=None,
        finish=lambda *_: None,
        record_identity=native_invalid,
        record_failure=reports.append,
    )
    tracker = CompletionTracker(backup, service)
    with pytest.warns(RuntimeWarning, match="auxiliary tracking"):
        with pytest.raises(ValueError, match="^NATIVE_IDENTITY_INVALID$"):
            with observe_failures(tracker):
                tracker.finish(SimpleNamespace(provenance={}), None)
    assert len(reports) == 1
    assert reports[0]["phase"] == "identity_write"
    assert reports[0]["exception_class"] == "ValueError"


def test_genuine_three_legacy_receipts_keep_their_frozen_v2_reader():
    from unified_normal_host_view import reader_identity

    candidate = ROOT.parent / "remaining-six-fixed-source13/next-mqar-delta303-17/candidate"
    request = json.loads((candidate / "payload/unified/request.json").read_bytes())
    assert len(request["predecessors"]) == 3
    for row in request["predecessors"]:
        original = Path(row["replay"]["source_request"]).parents[2]
        recorded = json.loads(Path(row["path"]).read_bytes())["host_reader"]
        assert reader_identity(candidate=original, recorded=recorded, inputs=request) == recorded
