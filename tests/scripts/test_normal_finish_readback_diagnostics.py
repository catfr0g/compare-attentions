"""Retain exact NORMAL readback guards without replaying scientific execution."""

import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import normal_launch_diagnostics as diagnostics
import normal_training_worker as worker
import pytest
import training_completion

ROOT = Path(
    "E:/HermesArtifacts/compare-attention/unx04/failed-output04/"
    "normal-820452d6863fa10d106107f0a4fbf4f1"
)
GUARDS = {
    "envelope": "NORMAL_RETURN_ENVELOPE",
    "receipt": "NORMAL_RETURN_RECEIPT",
    "fresh": "NORMAL_FRESH_UNSAMPLED_DRIFT",
    "native": "NORMAL_NATIVE_SERVICE_VALUES",
}


@pytest.fixture
def saved_finish(tmp_path, monkeypatch):
    """Genuine serialized producer records; native execution and transport are out of scope."""
    if not ROOT.is_dir():
        pytest.skip("Retained private producer records unavailable")
    root = tmp_path / "run"
    (root / "out").mkdir(parents=True)
    for name in (
        "training-task.json",
        "out/run_identity.json",
        "out/training-unsampled-readback.json",
        "out/metrics.jsonl",
    ):
        (root / name).write_bytes((ROOT / name).read_bytes())
    envelope = json.loads((root / "training-task.json").read_bytes())
    raw = json.loads((root / "out/training-unsampled-readback.json").read_bytes())

    def deny_network(*_args, **_kwargs):
        raise AssertionError("UNEXPECTED_NETWORK")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(
        training_completion.TrainingServiceTracker,
        "_production_readback",
        staticmethod(lambda _task: raw),
    )
    import normal_training_acceptance as acceptance

    monkeypatch.setattr(acceptance, "verify_sources", lambda *_args: None)
    monkeypatch.setattr(
        worker, "verify_native", lambda *_args: {"fixture_native_not_executed": True}
    )
    channel = tmp_path / "diagnostics"
    channel.mkdir()
    context = diagnostics.NormalDiagnostics(channel, "finish", "7" * 64, envelope["case"]["id"])
    return root, envelope, raw, context


@pytest.mark.parametrize("mutation", GUARDS)
def test_finish_retains_actual_consumer_guard_and_frame(saved_finish, mutation):
    root, envelope, raw, context = saved_finish
    case = copy.deepcopy(envelope["case"])
    if mutation == "envelope":
        case["seed"] += 1
    elif mutation == "receipt":
        envelope["receipt"]["flush_before_close"] = False
        (root / "training-task.json").write_text(json.dumps(envelope))
    elif mutation == "fresh":
        next(row for row in raw["events"] if training_completion.resource_event(row))["value"] += 1
    else:
        path = root / "out/metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row["type"] == "step")["loss"] += 1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    args = SimpleNamespace(output=root, package=root, prefix="normal-training/fixture")
    with pytest.raises(ValueError, match="^" + GUARDS[mutation] + "$"):
        context.capture(lambda: worker.finish(args, case, {}), "finish")
    report = context.read("finish", None)
    assert report["category"] == GUARDS[mutation]
    assert report["exception_class"] == "ValueError"
    assert report["phase"] == report["primary_phase"] == "finish"
    assert any(
        row["basename"] == "normal_training_acceptance.py"
        and row["function"] == "training_readback"
        for row in report["frames"]
    )
    if mutation == "native":
        assert any(row["function"] == "verify_native_service" for row in report["frames"])
    assert not (root / "training-fresh-readback.json").exists()
    assert not (root.parent / "run-publication.json").exists()


def test_valid_saved_readback_keeps_all_events_and_creates_no_diagnostic(saved_finish):
    from normal_training_acceptance import training_readback

    root, envelope, raw, context = saved_finish
    result = context.capture(lambda: training_readback(root, envelope["case"], {}), "finish")
    assert result["raw"] == raw
    assert result["receipt"] == envelope["receipt"]
    assert not list(context.root.iterdir())


@pytest.mark.parametrize("message", ["NORMAL_RETURN_ENVELOPE private-canary", "private-canary"])
def test_unknown_message_is_not_promoted_to_guard(tmp_path, message):
    context = diagnostics.NormalDiagnostics(tmp_path, "finish", "7" * 64, "fixture")
    report = context.report(ValueError(message), "finish")
    assert report["category"] == "UNCLASSIFIED"
    assert "private-canary" not in json.dumps(report)
