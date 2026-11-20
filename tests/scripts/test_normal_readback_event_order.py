"""Tie-order regressions: retain full rows, multiplicities, and original evidence."""

import copy
import json
from collections import Counter
from pathlib import Path

import normal_training_acceptance as acceptance
import pytest
import training_completion
from test_normal_finish_readback_diagnostics import saved_finish as saved_finish

CAPTURED = Path(
    "E:/HermesArtifacts/compare-attention/unified-normal-launch-contract-v1/"
    "normal-finish-cloud-probe01/outputs/first-fresh-clearml-response.json"
)


def test_genuine_captured_equal_iteration_permutation(saved_finish, monkeypatch):
    """Replay retained genuine response, not a new SDK read or fabricated success."""
    if not CAPTURED.is_file():
        pytest.skip("Retained private cloud response unavailable")
    root, envelope, stored, _context = saved_finish
    fresh = json.loads(CAPTURED.read_bytes())
    original = copy.deepcopy(fresh)

    def canonical(row):
        return json.dumps(row, sort_keys=True, allow_nan=False)

    assert fresh["events"] != stored["events"]
    assert Counter(map(canonical, fresh["events"])) == Counter(map(canonical, stored["events"]))
    monkeypatch.setattr(
        training_completion.TrainingServiceTracker,
        "_production_readback",
        staticmethod(lambda _task: fresh),
    )
    result = acceptance.training_readback(root, envelope["case"], {})
    assert result["receipt"] == envelope["receipt"]
    assert result["raw"] is fresh
    assert fresh == original
    assert json.loads((root / "out/training-unsampled-readback.json").read_bytes()) == stored


@pytest.fixture
def portable_finish(tmp_path, monkeypatch):
    """Complete tiny metadata-only producer, with the real receipt and native join."""
    config = {
        "optim": {"lr_schedule": "constant"},
        "schedule": {"train_flops": 10, "val_every_steps": 1},
        "events": ["configuration order remains semantic", "do not reorder"],
    }
    result = {
        "config": config,
        "steps": 1,
        "train_flops": 10,
        "stop_reason": "budget",
        "quality": {"train_loss": 1.0, "val_loss": 0.5, "val_accuracy": 0.25},
        "speed": {"train_tokens_per_second": 10, "peak_gpu_bytes": 100},
    }
    scalars = {
        "steps": 1,
        "train_flops": 10,
        "requested_train_flops": 10,
        "early_stopped": 0,
        "budget_exhausted": 1,
        **result["quality"],
        **result["speed"],
    }
    events = [
        {"metric": "train", "variant": "loss", "iter": 10, "value": 1.0},
        {"metric": "validation", "variant": "loss", "iter": 10, "value": 0.5},
        {"metric": "validation", "variant": "accuracy", "iter": 10, "value": 0.25},
        {"metric": ":monitor:machine", "variant": "cpu_usage", "iter": 10, "value": 4},
        {"metric": ":monitor:machine", "variant": "memory_free_gb", "iter": 10, "value": 8},
        *({"metric": "result", "variant": k, "iter": 1, "value": v} for k, v in scalars.items()),
    ]
    for row in events:
        row.update(timestamp=123, task="synthetic")
    raw = {
        "events": events,
        "status": "completed",
        "task_id": "synthetic",
        "total": len(events),
        "parameters": training_completion.clearml_domain(config),
        "pages": [{"events": copy.deepcopy(events), "total": len(events)}],
    }
    case = {
        "updates": 1,
        "validation_steps": [1],
        "config": config,
        "checkpoint_steps": [],
        "engineering_only": True,
    }
    receipt = training_completion.validate_unsampled_readback(
        raw,
        task_id="synthetic",
        updates=1,
        validations=1,
        config=config,
    )
    identity = {"status": "completed"}
    envelope = {
        "schema": "normal-training-return-v1",
        "case": case,
        "task_id": "synthetic",
        "name": "explicit-synthetic-fixture",
        "identity": identity,
        "receipt": receipt,
    }
    root = tmp_path / "run"
    (root / "out").mkdir(parents=True)
    for name, value in {
        "training-task.json": envelope,
        "out/run_identity.json": identity,
        "out/training-unsampled-readback.json": raw,
    }.items():
        (root / name).write_text(json.dumps(value))
    history = [
        {"type": "step", "flops": 10, "loss": 1.0},
        {"type": "validation", "flops": 10, "loss": 0.5, "accuracy": 0.25},
        result,
    ]
    (root / "out/metrics.jsonl").write_text("\n".join(map(json.dumps, history)))
    monkeypatch.setattr(acceptance, "verify_sources", lambda *_args: None)
    return root, envelope, raw, None


@pytest.mark.parametrize(
    "mutation", [None, "receipt", "parameters", "status", "pages", "outer", "config_events"]
)
def test_host_accepts_only_equal_iteration_permutation(portable_finish, monkeypatch, mutation):
    """Real host consumer; native model and HF transport are explicit stand-ins."""
    root, envelope, stored, _context = portable_finish
    fresh = copy.deepcopy(stored)
    fresh["events"][:3] = fresh["events"][1:3] + fresh["events"][:1]
    assert fresh["events"] != stored["events"]
    assert len({row["iter"] for row in fresh["events"][:3]}) == 1
    monkeypatch.setattr(
        training_completion.TrainingServiceTracker,
        "_production_readback",
        staticmethod(lambda _task: fresh),
    )
    native = {"explicit_native_stand_in_no_model_execution": True}
    monkeypatch.setattr(acceptance, "verify_native", lambda *_args: native)
    (root / "native-proof.json").write_text(json.dumps(native))
    previous = {
        "raw": copy.deepcopy(stored),
        "receipt": copy.deepcopy(envelope["receipt"]),
        "fresh_sdk_readback": True,
    }
    if mutation == "receipt":
        previous["receipt"]["flush_before_close"] = False
    elif mutation == "parameters":
        previous["raw"]["parameters"]["new"] = "changed"
    elif mutation == "status":
        previous["raw"]["status"] = "failed"
    elif mutation == "pages":
        previous["raw"]["pages"][0]["events"].reverse()
    elif mutation == "outer":
        previous["new"] = "changed"
    elif mutation == "config_events":
        previous["raw"]["parameters"]["events"] = "changed order"
    (root / "training-fresh-readback.json").write_text(json.dumps(previous))
    periodic = [
        {"step": step, "revision": str(step), "state": "verified", "files": {}}
        for step in envelope["case"]["checkpoint_steps"]
    ]
    (root / "periodic-readback.json").write_text(json.dumps({"publications": periodic}))
    import hashlib

    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    final = {
        "revision": "final-fixture",
        "state": "verified",
        "files": {
            n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()
        },
    }
    publication = {
        "case": envelope["case"],
        "schema": "normal-training-publication-v1",
        "final": final,
    }
    (root.parent / "run-publication.json").write_text(json.dumps(publication))
    import quality_services

    class FixtureHFTransport:
        """No service access: return the explicitly constructed byte fixture."""

        def __init__(self, _values):
            pass

        def read(self, row, revision):
            return {
                "private": True,
                "revision": revision,
                "files": {n: files[n] for n in row["files"]},
            }

    monkeypatch.setattr(quality_services, "HFTransport", FixtureHFTransport)
    if mutation is not None:
        with pytest.raises(ValueError, match="^NORMAL_HOST_SERVICE_DRIFT$"):
            acceptance.accept(root, root, envelope["case"], {})
        assert not (root.parent / "run-normal-accepted.json").exists()
        return
    result = acceptance.accept(root, root, envelope["case"], {})
    assert result["status"] == "ACCEPTED_NORMAL_LIFECYCLE"
    assert result["training"] == envelope["receipt"]
    assert fresh["events"] != stored["events"]
    assert json.loads((root / "training-fresh-readback.json").read_bytes()) == previous


@pytest.mark.parametrize(
    "mutation,guard",
    [
        (None, None),
        ("timestamp", "NORMAL_FRESH_UNSAMPLED_DRIFT"),
        ("value", "NORMAL_FRESH_UNSAMPLED_DRIFT"),
        ("axis", "CLEARML_VALIDATION_AXIS"),
        ("variant", "CLEARML_COMPLETE_EVENT_COUNTS"),
        ("missing", "CLEARML_UNSAMPLED_HISTORY"),
        ("added", "CLEARML_UNSAMPLED_HISTORY"),
        ("duplicate_replacement", "NORMAL_FRESH_UNSAMPLED_DRIFT"),
        ("resource_value", "NORMAL_FRESH_UNSAMPLED_DRIFT"),
        ("resource_identity", "NORMAL_FRESH_UNSAMPLED_DRIFT"),
        ("groups", "NORMAL_FRESH_UNSAMPLED_DRIFT"),
        ("parameters", "CLEARML_CONFIG_READBACK"),
    ],
)
def test_portable_producer_comparison(portable_finish, monkeypatch, mutation, guard):
    root, envelope, stored, _context = portable_finish
    fresh = copy.deepcopy(stored)
    fresh["events"][:3] = fresh["events"][1:3] + fresh["events"][:1]
    if mutation == "timestamp":
        fresh["events"][0]["timestamp"] += 1
    elif mutation == "value":
        # Two fields move together, so internal counts/quality still pass.
        fresh["events"][0]["value"] = 0.75
        next(r for r in fresh["events"] if r["variant"] == "val_loss")["value"] = 0.75
    elif mutation == "axis":
        fresh["events"][0]["iter"] += 1
    elif mutation == "variant":
        fresh["events"][0]["variant"] = "unknown"
    elif mutation == "missing":
        fresh["events"].pop()
    elif mutation == "added":
        fresh["events"].append(copy.deepcopy(fresh["events"][3]))
    elif mutation == "duplicate_replacement":
        fresh["events"][4] = copy.deepcopy(fresh["events"][3])
    elif mutation == "resource_value":
        fresh["events"][3]["value"] += 1
    elif mutation == "resource_identity":
        fresh["events"][3]["variant"] = "memory_used_gb"
    elif mutation == "groups":
        fresh["events"] = fresh["events"][5:] + fresh["events"][:5]
    elif mutation == "parameters":
        fresh["parameters"]["events"] = "changed"
    original = copy.deepcopy(fresh)
    monkeypatch.setattr(
        training_completion.TrainingServiceTracker,
        "_production_readback",
        staticmethod(lambda _task: fresh),
    )
    if guard:
        with pytest.raises(ValueError, match="^" + guard + "$"):
            acceptance.training_readback(root, envelope["case"], {})
    else:
        accepted = acceptance.training_readback(root, envelope["case"], {})
        assert accepted["raw"] is fresh
        assert accepted["receipt"] == envelope["receipt"]
    assert fresh == original
    assert json.loads((root / "out/training-unsampled-readback.json").read_bytes()) == stored
