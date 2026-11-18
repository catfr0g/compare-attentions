"""Fail-closed HOST witness tests; disposable fixtures never assert scientific success."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    "SEED202_BRIDGE_TEST_CANDIDATE" not in os.environ,
    reason="Requires an explicitly selected authenticated host candidate",
)


def _bridge(monkeypatch):
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    from seed202_pilot import qualified_bridge

    return qualified_bridge


def test_http_fixture_uses_real_sdk_serialization_and_bearer(monkeypatch):
    import types

    from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from seed202_qualified_http_fixture import HttpFixture

    seen = []
    transport = types.SimpleNamespace(
        call=lambda *args: (
            seen.append(args)
            or types.SimpleNamespace(
                ref="fixture/metadata-only",
                version_number=1,
                error=None,
            )
        ),
    )
    wire = HttpFixture(transport, monkeypatch)
    request = ApiSaveKernelRequest()
    request.slug = "fixture/metadata-only"
    request.text = "# No execution: HTTP serialization fixture only"
    response = KernelsApiClient(transport).save_kernel(request)
    assert response.ref == request.slug and response.version_number == 1
    assert len(seen) == len(wire.serialized) == 1
    assert wire.serialized[0][0] == "SaveKernel"


def test_http_fixture_duration_keeps_whole_days(monkeypatch):
    import datetime
    import types

    from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
    from kagglesdk.kernels.types import kernels_api_service as sdk

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from seed202_qualified_http_fixture import HttpFixture

    response = sdk.ApiGetAcceleratorQuotaStatisticsResponse()
    quota = sdk.ApiAcceleratorQuota()
    quota.total_time_allowed = datetime.timedelta(seconds=108000)
    response.gpu_quota = quota
    transport = types.SimpleNamespace(call=lambda *_args: response)
    HttpFixture(transport, monkeypatch)
    observed = KernelsApiClient(transport).get_accelerator_quota_statistics(
        sdk.ApiGetAcceleratorQuotaStatisticsRequest(),
    )
    assert observed.gpu_quota.total_time_allowed.total_seconds() == 108000


@pytest.mark.parametrize("missing", ["root", "archive", "accepted"])
def test_descriptor_requires_complete_replay_input_closure(tmp_path, monkeypatch, missing):
    bridge = _bridge(monkeypatch)
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    _path, value = bridge.descriptor(root)
    if missing == "root":
        value["roots"] = []
    else:
        selected = value["cases"]["sorting-deltanet-seed202"]
        path = selected["archive"] if missing == "archive" else selected["spec"]["output"]
        del value["files"][str(Path(path).resolve())]
    target = tmp_path / "payload/qualified-launch/bridge.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(value), encoding="utf8")
    with pytest.raises(ValueError, match="QUALIFIED_INPUT_CLOSURE"):
        bridge.descriptor(tmp_path)


def test_foreign_scientific_import_root_stops_before_provider_auth(tmp_path, monkeypatch):
    bridge = _bridge(monkeypatch)
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    from seed202_pilot import cli
    from seed202_pilot.seal import singleton_plan

    plan = singleton_plan(
        "mqar-k64-classic-seed202",
        root,
        tmp_path / "journals",
        [Path(os.environ["SEED202_BRIDGE_TEST_DELTA"])],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf8")
    auth = []
    monkeypatch.setattr(cli, "_provider", lambda _args: auth.append("auth") or None)
    assert (
        cli.main(
            [
                "prepare",
                "--candidate-dir",
                str(root),
                "--case",
                plan["case_id"],
                "--plan",
                str(plan_path),
                "--journal",
                plan["journals"]["publish"],
                "--action",
                "publish",
                "--science-dir",
                str(tmp_path / "foreign"),
            ]
        )
        == 1
    )
    assert auth == [], "foreign source import root reached provider authentication"
    assert bridge.descriptor(root)[1]["science"] != str(tmp_path / "foreign")


def test_descriptor_requires_actual_independent_transition_review(tmp_path, monkeypatch):
    bridge = _bridge(monkeypatch)
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    _path, value = bridge.descriptor(root)
    value.pop("transition_review", None)
    target = tmp_path / "payload/qualified-launch/bridge.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(value), encoding="utf8")
    with pytest.raises(ValueError, match="BRIDGE_DESCRIPTOR_SCHEMA"):
        bridge.descriptor(tmp_path)


def test_cached_inputs_reject_unpinned_import_shadow(tmp_path, monkeypatch):
    bridge = _bridge(monkeypatch)
    # This is a pure inventory unit test, not a preparation/lease proof. The
    # public SDK chain separately retains the actual production E path gate.
    from seed202_pilot import prepared_files

    monkeypatch.setattr(prepared_files, "safe_path", Path)
    original = tmp_path / "original.py"
    original.write_text("# authenticated fixture\n", encoding="utf8")
    value = {
        "files": {
            str(original): {"bytes": original.stat().st_size, "sha256": bridge.digest(original)}
        },
        "roots": [str(tmp_path)],
    }
    (tmp_path / "injected.py").write_text("# unexpected import shadow\n", encoding="utf8")
    with pytest.raises(ValueError, match="QUALIFIED_ROOT_INVENTORY"):
        bridge.verify_files(value)


@pytest.mark.parametrize("field", ["release", "data"])
def test_descriptor_rejects_foreign_scientific_import_or_data_root(tmp_path, monkeypatch, field):
    bridge = _bridge(monkeypatch)
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    _path, value = bridge.descriptor(root)
    value["cases"]["sorting-deltanet-seed202"]["spec"][field] = str(tmp_path / "foreign")
    target = tmp_path / "payload/qualified-launch/bridge.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(value), encoding="utf8")
    with pytest.raises(ValueError, match="QUALIFIED_SOURCE_PATH_BINDING"):
        bridge.descriptor(tmp_path)


@pytest.mark.parametrize("case_id", ["mqar-k64-classic-seed202", "mqar-k64-deltanet-seed202"])
def test_new_mqar_never_uses_qualified_policy(tmp_path, monkeypatch, case_id):
    bridge = _bridge(monkeypatch)
    acceptance = Path(os.environ["SEED202_BRIDGE_TEST_DELTA"])
    with pytest.raises(ValueError, match="WHOLE_ACCEPTANCE_REQUIRED"):
        bridge.replay_predecessor({"candidate_root": str(tmp_path)}, acceptance, case_id)


@pytest.mark.parametrize(
    "field",
    [
        "review",
        "native",
        "weight_binding",
        "greedy_val",
        "original_provider_errors",
        "original_telemetry",
        "qualification",
        "no_optimization",
        "services",
        "plan",
        "recovery_provider",
        "historical_source_and_service_evidence",
        "original_execution_outcome",
        "later_pilots_authorized",
        "schema",
        "extra_key",
    ],
)
def test_any_changed_qualified_field_rejected_before_replay(tmp_path, monkeypatch, field):
    bridge = _bridge(monkeypatch)
    actual_root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    _path, descriptor = bridge.descriptor(actual_root)
    genuine = Path(os.environ["SEED202_BRIDGE_TEST_DELTA"])
    value = json.loads(genuine.read_bytes())
    value[field] = "ALTERED_NOT_SCIENCE"
    changed = tmp_path / "altered-composite.json"
    changed.write_text(json.dumps(value), encoding="utf8")
    descriptor["cases"]["sorting-deltanet-seed202"]["spec"]["output"] = str(changed)
    descriptor["files"][str(changed.resolve())] = {
        "bytes": changed.stat().st_size,
        "sha256": bridge.digest(changed),
    }
    target = tmp_path / "payload/qualified-launch/bridge.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(descriptor), encoding="utf8")
    with pytest.raises(ValueError, match="EXACT_ACCEPTED_QUALIFIED_OBJECT"):
        bridge._qualified(tmp_path, "sorting-deltanet-seed202", changed)
