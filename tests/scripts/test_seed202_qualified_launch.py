"""Genuine-artifact launch regressions; paths are supplied by the operator."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    "SEED202_BRIDGE_TEST_CANDIDATE" not in os.environ,
    reason="Requires supplied immutable accepted artifacts; never fabricated science",
)


def test_public_next_uses_qualified_bridge(monkeypatch):
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    from seed202_pilot import cli

    assert (
        cli.main(
            [
                "next",
                "--candidate-dir",
                str(root),
                "--qualified-predecessor",
                str(root / "payload/predecessor/qualified-acceptance.json"),
                "--acceptance",
                os.environ["SEED202_BRIDGE_TEST_DELTA"],
            ]
        )
        == 0
    )


def _ordinary_transport_fixture(
    transport,
    root,
    monkeypatch,
    *,
    case_id="mqar-k64-classic-seed202",
    nonce="TRANSPORT_FIXTURE_NO_TRAINING",
):
    """Exercise ordinary whole-object persistence/replay, NOT the GPU scientific gate."""
    from seed202_pilot import science
    from seed202_pilot.acceptance import complete
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.storage import inventory

    case = require_authorized_case(case_id)
    downloaded = transport.root / ("FIXTURE-ONLY-download-" + nonce)
    run = downloaded / ("seed202-pilot-" + nonce)
    run.mkdir(parents=True)
    rows = {
        "intent.json": {
            "schema_version": 1,
            "case_id": case.case_id,
            "nonce": nonce,
            "case_deadline_seconds": case.case_deadline_seconds,
            "policy": "STOPALL_ANYFAIL",
        },
        "finish-returned.json": {
            "case_id": case.case_id,
            "status": "AWAITING_PROVIDER_DOWNLOAD_AND_HOST_WHOLE_ACCEPTANCE",
        },
        case.case_id + "-worker-accepted.json": {
            "case_id": case.case_id,
            "status": "AWAITING_HOST_WHOLE_ACCEPTANCE",
        },
    }
    for name, value in rows.items():
        (run / name).write_text(json.dumps(value), encoding="utf8")
    index = transport.root / ("FIXTURE-ONLY-index-" + nonce + ".json")
    index.write_text(json.dumps({"inventory": inventory(downloaded)}), encoding="utf8")
    snapshot = transport.root / ("FIXTURE-ONLY-service-snapshot-" + nonce)
    snapshot.mkdir()
    services = transport.root / ("FIXTURE-ONLY-services-" + nonce + ".json")
    services.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": "FRESH_REMOTE_SERVICE_SNAPSHOT",
                "snapshot": str(snapshot),
                "snapshot_inventory": {},
                "periodic_revisions": 11,
                "final_all_bytes": True,
                "clearml": {"status": "completed"},
            }
        ),
        encoding="utf8",
    )
    # The unchanged native science implementation is inherited, not replaced in
    # the candidate. Only this transport fixture replaces its leaf execution.
    monkeypatch.setattr(
        science,
        "replay",
        lambda **_kwargs: {
            "scope": "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY",
            "periodic": {"revision_count": 11, "full_count": 21},
            "engineering_fixture": "NO_GPU_NO_NATIVE_MQAR_RESULT",
        },
    )
    provider = {
        "typed_status": "COMPLETE",
        "head": 1,
        "version": 1,
        "private": True,
        "numeric_id": 999999999,
    }
    output = transport.root / ("FIXTURE-ONLY-ordinary-MQAR-acceptance-" + nonce + ".json")
    complete(
        case,
        nonce,
        downloaded,
        index,
        provider,
        provider,
        services,
        task_data=transport.mqar,
        wheel=root / "payload/inherited/training-wheel.whl",
        parent_archive=root / "payload/inherited/parent-candidate03.zip.bin",
        science_dir=transport.root,
        config_path=root / "payload/configs" / (case.case_id + ".yaml"),
        periodic_destination=transport.root / "FIXTURE-ONLY-periodic",
        output=output,
    )
    return output


def _post_training_transport(transport, wire, root, monkeypatch, case, plan, approval):
    """Public read/accept route with synthetic completion; NEVER a model result."""
    from seed202_pilot import host_services

    nonce = plan["nonce"]
    fixture = _ordinary_transport_fixture(
        transport,
        root,
        monkeypatch,
        case_id=case,
        nonce=nonce,
    )
    binding = json.loads(fixture.read_bytes())["replay_binding"]
    fixture_download = Path(binding["downloaded"])
    wire.outputs = {
        path.relative_to(fixture_download).as_posix(): path.read_bytes()
        for path in fixture_download.rglob("*")
        if path.is_file()
    }
    transport.kernels[plan["deployment"]["kernel_target"]].status = "COMPLETE"
    downloaded = transport.root / "CLI-FIXTURE-ONLY-collected"
    index = transport.root / "CLI-FIXTURE-ONLY-index.json"
    journal = Path(plan["journals"]["collect"])
    assert (
        transport.command(
            "collect",
            case,
            "--approval",
            approval,
            "--destination",
            downloaded,
            "--index",
            index,
            journal=journal,
        )
        == 0
    ), transport.last_trace
    # Service semantics are inherited unchanged; this fixture exercises the
    # public saved-file contract without contacting a real service or a task.
    monkeypatch.setattr(
        host_services,
        "consume",
        lambda *_args: json.loads(Path(binding["services_evidence"]).read_bytes()),
    )
    services = transport.root / "CLI-FIXTURE-ONLY-services.json"
    assert (
        transport.command(
            "services",
            case,
            "--approval",
            approval,
            "--destination",
            downloaded,
            "--output",
            services,
            journal=journal,
        )
        == 0
    )
    accepted = transport.root / "CLI-FIXTURE-ONLY-accepted.json"
    assert (
        transport.command(
            "accept",
            case,
            "--approval",
            approval,
            "--destination",
            downloaded,
            "--index",
            index,
            "--services-evidence",
            services,
            "--task-data",
            transport.mqar,
            "--wheel",
            root / "payload/inherited/training-wheel.whl",
            "--config-path",
            root / "payload/configs" / (case + ".yaml"),
            "--periodic-destination",
            transport.root / "CLI-FIXTURE-ONLY-periodic",
            "--output",
            accepted,
            journal=journal,
        )
        == 0
    ), transport.last_trace
    value = json.loads(accepted.read_bytes())
    assert value["status"] == "ACCEPTED_SEED202_FULL"
    assert value["scientific"]["engineering_fixture"] == "NO_GPU_NO_NATIVE_MQAR_RESULT"
    return accepted


@pytest.mark.parametrize("case", ["mqar-k64-classic-seed202", "mqar-k64-deltanet-seed202"])
def test_genuine_prefix_public_sdk_chain_stops_on_unaccepted_mqar(monkeypatch, case):
    """Real predecessors; synthetic SDK I/O only; no MQAR scientific success claim."""
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    monkeypatch.syspath_prepend(str(root / "payload/source/tests/seed202_pilot"))
    from fresh_sdk_fixture import FreshTransport
    from seed202_pilot import cli
    from test_fresh_public_chain_integration06 import _approve, _prepare, _probe

    transport = FreshTransport(root, monkeypatch)
    original_provider = cli._provider
    source_probe = root / "payload/source/src/seed202_pilot/qualified_imports.py"

    def leased_provider(args):
        before = source_probe.read_bytes()
        with pytest.raises(PermissionError):
            with source_probe.open("r+b"):
                pass  # OPEN_EXISTING only: never truncate or write any bytes.
        assert source_probe.read_bytes() == before
        return original_provider(args)

    monkeypatch.setattr(cli, "_provider", leased_provider)
    transport.enable_mutations()
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from seed202_qualified_http_fixture import HttpFixture

    wire = HttpFixture(transport, monkeypatch)
    prefix = ["--acceptance", os.environ["SEED202_BRIDGE_TEST_DELTA"]]
    expected_scopes = ["QUALIFIED_SORTING_CLASSIC_TAIL_V1", "QUALIFIED_SORTING_DELTA_TAIL_V1"]
    if case == "mqar-k64-deltanet-seed202":
        ordinary = _ordinary_transport_fixture(transport, root, monkeypatch)
        prefix += ["--acceptance", ordinary]
        expected_scopes.append("AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY")
    plan_path = transport.root / "qualified-plan.json"
    assert (
        transport.command(
            "make-plan",
            case,
            *prefix,
            "--output",
            plan_path,
            journal=transport.root / "journals",
        )
        == 0
    ), transport.last_trace
    plan = json.loads(plan_path.read_bytes())
    witnesses = plan["qualified_launch_bridge"]["witnesses"]
    assert set(witnesses) == {"sorting-classic-seed202", "sorting-deltanet-seed202"}
    assert all(w["later_pilots_authorized"] is False for w in witnesses.values())
    import hashlib

    from seed202_pilot.storage import canonical_bytes

    review_path = transport.root / "FIXTURE-ONLY-review.json"
    review_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "independent": True,
                "reviewer": "SYNTHETIC_TRANSPORT_FIXTURE_NOT_PRODUCTION_REVIEW",
                "scope": "SEED202_PILOT_SINGLETON_FULL",
                "plan_sha256": hashlib.sha256(canonical_bytes(plan)).hexdigest(),
            }
        ),
        encoding="utf8",
    )
    for action in ("publish", "submit"):
        receipt = _prepare(transport, case, plan, plan_path, review_path, action)
        proof = _probe(transport, case, plan, plan_path, review_path, action, receipt)
        observed = json.loads(proof.read_bytes())["predecessor_evidence"]
        assert [row["replay_scope"] for row in observed] == expected_scopes
        approval = _approve(transport, case, plan, plan_path, review_path, proof, action)
        journal = Path(plan["journals"][action])
        transport.lost_ack = action
        assert (
            transport.command(
                action,
                case,
                "--approval",
                approval,
                "--review",
                review_path,
                "--preparation",
                receipt,
                journal=journal,
            )
            == 1
        )
        before = list(transport.mutations)
        auth = []
        provider = cli._provider
        monkeypatch.setattr(
            cli, "_provider", lambda _args, auth=auth: auth.append(1) or transport.provider
        )
        assert (
            transport.command(
                action,
                case,
                "--approval",
                approval,
                "--review",
                review_path,
                "--preparation",
                receipt,
                journal=journal,
            )
            == 1
        )
        assert not auth
        monkeypatch.setattr(cli, "_provider", provider)
        assert (
            transport.command(
                "reconcile", case, "--action", action, "--approval", approval, journal=journal
            )
            == 0
        ), transport.last_trace
        assert transport.mutations == before
        result = json.loads((journal / "reconciled-ack.json").read_bytes())
        assert result["state"] == "RECONCILED_GET_ONLY" and result["remote_mutations"] == 0
    accepted = _post_training_transport(transport, wire, root, monkeypatch, case, plan, approval)
    if case == "mqar-k64-classic-seed202":
        assert (
            cli.main(
                [
                    "next",
                    "--candidate-dir",
                    str(root),
                    "--qualified-predecessor",
                    str(root / "payload/predecessor/qualified-acceptance.json"),
                    "--acceptance",
                    os.environ["SEED202_BRIDGE_TEST_DELTA"],
                    "--acceptance",
                    str(accepted),
                ]
            )
            == 0
        )
    invalid = transport.root / "MQAR-NOT-ACCEPTED.json"
    invalid.write_text('{"status":"ERROR"}', encoding="utf8")
    assert (
        cli.main(
            [
                "next",
                "--candidate-dir",
                str(root),
                "--qualified-predecessor",
                str(root / "payload/predecessor/qualified-acceptance.json"),
                "--acceptance",
                os.environ["SEED202_BRIDGE_TEST_DELTA"],
                "--acceptance",
                str(invalid),
            ]
        )
        == 1
    )
    assert {method for method, _fields in wire.serialized} >= {
        "StartBlobUpload",
        "CreateDataset",
        "SaveKernel",
        "GetDataset",
        "GetKernel",
    }
    assert [kind for kind, _target in transport.mutations if kind != "transport"] == [
        "CreateDataset",
        "SaveKernel",
    ]


def test_public_make_plan_accepts_genuine_qualified_delta(tmp_path, monkeypatch):
    """The unchanged public planner must accept the separately qualified predecessor."""
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    acceptance = Path(os.environ["SEED202_BRIDGE_TEST_DELTA"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    from seed202_pilot.cli import main

    output = tmp_path / "plan.json"
    code = main(
        [
            "make-plan",
            "--candidate-dir",
            str(root),
            "--case",
            "mqar-k64-classic-seed202",
            "--acceptance",
            str(acceptance),
            "--journal",
            str(tmp_path / "journals"),
            "--output",
            str(output),
        ]
    )
    assert code == 0, "genuine qualified Delta rejected by ordinary-only planner"
    plan = json.loads(output.read_bytes())
    assert plan["case_id"] == "mqar-k64-classic-seed202"
    assert plan["predecessors"][0]["sha256"] == (
        "d5fbc3a20b67b3be0400f786fbd461993ad07fe640bdca6a718364eeaa160a77"
    )
    assert not (tmp_path / "journals").exists()


def test_cached_native_replay_requires_same_complete_input_bytes(monkeypatch):
    """Memoization is only for exact pure replay; all raw pins are checked again."""
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    from seed202_pilot import qualified_bridge as bridge

    calls = []
    actual = bridge.subprocess.run

    def counted(*args, **kwargs):
        for key in ("UV_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            assert Path(kwargs["env"].get(key, "UNBOUND")).is_relative_to(
                Path(os.environ["TMPDIR"])
            )
        calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(bridge.subprocess, "run", counted)
    getattr(bridge, "_REPLAY_CACHE", {}).clear()
    acceptance = Path(os.environ["SEED202_BRIDGE_TEST_DELTA"])
    first = bridge._qualified(root, "sorting-deltanet-seed202", acceptance)
    second = bridge._qualified(root, "sorting-deltanet-seed202", acceptance)
    assert second == first
    assert len(calls) == 1


def test_ordinary_mqar_cannot_use_delta_qualified_acceptance(monkeypatch):
    """Recovery classification must never become a new-MQAR acceptance policy."""
    root = Path(os.environ["SEED202_BRIDGE_TEST_CANDIDATE"])
    acceptance = Path(os.environ["SEED202_BRIDGE_TEST_DELTA"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    from seed202_pilot.state import replay_acceptance

    with pytest.raises(ValueError, match="WHOLE_ACCEPTANCE_REQUIRED"):
        replay_acceptance(acceptance, "mqar-k64-classic-seed202")
