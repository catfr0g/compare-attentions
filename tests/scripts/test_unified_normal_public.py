"""Audited public command chain with genuine prefix and SDK transport-only fixtures."""

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest


@pytest.fixture
def operator(monkeypatch):
    logging.disable(logging.CRITICAL)
    root = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    monkeypatch.syspath_prepend(str(root / "payload/source/tests/seed202_pilot"))
    monkeypatch.syspath_prepend(str(root / "payload/source/tests/qualified_launch"))
    return root


@pytest.mark.parametrize("command", ["services", "accept"])
def test_public_host_routes_to_actual_normal_consumer(operator, monkeypatch, command):
    import types

    import unified_normal_host
    from seed202_pilot.host_operator import _execute

    root = operator
    case = "mqar-k64-deltanet-seed202"
    args = types.SimpleNamespace(candidate_dir=root, case=case, command=command)
    global_plan = json.loads((root / "PLAN-UNAPPROVED.json").read_bytes())
    calls = []

    def boundary(actual_args, approval, provider, metadata, source):
        assert actual_args is args and approval == {"fixture": "routing-only"}
        assert provider is None and metadata["id"].startswith("zimike/seed202-pilot-")
        assert b"normal-plan.json" in source
        calls.append(command)
        return {"scope": "ROUTE_TEST_NOT_SCIENTIFIC_ACCEPTANCE"}

    monkeypatch.setattr(unified_normal_host, "run_public", boundary)
    result = _execute(args, {}, global_plan, {"fixture": "routing-only"}, None)
    assert result == {"scope": "ROUTE_TEST_NOT_SCIENTIFIC_ACCEPTANCE"} and calls == [command]


def test_genuine_prefix_public_preparation_live_seal_sdk_unknown_ack(operator, monkeypatch):
    import unified_normal_contract as shared
    from fresh_sdk_fixture import FreshTransport
    from seed202_pilot import cli
    from seed202_qualified_http_fixture import HttpFixture
    from test_fresh_public_chain_integration06 import _approve, _prepare, _probe
    from unified_normal_contract import descriptor, expected_evidence

    # A predecessor callback inside LIVE must consume existing write-denial
    # handles; full-byte acquisition belongs before the fresh 120-second window.
    original_lease = shared.request_lease
    lease_events = []

    def traced_lease(candidate):
        lease_events.append(bool(shared._HELD_INPUTS))
        return original_lease(candidate)

    monkeypatch.setattr(shared, "request_lease", traced_lease)

    root = operator
    transport = FreshTransport(root, monkeypatch)
    actual_provider = cli._provider
    protected = root / "payload/source/src/unified_normal_host.py"

    def leased_provider(args):
        before = protected.read_bytes()
        with pytest.raises(PermissionError):
            with protected.open("r+b"):
                pass
        assert protected.read_bytes() == before
        return actual_provider(args)

    monkeypatch.setattr(cli, "_provider", leased_provider)
    transport.enable_mutations()
    wire = HttpFixture(transport, monkeypatch)
    _path, request = descriptor(root)
    case = request["run"]["id"]
    plan_path = transport.root / "unified-plan.json"
    prefix = [
        token for row in request["predecessors"][1:] for token in ("--acceptance", row["path"])
    ]
    assert (
        transport.command(
            "make-plan", case, *prefix, "--output", plan_path, journal=transport.root / "journals"
        )
        == 0
    ), transport.last_trace
    plan = json.loads(plan_path.read_bytes())
    assert plan["source_authority"]["fresh"]["normal_wheel_sha256"].startswith("a1e1db")
    review = transport.root / "SYNTHETIC-HTTP-REVIEW-NOT-AUTHORITY.json"
    review.write_text(
        json.dumps(
            {
                "status": "PASS",
                "independent": True,
                "reviewer": "SYNTHETIC-HTTP-FIXTURE-NOT-INDEPENDENT-REVIEW",
                "scope": "SEED202_PILOT_SINGLETON_FULL",
                "plan_sha256": hashlib.sha256(
                    json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    )
    for action in ("publish", "submit"):
        prepared = _prepare(transport, case, plan, plan_path, review, action)
        proof = _probe(transport, case, plan, plan_path, review, action, prepared)
        assert json.loads(proof.read_bytes())["predecessor_evidence"] == expected_evidence(plan)
        approval = _approve(transport, case, plan, plan_path, review, proof, action)
        journal = Path(plan["journals"][action])
        transport.lost_ack = action
        assert (
            transport.command(
                action,
                case,
                "--approval",
                approval,
                "--review",
                review,
                "--preparation",
                prepared,
                journal=journal,
            )
            == 1
        )
        before = list(transport.mutations)
        auth = []
        original = cli._provider
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
                review,
                "--preparation",
                prepared,
                journal=journal,
            )
            == 1
        )
        assert auth == [] and transport.mutations == before
        monkeypatch.setattr(cli, "_provider", original)
        assert (
            transport.command(
                "reconcile", case, "--action", action, "--approval", approval, journal=journal
            )
            == 0
        ), transport.last_trace
        assert transport.mutations == before
        receipt = json.loads((journal / "reconciled-ack.json").read_bytes())
        assert receipt["remote_mutations"] == 0 and receipt["state"] == "RECONCILED_GET_ONLY"
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
