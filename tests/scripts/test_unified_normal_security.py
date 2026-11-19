"""Fail-closed public/source authority controls; no credential or scientific execution."""

import json
import os
import shutil
import types
from pathlib import Path

import pytest


@pytest.fixture
def candidate(monkeypatch):
    root = Path(os.environ["UNIFIED_NORMAL_CANDIDATE"])
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    return root


def test_profile_rejects_injected_import_file(candidate, tmp_path):
    from unified_normal_contract import descriptor
    from unified_normal_source import PROFILE, profile

    _path, request = descriptor(candidate)
    package = tmp_path / "normal-package"
    original = Path(request["normal"]["package"])
    shutil.copytree(original, package)
    request["normal"]["package"] = str(package)
    request["roots"] = [str(package) if p == str(original) else p for p in request["roots"]]
    request["files"] = {
        str(package / Path(p).relative_to(original)) if Path(p).is_relative_to(original) else p: pin
        for p, pin in request["files"].items()
    }
    shadow = tmp_path / "shadow"
    path = shadow / "payload/unified/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(request))
    assert profile(shadow, {"source_profile": PROFILE})["normal_profile"] == PROFILE
    (package / "src/injected_import.py").write_text("raise RuntimeError('INJECTED')\n")
    with pytest.raises(ValueError, match="UNIFIED_NORMAL_ROOT_CLOSURE"):
        profile(shadow, {"source_profile": PROFILE})


@pytest.mark.parametrize(
    "mutation", ["independent", "plan", "case", "extra", "approved", "review_pin"]
)
def test_execution_authority_rejects_cross_scope_and_extras(candidate, tmp_path, mutation):
    from unified_normal_authority import pin, validate

    plan = candidate / "payload/unified/normal-plan.json"
    value = json.loads(plan.read_bytes())
    common = {
        "case_id": next(iter(value["cases"])),
        "source_identity_sha256": value["source_identity_sha256"],
        "plan_sha256": pin(plan)["sha256"],
    }
    review = {
        **common,
        "status": "PASS",
        "independent": True,
        "reviewer": "SYNTHETIC_SCHEMA_FIXTURE_NOT_PRODUCTION_REVIEW",
    }
    if mutation == "independent":
        review["independent"] = False
    elif mutation == "plan":
        review["plan_sha256"] = "0" * 64
    elif mutation == "case":
        review["case_id"] = "mqar-k64-deltanet-seed404"
    elif mutation == "extra":
        review["verified"] = True
    rp = tmp_path / "review.json"
    rp.write_text(json.dumps(review))
    decision = {
        **common,
        "action": "NORMAL_FRESH_TRAINING",
        "approved": True,
        "review_sha256": pin(rp)["sha256"],
    }
    if mutation == "approved":
        decision["approved"] = 1
    elif mutation == "review_pin":
        decision["review_sha256"] = "0" * 64
    dp = tmp_path / "decision.json"
    dp.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="UNIFIED_AUTHORITY_"):
        validate(plan, rp, dp)


def test_unbound_source_authority_stops_mutations_before_auth(candidate):
    from unified_normal_authority import before_auth

    for command in ("publish", "submit"):
        with pytest.raises(ValueError, match="UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED"):
            before_auth(types.SimpleNamespace(command=command, candidate_dir=candidate))


@pytest.mark.parametrize(
    "case",
    [
        "mqar-k64-deltanet-seed303",
        "mqar-k64-deltanet-seed404",
        "sorting-deltanet-seed202",
        "mqar-k64-classic-seed202",
    ],
)
def test_nonrequested_cases_cannot_create_plans(candidate, tmp_path, case):
    from seed202_pilot.seal import singleton_plan

    with pytest.raises(ValueError, match="UNIFIED_REQUESTED_CASE_ONLY"):
        singleton_plan(case, candidate, tmp_path / "no-journal", [])
    assert not (tmp_path / "no-journal").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_identity_sha256", "0" * 64),
        ("wheel_sha256", "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"),
        ("profile", "OLD_CLOUD_RUNNER"),
    ],
)
def test_request_cannot_downgrade_source(candidate, tmp_path, field, value):
    from unified_normal_contract import descriptor
    from unified_normal_source import PROFILE, profile

    _path, request = descriptor(candidate)
    request["normal"][field] = value
    root = tmp_path / field
    path = root / "payload/unified/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="UNIFIED_EXPLICIT_NORMAL_PROFILE"):
        profile(root, {"source_profile": PROFILE})


@pytest.mark.parametrize("mutation", ["data", "cadence", "rank_flops", "namespace", "valid"])
def test_operator_profile_joins_actual_data_cadence_and_names(candidate, tmp_path, mutation):
    from unified_normal_contract import descriptor, digest
    from unified_normal_source import PROFILE, operator_profile

    _path, request = descriptor(candidate)
    case_id = request["run"]["id"]
    root = tmp_path / "metadata-only-not-launch-authority"
    plan = json.loads((candidate / "payload/unified/normal-plan.json").read_bytes())
    if mutation == "data":
        plan["cases"][case_id]["data_files"]["val_labels.npy"]["sha256"] = "0" * 64
    elif mutation == "cadence":
        request["run"]["validation_steps"][0] += 1
    elif mutation == "rank_flops":
        request["run"]["rank_step_flops"] -= 1
    p = root / "payload/unified/normal-plan.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(plan))
    request["run"]["plan_sha256"] = digest(p)
    (p.parent / "request.json").write_text(json.dumps(request))
    shutil.copyfile(
        candidate / "payload/unified/normal-authority.json", p.parent / "normal-authority.json"
    )
    config = root / "payload/configs" / (case_id + ".yaml")
    config.parent.mkdir(parents=True)
    shutil.copyfile(candidate / "payload/configs" / config.name, config)
    deployment = {
        "source_profile": PROFILE,
        "nonce": "d" * 32,
        "case_id": case_id,
        "normal_plan_sha256": digest(p),
    }
    if mutation == "namespace":
        deployment["nonce"] = "INVALID-LITERAL-NONCE"
    errors = {
        "data": "REQUESTED_DATA_BINDING",
        "cadence": "CADENCE_AND_IDENTITY",
        "rank_flops": "CADENCE_AND_IDENTITY",
        "namespace": "SERVICE_DECLARATION",
    }
    if mutation == "valid":
        result = operator_profile(root, deployment)
        assert result["normal_service_prefix"] == "normal-training/" + "d" * 32
        assert result["normal_authority_bound"] is False
    else:
        with pytest.raises(ValueError, match="UNIFIED_NORMAL_" + errors[mutation]):
            operator_profile(root, deployment)
