"""Internally consistent corruptions must fail semantic gates, not just public checksums."""

import copy
import json
from pathlib import Path

import pytest
from mqar_final_tail_test_support import World


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    world = World(tmp_path_factory.mktemp("tail-security"), patch)
    world.receipt, world.approval = world.prepare("publish")
    yield world
    patch.undo()


@pytest.mark.parametrize(
    "field,value",
    [
        ("quota.total", True),
        ("quota.reserved", 1),
        ("quota.required", 1),
        ("quota.planning", 999999),
        ("source", "PREPARATION_CACHE"),
        ("namespace.complete_owner_namespace", False),
        ("namespace.controls.exact_metadata_status", 404),
        ("private_evidence.private_env_equal_memory_only", False),
        ("service_evidence.nonce", "0" * 32),
        ("preparation.revalidation.write_denial", "STAT_ONLY"),
        ("preparation.receipt.action", "submit"),
        ("plan_sha256", "f" * 64),
        ("action", "submit"),
        ("active", [{"target": "active"}]),
    ],
)
def test_live_semantic_corruptions_rejected_after_valid_mac(prepared, field, value):
    from mqar_final_tail_host_live import validate_live
    from seed202_pilot.preparation_auth import tag

    proof = copy.deepcopy(json.loads(prepared.approval.read_bytes())["proof"])
    at = proof
    *keys, last = field.split(".")
    for key in keys:
        at = at[key]
    at[last] = value
    proof["authentication"] = tag(prepared.plan, proof)
    with pytest.raises((ValueError, KeyError)):
        validate_live(prepared.plan, proof, "publish", fresh=False)


def test_expired_live_never_uses_prepared_quota(prepared):
    from mqar_final_tail_host_live import validate_live
    from seed202_pilot.preparation_auth import tag

    proof = copy.deepcopy(json.loads(prepared.approval.read_bytes())["proof"])
    proof["started"] -= 121
    proof["finished"] -= 121
    proof["authentication"] = tag(prepared.plan, proof)
    with pytest.raises(ValueError):
        validate_live(prepared.plan, proof, "publish")


def test_receipt_edit_is_not_authenticated_by_recomputed_public_hash(prepared):
    from mqar_final_tail_host_prepare import session

    receipt = json.loads(prepared.receipt.read_bytes())
    receipt["action"] = "submit"
    with pytest.raises(ValueError):
        with session(prepared.plan, receipt, "submit", prepared.review):
            pytest.fail("FORGED_RECEIPT_ACCEPTED")


def test_real_lease_denies_actual_file_write(prepared):
    from mqar_final_tail_host_prepare import session

    with session(
        prepared.plan, json.loads(prepared.receipt.read_bytes()), "publish", prepared.review
    ):
        path = Path(prepared.plan["journal_directory"]) / "dataset/recovery-operator.zip.bin"
        with pytest.raises(PermissionError):
            path.open("wb")


def test_cross_plan_nonce_and_review_policy_fail_before_authentication(prepared, monkeypatch):
    from mqar_final_tail_host_contract import validate_plan
    from mqar_final_tail_host_seal import seal

    proof = json.loads(prepared.approval.read_bytes())["proof"]
    plan = copy.deepcopy(prepared.plan)
    plan["nonce"] = "0" * 32
    with pytest.raises(ValueError):
        validate_plan(plan)
    review = {**prepared.review, "policy": "WAIVE_ORIGINAL_MANDATORY_GAPS"}
    with pytest.raises(ValueError):
        seal(prepared.plan, review, proof)


def test_predecessor_error_and_403_classification_cannot_be_relabelled(prepared):
    from mqar_final_tail_host_live import validate_live
    from seed202_pilot.preparation_auth import tag

    for mode in ("predecessor", "403"):
        proof = copy.deepcopy(json.loads(prepared.approval.read_bytes())["proof"])
        if mode == "predecessor":
            proof["predecessors"][0]["typed_status"] = "COMPLETE"
        else:
            row = next(iter(proof["namespace"]["observations"].values()))
            row["classification"] = "HTTP_NOT_FOUND"
        proof["authentication"] = tag(prepared.plan, proof)
        with pytest.raises(ValueError):
            validate_live(prepared.plan, proof, "publish", fresh=False)


@pytest.mark.parametrize(
    "drift",
    [
        "head",
        "private",
        "source-bytes",
        "private-bytes",
        "predecessor",
        "active",
        "expired-lifetime",
    ],
)
def test_fresh_drift_inhibits_before_mutation(prepared, monkeypatch, drift):
    from mqar_final_tail_host_contract import PRIVATE, SCIENCE

    if drift in {"head", "private"}:
        monkeypatch.setitem(
            prepared.drift,
            SCIENCE,
            {"current_version_number": 2} if drift == "head" else {"is_private": False},
        )
    elif drift == "source-bytes":
        monkeypatch.setitem(prepared.datasets[SCIENCE], "scientific.zip.bin", b"CORRUPTED")
    elif drift == "private-bytes":
        monkeypatch.setitem(prepared.datasets[PRIVATE], "private.env", b"SYNTHETIC_WRONG")
    elif drift == "predecessor":
        monkeypatch.setattr(
            prepared.kernels[prepared.plan["predecessors"][0]["target"]], "status", "COMPLETE"
        )
    elif drift == "active":
        monkeypatch.setattr(
            prepared.kernels[prepared.plan["predecessors"][0]["target"]], "status", "RUNNING"
        )
    else:
        raw = prepared.lifetime.read_bytes()
        value = json.loads(raw)
        value["observed_unix_seconds"] -= 43201
        prepared.lifetime.write_text(json.dumps(value))
    try:
        assert (
            prepared.command(
                "probe", receipt=prepared.receipt, output=prepared.root / (drift + ".json")
            )
            == 1
        )
        assert prepared.writes == []
    finally:
        if drift == "expired-lifetime":
            prepared.lifetime.write_bytes(raw)


@pytest.mark.parametrize(
    "flag,value", [("enable_gpu", False), ("enable_tpu", True), ("enable_internet", False)]
)
def test_sdk_save_wrong_runtime_flags_stop_at_presend_guard(prepared, flag, value):
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from mqar_final_tail_host_sdk import TailCapability

    cap = TailCapability.__new__(TailCapability)
    cap.plan, cap.action, cap.state = prepared.plan, "submit", "NEW"
    cap.check = lambda: None  # Leaf fixture; normal routes exercise the actual live capability.
    request = ApiSaveKernelRequest()
    request.slug = prepared.plan["kernel_target"]
    request.text = (prepared.source / "kernel/kernel.py").read_text()
    request.dataset_data_sources = prepared.plan["requested_versioned_inputs"]
    request.machine_shape, request.is_private = "NvidiaTeslaT4", True
    request.enable_gpu, request.enable_tpu, request.enable_internet = True, False, True
    setattr(request, flag, value)
    with pytest.raises(ValueError):
        cap.transport("SaveKernel", request)
