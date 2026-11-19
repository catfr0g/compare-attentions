"""Engineering contract verification, genuine immutable inputs, no SDK or training."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path("E:/HermesArtifacts/compare-attention/unified-normal-launch-contract-v1")


@pytest.fixture
def contract(monkeypatch):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    packet = Path(os.environ.get("UNIFIED_CONTRACT_PACKET", str(ROOT / "engineering-prefix01")))
    monkeypatch.syspath_prepend(str(packet / "source"))
    old = json.loads((ROOT / "inputs.json").read_bytes())["sorting_candidate"]
    monkeypatch.syspath_prepend(str(Path(old) / "payload/source/src"))
    monkeypatch.syspath_prepend(
        os.environ.get("UNIFIED_CONTRACT_MODULE_ROOT", str(packet / "source"))
    )
    import unified_normal_contract

    return packet, unified_normal_contract


def test_genuine_whole_mqar_typed_verifier(contract):
    packet, module = contract
    _path, request = module.descriptor(packet)
    witness = module._replay(packet, request["predecessors"][-1])
    assert isinstance(witness, module.VerifiedPredecessor)
    assert witness.policy is module.Policy.MQAR_CONTINUATION
    assert witness.acceptance_sha256 == (
        "9a75cfe268660608e835582acf74f1a4438ccc1281e3143b07c6c84a3cd347ed"
    )
    assert "verified" not in witness.serialized()


@pytest.mark.parametrize("offset", [0, 1])
def test_genuine_sorting_compatibility(contract, offset):
    packet, module = contract
    _path, request = module.descriptor(packet)
    row = request["predecessors"][offset]
    witness = module._replay(packet, row)
    assert witness.case_id == row["case_id"]
    assert witness.acceptance_sha256 == row["sha256"]
    assert witness.policy.value == row["policy"]


def test_reviewed_source_profile_uses_genuine_t4_reference(contract):
    packet, _module = contract
    from unified_normal_source import PROFILE, SOURCE, WHEEL, profile

    value = profile(packet, {"source_profile": PROFILE})
    assert value["normal_source_identity_sha256"] == SOURCE
    assert value["normal_wheel_sha256"] == WHEEL
    assert value["normal_runtime_sha256"] != "0" * 64


@pytest.mark.parametrize(
    "case,policy,sha,valid",
    [
        (
            "mqar-k64-classic-seed202",
            "QUALIFIED_MQAR_CONTINUATION_TAIL_V1",
            "9a75cfe268660608e835582acf74f1a4438ccc1281e3143b07c6c84a3cd347ed",
            True,
        ),
        ("mqar-k64-deltanet-seed202", "QUALIFIED_MQAR_CONTINUATION_TAIL_V1", "a" * 64, False),
        ("mqar-k64-classic-seed202", "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY", "a" * 64, False),
    ],
)
def test_seal_evidence_policy_is_not_download_controlled(contract, case, policy, sha, valid):
    _packet, module = contract
    assert (
        module.valid_evidence_row(
            {"case_id": case, "replay_scope": policy, "acceptance_sha256": sha}
        )
        is valid
    )


def test_raw_verified_bool_is_not_a_declared_policy(contract):
    packet, module = contract
    forged = {"verified": True, "case_id": "mqar-k64-classic-seed202"}
    with pytest.raises(ValueError, match="UNIFIED_DECLARED_POLICY_REQUIRED"):
        module._replay(packet, forged)


def test_exact_replay_cache_is_bound_to_full_locked_bytes(contract, monkeypatch):
    packet, module = contract
    _path, request = module.descriptor(packet)
    calls = []
    actual = module.subprocess.run

    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", counted)
    getattr(module, "_REPLAY_CACHE", {}).clear()
    first = module._replay(packet, request["predecessors"][-1])
    second = module._replay(packet, request["predecessors"][-1])
    assert first == second
    assert len(calls) == 1


def test_qualified_mqar_transition_review_is_mandatory(contract):
    import subprocess
    import sys
    import uuid

    packet, module = contract
    _path, request = module.descriptor(packet)
    changed = json.loads(json.dumps(request))
    root = ROOT / ("ENGINEERING-NEGATIVE-" + uuid.uuid4().hex)
    target = root / "payload/unified/request.json"
    target.parent.mkdir(parents=True)
    review = json.loads(
        Path(changed["predecessors"][-1]["replay"]["transition_review"]).read_bytes()
    )
    review["independent"] = False
    review_path = root / "rejected-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    changed["predecessors"][-1]["replay"]["transition_review"] = str(review_path)
    target.write_text(json.dumps(changed), encoding="utf-8")
    child = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-B",
            str(Path(module.__file__).with_name("unified_normal_replay.py")),
            "--request",
            str(target),
            "--case",
            "mqar-k64-classic-seed202",
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert child.returncode == 1
    result = json.loads(child.stdout)
    assert result["status"] == "REJECTED" and result["class"] == "ValueError"
