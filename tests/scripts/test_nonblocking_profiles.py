"""A future-only telemetry derivative preserves the entire scientific wheel."""

import hashlib
import json
from pathlib import Path

import build_normal_training_lifecycle as builder
import pytest

PARENT = Path(
    "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/"
    "remaining-six-fixed-source13/next-mqar-delta303-17/candidate/payload/normal-package"
)
REPO = Path(__file__).resolve().parents[2]


def test_nonblocking_release_keeps_all_scientific_members(tmp_path):
    target = tmp_path / "normal-package"
    builder.build_nonblocking_release(PARENT, REPO, target)
    wheel = "compare_attention-0.1.0-py3-none-any.whl"
    assert (target / wheel).read_bytes() == (PARENT / wheel).read_bytes()
    identity = json.loads((target / "normal-source-identity.json").read_bytes())
    assert identity["tracking_policy"] == "AUXILIARY_NONBLOCKING_V1"
    assert (
        identity["completion_parent_identity_sha256"]
        == hashlib.sha256((PARENT / "normal-source-identity.json").read_bytes()).hexdigest()
    )
    for name in (
        "training_completion",
        "training_diagnostics",
        "normal_training_acceptance",
        "normal_training_worker",
        "unified_normal_host",
    ):
        assert (target / "src" / (name + ".py")).read_bytes() == (
            REPO / "scripts" / (name + ".py")
        ).read_bytes()
    assert (target / "inherited-source/training_diagnostics.py").is_file()
    assert set(identity["files"]) == {
        p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()
    } - {"normal-source-identity.json"}


def test_four_profile_binds_real_nonblocking_source_only():
    from unified_normal_profiles import authenticate_package, is_remaining_profile, selected

    profile = "REMAINING_FOUR_NONBLOCKING_TRACKING_V1"
    assert is_remaining_profile(profile)
    release = json.loads(
        Path(
            "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/tracking-nonblocking-repair24/source-release-final.json"
        ).read_bytes()
    )
    assert selected(profile)["source"] == release["source"]
    assert selected(profile)["archive"] == release["archive_sha256"]
    assert (
        authenticate_package(release["package"], profile)["tracking_policy"]
        == "AUXILIARY_NONBLOCKING_V1"
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "sorting-classic-seed303",
        "sorting-deltanet-seed303",
        "mqar-k64-classic-seed303",
        "mqar-k64-deltanet-seed303",
    ],
)
def test_four_profile_rejects_completed303_before_permission(case_id):
    from unified_normal_generation import intent_value

    value = {
        "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
        "nonce": "b" * 32,
        "case_id": case_id,
        "source_identity_sha256": "0" * 64,
        "authorization": {"path": "MUST_NOT_READ", "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="^REMAINING_FOUR_CASE_SCOPE$"):
        intent_value(value, "REMAINING_FOUR_NONBLOCKING_TRACKING_V1")


def test_four_profile_requires_its_four_actual_plan_bytes():
    from unified_normal_remaining import SIX_PLAN_PINS, require_plan_pin

    packet = json.loads(
        Path(
            "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/tracking-nonblocking-repair24/four-raw-subjects-final.json"
        ).read_bytes()
    )
    assert len(packet["subjects"]) == 4
    for row in packet["subjects"]:
        require_plan_pin(
            row["case_id"], row["normal_plan_sha256"], "REMAINING_FOUR_NONBLOCKING_TRACKING_V1"
        )
        with pytest.raises(ValueError, match="^REMAINING_FOUR_FROZEN_PLAN$"):
            require_plan_pin(
                row["case_id"],
                SIX_PLAN_PINS[row["case_id"]],
                "REMAINING_FOUR_NONBLOCKING_TRACKING_V1",
            )


def test_only_new_four_prefix_admits_truthful_case3_qualification():
    from normal_remaining_scope import CASE_IDS
    from unified_normal_contract import Policy, valid_evidence_row
    from unified_normal_remaining import prefix_policies

    ordinary = "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY"
    qualified = "QUALIFIED_MQAR_DELTA303_COMPLETED_NATIVE_V1"
    rows = [{"case_id": cid, "policy": ordinary} for cid in CASE_IDS[:4]]
    rows[3]["policy"] = qualified
    prefix_policies(rows, "REMAINING_FOUR_NONBLOCKING_TRACKING_V1")
    assert valid_evidence_row(
        {"case_id": CASE_IDS[3], "acceptance_sha256": "a" * 64, "replay_scope": qualified}
    )
    assert not valid_evidence_row(
        {"case_id": CASE_IDS[2], "acceptance_sha256": "a" * 64, "replay_scope": qualified}
    )
    assert Policy(qualified)
    with pytest.raises(ValueError, match="REMAINING_ORDINARY_PREFIX_REQUIRED"):
        prefix_policies(rows, "REMAINING_SIX_ELASTIC_FIXED_V1")
    rows[3]["policy"] = ordinary
    with pytest.raises(ValueError, match="REMAINING_QUALIFIED_CASE3_PREFIX_REQUIRED"):
        prefix_policies(rows, "REMAINING_FOUR_NONBLOCKING_TRACKING_V1")


def test_staged_legacy_completion_includes_nonblocking_dependency(tmp_path):
    archive = Path(
        "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/mqar-classic-resume-repair-v1/mount-postreview-real03/sorting-mqar-seed202-pilot-v1.zip"
    )
    staged = builder.stage_source(archive, tmp_path / "source")
    assert (staged / "training_diagnostics.py").read_bytes() == (
        REPO / "scripts/training_diagnostics.py"
    ).read_bytes()
