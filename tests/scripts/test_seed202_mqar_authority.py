"""Postreview engineering decisions bind exact bytes without archive selfhash."""

import json
import zipfile
from pathlib import Path

import pytest


def test_postreview_descriptor_identifies_same_derived_archive_and_unique_member_total(tmp_path):
    from seed202_mqar_authority import canonical, pin
    from seed202_mqar_binding_fixture import engineering_bind
    from seed202_mqar_package import inventory, write_zip
    from seed202_mqar_postreview import CASE, normalized_kernel

    candidate = tmp_path / "ENGINEERING_ONLY_prebind"
    payload = candidate / "payload"
    deploy = payload / "deploy" / CASE
    inputs = {
        deploy / "dataset/seed202-pilot.zip.bin": b"ENGINEERING_WORKER",
        deploy / "resume-input/resume.zip.bin": b"ENGINEERING_RESUME",
        deploy / "kernel/kernel-metadata.json": canonical({}),
        deploy / "kernel/kernel.py": b"BINDING = {}\n",
        payload / "configs" / (CASE + ".yaml"): canonical({}),
        payload / "inherited/sorting-science.zip.bin": b"ENGINEERING_SCIENCE",
    }
    for path, raw in inputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    evidence = {
        "scope": "MQAR_EVIDENCE_ONLY_V1",
        "mode": "MQAR_CLASSIC_STEP980_V1",
        "case": CASE,
        "nonce": "4a" * 16,
        "mounted": {"resume.zip.bin": pin(inputs[deploy / "resume-input/resume.zip.bin"])},
        "normalized_kernel": pin(normalized_kernel("BINDING = {}\n")),
    }
    for key, path in (
        ("worker", deploy / "dataset/seed202-pilot.zip.bin"),
        ("metadata", deploy / "kernel/kernel-metadata.json"),
        ("config", payload / "configs" / (CASE + ".yaml")),
        ("science", payload / "inherited/sorting-science.zip.bin"),
    ):
        evidence[key] = pin(inputs[path])
    (deploy / "evidence-required.json").write_bytes(canonical(evidence))
    (payload / "PLAN-UNAPPROVED.json").write_bytes(canonical({"deployments": [{"case_id": CASE}]}))
    archive = candidate / "sorting-mqar-seed202-pilot-v1.zip"
    manifest = write_zip(
        archive, {name: payload / name for name in inventory(payload)}, "ENGINEERING_ONLY_PREBIND"
    )
    (payload / "manifest.json").write_bytes(manifest)
    with zipfile.ZipFile(archive) as saved:
        old_count = len(saved.namelist())
    descriptor = {
        "status": "COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
        "archive": str(archive),
        "members": old_count,
        "sha256": pin(archive.read_bytes())["sha256"],
    }
    (candidate / "candidate.json").write_bytes(canonical(descriptor))
    before = inventory(candidate)
    derived = engineering_bind(candidate, tmp_path / "ENGINEERING_ONLY_bound")
    actual = derived / archive.name
    bound = json.loads((derived / "candidate.json").read_bytes())
    with zipfile.ZipFile(actual) as saved:
        names = saved.namelist()
    assert len(names) == len(set(names)) == old_count + 4
    assert {
        "archive": Path(bound["archive"]).resolve(),
        "members": bound["members"],
        "sha256": bound["sha256"],
    } == {
        "archive": actual.resolve(),
        "members": len(names),
        "sha256": pin(actual.read_bytes())["sha256"],
    }
    assert inventory(candidate) == before
    delivery = json.loads((derived / "artifact-index.json").read_bytes())
    assert delivery["members"] == len(names)
    assert delivery["scope"] == "POSTREVIEW_EVIDENCE_BOUND_HOST_ACTION_SEALS_STILL_REQUIRED"
    for entry in delivery["files"].values():
        path = Path(entry["path"])
        assert path.resolve().is_relative_to(derived.resolve())
        assert {key: entry[key] for key in ("bytes", "sha256")} == pin(path.read_bytes())


def test_postreview_bytes_inventory_rejects_missing_changed_and_extra(tmp_path):
    from seed202_mqar_authority import bind_bytes, canonical, pin, validate

    evidence = {
        "scope": "MQAR_EVIDENCE_ONLY_V1",
        "mode": "MQAR_CLASSIC_STEP980_V1",
        "case": "mqar-k64-classic-seed202",
        "nonce": "4a" * 16,
        "mounted": {"resume.zip.bin": pin(b"ENGINEERING_ONLY")},
    }
    digest = pin(canonical(evidence))["sha256"]
    review = canonical(
        {
            "scope": "INDEPENDENT_EVIDENCE_REVIEW_V1",
            "evidence_sha256": digest,
            "decision": "PASS",
            "reviewer": "ENGINEERING_FIXTURE_ONLY",
        }
    )
    decision = canonical(
        {
            "scope": "EXECUTION_DECISION_V1",
            "evidence_sha256": digest,
            "review_sha256": pin(review)["sha256"],
            "decision": "ALLOW",
            "nonce": evidence["nonce"],
        }
    )
    (tmp_path / "resume.zip.bin").write_bytes(b"ENGINEERING_ONLY")
    bind_bytes(tmp_path, evidence, review, decision)
    result = validate(tmp_path, digest)
    assert set(result["files"]) == {
        "resume.zip.bin",
        "evidence.json",
        "independent-review.json",
        "execution-decision.json",
        "authority-inventory.json",
    }
    authority = (tmp_path / "execution-decision.json").read_bytes()
    (tmp_path / "execution-decision.json").write_bytes(authority + b" ")
    # Semantically identical JSON is still different review/execution authority bytes.
    with pytest.raises(ValueError, match="AUTHORITY_INVENTORY"):
        validate(tmp_path, digest)
    (tmp_path / "execution-decision.json").write_bytes(authority)
    (tmp_path / "extra.py").write_text("# shadow", encoding="utf-8")
    with pytest.raises(ValueError, match="AUTHORITY_INVENTORY"):
        validate(tmp_path, digest)
    (tmp_path / "extra.py").unlink()
    (tmp_path / "independent-review.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(tmp_path, digest)
    assert json.loads(review)["reviewer"] == "ENGINEERING_FIXTURE_ONLY"
