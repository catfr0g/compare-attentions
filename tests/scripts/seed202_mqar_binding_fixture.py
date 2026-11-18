"""SYNTHETIC engineering decisions; never called by production packaging or binding."""

import json
import shutil
from pathlib import Path

from seed202_mqar_authority import canonical, pin
from seed202_mqar_postreview import bind_candidate


def engineering_candidate(staged, destination):
    staged, destination = Path(staged), Path(destination)
    destination.mkdir(parents=True)
    shutil.copytree(staged / "payload", destination / "payload")
    for name in ("sorting-mqar-seed202-pilot-v1.zip", "PLAN-UNAPPROVED.json", "candidate.json"):
        shutil.copyfile(staged / name, destination / name)
    value = json.loads((destination / "candidate.json").read_bytes())
    value.update(
        status="COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
        engineering_fixture="ENGINEERING_FIXTURE_ONLY",
    )
    (destination / "candidate.json").write_bytes(canonical(value))
    return engineering_bind(destination, destination.with_name(destination.name + "-bound"))


def engineering_bind(candidate, destination):
    candidate, destination = Path(candidate), Path(destination)
    evidence = json.loads(
        (
            candidate / "payload/deploy/mqar-k64-classic-seed202" / "evidence-required.json"
        ).read_bytes()
    )
    digest = pin(canonical(evidence))["sha256"]
    documents = destination.parent / (destination.name + "-FIXTURE-ONLY-decisions")
    documents.mkdir(parents=True)
    review = documents / "independent-review.json"
    review.write_bytes(
        canonical(
            {
                "scope": "INDEPENDENT_EVIDENCE_REVIEW_V1",
                "evidence_sha256": digest,
                "decision": "PASS",
                "reviewer": "ENGINEERING_FIXTURE_ONLY",
            }
        )
    )
    decision = documents / "execution-decision.json"
    decision.write_bytes(
        canonical(
            {
                "scope": "EXECUTION_DECISION_V1",
                "evidence_sha256": digest,
                "review_sha256": pin(review.read_bytes())["sha256"],
                "decision": "ALLOW",
                "nonce": evidence["nonce"],
            }
        )
    )
    # This descriptor is itself engineering-only; original staged/frozen bytes stay intact.
    descriptor = candidate / "candidate.json"
    value = json.loads(descriptor.read_bytes())
    if value["status"] != "COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED":
        raise ValueError("TEST_FIXTURE_DESCRIPTOR_REQUIRED")
    return bind_candidate(candidate, review, decision, destination)
