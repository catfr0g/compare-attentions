"""Close exact mounted human/reviewer authority bytes to the new inference namespace."""

import base64
import hashlib
import json
from pathlib import Path

from mqar_final_tail_native import canonical, digest, require


def capture(decision):
    decision = Path(decision)
    raw, review = decision.read_bytes(), decision.with_name("independent-review.json").read_bytes()
    return {
        "decision_base64": base64.b64encode(raw).decode(),
        "review_base64": base64.b64encode(review).decode(),
        "decision_sha256": hashlib.sha256(raw).hexdigest(),
        "review_sha256": hashlib.sha256(review).hexdigest(),
        "hf11_sha256": digest(decision.with_name("hf11-proof.json")),
    }


def verify(value, release, nonce, target, hf_proof):
    from mqar_final_tail_composite import require_review

    require(
        type(value) is dict
        and set(value)
        == {"decision_base64", "review_base64", "decision_sha256", "review_sha256", "hf11_sha256"},
        "EXACT_MOUNTED_AUTHORITY_SCHEMA",
    )
    raw = base64.b64decode(value["decision_base64"], validate=True)
    raw_review = base64.b64decode(value["review_base64"], validate=True)
    require(
        hashlib.sha256(raw).hexdigest() == value["decision_sha256"]
        and hashlib.sha256(raw_review).hexdigest() == value["review_sha256"]
        and digest(hf_proof) == value["hf11_sha256"],
        "ALL_MOUNTED_AUTHORITY_BYTES",
    )
    decision, review = json.loads(raw), json.loads(raw_review)
    require_review(review, digest(Path(release) / "manifest.json"))
    require(
        set(decision)
        == {
            "status",
            "independent",
            "scope",
            "manifest_sha256",
            "execution_authorized",
            "telemetry_policy",
            "independent_review_sha256",
            "historical_hf_proof_sha256",
            "nonce",
            "kernel_target",
            "allocations_permitted",
        }
        and decision["status"] == "PASS"
        and decision["independent"] is True
        and decision["scope"] == "MQAR202_FINAL10780_INFERENCE_PUBLICATION_ONLY"
        and decision["manifest_sha256"] == digest(Path(release) / "manifest.json")
        and decision["execution_authorized"] is True
        and decision["nonce"] == nonce
        and decision["kernel_target"] == target == "zimike/mqar202-final10780-tail-" + nonce[:16]
        and type(decision["allocations_permitted"]) is int
        and decision["allocations_permitted"] == 1
        and decision["independent_review_sha256"] == value["review_sha256"]
        and decision["historical_hf_proof_sha256"] == value["hf11_sha256"]
        and decision["telemetry_policy"]
        == "EXPLICITLY_ACCEPT_ORIGINAL_GAP_WITH_QUALIFIED_DERIVED_NATIVE_EVIDENCE",
        "ACTUAL_EXACT_NEW_NONCE_EXECUTION_AUTHORITY",
    )
    return {
        "execution_decision": decision,
        "independent_review": review,
        "mounted_file_pins": {
            "execution-decision.json": value["decision_sha256"],
            "independent-review.json": value["review_sha256"],
            "hf11-proof.json": value["hf11_sha256"],
        },
        "canonical_review": canonical(review),
    }
