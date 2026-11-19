"""Synthetic decisions only for denied-network SDK transport fixtures."""

import json
import uuid

from mqar_final_tail_test_layout import ART, CANDIDATE, DATA


def fixture_plan(tmp_path):
    from mqar_final_tail_host_contract import bind
    from mqar_final_tail_native import digest

    folder = ART / "ENGINEERING-ONLY-publications" / ("f-" + uuid.uuid4().hex[:16])
    folder.mkdir(parents=True)
    review = folder / "independent-review.json"
    review.write_text(
        json.dumps(
            {
                "status": "PASS",
                "independent": True,
                "scope": "MQAR202_FINAL10780_TAIL_RECOVERY",
                "candidate_sha256": digest(CANDIDATE / "payload/manifest.json"),
                "policy": "RECOVERED_TAIL_WITH_ORIGINAL_TELEMETRY_GAP",
            }
        )
    )
    receipts = json.loads((CANDIDATE / "payload/evidence/recovery-receipts.json").read_bytes())
    expected = [
        {k: r[k] for k in ("step", "revision", "files")}
        for r in receipts["entries"]
        if r["state"] == "verified"
    ]
    hf = folder / "hf11-proof.json"
    hf.write_text(
        json.dumps(
            {
                "schema": "MQAR202_HF11_FRESH_PROOF_V1",
                "scope": "FRESH_ALL_IMMUTABLE_REVISIONS_ALL_FILES_ALL_BYTES",
                "private": True,
                "get_only": True,
                "entries": expected,
                "missing_original_first_backup": True,
                "raw_evidence": {
                    "SYNTHETIC_HTTP_ENGINEERING_ONLY_NOT_FRESH_HF_PROOF": {
                        "bytes": 1,
                        "sha256": "0" * 64,
                    }
                },
            }
        )
    )
    frozen = json.loads((CANDIDATE / "plan.json").read_bytes())
    decision = folder / "execution-decision.json"
    decision.write_text(
        json.dumps(
            {
                "status": "PASS",
                "independent": True,
                "scope": "MQAR202_FINAL10780_INFERENCE_PUBLICATION_ONLY",
                "manifest_sha256": digest(CANDIDATE / "payload/manifest.json"),
                "execution_authorized": True,
                "nonce": frozen["nonce"],
                "kernel_target": frozen["target"],
                "allocations_permitted": 1,
                "telemetry_policy": (
                    "EXPLICITLY_ACCEPT_ORIGINAL_GAP_WITH_QUALIFIED_DERIVED_NATIVE_EVIDENCE"
                ),
                "independent_review_sha256": digest(review),
                "historical_hf_proof_sha256": digest(hf),
            }
        )
    )
    return bind(CANDIDATE, folder / "journals", decision, DATA)
