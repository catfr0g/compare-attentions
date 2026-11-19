"""Truthful old ERROR tasks, strict scalar repair, and external fresh HF11 evidence."""

import json
from pathlib import Path

from mqar_final_tail_native import FINAL, FULL_SHA, canonical, digest, require


def verify_hf11(value, expected):
    """Exact step/revision/all-assets; never turn a local receipt into a fresh GET proof."""
    require(
        type(value) is dict
        and set(value)
        == {
            "schema",
            "scope",
            "private",
            "get_only",
            "entries",
            "raw_evidence",
            "missing_original_first_backup",
        },
        "EXTERNAL_HF11_PROOF_SCHEMA",
    )
    require(
        value["schema"] == "MQAR202_HF11_FRESH_PROOF_V1"
        and value["scope"] == "FRESH_ALL_IMMUTABLE_REVISIONS_ALL_FILES_ALL_BYTES"
        and value["private"] is True
        and value["get_only"] is True
        and value["missing_original_first_backup"] is True,
        "EXTERNAL_AUTHENTICATED_FRESH_HF11_REQUIRED",
    )
    require(canonical(value["entries"]) == canonical(expected), "EXACT_ELEVEN_STEP_REVISION_FILES")
    require(
        type(value["raw_evidence"]) is dict and bool(value["raw_evidence"]),
        "HF11_RAW_INDEX_REQUIRED",
    )
    for name, pin in value["raw_evidence"].items():
        require(
            type(name) is str
            and name
            and type(pin) is dict
            and set(pin) == {"bytes", "sha256"}
            and type(pin["bytes"]) is int
            and pin["bytes"] > 0
            and type(pin["sha256"]) is str
            and len(pin["sha256"]) == 64,
            "HF11_RAW_PIN",
        )
    return value


def verify(release, hf_proof=None):
    from mqar_final_tail_completion import validate_unsampled_readback
    from mqar_final_tail_package import check

    release = Path(release)
    check(release)
    diagnosis = json.loads((release / "evidence/diagnosis.json").read_bytes())
    require(
        [r["numeric_id"] for r in diagnosis["providers"]] == [145778849, 145850896]
        and all(r["typed_status"] == "ERROR" for r in diagnosis["providers"]),
        "OLD_ERROR_IDENTITIES",
    )
    raw = json.loads((release / "evidence/original-clearml-all-events.json").read_bytes())
    config = json.loads((release / "evidence/original-config.json").read_bytes())
    scalar = validate_unsampled_readback(
        raw,
        task_id="17a3c249cedc4f37a358ca99f653ac52",
        updates=10780,
        validations=11,
        config=config,
    )
    require(
        scalar["scientific_event_count"] == 10812 and scalar["resource_event_count"] == 440,
        "GENUINE_SCALAR_REPLAY",
    )
    receipt = json.loads((release / "evidence/recovery-receipts.json").read_bytes())
    expected = [
        {k: row[k] for k in ("step", "revision", "files")}
        for row in receipt["entries"]
        if row["state"] == "verified"
    ]
    require(
        len(expected) == 11
        and [r["step"] for r in expected] == list(range(980, 10781, 980))
        and expected[-1]["revision"] == "ac58dd57dd1d347fc7bf0c2252ef1147c01ae15c"
        and expected[-1]["files"]["step_00010780.pt"]["sha256"] == FULL_SHA,
        "EXACT_LOCAL_CHECKPOINT_RECEIPTS",
    )
    require(hf_proof is not None, "FRESH_EXTERNAL_HF11_PROOF_INPUT_REQUIRED")
    verify_hf11(hf_proof, expected)
    return {
        "original_error_jobs": diagnosis["providers"],
        "clearml": scalar,
        "original_clearml_task": "17a3c249cedc4f37a358ca99f653ac52",
        "actual_final_parameters_sha256": FINAL,
        "periodic_hf11": hf_proof,
        "missing_original_first_hf_backup": "PRESERVED_NEW_CONTINUATION_PREFIX_UPLOAD_NOT_BACKFILL",
        "original_inner_exception": "KeyError task",
        "resume_inner_exception": "UNKNOWN_NOT_PERSISTED",
        "original_final_envelopes": "ABSENT_NOT_MANUFACTURED",
        "completion_module_sha256": digest(
            Path(__file__).with_name("mqar_final_tail_completion.py")
        ),
    }
