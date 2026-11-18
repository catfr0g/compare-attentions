"""Mandatory original ERROR, preserved history and native restore evidence join."""

import json
from pathlib import Path


def requested(config_path, spec):
    """Read the frozen request, never infer resume mode from downloaded evidence."""
    from seed202_pilot.mqar_authority import MODE, require
    from seed202_pilot.mqar_host_resume import verify_extracted

    candidate = Path(config_path).resolve().parents[2]
    verify_extracted(candidate)
    plan = json.loads((candidate / "payload/PLAN-UNAPPROVED.json").read_bytes())
    rows = [row for row in plan["deployments"] if row["case_id"] == spec.case_id]
    require(len(rows) == 1, "FROZEN_REQUESTED_CASE")
    return rows[0] if rows[0].get("mode") == MODE else None


def consume(root, case, config_path, spec):
    binding = requested(config_path, spec)
    return {"continuation": verify(root, case, binding)} if binding is not None else {}


def verify(root, case, requested):
    from seed202_pilot.mqar_authority import MODE, validate
    from seed202_pilot.mqar_continuation import provenance
    from seed202_pilot.mqar_restore_gate import FIELDS
    from seed202_pilot.mqar_resume_pins import FILES
    from seed202_pilot.storage import canonical_bytes, digest

    def require(value, code):
        if not value:
            raise ValueError(code)

    root = Path(root)
    require(case["id"] == "mqar-k64-classic-seed202", "COMPOSITE_EXACT_MQAR_CASE")
    require(
        requested["mode"] == MODE and requested["case_id"] == case["id"],
        "FROZEN_CONTINUATION_REQUEST_REQUIRED",
    )
    authority = validate(
        root / "mounted-authority", requested["evidence_sha256"], requested["authority_sha256"]
    )
    require(
        authority["files"] == requested["mounted_authority_files"],
        "EXECUTION_REVIEW_DECISION_ALL_MOUNTED_FILES",
    )
    expected = provenance()
    actual = json.loads((root / "resume-provenance.json").read_bytes())
    require(canonical_bytes(actual) == canonical_bytes(expected), "EXACT_ORIGINAL_ERROR_PROVENANCE")
    prefix = Path(__file__).with_name("mqar-prefix-metrics.jsonl")
    require(digest(prefix) == FILES["out/metrics.jsonl"]["sha256"], "AUTHENTIC_PREFIX_BYTES")
    old = [json.loads(line) for line in prefix.read_text(encoding="utf-8").splitlines()]
    new = [
        json.loads(line)
        for line in (root / "out/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    require(new[: len(old)] == old and len(new) > len(old), "EXACT_HISTORICAL_PREFIX")
    require(
        new[len(old)]["type"] == "step" and new[len(old)]["flops"] == 981 * 463856467968,
        "EXACT_FIRST_NEW_UPDATE_981",
    )
    for name, pin in FILES.items():
        if name.startswith("diagnostic/"):
            path = root / name
            require(
                path.stat().st_size == pin["bytes"] and digest(path) == pin["sha256"],
                "PRESERVED_ORIGINAL_TWO_RANK_CAPTURE",
            )
    receipts = []
    for rank in (0, 1):
        path = root / f"restore-equality-rank-{rank}.json"
        receipt = json.loads(path.read_bytes())
        require(
            receipt
            == {
                "scope": "ACTUAL_TARGET_NATIVE_RESTORE_EQUALITY_BEFORE_981",
                "rank": rank,
                "step": 980,
                "next_step": 981,
                "fields": list(FIELDS),
                "native_restore_unchanged": True,
                "rng_checked_after_history_replay": True,
            },
            "BOTH_TARGET_NATIVE_RESTORE_EQUALITY_RECEIPTS",
        )
        receipts.append({"rank": rank, "sha256": digest(path)})
    return {
        "scope": "ORIGINAL_ERROR_980_PLUS_NATIVE_CONTINUATION_TO_10780",
        "original": expected,
        "original_status_preserved": "ERROR",
        "prefix_metric_events": len(old),
        "prefix_updates": 980,
        "remaining_updates": 9800,
        "first_new_update": 981,
        "restore_receipts": receipts,
        "preserved980_publication": "SALVAGED_BOUNDARY_NOT_ORIGINAL_BACKUP",
        "mounted_authority": authority,
    }
