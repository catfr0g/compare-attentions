"""MQAR final10780 only, distinct authority for both tail publication actions."""

import json
from pathlib import Path

from mqar_final_tail_native import canonical, digest, require
from mqar_final_tail_package import check, inventory

ACTIONS = ("publish", "submit")
SCIENCE = "zimike/sorting-quality-input-fcff1b5daf574b70"
DATA = "zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4"
PRIVATE = "zimike/k64-private-20261109a1b2"
CONTROL = "zimike/experiments-fp16-k64-small-13919d-272c2da7a9ef"
POLICY = "EXPLICITLY_ACCEPT_ORIGINAL_GAP_WITH_QUALIFIED_DERIVED_NATIVE_EVIDENCE"


def value_sha(value):
    import hashlib

    return hashlib.sha256(canonical(value).encode()).hexdigest()


def exact(left, right):
    return canonical(left) == canonical(right)


def pin(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def binding(candidate, journal_root, decision, data):
    from mqar_final_tail_cli import decision as validate_decision
    from mqar_final_tail_render import render

    candidate, journal_root, decision = map(
        lambda p: Path(p).resolve(), (candidate, journal_root, decision)
    )
    check(candidate)
    frozen = json.loads((candidate / "plan.json").read_bytes())
    source, metadata = render(
        frozen["nonce"],
        frozen["operator_sha256"],
        frozen["input_sha256"],
        guard_path=candidate / "payload/src/inherited_kernel.py",
    )
    require(source.encode() == (candidate / "kernel/kernel.py").read_bytes(), "MQAR_SAVED_SOURCE")
    require(
        exact(metadata, json.loads((candidate / "kernel/kernel-metadata.json").read_bytes())),
        "MQAR_METADATA",
    )
    require(frozen["original_errors"] == [145778849, 145850896], "MQAR_ORIGINAL_ERRORS")
    for key, name in (
        ("input_sha256", "recovery-input.zip.bin"),
        ("operator_sha256", "recovery-operator.zip.bin"),
    ):
        require(digest(candidate / "dataset" / name) == frozen[key], "MQAR_ARCHIVE")
    from mqar_final_tail_data import MQAR_DATA

    data = Path(data).resolve()
    require(
        all(
            (data / n).is_file() and (data / n).stat().st_size == p["bytes"]
            for n, p in MQAR_DATA["files"].items()
        ),
        "EXACT_MQAR_DATA_FILESET_SIZES",
    )
    validate_decision(decision, candidate / "payload", frozen["nonce"])
    diagnosis = json.loads((candidate / "payload/evidence/diagnosis.json").read_bytes())
    predecessors = diagnosis["providers"]
    require([r["numeric_id"] for r in predecessors] == [145778849, 145850896], "MQAR_PREDECESSORS")
    require(all(r["typed_status"] == "ERROR" for r in predecessors), "MQAR_ERROR_PRESERVED")
    review = decision.with_name("independent-review.json")
    return {
        "schema": "MQAR202_FINAL10780_TAIL_HOST_TWO_STAGE_V1",
        "implementation_owner": "mqar-final-tail-implementation-owner",
        "candidate_root": str(candidate),
        "candidate_manifest_sha256": digest(candidate / "manifest.json"),
        "nonce": frozen["nonce"],
        "actions": list(ACTIONS),
        "journal_directory": str(journal_root),
        "journals": {a: str(journal_root / a) for a in ACTIONS},
        "kernel_target": metadata["id"],
        "dataset_target": metadata["dataset_sources"][0],
        "requested_versioned_inputs": metadata["dataset_sources"],
        "kernel_sha256": frozen["kernel_sha256"],
        "metadata_sha256": digest(candidate / "kernel/kernel-metadata.json"),
        "decision_path": str(decision),
        "review_path": str(review),
        "hf11_path": str(decision.with_name("hf11-proof.json")),
        "dataset_files": {
            "recovery-input.zip.bin": pin(candidate / "dataset/recovery-input.zip.bin"),
            "recovery-operator.zip.bin": pin(candidate / "dataset/recovery-operator.zip.bin"),
            "execution-decision.json": pin(decision),
            "independent-review.json": pin(review),
            "hf11-proof.json": pin(decision.with_name("hf11-proof.json")),
        },
        "data_directory": str(data),
        "data_files": MQAR_DATA["files"],
        "predecessors": predecessors,
        "optimizer_updates_permitted": 0,
        "scope": "MQAR202_FINAL10780_INFERENCE_PUBLICATION_ONLY",
        "case_deadline_seconds": 7200,
        "acceptance_tail_seconds": 1800,
        "original_telemetry_policy": "CONDITIONAL_EXPLICIT_INDEPENDENT_REVIEW_REQUIRED",
        "science_sha256": "71872615b0cfbe2406cd08762cc02c390c88a70b6bb5f4bb38d6571830c8a3bb",
        "archive_predecessor_sha256": (
            "3a95667c7c5b733b8a8ebdbe4224d1daaef64acf38cb626a5aff3f2b5eed42ab"
        ),
    }


def bind(candidate, journal_root, decision, data):
    plan = binding(candidate, journal_root, decision, data)
    require(
        not any((Path(plan["journals"][a]) / "intent.json").exists() for a in ACTIONS),
        "PENDING_GET_ONLY",
    )
    return plan


def validate_plan(plan):
    require(
        type(plan) is dict
        and exact(
            plan,
            binding(
                plan["candidate_root"],
                plan["journal_directory"],
                plan["decision_path"],
                plan["data_directory"],
            ),
        ),
        "EXACT_MQAR_TAIL_PLAN",
    )
    return plan


def local_paths(plan):
    candidate = Path(plan["candidate_root"])
    return (
        [str(candidate / n) for n in sorted(inventory(candidate))]
        + [plan["decision_path"], plan["review_path"], plan["hf11_path"]]
        + [str(Path(plan["data_directory"]) / n) for n in plan["data_files"]]
    )


def activate(plan):
    """Keep unchanged audited host package; reject ambient mismatched import aliases."""
    import sys

    root = Path(plan["candidate_root"]) / "inherited-two-stage-host"
    sys.path.insert(0, str(root))
    import seed202_pilot

    for name, module in tuple(sys.modules.items()):
        if name.startswith("seed202_pilot.") and getattr(module, "__file__", None):
            expected = root / Path(*name.split(".")).with_suffix(".py")
            require(
                expected.is_file() and digest(expected) == digest(Path(module.__file__)),
                "HOST_IMPORT_SOURCE",
            )
    seed202_pilot.__path__ = [str(root / "seed202_pilot")]
    return root
