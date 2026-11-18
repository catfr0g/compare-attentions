"""Finite byte authority shared by postreview host and mounted worker consumers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODE = "MQAR_CLASSIC_STEP980_V1"
NAMES = ("evidence.json", "independent-review.json", "execution-decision.json")
INVENTORY = "authority-inventory.json"


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def files(root, *, publication_folder=False):
    root = Path(root)
    result = {}
    for path in root.iterdir():
        require(
            not path.is_symlink() and not path.is_junction() and path.is_file(),
            "AUTHORITY_INVENTORY",
        )
        if not publication_folder or path.name != "dataset-metadata.json":
            result[path.name] = pin(path.read_bytes())
    return result


def documents(evidence, review_raw, decision_raw):
    digest = pin(canonical(evidence))["sha256"]
    review, decision = json.loads(review_raw), json.loads(decision_raw)
    require(
        evidence["scope"] == "MQAR_EVIDENCE_ONLY_V1"
        and evidence["mode"] == MODE
        and evidence["case"] == "mqar-k64-classic-seed202",
        "EXACT_AUTHORITY_SCOPE",
    )
    require(
        set(review) == {"scope", "evidence_sha256", "decision", "reviewer"}
        and review["scope"] == "INDEPENDENT_EVIDENCE_REVIEW_V1"
        and review["evidence_sha256"] == digest
        and review["decision"] == "PASS"
        and type(review["reviewer"]) is str
        and bool(review["reviewer"].strip()),
        "INDEPENDENT_REVIEW_REQUIRED",
    )
    require(
        decision
        == {
            "scope": "EXECUTION_DECISION_V1",
            "evidence_sha256": digest,
            "review_sha256": pin(review_raw)["sha256"],
            "decision": "ALLOW",
            "nonce": evidence["nonce"],
        },
        "EXACT_EXECUTION_DECISION",
    )
    return digest


def bind_bytes(root, evidence, review_raw, decision_raw, *, publication_folder=False):
    """Consume supplied decisions; never manufacture a review or execution approval."""
    root = Path(root)
    digest = documents(evidence, review_raw, decision_raw)
    require(
        files(root, publication_folder=publication_folder) == evidence["mounted"],
        "EXACT_PREBIND_MOUNTED_BYTES",
    )
    for name, raw in zip(NAMES, (canonical(evidence), review_raw, decision_raw), strict=True):
        with (root / name).open("xb") as stream:
            stream.write(raw)
    inventory = canonical(
        {
            "scope": "ALL_MOUNTED_FILES_V1",
            "files": files(root, publication_folder=publication_folder),
        }
    )
    with (root / INVENTORY).open("xb") as stream:
        stream.write(inventory)
    return {"evidence_sha256": digest, "authority_sha256": pin(inventory)["sha256"]}


def validate(root, evidence_sha256, authority_sha256=None, *, publication_folder=False):
    """Exact inventory before imports, after imports, at worker entry and acceptance."""
    root = Path(root)
    raw = (root / INVENTORY).read_bytes()
    if authority_sha256 is not None:
        require(pin(raw)["sha256"] == authority_sha256, "AUTHORITY_INVENTORY_PIN")
    inventory = json.loads(raw)
    actual = files(root, publication_folder=publication_folder)
    del actual[INVENTORY]
    require(inventory == {"scope": "ALL_MOUNTED_FILES_V1", "files": actual}, "AUTHORITY_INVENTORY")
    evidence_raw, review, decision = [(root / name).read_bytes() for name in NAMES]
    evidence = json.loads(evidence_raw)
    require(
        pin(evidence_raw)["sha256"] == evidence_sha256 and canonical(evidence) == evidence_raw,
        "FROZEN_EVIDENCE_PIN",
    )
    documents(evidence, review, decision)
    require(
        {key: value for key, value in actual.items() if key not in NAMES} == evidence["mounted"],
        "EXACT_MOUNTED_BYTES",
    )
    return {
        "evidence_sha256": evidence_sha256,
        "authority_sha256": pin(raw)["sha256"],
        "files": files(root, publication_folder=publication_folder),
    }


def import_inventory(release, science):
    """Snapshot all authenticated import files, including non-Python companions."""
    result = {}
    for root in (Path(release), Path(science)):
        for path in root.rglob("*"):
            require(not path.is_symlink() and not path.is_junction(), "IMPORT_TREE_LINK")
            if path.is_file():
                result[str(path.resolve())] = pin(path.read_bytes())
    return result


def validate_args(args):
    result = validate(args.authority_root, args.evidence_sha256, args.authority_sha256)
    evidence = json.loads((Path(args.authority_root) / "evidence.json").read_bytes())
    require(
        evidence["case"] == args.case and evidence["nonce"] == args.nonce,
        "EXACT_MOUNTED_CASE_NONCE",
    )
    return result


def check_imports(release, science, before):
    import sys

    require(import_inventory(release, science) == before, "POSTIMPORT_INVENTORY_DRIFT")
    roots = (Path(release).resolve(), Path(science).resolve())
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("seed202_pilot", "quality_")) or name in {"process_control", "session"}:
            filename = getattr(module, "__file__", None)
            require(
                filename is not None
                and any(Path(filename).resolve().is_relative_to(root) for root in roots),
                "UNEXPECTED_IMPORT_SHADOW",
            )
