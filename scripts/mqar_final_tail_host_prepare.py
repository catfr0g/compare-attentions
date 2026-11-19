"""Tail adapter over unchanged byte leases, remote object binding and DPAPI MAC."""

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

from mqar_final_tail_host_contract import (
    ACTIONS,
    DATA,
    SCIENCE,
    exact,
    local_paths,
    pin,
    validate_plan,
    value_sha,
)
from mqar_final_tail_native import require
from mqar_final_tail_runner import create_json

ACTIVE = {}


def prepare_local(plan):
    """Finite new owned packaging; never alter the frozen operator/input or private input."""
    validate_plan(plan)
    folder = Path(plan["journal_directory"]) / "dataset"
    candidate = Path(plan["candidate_root"])
    # Kaggle 2.2.5 flattens the full absolute upload path into ONE cache filename.
    # Reject before credentials/StartBlobUpload, not after the SDK creates an unknown ACK.
    for name in plan["dataset_files"]:
        encoded = str((folder / name).resolve()).replace(os.path.sep, "_").replace(":", "_")
        require(len(os.fsencode(encoded + ".json")) <= 255, "SDK_UPLOAD_STATE_FILENAME_LIMIT")
    if not folder.exists():
        folder.mkdir(parents=True)
        for name in plan["dataset_files"]:
            external = {
                "execution-decision.json": plan["decision_path"],
                "independent-review.json": plan["review_path"],
                "hf11-proof.json": plan["hf11_path"],
            }
            source = Path(external[name]) if name in external else candidate / "dataset" / name
            shutil.copyfile(source, folder / name)
        shutil.copyfile(
            candidate / "dataset/dataset-metadata.json", folder / "dataset-metadata.json"
        )
    require(
        {p.name for p in folder.iterdir()}
        == set(plan["dataset_files"]) | {"dataset-metadata.json"},
        "TAIL_PACKAGING_SET",
    )
    require(
        all(exact(pin(folder / n), p) for n, p in plan["dataset_files"].items()),
        "TAIL_PACKAGING_BYTES",
    )
    require(
        (folder / "dataset-metadata.json").read_bytes()
        == (candidate / "dataset/dataset-metadata.json").read_bytes(),
        "TAIL_PACKAGING_METADATA",
    )
    return local_paths(plan) + [str(p) for p in sorted(folder.iterdir())]


def paths(plan, science_archive, parent_archive, science_directory):
    from mqar_final_tail_host_science import paths as science_paths
    from seed202_pilot.inherit import read_parent_member

    require(pin(science_archive)["sha256"] == plan["science_sha256"], "TAIL_SCIENCE_ARCHIVE")
    read_parent_member(parent_archive, "dependencies/vendor/inherited_deployment_seal.py")
    scientific, _pins = science_paths(science_directory, science_archive, plan["science_sha256"])
    return (
        prepare_local(plan)
        + scientific
        + [str(Path(p).resolve()) for p in (science_archive, parent_archive)]
    )


def prepare(
    plan, action, review, provider, science_archive, parent_archive, output, science_directory
):
    """All bytes before LIVE; no quota/namespace/credential equality cached as live evidence."""
    from mqar_final_tail_host_live import predecessors
    from seed202_pilot.preparation_auth import tag
    from seed202_pilot.prepared_files import check, lease
    from seed202_pilot.prepared_remote import capture, metadata, private_identity
    from seed202_pilot.provider import dataset_files_readback

    require(action in ACTIONS, "TAIL_ACTION")
    require(not Path(output).exists(), "CREATE_ONLY_RECEIPT")
    files = paths(plan, science_archive, parent_archive, science_directory)
    scientific_pins = json.loads((Path(science_directory) / "manifest.json").read_bytes())["files"]
    started = time.time()
    with lease(files) as local:
        require(
            all(
                local[str(Path(science_directory).resolve() / n)]["sha256"] == p["sha256"]
                for n, p in scientific_pins.items()
            ),
            "TAIL_SCIENCE_LEASE_BYTES",
        )
        selected = {
            SCIENCE: {"scientific.zip.bin": Path(science_archive)},
            DATA: {n: Path(plan["data_directory"]) / n for n in plan["data_files"]},
        }
        if action == "submit":
            selected[plan["dataset_target"][:-2]] = {
                n: Path(plan["journal_directory"]) / "dataset" / n for n in plan["dataset_files"]
            }
        before = {ref: metadata(provider, ref) for ref in selected}
        private = private_identity(provider)
        with capture(provider) as objects:
            sources = {
                ref: dataset_files_readback(provider, ref, fs) for ref, fs in selected.items()
            }
        require(set(objects) == set(before), "PREPARATION_REMOTE_SET")
        for ref in before:
            require(set(objects[ref]) == set(before[ref]["files"]), "PREPARATION_REMOTE_FILES")
            require(exact(before[ref], metadata(provider, ref)), "PREPARATION_REMOTE_DRIFT")
        prior = predecessors(plan, provider)
        require(exact(private, private_identity(provider)), "PREPARATION_PRIVATE_DRIFT")
        check(local)
        receipt = {
            "schema": "TAIL_IMMUTABLE_BYTES_NOT_AUTHORIZED_V1",
            "mutation_authority": False,
            "plan_sha256": value_sha(plan),
            "review_sha256": value_sha(review) if review else None,
            "action": action,
            "started": started,
            "finished": time.time(),
            "local": local,
            "science_archive": str(Path(science_archive).resolve()),
            "science_directory": str(Path(science_directory).resolve()),
            "science_files": scientific_pins,
            "parent_archive": str(Path(parent_archive).resolve()),
            "private_identity": private,
            "predecessors": prior,
            "source_evidence": sources,
            "remote": {ref: {"metadata": before[ref], "objects": objects[ref]} for ref in before},
        }
        receipt["authentication"] = tag(plan, receipt, create=True)
        create_json(output, receipt)
    return receipt


def validate_receipt(plan, receipt, action, review=None):
    from seed202_pilot.preparation_auth import verify

    verify(plan, receipt)
    require(
        set(receipt)
        == {
            "schema",
            "mutation_authority",
            "plan_sha256",
            "review_sha256",
            "action",
            "started",
            "finished",
            "local",
            "science_archive",
            "science_directory",
            "science_files",
            "parent_archive",
            "private_identity",
            "predecessors",
            "source_evidence",
            "remote",
            "authentication",
        },
        "TAIL_RECEIPT_SCHEMA",
    )
    require(
        receipt["schema"] == "TAIL_IMMUTABLE_BYTES_NOT_AUTHORIZED_V1"
        and receipt["mutation_authority"] is False,
        "TAIL_RECEIPT_NOT_AUTHORIZATION",
    )
    require(
        receipt["plan_sha256"] == value_sha(plan)
        and receipt["action"] == action
        and action in ACTIONS,
        "TAIL_RECEIPT_SCOPE",
    )
    require(review is None or receipt["review_sha256"] == value_sha(review), "TAIL_REVIEW_BINDING")
    require(
        type(receipt["started"]) in (int, float)
        and type(receipt["finished"]) in (int, float)
        and 0 <= receipt["started"] <= receipt["finished"],
        "TAIL_RECEIPT_TIMES",
    )
    candidate = Path(plan["candidate_root"])
    names = json.loads((candidate / "manifest.json").read_bytes())["files"]
    expected_local = (
        {str(candidate / n) for n in [*names, "manifest.json"]}
        | {
            plan["decision_path"],
            plan["review_path"],
            plan["hf11_path"],
            receipt["science_archive"],
            receipt["parent_archive"],
        }
        | {str(Path(plan["data_directory"]) / n) for n in plan["data_files"]}
        | {
            str(Path(receipt["science_directory"]) / n)
            for n in [*receipt["science_files"], "manifest.json"]
        }
        | {
            str(Path(plan["journal_directory"]) / "dataset" / n)
            for n in [*plan["dataset_files"], "dataset-metadata.json"]
        }
    )
    require(set(receipt["local"]) == expected_local, "TAIL_RECEIPT_LOCAL_SET")
    wanted = {SCIENCE, DATA} | ({plan["dataset_target"][:-2]} if action == "submit" else set())
    require(
        set(receipt["remote"]) == set(receipt["source_evidence"]) == wanted,
        "TAIL_RECEIPT_REMOTE_SET",
    )
    for ref, row in receipt["remote"].items():
        require(
            set(row) == {"metadata", "objects"}
            and set(row["objects"]) == set(row["metadata"]["files"]),
            "TAIL_RECEIPT_REMOTE_FILES",
        )
        expected = (
            {
                "scientific.zip.bin": {
                    "bytes": receipt["local"][receipt["science_archive"]]["identity"]["bytes"],
                    "sha256": plan["science_sha256"],
                }
            }
            if ref == SCIENCE
            else plan["data_files"]
            if ref == DATA
            else plan["dataset_files"]
        )
        source = {
            "target": ref + "/1",
            "head": 1,
            "private": True,
            "ready": True,
            "file_count": len(expected),
            "all_bytes_equal": True,
            "files": expected,
        }
        require(exact(receipt["source_evidence"][ref], source), "TAIL_RECEIPT_SOURCE")
    return receipt


@contextlib.contextmanager
def session(plan, receipt, action, review):
    """Rehash under real Win32 read leases BEFORE provider construction and keep through write."""
    from seed202_pilot.prepared_files import lease

    validate_plan(plan)
    validate_receipt(plan, receipt, action, review)
    started = time.time()
    with lease(
        paths(
            plan,
            receipt["science_archive"],
            receipt["parent_archive"],
            receipt["science_directory"],
        )
    ) as local:
        require(exact(local, receipt["local"]), "TAIL_LOCAL_BYTES_DRIFT")
        wrapper = {
            "receipt": receipt,
            "sha256": value_sha(receipt),
            "revalidation": {
                "started": started,
                "finished": time.time(),
                "write_denial": "WIN32_FILE_SHARE_READ",
            },
        }
        ACTIVE[wrapper["sha256"]] = (local, wrapper)
        try:
            yield wrapper
        finally:
            ACTIVE.pop(wrapper["sha256"], None)


def assert_lease(wrapper):
    from seed202_pilot.prepared_files import check

    active = ACTIVE.get(wrapper["sha256"])
    require(
        active is not None
        and exact(active[1], wrapper)
        and exact(active[0], wrapper["receipt"]["local"]),
        "TAIL_LEASE_NOT_HELD",
    )
    check(active[0])
