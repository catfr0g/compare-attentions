"""Create-only portable MQAR candidate from owned source and authenticated inputs."""

import argparse
import json
import shutil
import uuid
import zipfile
from pathlib import Path

from mqar_final_tail_native import digest, require
from mqar_final_tail_package import check, inventory
from mqar_final_tail_render import render
from mqar_final_tail_runner import create_json


def archive(root, destination):
    require(not destination.exists(), "CREATE_ONLY_ARCHIVE")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as target:
        for name in sorted(inventory(root)):
            target.write(root / name, name)
    with zipfile.ZipFile(destination) as target:
        require(len(target.namelist()) == len(set(target.namelist())), "UNIQUE_MEMBERS")
        require(set(target.namelist()) == set(inventory(root)), "ARCHIVE_FULL_KEYSET")
        require(
            all(target.read(n) == (root / n).read_bytes() for n in target.namelist()),
            "ALL_ARCHIVE_MEMBER_BYTES",
        )


def build(workspace, source, destination):
    workspace, source, destination = map(Path, (workspace, source, destination))
    require(not destination.exists(), "CREATE_ONLY_CANDIDATE")
    base = workspace / "candidate"
    check(workspace / "input")
    destination.mkdir(parents=True)
    shutil.copytree(base / "payload", destination / "payload")
    shutil.copytree(base / "inherited-two-stage-host", destination / "inherited-two-stage-host")
    payload = destination / "payload"
    for path in source.glob("mqar_final_tail_*.py"):
        if not any(
            x in path.stem
            for x in (
                "test_support",
                "fixture",
                "sdk_fixture",
                "test_layout",
                "chain_child",
                "observer_child",
                "_child",
            )
        ):
            shutil.copyfile(path, payload / "src" / path.name)
    evidence = payload / "evidence"
    shutil.copyfile(
        workspace / "input/original/original-clearml-all-events.json",
        evidence / "original-clearml-all-events.json",
    )
    result = json.loads(
        (workspace / "input/original/out/metrics.jsonl").read_text().splitlines()[-1]
    )
    create_json(evidence / "original-config.json", result["config"])
    old_receipts = workspace / "historical-recovery-receipts.json"
    require(
        digest(old_receipts) == "58865908f62c93759a7ed6eea66eadd2374752e6f5c2cd69e8a647db89ebfce6",
        "ORIGINAL_RECEIPT_SOURCE",
    )
    shutil.copyfile(old_receipts, evidence / "recovery-receipts.json")
    create_json(payload / "manifest.json", {"files": inventory(payload)})
    dataset = destination / "dataset"
    dataset.mkdir()
    archive(payload, dataset / "recovery-operator.zip.bin")
    archive(workspace / "input", dataset / "recovery-input.zip.bin")
    nonce = uuid.uuid4().hex
    code, metadata = render(
        nonce,
        digest(dataset / "recovery-operator.zip.bin"),
        digest(dataset / "recovery-input.zip.bin"),
        guard_path=payload / "src/inherited_kernel.py",
    )
    kernel = destination / "kernel"
    kernel.mkdir()
    (kernel / "kernel.py").write_text(code, encoding="utf-8", newline="\n")
    create_json(kernel / "kernel-metadata.json", metadata)
    create_json(
        dataset / "dataset-metadata.json",
        {
            "id": metadata["dataset_sources"][0][:-2],
            "title": metadata["dataset_sources"][0].split("/")[1],
            "licenses": [{"name": "CC0-1.0"}],
        },
    )
    plan = {
        "schema": "MQAR202_FINAL10780_FROZEN_INFERENCE_CANDIDATE_V1",
        "nonce": nonce,
        "target": metadata["id"],
        "kernel_sha256": digest(kernel / "kernel.py"),
        "operator_sha256": digest(dataset / "recovery-operator.zip.bin"),
        "input_sha256": digest(dataset / "recovery-input.zip.bin"),
        "original_errors": [145778849, 145850896],
        "optimizer_updates_permitted": 0,
        "files": check(payload)["files"],
        "scientific_wheel_sha256": (
            "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
        ),
        "external_hf11_proof": "REQUIRED_AUTHENTICATED_SEPARATE_REVIEW_INPUT_NOT_GENERATED",
    }
    create_json(destination / "plan.json", plan)
    create_json(destination / "manifest.json", {"files": inventory(destination)})
    check(payload)
    check(destination)
    return {
        "candidate": str(destination),
        "manifest_sha256": digest(destination / "manifest.json"),
        "members": len(inventory(destination)),
        "review_status": "NOT_REVIEWED_NOT_AUTHORIZED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("workspace", "source", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.workspace, args.source, args.destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
