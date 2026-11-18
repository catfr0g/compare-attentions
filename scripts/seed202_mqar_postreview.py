"""Create-only postreview binding; supplied decisions remain separate byte authority."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import zipfile
from pathlib import Path

from seed202_mqar_authority import bind_bytes, canonical, pin, validate

CASE = "mqar-k64-classic-seed202"


def kernel_binding(source, **changes):
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BINDING" for target in node.targets)
    )
    binding = ast.literal_eval(node.value)
    binding.update(changes)
    return source.replace(ast.get_source_segment(source, node), "BINDING = " + repr(binding))


def normalized_kernel(source):
    return kernel_binding(source, evidence_sha256="0" * 64, authority_sha256="0" * 64).encode()


def bind_candidate(candidate, review, decision, destination):
    """Require external independent documents; no candidate file is changed in place."""
    from seed202_mqar_package import inventory, write_zip

    candidate, destination = Path(candidate), Path(destination)
    descriptor = json.loads((candidate / "candidate.json").read_bytes())
    if descriptor["status"] != "COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED" or destination.exists():
        raise ValueError("FROZEN_CREATE_ONLY_POSTREVIEW_BINDING")
    payload = candidate / "payload"
    archive_path = candidate / "sorting-mqar-seed202-pilot-v1.zip"
    if pin(archive_path.read_bytes())["sha256"] != descriptor["sha256"]:
        raise ValueError("FROZEN_ARCHIVE_PIN")
    with zipfile.ZipFile(archive_path) as archive:
        manifest_raw = archive.read("manifest.json")
    actual = inventory(payload)
    if (payload / "manifest.json").read_bytes() != manifest_raw or actual != {
        **json.loads(manifest_raw)["files"],
        "manifest.json": pin(manifest_raw),
    }:
        raise ValueError("FROZEN_EXTRACTED_INVENTORY")
    evidence = json.loads((payload / "deploy" / CASE / "evidence-required.json").read_bytes())
    # Validate supplied decisions before creating any output directory.
    from seed202_mqar_authority import documents

    review_raw, decision_raw = Path(review).read_bytes(), Path(decision).read_bytes()
    documents(evidence, review_raw, decision_raw)
    deploy = payload / "deploy" / CASE
    for path, key in (
        (deploy / "dataset/seed202-pilot.zip.bin", "worker"),
        (deploy / "kernel/kernel-metadata.json", "metadata"),
        (payload / "configs" / (CASE + ".yaml"), "config"),
        (payload / "inherited/sorting-science.zip.bin", "science"),
    ):
        if pin(path.read_bytes()) != evidence[key]:
            raise ValueError("REVIEWED_EXECUTION_INPUT_BYTES")
    source = (payload / "deploy" / CASE / "kernel/kernel.py").read_text(encoding="utf-8")
    if pin(normalized_kernel(source)) != evidence["normalized_kernel"]:
        raise ValueError("NORMALIZED_REVIEWED_KERNEL_BYTES")
    destination.mkdir(parents=True)
    shutil.copytree(payload, destination / "payload")
    deploy = destination / "payload/deploy" / CASE
    bound = bind_bytes(
        deploy / "resume-input",
        evidence,
        review_raw,
        decision_raw,
        publication_folder=True,
    )
    kernel = deploy / "kernel/kernel.py"
    kernel.write_text(kernel_binding(source, **bound), encoding="utf-8", newline="\n")
    result = validate(deploy / "resume-input", **bound, publication_folder=True)
    plan = json.loads((destination / "payload/PLAN-UNAPPROVED.json").read_bytes())
    row = next(row for row in plan["deployments"] if row["case_id"] == CASE)
    row.update(
        **bound,
        mounted_authority_files=result["files"],
        kernel_source_sha256=pin(kernel.read_bytes())["sha256"],
    )
    plan["review_file_allowlist_excluding_plan_and_manifest"] = {
        name: value
        for name, value in inventory(destination / "payload").items()
        if name not in {"PLAN-UNAPPROVED.json", "manifest.json"}
    }
    for path in (
        destination / "PLAN-UNAPPROVED.json",
        destination / "payload/PLAN-UNAPPROVED.json",
    ):
        path.write_bytes(canonical(plan))
    archive = destination / "sorting-mqar-seed202-pilot-v1.zip"
    files = {
        name: destination / "payload" / name
        for name in inventory(destination / "payload")
        if name != "manifest.json"
    }
    manifest = write_zip(archive, files, "POSTREVIEW_EVIDENCE_BOUND_NOT_EXECUTION_SEAL")
    (destination / "payload/manifest.json").write_bytes(manifest)
    with zipfile.ZipFile(archive) as saved:
        names = saved.namelist()
    if len(names) != len(set(names)):
        raise ValueError("UNIQUE_DERIVED_ARCHIVE_MEMBERS")
    descriptor.update(
        archive=str(archive.resolve()),
        members=len(names),
        sha256=pin(archive.read_bytes())["sha256"],
        postreview_binding=result,
        frozen_prebinding_sha256=descriptor["sha256"],
    )
    (destination / "candidate.json").write_bytes(canonical(descriptor))
    indexed = {
        "archive": archive,
        "descriptor": destination / "candidate.json",
        "manifest": destination / "payload/manifest.json",
        "plan": destination / "PLAN-UNAPPROVED.json",
    }
    indexed.update({name: deploy / "resume-input" / name for name in result["files"]})
    (destination / "artifact-index.json").write_bytes(
        canonical(
            {
                "scope": "POSTREVIEW_EVIDENCE_BOUND_HOST_ACTION_SEALS_STILL_REQUIRED",
                "members": descriptor["members"],
                "frozen_prebinding_sha256": descriptor["frozen_prebinding_sha256"],
                "files": {
                    name: {"path": str(path.resolve()), **pin(path.read_bytes())}
                    for name, path in indexed.items()
                },
            }
        )
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("candidate", "review", "decision", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError("CPU_MASK_BEFORE_IMPORTS_REQUIRED")
    print(bind_candidate(args.candidate, args.review, args.decision, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
