"""Create-only staging of complete MQAR worker and original-checkpoint inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from build_seed202_mqar_continuation import EXECUTION, original_files, sha, stage_source
from seed202_mqar_host_integration import BASELINE_SHA, MODE, NONCE


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink() or path.is_junction():
            raise ValueError("NO_LINKED_PACKAGE_INPUTS")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
    return result


def write_zip(path, files, scope):
    """Write and re-read safe unique regular members plus exact per-member manifest."""
    pins = {}
    for name, source in files.items():
        part = PurePosixPath(name)
        if part.is_absolute() or ".." in part.parts or "\\" in name or name == "manifest.json":
            raise ValueError("SAFE_UNIQUE_ARCHIVE_MEMBER")
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        pins[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = json_bytes({"schema_version": 1, "scope": scope, "files": pins})
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted([*files, "manifest.json"]):
            raw = manifest if name == "manifest.json" else files[name]
            if not isinstance(raw, bytes):
                raw = Path(raw).read_bytes()
            info = zipfile.ZipInfo(name, (2026, 11, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            archive.writestr(info, raw)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(files) | {"manifest.json"}:
            raise ValueError("ARCHIVE_UNIQUE_READBACK")
        for name, pin in pins.items():
            raw = archive.read(name)
            if pin != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                raise ValueError("ARCHIVE_MEMBER_READBACK")
    return manifest


def stage_candidate(parent, baseline, history, destination):
    """Stage review bytes; this function does not claim completed verification."""
    parent, destination = Path(parent), Path(destination)
    if sha(baseline) != BASELINE_SHA or destination.exists():
        raise ValueError("AUTHENTIC_CREATE_ONLY_CANDIDATE")
    destination.mkdir(parents=True)
    payload = destination / "payload"
    shutil.copytree(parent, payload)
    source = stage_source(parent, history, destination / "source-build")
    for path in source.rglob("*"):
        if path.is_file():
            target = payload / "source/src" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    scripts = Path(__file__).parent
    for pattern, target in (
        ("seed202_mqar*.py", payload / "source/scripts"),
        ("build_seed202_mqar*.py", payload / "source/scripts"),
        ("seed202_native_schema.py", payload / "source/scripts"),
    ):
        target.mkdir(parents=True, exist_ok=True)
        for path in scripts.glob(pattern):
            shutil.copyfile(path, target / path.name)
    tests = scripts.parent / "tests/scripts"
    target = payload / "source/tests/scripts"
    target.mkdir(parents=True, exist_ok=True)
    for pattern in ("test_seed202_mqar*.py", "seed202_mqar*.py", "test_seed202_native_schema.py"):
        for path in tests.glob(pattern):
            shutil.copyfile(path, target / path.name)
    runbook = scripts.parent / "docs/mqar202-continuation-runbook.ru.md"
    (payload / "docs").mkdir(exist_ok=True)
    shutil.copyfile(runbook, payload / "docs" / runbook.name)
    (payload / "source/docs").mkdir(exist_ok=True)
    shutil.copyfile(runbook, payload / "source/docs" / runbook.name)
    shutil.copyfile(baseline, payload / "inherited/mqar-original.zip.bin")
    evidence = payload / "evidence/mqar-original"
    evidence.mkdir(parents=True)
    for name, checksum in (
        ("submit-proof.json", "d1102e3a4f9592d8412895b226cc6f57e8236bf5eb81b6403287fc536a0c2464"),
        (
            "failed-download-index-v1.json",
            "464c032ad042342a720ee77da19a22683b8d1d70df74c29acc33479b58c41652",
        ),
    ):
        path = EXECUTION / name
        if sha(path) != checksum:
            raise ValueError("AUTHENTIC_ORIGINAL_SOURCE_EVIDENCE")
        shutil.copyfile(path, evidence / name)
    with zipfile.ZipFile(parent / "worker/seed202-pilot.zip.bin") as inherited:
        worker_files = {
            name: inherited.read(name)
            for name in inherited.namelist()
            if name.startswith(("cloud/", "configs/"))
        }
    worker_files.update(
        {
            "src/" + name: source / name
            for name in inventory(source)
            if name not in {"quality_native.py", "quality_contract.py"}
        }
    )
    worker = payload / "worker/seed202-pilot.zip.bin"
    # Existing copied worker remains preserved inside mqar-original.zip.bin.
    worker_new = destination / "worker-new.zip.bin"
    write_zip(worker_new, worker_files, "MQAR_CONTINUATION_WORKER_OVERLAY")
    shutil.copyfile(worker_new, worker)
    original, pins = original_files()
    resume_new = destination / "resume-new.zip.bin"
    resume_files = {"original/" + name: original / name for name in pins}
    with zipfile.ZipFile(baseline) as archive:
        prefix = "deploy/mqar-k64-classic-seed202/kernel/"
        resume_files["original-kernel.py"] = archive.read(prefix + "kernel.py")
        resume_files["original-kernel-metadata.json"] = archive.read(
            prefix + "kernel-metadata.json"
        )
    write_zip(resume_new, resume_files, "EXACT_ORIGINAL_MQAR_ERROR_FULL_INPUT")
    plan = json.loads((parent / "PLAN-UNAPPROVED.json").read_bytes())
    prior_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "seed202_pilot" or name.startswith("seed202_pilot.")
    }
    for name in prior_modules:
        del sys.modules[name]
    sys.path.insert(0, str(source))
    try:
        from seed202_mqar_render import render_kernel

        for row in plan["deployments"]:
            deploy = payload / "deploy" / row["case_id"]
            if row["case_id"] == "mqar-k64-classic-seed202":
                row.update(
                    mode=MODE,
                    nonce=NONCE,
                    dataset_target="zimike/sorting-mqar-seed202-" + NONCE[:16] + "/1",
                    kernel_target="zimike/seed202-pilot-" + NONCE[:16],
                    resume_input_target="zimike/seed202-resume-input-" + NONCE[:16] + "/1",
                    resume_input_sha256=sha(resume_new),
                    original_error_kernel="zimike/seed202-pilot-786acf2c5ec864ce/1",
                )
                code, metadata = render_kernel(
                    row["case_id"],
                    NONCE,
                    sha(worker),
                    parent / "inherited/runtime-order.zip.bin",
                    resume_sha256=sha(resume_new),
                )
                from seed202_mqar_authority import pin
                from seed202_mqar_postreview import kernel_binding, normalized_kernel
                from seed202_pilot.contract import MQAR_DATA

                with zipfile.ZipFile(worker) as worker_archive:
                    worker_manifest = pin(worker_archive.read("manifest.json"))
                evidence_required = {
                    "scope": "MQAR_EVIDENCE_ONLY_V1",
                    "mode": MODE,
                    "case": row["case_id"],
                    "nonce": NONCE,
                    "mounted": {"resume.zip.bin": pin(resume_new.read_bytes())},
                    "worker": pin(worker.read_bytes()),
                    "worker_manifest": worker_manifest,
                    "normalized_kernel": pin(normalized_kernel(code)),
                    "metadata": pin(json_bytes(metadata)),
                    "config": pin((payload / "configs" / (row["case_id"] + ".yaml")).read_bytes()),
                    "requested_versioned_inputs": metadata["dataset_sources"],
                    "science": pin((parent / "inherited/sorting-science.zip.bin").read_bytes()),
                    "mqar_data": MQAR_DATA,
                    "private_input": "VERSIONED_REF_PLUS_OWNER_HELD_BYTES_NO_DISCLOSED_SECRET_PINS",
                }
                row["evidence_sha256"] = pin(json_bytes(evidence_required))["sha256"]
                row["authority_sha256"] = "0" * 64
                code = kernel_binding(code, evidence_sha256=row["evidence_sha256"])
                (deploy / "evidence-required.json").write_bytes(json_bytes(evidence_required))
                (deploy / "resume-input").mkdir()
                shutil.copyfile(resume_new, deploy / "resume-input/resume.zip.bin")
                (deploy / "resume-input/dataset-metadata.json").write_bytes(
                    json_bytes(
                        {
                            "id": row["resume_input_target"].rsplit("/", 1)[0],
                            "title": row["resume_input_target"].split("/")[1],
                            "licenses": [{"name": "CC0-1.0"}],
                        }
                    )
                )
                (deploy / "kernel/kernel.py").write_text(code, encoding="utf-8", newline="\n")
                (deploy / "kernel/kernel-metadata.json").write_bytes(json_bytes(metadata))
            else:
                from seed202_pilot.render import render_kernel as fresh_render

                code, metadata = fresh_render(
                    row["case_id"],
                    row["nonce"],
                    sha(worker),
                    parent / "inherited/runtime-order.zip.bin",
                )
                (deploy / "kernel/kernel.py").write_text(code, encoding="utf-8", newline="\n")
                (deploy / "kernel/kernel-metadata.json").write_bytes(json_bytes(metadata))
            shutil.copyfile(worker, deploy / "dataset/seed202-pilot.zip.bin")
            (deploy / "dataset/dataset-metadata.json").write_bytes(
                json_bytes(
                    {
                        "id": row["dataset_target"].rsplit("/", 1)[0],
                        "title": row["dataset_target"].split("/")[1],
                        "licenses": [{"name": "CC0-1.0"}],
                    }
                )
            )
            row.update(
                worker_sha256=sha(worker),
                kernel_source_sha256=sha(deploy / "kernel/kernel.py"),
                kernel_metadata_sha256=sha(deploy / "kernel/kernel-metadata.json"),
            )
    finally:
        sys.path.remove(str(source))
        for name in list(sys.modules):
            if name == "seed202_pilot" or name.startswith("seed202_pilot."):
                del sys.modules[name]
        sys.modules.update(prior_modules)
    plan["source_authority"]["source_files"] = inventory(payload / "source")
    old_source = {
        name.removeprefix("source/src/"): pin
        for name, pin in json.loads((parent / "manifest.json").read_bytes())["files"].items()
        if name.startswith("source/src/")
    }
    new_source = inventory(payload / "source/src")
    classification = {
        "scope": "AUXILIARY_BOUNDARY_ONLY_NO_SCIENTIFIC_WHEEL_OR_RECIPE_CHANGE",
        "modified_inherited": {
            name: {"before": value, "after": new_source.get(name)}
            for name, value in old_source.items()
            if new_source.get(name) != value
        },
        "added_runtime": {
            name: value for name, value in new_source.items() if name not in old_source
        },
        "training_wheel": {
            "sha256": sha(payload / "inherited/training-wheel.whl"),
            "unchanged": True,
        },
        "independent_review": "REQUIRED_NOT_ISSUED",
        "target_restore": "PENDING_FUTURE_AUTHORIZED_CUDA_JOB",
    }
    (payload / "evidence/mqar-source-classification.json").write_bytes(json_bytes(classification))
    plan["source_authority"]["worker"] = {"bytes": worker.stat().st_size, "sha256": sha(worker)}
    plan["source_authority"]["inherited_files"]["inherited/mqar-original.zip.bin"] = {
        "bytes": Path(baseline).stat().st_size,
        "sha256": BASELINE_SHA,
    }
    plan["review_file_allowlist_excluding_plan_and_manifest"] = {
        name: pin
        for name, pin in inventory(payload).items()
        if name not in {"PLAN-UNAPPROVED.json", "manifest.json"}
    }
    (payload / "PLAN-UNAPPROVED.json").write_bytes(json_bytes(plan))
    (destination / "PLAN-UNAPPROVED.json").write_bytes(json_bytes(plan))
    files = {name: payload / name for name in inventory(payload) if name != "manifest.json"}
    manifest = write_zip(
        destination / "sorting-mqar-seed202-pilot-v1.zip",
        files,
        "MQAR_CONTINUATION_STAGED_ENGINEERING_NOT_YET_REVIEW_READY",
    )
    (payload / "manifest.json").write_bytes(manifest)
    (destination / "candidate.json").write_bytes(
        json_bytes(
            {
                "status": "STAGED_FOR_ENGINEERING_TESTS_NOT_READY",
                "sha256": sha(destination / "sorting-mqar-seed202-pilot-v1.zip"),
                "independent_review": "REQUIRED_NOT_ISSUED",
                "cloud_writes": 0,
            }
        )
    )
    return destination


def freeze_candidate(staged, verification_dir, destination):
    """Freeze only the exact source bytes that passed the complete affected matrix."""
    from seed202_mqar_quality import junit_stats

    staged, verification_dir, destination = map(Path, (staged, verification_dir, destination))
    report = json.loads((verification_dir / "verification.json").read_bytes())
    required = (
        "test_public_both_inputs_unknown_ack_get_only_with_genuine_original_error",
        "test_frozen_saved_main_mounts_checkpoint_before_selected_uv_metadata_boundary",
        "test_complete_native_metadata_validator_on_genuine_full",
        "test_rehashed_container_type_changes_do_not_weaken_native_predicates[model_structure]",
        "test_rehashed_container_type_changes_do_not_weaken_native_predicates[optimizer_betas]",
        "test_genuine_full_first_callback_sdk_http_readback[False-False]",
        "test_genuine_full_first_callback_sdk_http_readback[True-False]",
        "test_genuine_full_first_callback_sdk_http_readback[False-True]",
        "test_actual_public_normal_return_reaches_host_receipt_consumer",
        "test_postreview_bytes_inventory_rejects_missing_changed_and_extra",
        "test_postreview_descriptor_identifies_same_derived_archive_and_unique_member_total",
        "test_full_ordinary_native_eleven_publications_and_required_continuation",
        "test_profile_return_calls_native_local_call_before_any_update[0]",
        "test_profile_return_calls_native_local_call_before_any_update[1]",
        "test_real_saved_hardware_guard_accepts_parent_wrong_torch_and_rejects_topology",
        "test_actual_import_shadow_and_postimport_inventory_guards",
    )
    stats = junit_stats([verification_dir / "affected.xml"])
    names = [name for _classname, name in stats["unique_ids"]]
    valid = (
        report["status"] == "OWNER_AFFECTED_CHECKS_PASSED_NOT_INDEPENDENT_REVIEW"
        and report["staged_archive_sha256"] == sha(staged / "sorting-mqar-seed202-pilot-v1.zip")
        and report["source_inventory"] == inventory(staged / "payload/source")
        and json_bytes(stats) == json_bytes(report["tests"])
        and all(any(name.startswith(prefix) for name in names) for prefix in required)
        and sum(
            classname.endswith("test_seed202_native_schema")
            for classname, _name in stats["unique_ids"]
        )
        == 23
        and report["preservation_before"] == report["preservation_after"]
        and not destination.exists()
    )
    if not valid:
        raise ValueError("COMPLETE_SOURCE_BOUND_AFFECTED_VERIFICATION_REQUIRED")
    for check in report["checks"].values():
        if check["exit_code"] != 0 or sha(verification_dir / check["log"]) != check["sha256"]:
            raise ValueError("QUALITY_LOG_DRIFT")
    for layout in ("flat", "owner"):
        command = report["saved_main_commands"][layout]
        path = verification_dir / command["file"]
        if (
            sha(path) != command["sha256"]
            or json.loads(path.read_bytes())["tested_input_archive_sha256"]
            != report["staged_archive_sha256"]
        ):
            raise ValueError("SAVED_COMMAND_RECEIPT_DRIFT")
    destination.mkdir(parents=True)
    payload = destination / "payload"
    shutil.copytree(staged / "payload", payload)
    evidence = payload / "evidence/mqar-continuation-owner"
    evidence.mkdir(parents=True)
    for name in (
        "verification.json",
        "affected.xml",
        "ruff.log",
        "format.log",
        "pylint.log",
        "pytest.log",
        "kernels-ruff.log",
        "kernels-pylint.log",
        "saved-main-flat-command.json",
        "saved-main-owner-command.json",
    ):
        shutil.copyfile(verification_dir / name, evidence / name)
    plan = json.loads((payload / "PLAN-UNAPPROVED.json").read_bytes())
    plan["review_file_allowlist_excluding_plan_and_manifest"] = {
        name: pin
        for name, pin in inventory(payload).items()
        if name not in {"PLAN-UNAPPROVED.json", "manifest.json"}
    }
    (payload / "PLAN-UNAPPROVED.json").write_bytes(json_bytes(plan))
    (destination / "PLAN-UNAPPROVED.json").write_bytes(json_bytes(plan))
    for path in (payload / "deploy").glob("*/kernel/kernel.py"):
        compile(path.read_bytes(), str(path), "exec")
    files = {name: payload / name for name in inventory(payload) if name != "manifest.json"}
    archive = destination / "sorting-mqar-seed202-pilot-v1.zip"
    manifest = write_zip(archive, files, "COMPLETE_MQAR_CONTINUATION_REVIEW_CANDIDATE")
    (payload / "manifest.json").write_bytes(manifest)
    descriptor = {
        "status": "COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
        "archive": str(archive.resolve()),
        "sha256": sha(archive),
        "members": len(files) + 1,
        "source_inventory": report["source_inventory"],
        "tests": stats,
        "junit_sha256": sha(evidence / "affected.xml"),
        "preservation": report["preservation_after"],
        "cloud_writes": 0,
        "allocations": 0,
        "independent_review": "REQUIRED_NOT_PERFORMED_NO_SELF_APPROVAL",
        "actual_target_restore": "PENDING_FUTURE_SINGLE_APPROVED_CUDA_JOB",
        "postreview_mounted_authority": "REQUIRED_EXTERNAL_REVIEW_AND_EXECUTION_DECISION_BYTES",
        "qualification": (
            "OWNER_ENGINEERING_CANDIDATE_NOT_SCIENTIFIC_ACCEPTANCE_OR_EXECUTION_PERMISSION"
        ),
    }
    (destination / "candidate.json").write_bytes(json_bytes(descriptor))
    paths = {
        "archive": archive,
        "descriptor": destination / "candidate.json",
        "manifest": payload / "manifest.json",
        "classification": payload / "evidence/mqar-source-classification.json",
        "verification": evidence / "verification.json",
        "junit": evidence / "affected.xml",
        "runbook": payload / "docs/mqar202-continuation-runbook.ru.md",
        "saved_main_flat_command": evidence / "saved-main-flat-command.json",
        "saved_main_owner_command": evidence / "saved-main-owner-command.json",
        "postreview_evidence": payload / "deploy/mqar-k64-classic-seed202/evidence-required.json",
        "staging_archive": staged / "sorting-mqar-seed202-pilot-v1.zip",
        "staging_descriptor": staged / "candidate.json",
    }
    (destination / "artifact-index.json").write_bytes(
        json_bytes(
            {
                "scope": "FROZEN_OWNER_ENGINEERING_CANDIDATE_INDEPENDENT_REVIEW_REQUIRED",
                "files": {
                    name: {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha(path),
                    }
                    for name, path in paths.items()
                },
                "tests": stats["unique_passed"],
                "cloud_writes": 0,
                "allocations": 0,
                "source_inventory": report["source_inventory"],
                "archive_readback": {
                    "unique_regular_members": descriptor["members"],
                    "all_member_bytes_equal": True,
                },
            }
        )
    )
    return destination


def main(argv=None):
    """Local create-only staging command. No SDK, credentials, upload or launch."""
    import argparse
    import os

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("parent", "baseline", "history", "staged", "verification-dir", "destination"):
        parser.add_argument("--" + name, type=Path, required=name == "destination")
    args = parser.parse_args(argv)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError("MASK_AND_NO_BYTECODE_REQUIRED_BEFORE_IMPORTS")
    if args.staged and args.verification_dir:
        root = freeze_candidate(args.staged, args.verification_dir, args.destination)
    else:
        if not all((args.parent, args.baseline, args.history)):
            parser.error(
                "stage requires --parent --baseline --history; "
                "freeze requires --staged --verification-dir"
            )
        root = stage_candidate(args.parent, args.baseline, args.history, args.destination)
    descriptor = json.loads((root / "candidate.json").read_bytes())
    print(
        json_bytes(
            {"status": descriptor["status"], "root": str(root), "sha256": descriptor["sha256"]}
        ).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
