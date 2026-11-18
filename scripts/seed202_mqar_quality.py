"""Offline CPU affected checks and source-bound evidence; never independent approval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

from build_seed202_mqar_continuation import EXECUTION, sha, stage_science_helpers
from seed202_mqar_package import inventory, json_bytes

INHERITED_TESTS = (
    "test_completion.py",
    "test_cloud.py",
    "test_cloud_observer.py",
    "test_validator_binding.py",
    "test_task_metrics.py",
    "test_metrics.py",
    "test_preflight_integration02.py",
    "test_seal_integration03.py",
    "test_publish_integration02.py",
    "test_acceptance_integration02.py",
)


def junit_stats(paths):
    """Use testcase identities, not totals from overlapping test runs."""
    identities = set()
    for path in paths:
        cases = ET.parse(path).getroot().findall(".//testcase")
        if not cases:
            raise ValueError("AFFECTED_TESTS_NOT_ALL_PASSED")
        for case in cases:
            if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
                raise ValueError("AFFECTED_TESTS_NOT_ALL_PASSED")
            identities.add((case.attrib["classname"], case.attrib["name"]))
    return {
        "unique_passed": len(identities),
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "unique_ids": sorted(identities),
    }


def _run(command, env, log):
    with Path(log).open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False
        )
    if result.returncode:
        raise ValueError("QUALITY_CHECK_FAILED_" + Path(log).stem.upper())
    return {"exit_code": result.returncode, "log": Path(log).name, "sha256": sha(log)}


def _preservation(parent):
    expected = json.loads((EXECUTION / "failed-download-index-v1.json").read_bytes())["inventory"]
    actual = inventory(EXECUTION / "failed-download-v1")
    if (
        actual != expected
        or len(actual) != 1402
        or sum(p["bytes"] for p in actual.values()) != 137928809
    ):
        raise ValueError("ORIGINAL_DOWNLOAD_MUTATION")
    baseline = EXECUTION.parent / "development-07/sorting-mqar-seed202-pilot-v1.zip"
    if sha(baseline) != "581406e1f8569fcfa50d319033515209962a0b4c4f67d98947905cd226aa1712":
        raise ValueError("AUTHENTIC_BASELINE_ARCHIVE_REQUIRED")
    with zipfile.ZipFile(baseline) as saved:
        manifest_raw = saved.read("manifest.json")
    if manifest_raw != (Path(parent) / "manifest.json").read_bytes():
        raise ValueError("AUTHENTIC_BASELINE_MANIFEST_REQUIRED")
    manifest = json.loads(manifest_raw)["files"]
    if inventory(parent) != {
        **manifest,
        "manifest.json": {
            "bytes": (Path(parent) / "manifest.json").stat().st_size,
            "sha256": sha(Path(parent) / "manifest.json"),
        },
    }:
        raise ValueError("ORIGINAL_BASELINE_EXTRACTION_MUTATION")
    return {
        "raw_files": len(actual),
        "raw_bytes": sum(p["bytes"] for p in actual.values()),
        "raw_inventory_sha256": __import__("hashlib").sha256(json_bytes(actual)).hexdigest(),
        "training_wheel_sha256": sha(Path(parent) / "inherited/training-wheel.whl"),
        "baseline_archive_sha256": sha(baseline),
        "protected_parent_snapshot": _protected_snapshot(),
    }


def _protected_snapshot():
    """Close all1802 original/core/WIP bytes and link counts from the read-only parent."""
    path = (
        EXECUTION.parents[2]
        / "mqar-classic-resume-repair-v1/parent-verification-02/preservation-before.json"
    )
    expected_sha = "cdf5d099256d48f1dbddb1c72b879e3138733a46036563070e78ffc74caee58c"
    if sha(path) != expected_sha:
        raise ValueError("AUTHENTIC_PARENT_PRESERVATION_SNAPSHOT")
    value = json.loads(path.read_bytes())
    if len(value["files"]) != 1802:
        raise ValueError("ALL_1802_PROTECTED_FILES_REQUIRED")
    for name, expected in value["files"].items():
        item = Path(name)
        stat = item.stat()
        if (
            item.is_symlink()
            or item.is_junction()
            or expected != {"bytes": stat.st_size, "nlink": stat.st_nlink, "sha256": sha(item)}
        ):
            raise ValueError("PROTECTED_ORIGINAL_CORE_WIP_BYTES_OR_LINKS_CHANGED")
    repository = Path("C:/Users/Michael/Documents/projects/compare-attention")
    index = repository / ".git/index"
    metadata = index.stat()
    if value["git_index"] != {
        "bytes": metadata.st_size,
        "nlink": metadata.st_nlink,
        "sha256": sha(index),
    }:
        raise ValueError("GIT_INDEX_CHANGED")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != value["git_head"]:
        raise ValueError("GIT_HEAD_CHANGED")
    return {
        "snapshot_sha256": expected_sha,
        "files": 1802,
        "all_bytes_and_nlinks_equal": True,
        "git_head": head,
        "git_index_unchanged": True,
    }


def _extract_wheel(parent, destination):
    archive = Path(parent) / "inherited/training-wheel.whl"
    if sha(archive) != "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3":
        raise ValueError("AUTHENTIC_WHEEL_REQUIRED")
    destination = Path(destination)
    destination.mkdir()
    with zipfile.ZipFile(archive) as saved:
        if len(saved.namelist()) != len(set(saved.namelist())):
            raise ValueError("UNIQUE_WHEEL_MEMBERS")
        for info in saved.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename or info.is_dir():
                raise ValueError("SAFE_WHEEL_MEMBERS")
            target = destination / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(saved.read(info))
    return destination


def verify(staged, destination, parent):
    """Lint first, then one fresh affected pytest run against frozen source bytes."""
    staged, destination, parent = Path(staged), Path(destination), Path(parent)
    if destination.exists():
        raise ValueError("CREATE_ONLY_QUALITY_EVIDENCE")
    destination.mkdir(parents=True)
    before = _preservation(parent)
    helpers = stage_science_helpers(parent, destination / "science-helpers")
    wheel = _extract_wheel(parent, destination / "authenticated-wheel")
    cloud = destination / "authenticated-cloud"
    cloud.mkdir()
    with zipfile.ZipFile(staged / "payload/worker/seed202-pilot.zip.bin") as saved:
        pins = json.loads(saved.read("manifest.json"))["files"]
        for name in ("quality_native_v2.py", "quality_runtime_join.py"):
            raw = saved.read("cloud/" + name)
            target = cloud / name
            target.write_bytes(raw)
            if pins["cloud/" + name] != {"bytes": len(raw), "sha256": sha(target)}:
                raise ValueError("AUTHENTIC_CLOUD_HELPER_MEMBERS")
    fixture = destination / "ENGINEERING-ONLY-candidate"
    fixture.mkdir()
    shutil.copytree(staged / "payload", fixture / "payload")
    for name in ("PLAN-UNAPPROVED.json", "candidate.json", "sorting-mqar-seed202-pilot-v1.zip"):
        shutil.copyfile(staged / name, fixture / name)
    descriptor = json.loads((fixture / "candidate.json").read_bytes())
    descriptor.update(
        status="COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
        engineering_fixture="CPU_SDK_FIXTURE_NOT_PRODUCTION_REVIEW_AUTHORITY",
    )
    (fixture / "candidate.json").write_bytes(json_bytes(descriptor))
    source = staged / "payload/source"
    script_root, tests = source / "scripts", source / "tests"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SEED202_FROZEN_CANDIDATE": str(fixture),
        "MQAR_TEST_CANDIDATE": str(staged),
        "MQAR_REPAIR_SOURCE": str(staged / "payload/source/src"),
        "SEED202_EXTRACTED_WHEEL": str(wheel),
        "TMPDIR": "C:/Users/Michael/AppData/Local/hermes/cache/scratch",
    }
    env["PYTHONPATH"] = os.pathsep.join(
        map(
            str,
            (
                script_root,
                tests / "scripts",
                source / "src",
                wheel,
                cloud,
                helpers / "operator",
                helpers / "operator/inherited",
                helpers / "tools",
                tests / "seed202_pilot",
                tests / "qualified_launch",
            ),
        )
    )
    base = [shutil.which("uv"), "run", "--offline", "--no-sync", "python", "-B"]
    maintained = [*script_root.glob("*mqar*.py"), *tests.glob("scripts/*mqar*.py")]
    maintained += [script_root / "seed202_native_schema.py"]
    package = source / "src/seed202_pilot"
    generated = [*package.glob("mqar_*.py"), package / "resume_router.py"]
    generated += [package / "native_schema.py"]
    generated += [
        package / (name + ".py")
        for name in (
            "cloud_native",
            "fresh_mode",
            "host_operator",
            "preparation",
            "preflight",
            "seal",
            "cli",
            "science",
            "periodic",
        )
    ]
    targets = list(map(str, [*maintained, *generated]))
    checks = {
        "ruff": _run([*base, "-m", "ruff", "check", *targets], env, destination / "ruff.log"),
        "format": _run(
            [*base, "-m", "ruff", "format", "--check", *targets], env, destination / "format.log"
        ),
        "pylint": _run(
            [*base, "-m", "pylint", "--disable=all", "--enable=E,F", *targets],
            env,
            destination / "pylint.log",
        ),
    }
    kernels = sorted((staged / "payload/deploy").glob("*/kernel/kernel.py"))
    for kernel in kernels:
        compile(kernel.read_bytes(), str(kernel), "exec")
    checks["saved_kernels_ruff"] = _run(
        [*base, "-m", "ruff", "check", "--select", "E,F", *map(str, kernels)],
        env,
        destination / "kernels-ruff.log",
    )
    checks["saved_kernels_pylint"] = _run(
        [*base, "-m", "pylint", "--disable=all", "--enable=E,F", *map(str, kernels)],
        env,
        destination / "kernels-pylint.log",
    )
    if any(
        len(path.read_text(encoding="utf-8").splitlines()) >= 500 for path in map(Path, targets)
    ):
        raise ValueError("SOURCE_MODULE_MUST_BE_LESS_THAN_500_LINES")
    junit = destination / "affected.xml"
    test_scratch = Path(env["TMPDIR"]) / (
        "mqar202-" + destination.name + "-" + uuid.uuid4().hex[:12]
    )
    if test_scratch.exists():
        raise ValueError("CREATE_ONLY_TEST_SCRATCH_REQUIRED")
    selected = sorted(tests.glob("scripts/test_seed202_mqar*.py"))
    selected += [tests / "scripts/test_seed202_native_schema.py"]
    selected += [tests / "seed202_pilot" / name for name in INHERITED_TESTS]
    checks["pytest"] = _run(
        [
            *base,
            "-m",
            "pytest",
            "--rootdir=" + str(source),
            *map(str, selected),
            "-q",
            "--tb=short",
            "--show-capture=no",
            "-p",
            "no:cacheprovider",
            "--basetemp=" + str(test_scratch),
            "--junitxml=" + str(junit),
        ],
        env,
        destination / "pytest.log",
    )
    after = _preservation(parent)
    if before != after:
        raise ValueError("PROTECTED_BYTES_MUTATED")
    with zipfile.ZipFile(staged / "sorting-mqar-seed202-pilot-v1.zip") as archive:
        recorded = json.loads(archive.read("manifest.json"))["files"]
    actual = inventory(staged / "payload")
    del actual["manifest.json"]
    if actual != recorded:
        raise ValueError("SOURCE_BOUND_TEST_INPUT_MUTATION")
    saved_commands = {}
    for case in ET.parse(junit).getroot().findall(".//testcase"):
        for prop in case.findall("./properties/property"):
            if prop.attrib["name"] != "uv_command_receipt":
                continue
            receipt = Path(prop.attrib["value"])
            value = json.loads(receipt.read_bytes())
            if (
                value["tested_input_archive_sha256"]
                != sha(staged / "sorting-mqar-seed202-pilot-v1.zip")
                or value["exit_code"] != 73
                or value["mask"] != "-1"
                or value["bytecode_disabled"] != "1"
                or sha(value["kernel"]) != value["kernel_sha256"]
            ):
                raise ValueError("EXACT_SOURCE_BOUND_SAVED_UV_COMMAND")
            name = "saved-main-" + value["layout"] + "-command.json"
            shutil.copyfile(receipt, destination / name)
            saved_commands[value["layout"]] = {"file": name, "sha256": sha(destination / name)}
    if set(saved_commands) != {"flat", "owner"}:
        raise ValueError("BOTH_SAVED_MAIN_COMMAND_RECEIPTS_REQUIRED")
    result = {
        "status": "OWNER_AFFECTED_CHECKS_PASSED_NOT_INDEPENDENT_REVIEW",
        "checks": checks,
        "tests": junit_stats([junit]),
        "source_inventory": inventory(source),
        "preservation_before": before,
        "preservation_after": after,
        "staged_archive_sha256": sha(staged / "sorting-mqar-seed202-pilot-v1.zip"),
        "target_restore": "PENDING_FUTURE_SINGLE_APPROVED_CUDA_JOB",
        "cloud_writes": 0,
        "test_scratch": str(test_scratch),
        "saved_main_commands": saved_commands,
        "independent_review": "REQUIRED_NOT_PERFORMED",
    }
    (destination / "verification.json").write_bytes(json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("staged", "destination", "parent"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError("CPU_MASK_REQUIRED_BEFORE_IMPORTS")
    result = verify(args.staged, args.destination, args.parent)
    print(
        json_bytes({"status": result["status"], "tests": result["tests"]["unique_passed"]}).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
