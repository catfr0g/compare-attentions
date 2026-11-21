"""Prepare NORMAL completion source in a new directory; never deploy or authorize."""

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

PARENT_SHA256 = "e10e7e693e5eeca90ba732d2644d8c00a303bd32533abe04b3789955c5b499f5"


def stage_source(parent_archive, destination):
    """Extract manifest-authenticated Python only, without scientific data copies."""
    parent_archive, destination = Path(parent_archive), Path(destination)
    with parent_archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != PARENT_SHA256 or destination.exists():
        raise ValueError("NORMAL_SOURCE_PARENT_OR_CREATE_ONLY")
    members = {}
    with zipfile.ZipFile(parent_archive) as saved:
        names = saved.namelist()
        if len(names) != len(set(names)):
            raise ValueError("NORMAL_SOURCE_DUPLICATE_MEMBERS")
        manifest = json.loads(saved.read("manifest.json"))["files"]
        for name in names:
            part = PurePosixPath(name)
            if part.is_absolute() or ".." in part.parts or "\\" in name or ":" in name:
                raise ValueError("NORMAL_SOURCE_UNSAFE_MEMBER")
            if name.startswith("source/src/") and name.endswith(".py"):
                raw = saved.read(name)
                if manifest[name] != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                    raise ValueError("NORMAL_SOURCE_MEMBER_PIN")
                members[name.removeprefix("source/src/")] = raw
    if "seed202_pilot/cloud_completion.py" not in members:
        raise ValueError("NORMAL_SOURCE_COMPLETION_REQUIRED")
    old = members["seed202_pilot/cloud_completion.py"]
    if (
        hashlib.sha256(old).hexdigest()
        != "b5f48c2b1da5d78d2033cc9ab0fee4bdeef36a075d756d1255436535df635920"
    ):
        raise ValueError("NORMAL_COMPLETION_PARENT_PIN")
    members["seed202_pilot/cloud_completion.py"] = (
        Path(__file__).with_name("training_completion.py").read_bytes()
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in members.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return destination


def _pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(root, name, raw):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def build_package(parent_archive, repository, destination):
    """Create a source-authenticated future NORMAL derivative, never a launch plan."""
    parent_archive, repository, destination = map(Path, (parent_archive, repository, destination))
    stage_source(parent_archive, destination / "inherited-source")
    with zipfile.ZipFile(parent_archive) as parent:
        old_wheel = parent.read("inherited/training-wheel.whl")
        if _pin(old_wheel)["sha256"] != (
            "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
        ):
            raise ValueError("NORMAL_ORIGINAL_WHEEL")
        science_raw = parent.read("inherited/sorting-science.zip.bin")
        package_raw = parent.read("inherited/parent-candidate03.zip.bin")
    if _pin(package_raw)["sha256"] != (
        "b3e5c1824b05999f475e844186ba197f4a0a2808dd6638c463ea7664222ca57d"
    ):
        raise ValueError("NORMAL_PURE_CONTRACT_PARENT")
    with zipfile.ZipFile(io.BytesIO(package_raw)) as pure:
        _write(
            destination,
            "inherited/operator/quality_contract.py",
            pure.read("dependencies/package/src/quality_contract.py"),
        )
    if _pin(science_raw)["sha256"] != (
        "71872615b0cfbe2406cd08762cc02c390c88a70b6bb5f4bb38d6571830c8a3bb"
    ):
        raise ValueError("NORMAL_SCIENCE_PARENT")
    with zipfile.ZipFile(io.BytesIO(science_raw)) as science:
        for name in science.namelist():
            if name.startswith(("operator/", "tools/")) and name.endswith(".py"):
                part = PurePosixPath(name)
                if part.is_absolute() or ".." in part.parts or "\\" in name or ":" in name:
                    raise ValueError("NORMAL_SCIENCE_UNSAFE")
                _write(destination, "inherited/" + name, science.read(name))
    with zipfile.ZipFile(io.BytesIO(old_wheel)) as wheel:
        members = {name: wheel.read(name) for name in wheel.namelist()}
    changed = []
    for filename in ("backup", "distributed", "identity", "loop", "tracker", "lifecycle"):
        name = "compare_attention/train/" + filename + ".py"
        raw = (repository / "src" / name).read_bytes()
        if filename == "loop":

            def functions(value):
                return {
                    node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(value).body
                    if isinstance(node, ast.FunctionDef) and node.name != "_train"
                }

            if functions(raw) != functions(members[name]):
                raise ValueError("NORMAL_NUMERICAL_FUNCTION_DRIFT")
        if members.get(name) != raw:
            changed.append(name)
        members[name] = raw
    record_name = "compare_attention-0.1.0.dist-info/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, raw in sorted(members.items()):
        if name != record_name:
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            writer.writerow((name, "sha256=" + digest, len(raw)))
    writer.writerow((record_name, "", ""))
    members[record_name] = record.getvalue().encode()
    wheel_path = destination / "compare_attention-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "x", compression=zipfile.ZIP_DEFLATED) as saved:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            saved.writestr(info, raw)
    scripts = [
        "training_completion",
        "normal_training_plan",
        "normal_training_native",
        "normal_training_worker",
        "normal_training_acceptance",
    ]
    for name in scripts:
        _write(
            destination,
            "src/" + name + ".py",
            (repository / "scripts" / (name + ".py")).read_bytes(),
        )
    launch = (repository / "scripts/normal_training_launch.py").read_bytes()
    _write(destination, "setup.py", launch)
    _write(
        destination,
        "normal_driver_environment.py",
        (repository / "scripts/normal_driver_environment.py").read_bytes(),
    )
    for helper in ("normal_launch_diagnostics", "normal_torchrun_entry"):
        _write(
            destination, helper + ".py", (repository / "scripts" / (helper + ".py")).read_bytes()
        )
    _write(
        destination,
        "worker.py",
        (
            b'"""Saved public NORMAL worker, no import-time scientific execution."""\n\n'
            b"import sys\nfrom pathlib import Path\n\n"
            b'sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))\n'
            b"from normal_launch_diagnostics import normal_cli\n\n\n"
            b"def entry():\n"
            b"    from normal_training_worker import main\n\n"
            b"    return main()\n\n\n"
            b'if __name__ == "__main__":\n    raise SystemExit(normal_cli(entry, "pre-worker"))\n'
        ),
    )
    files = {
        p.relative_to(destination).as_posix(): _pin(p.read_bytes())
        for p in destination.rglob("*")
        if p.is_file()
    }
    identity = {
        "schema": "normal-source-identity-v1",
        "scope": "FUTURE_NORMAL_TRAINING_LIFECYCLE_NOT_OLD_RESUME",
        "parent_archive_sha256": PARENT_SHA256,
        "original_wheel_sha256": _pin(old_wheel)["sha256"],
        "wheel_sha256": _pin(wheel_path.read_bytes())["sha256"],
        "wheel_changed_source_members": sorted(changed),
        "wheel_packaging_changed": [record_name],
        "whole_wheel_byte_identity": False,
        "files": files,
    }
    _write(
        destination,
        "normal-source-identity.json",
        json.dumps(identity, sort_keys=True, indent=2).encode() + b"\n",
    )
    return destination


def build_observer_release(parent_package, repository, destination):
    """Derive only launch observability from the authenticated remaining release."""
    import shutil

    from unified_normal_profiles import REMAINING, authenticate_package

    parent_package, repository, destination = map(Path, (parent_package, repository, destination))
    identity = authenticate_package(parent_package, REMAINING)
    if destination.exists():
        raise ValueError("NORMAL_SOURCE_PARENT_OR_CREATE_ONLY")
    shutil.copytree(parent_package, destination)
    for name, target in (
        ("normal_training_launch.py", "setup.py"),
        ("normal_torchrun_entry.py", "normal_torchrun_entry.py"),
    ):
        _write(destination, target, (repository / "scripts" / name).read_bytes())
    files = {
        p.relative_to(destination).as_posix(): _pin(p.read_bytes())
        for p in destination.rglob("*")
        if p.is_file() and p.name != "normal-source-identity.json"
    }
    updated = {
        **identity,
        "observability_parent_identity_sha256": _pin(
            (parent_package / "normal-source-identity.json").read_bytes()
        )["sha256"],
        "files": files,
    }
    _write(
        destination,
        "normal-source-identity.json",
        json.dumps(updated, sort_keys=True, indent=2).encode() + b"\n",
    )
    if (destination / "compare_attention-0.1.0-py3-none-any.whl").read_bytes() != (
        parent_package / "compare_attention-0.1.0-py3-none-any.whl"
    ).read_bytes():
        raise ValueError("NORMAL_OBSERVER_SCIENCE_DRIFT")
    return destination


def main(argv=None):
    """Build only; no credentials, provider SDK, review creation or Git mutation."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("parent", "repository", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    build_package(args.parent, args.repository, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
