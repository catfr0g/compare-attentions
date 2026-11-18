"""Build a fresh HOST-only V7 qualified-predecessor candidate, never a cloud approval.

All raw artifact locations are supplied in an input JSON. No credentials are used.
The accepted V7 and recovery archives remain immutable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from seed202_qualified_bridge import ACCEPTED, ARCHIVES, canonical, digest, require

V7 = "8b3a9ad9a28ed10e0379456b972f94c14509ab5c12745c39e5c07f5cf5bbee21"
HOST_FILES = {"seal.py", "host_operator.py", "audited_publish.py", "preparation.py", "cli.py"}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")


def _inventory(root):
    result = {}
    for path in Path(root).rglob("*"):
        require(not path.is_symlink() and not path.is_junction(), "NO_LINKED_INPUTS")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    return result


def _replace(root, name, before, after):
    path = root / name
    text = path.read_text(encoding="utf-8")
    require(text.count(before) == 1, "EXACT_V7_HOST_INTERFACE_" + name)
    path.write_text(text.replace(before, after), encoding="utf-8", newline="\n")


def _integrate(root):
    _replace(
        root,
        "seal.py",
        "    from seed202_pilot.state import replay_acceptance\n\n    predecessors = []",
        "    from seed202_pilot.qualified_bridge import binding, replay_predecessor\n"
        "\n    bridge = binding(candidate)\n    predecessors = []",
    )
    _replace(
        root,
        "seal.py",
        "        replay_acceptance(resolved, expected.case_id)",
        "        replay_predecessor(\n"
        '            {"candidate_root": str(candidate), "qualified_launch_bridge": bridge},\n'
        "            resolved, expected.case_id,\n"
        "        )",
    )
    _replace(
        root,
        "seal.py",
        '        "qualified_predecessor": transition,',
        '        "qualified_predecessor": transition,\n        "qualified_launch_bridge": bridge,',
    )
    path = root / "seal.py"
    text = path.read_text()
    start = text.index("    expected = [", text.index("def require_predecessor_evidence"))
    end = text.index('    _require(proof.get("predecessor_evidence")', start)
    text = (
        text[:start] + "    from seed202_pilot.qualified_bridge import expected_evidence\n"
        "\n    expected = expected_evidence(plan)\n" + text[end:]
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    _replace(
        root,
        "seal.py",
        '            and row["replay_scope"] == "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY"',
        "            and (\n"
        "                (\n"
        '                    row["case_id"] == "sorting-classic-seed202"\n'
        '                    and row["replay_scope"] == "QUALIFIED_SORTING_CLASSIC_TAIL_V1"\n'
        "                )\n"
        "                or (\n"
        '                    row["case_id"] == "sorting-deltanet-seed202"\n'
        '                    and row["replay_scope"] == "QUALIFIED_SORTING_DELTA_TAIL_V1"\n'
        "                )\n"
        '                or row["replay_scope"] == "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY"\n'
        "            )",
    )
    path = root / "host_operator.py"
    text = path.read_text()
    start = text.index("    def predecessors():")
    end = text.index("    def local():", start)
    text = (
        text[:start] + "    def predecessors():\n"
        "        from seed202_pilot.qualified_bridge import proof_rows\n"
        '\n        plan = approval["plan"] if approval else _load(args.plan)\n'
        "        return proof_rows(plan)\n\n" + text[end:]
    )
    text = text.replace("    from seed202_pilot.state import replay_acceptance\n", "")
    path.write_text(text, encoding="utf-8", newline="\n")
    path = root / "audited_publish.py"
    text = path.read_text()
    start = text.index("        for row in predecessors:", text.index("def guarded_mutation"))
    end = text.index("        if two_stage:", start)
    text = (
        text[:start] + "        from seed202_pilot.qualified_bridge import proof_rows\n"
        '\n        _require(proof_rows(plan) == proof["predecessor_evidence"], '
        '"PREDECESSOR_WHOLE_REPLAY")\n' + text[end:]
    )
    text = text.replace("    predecessors = _prefix(plan)", "    _prefix(plan)")
    path.write_text(text, encoding="utf-8", newline="\n")
    _replace(
        root,
        "preparation.py",
        "    return list(map(str, files))",
        "    from seed202_pilot.qualified_bridge import evidence_paths\n"
        "\n    files += [Path(path) for path in evidence_paths(candidate)]\n"
        "    return list(dict.fromkeys(map(str, files)))",
    )
    _replace(
        root,
        "cli.py",
        "            selected = next_case(args.qualified_predecessor, args.acceptance)",
        "            if args.candidate_dir:\n"
        "                from seed202_pilot.qualified_bridge import next_qualified_case\n"
        "\n                selected = next_qualified_case(\n"
        "                    args.candidate_dir, args.qualified_predecessor, args.acceptance\n"
        "                )\n"
        "            else:\n"
        "                selected = next_case(args.qualified_predecessor, args.acceptance)",
    )
    _replace(
        root,
        "cli.py",
        "            from seed202_pilot.sdk_scope import suppressed",
        "            from seed202_pilot.qualified_imports import authenticated_imports\n"
        "            from seed202_pilot.sdk_scope import suppressed",
    )
    _replace(
        root,
        "cli.py",
        '            with command_scope(args, locals().get("approval")), suppressed():',
        '            with command_scope(args, locals().get("approval")), suppressed(), '
        "authenticated_imports(args):",
    )


def build(v7_archive, inputs, destination, *, test_root=None):
    """Exact archive inheritance plus finite host interfaces, with a complete diff index."""
    require(digest(v7_archive) == V7, "VERIFIED_V7_REQUIRED")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / "payload"
    with zipfile.ZipFile(v7_archive) as source:
        names = source.namelist()
        require(len(names) == len(set(names)) == 121, "V7_MEMBER_SET")
        require(
            all(
                not Path(n).is_absolute() and ".." not in Path(n).parts and chr(92) not in n
                for n in names
            ),
            "V7_SAFE_MEMBERS",
        )
        source.extractall(payload)
    before = _inventory(payload)
    require(
        set(inputs) == {"cases", "science", "science_archive", "transition_review"}
        and set(inputs["cases"]) == set(ACCEPTED),
        "INPUT_SPEC_SCHEMA",
    )
    external = {}
    closed_roots = set()
    for key in ("science_archive", "transition_review"):
        path = Path(inputs[key]).resolve()
        external[str(path)] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    for case_id, row in inputs["cases"].items():
        require(set(row) == {"archive", "root", "spec"}, "QUALIFIED_CASE_INPUTS")
        require(digest(row["archive"]) == ARCHIVES[case_id], "QUALIFIED_REVIEWED_ARCHIVE")
        require(digest(row["spec"]["output"]) == ACCEPTED[case_id], "GENUINE_ACCEPTANCE_REQUIRED")
        roots = [
            Path(row["root"]),
            Path(row["spec"]["input"]),
            Path(row["spec"]["result"]),
            Path(row["spec"]["data"]),
        ]
        files = [
            Path(row["archive"]),
            *[
                Path(row["spec"][key])
                for key in ("native", "services", "provider", "review", "plan", "output")
            ],
        ]
        for root in roots:
            closed_roots.add(str(root.resolve()))
            files.extend(root / name for name in _inventory(root))
        for path in files:
            require(path.suffix.lower() not in {".env", ".pem", ".key"}, "NO_PRIVATE_INPUTS")
            external[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    science = Path(inputs["science"])
    closed_roots.add(str(science.resolve()))
    for name, pin in _inventory(science).items():
        external[str((science / name).resolve())] = pin
    bridge = {
        "schema": "SEED202_QUALIFIED_LAUNCH_INPUTS_V1",
        **inputs,
        "roots": sorted(closed_roots),
        "files": external,
    }
    _save(payload / "qualified-launch/bridge.json", bridge)
    source_root = payload / "source/src/seed202_pilot"
    shutil.copyfile(
        Path(__file__).with_name("seed202_qualified_bridge.py"), source_root / "qualified_bridge.py"
    )
    shutil.copyfile(
        Path(__file__).with_name("seed202_qualified_imports.py"),
        source_root / "qualified_imports.py",
    )
    _integrate(source_root)
    _replace(
        payload / "source/tests/seed202_pilot",
        "fresh_sdk_fixture.py",
        '            "--science-dir",\n            str(self.root),\n            "--lifetime",',
        '            "--science-dir",\n'
        '            json.loads((self.candidate / "payload/qualified-launch/bridge.json")'
        '.read_bytes())["science"],\n'
        '            "--lifetime",',
    )
    # Format only the new HOST projection. Never touch inherited scientific bytes.
    subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-B",
            "-m",
            "ruff",
            "format",
            "--line-length",
            "100",
            *[str(source_root / name) for name in sorted(HOST_FILES)],
            str(payload / "source/tests/seed202_pilot/fresh_sdk_fixture.py"),
        ],
        check=True,
        capture_output=True,
    )
    # Retain the exact maintained builder and regression bytes for independent replay.
    for name in (
        "build_seed202_qualified_bridge.py",
        "seed202_qualified_bridge.py",
        "seed202_qualified_imports.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), payload / "qualified-launch" / name)
    tests = (
        Path(test_root)
        if test_root is not None
        else (Path(__file__).resolve().parents[1] / "tests/scripts")
    )
    frozen_tests = payload / "source/tests/qualified_launch"
    frozen_tests.mkdir()
    for name in (
        "test_seed202_qualified_launch.py",
        "test_seed202_qualified_security.py",
        "seed202_qualified_http_fixture.py",
    ):
        shutil.copyfile(tests / name, frozen_tests / name)
    shutil.copyfile(payload / "manifest.json", payload / "qualified-launch/parent-v7-manifest.json")
    (payload / "manifest.json").unlink()
    files = _inventory(payload)
    _save(
        payload / "manifest.json",
        {"schema_version": 1, "scope": "COMPLETE_SEED202_PILOT_REVIEW_CANDIDATE", "files": files},
    )
    archive = destination / "sorting-mqar-seed202-pilot-v1.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name in [*files, "manifest.json"]:
            output.write(payload / name, name)
    shutil.copyfile(payload / "PLAN-UNAPPROVED.json", destination / "PLAN-UNAPPROVED.json")
    descriptor = {
        "status": "COMPLETE_FROZEN_CANDIDATE_REVIEW_REQUIRED",
        "archive": str(archive),
        "sha256": digest(archive),
        "members": len(files) + 1,
        "worker_sha256": digest(payload / "worker/seed202-pilot.zip.bin"),
        "cloud_mutations": 0,
        "independent_review": "REQUIRED_NOT_SELF_ISSUED",
        "launch_approval": "NOT_ISSUED",
    }
    _save(destination / "candidate.json", descriptor)
    after = _inventory(payload)
    changed = {
        name: {"before": before.get(name), "after": pin}
        for name, pin in after.items()
        if before.get(name) != pin
    }
    allowed = {"source/src/seed202_pilot/" + name for name in HOST_FILES} | {
        "source/src/seed202_pilot/qualified_bridge.py",
        "source/src/seed202_pilot/qualified_imports.py",
        "source/tests/seed202_pilot/fresh_sdk_fixture.py",
        "qualified-launch/seed202_qualified_imports.py",
        "qualified-launch/bridge.json",
        "qualified-launch/parent-v7-manifest.json",
        "manifest.json",
        "qualified-launch/build_seed202_qualified_bridge.py",
        "qualified-launch/seed202_qualified_bridge.py",
        "source/tests/qualified_launch/test_seed202_qualified_launch.py",
        "source/tests/qualified_launch/test_seed202_qualified_security.py",
        "source/tests/qualified_launch/seed202_qualified_http_fixture.py",
    }
    require(set(changed) == allowed and set(before) <= set(after), "HOST_ONLY_DIFF")
    _save(
        destination / "HOST-DIFF.json",
        {"parent_sha256": V7, "classification": "HOST_ONLY_NOT_REVIEWED", "files": changed},
    )
    with zipfile.ZipFile(archive) as frozen:
        require(
            len(frozen.namelist()) == len(set(frozen.namelist())) == descriptor["members"],
            "FINAL_UNIQUE_MEMBERS",
        )
        require(
            all(frozen.read(n) == (payload / n).read_bytes() for n in frozen.namelist()),
            "FINAL_ARCHIVE_READBACK",
        )
    return descriptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--v7-archive", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.v7_archive,
                json.loads(args.inputs.read_bytes()),
                args.destination,
                test_root=args.test_root,
            ),
            sort_keys=True,
        )
    )
