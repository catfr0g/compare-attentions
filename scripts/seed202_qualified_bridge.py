"""HOST-only tagged predecessor policies for two specifically accepted Sorting tails.

This module is copied into a new authenticated V7 host candidate. It is never
included in the scientific worker. Ordinary acceptance is not modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ACCEPTED = {
    "sorting-classic-seed202": "d0376599d2934db6ea87b5e455a9ca90f524cbf10d98ec5129bf56f931c14494",
    "sorting-deltanet-seed202": "d5fbc3a20b67b3be0400f786fbd461993ad07fe640bdca6a718364eeaa160a77",
}
ARCHIVES = {
    "sorting-classic-seed202": "3a95667c7c5b733b8a8ebdbe4224d1daaef64acf38cb626a5aff3f2b5eed42ab",
    "sorting-deltanet-seed202": "e103d5b1208ef5e5153d9ffe42682df4ac03fcaa46a60ee05cee6c0c5777aaee",
}
SCOPES = {
    "sorting-classic-seed202": "QUALIFIED_SORTING_CLASSIC_TAIL_V1",
    "sorting-deltanet-seed202": "QUALIFIED_SORTING_DELTA_TAIL_V1",
}
ORDINARY = "AUTHENTIC_HOST_NATIVE_V2_FULL_REPLAY"
SCHEMA = "SEED202_TAGGED_PREDECESSOR_WITNESS_V1"
_REPLAY_CACHE = {}  # Process-local pure result cache; never persisted authority.


def require(condition, code):
    """Use finite errors; never include provider errors or credential material."""
    if not condition:
        raise ValueError(code)


def canonical(value):
    """Canonical comparison preserves complete objects and JSON scalar types."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    """Hash public evidence only, not environment files or credentials."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    """Read a complete saved object without projecting its fields."""
    return json.loads(Path(path).read_bytes())


def descriptor(candidate):
    """The enclosing candidate manifest/plan pins this descriptor and the driver."""
    path = Path(candidate) / "payload/qualified-launch/bridge.json"
    value = read(path)
    require(
        set(value)
        == {"schema", "cases", "science", "science_archive", "transition_review", "roots", "files"}
        and value["schema"] == "SEED202_QUALIFIED_LAUNCH_INPUTS_V1"
        and set(value["cases"]) == set(ACCEPTED),
        "BRIDGE_DESCRIPTOR_SCHEMA",
    )
    roots = {str(Path(value["science"]).resolve())}
    files = {str(Path(value[key]).resolve()) for key in ("science_archive", "transition_review")}
    for selected in value["cases"].values():
        roots.update(
            str(Path(selected["spec"][key]).resolve()) for key in ("input", "data", "result")
        )
        roots.add(str(Path(selected["root"]).resolve()))
        files.add(str(Path(selected["archive"]).resolve()))
        files.update(
            str(Path(selected["spec"][key]).resolve())
            for key in ("native", "services", "provider", "review", "plan", "output")
        )
        require(set(selected) == {"archive", "root", "spec"}, "QUALIFIED_CASE_SCHEMA")
        spec = selected["spec"]
        require(
            set(spec)
            == {
                "release",
                "result",
                "native",
                "services",
                "provider",
                "review",
                "plan",
                "data",
                "input",
                "output",
            },
            "QUALIFIED_REPLAY_SCHEMA",
        )
        require(
            Path(spec["release"]).resolve() == (Path(selected["root"]) / "payload").resolve()
            and Path(spec["data"]).resolve() == (Path(value["science"]) / "data/sorting").resolve(),
            "QUALIFIED_SOURCE_PATH_BINDING",
        )
    require(
        type(value["roots"]) is list
        and len(value["roots"]) == len(roots)
        and set(value["roots"]) == roots
        and files <= set(value["files"]),
        "QUALIFIED_INPUT_CLOSURE",
    )
    require(
        digest(value["transition_review"])
        == "4bc89cec199cc50cb7470e26f36e3482225575c61c7e8009bb2eb82a934fc626",
        "INDEPENDENT_TRANSITION_REVIEW_PIN",
    )
    return path, value


def evidence_paths(candidate):
    """Include every external replay/import input in existing write-denial leases."""
    _path, value = descriptor(candidate)
    return list(value["files"])


def verify_files(value):
    """No link, inventory, or raw-byte drift is admitted before replay."""
    from seed202_pilot.prepared_files import safe_path

    for root_name in value["roots"]:
        root = Path(root_name).resolve()
        members = set()
        require(not root.is_symlink() and not root.is_junction(), "QUALIFIED_ROOT_LINK")
        for member in root.rglob("*"):
            require(not member.is_symlink() and not member.is_junction(), "QUALIFIED_ROOT_LINK")
            if member.is_file():
                members.add(str(member.resolve()))
        expected = {name for name in value["files"] if Path(name).is_relative_to(root)}
        require(members == expected, "QUALIFIED_ROOT_INVENTORY")
    for name, pin in value["files"].items():
        path = safe_path(Path(name))
        require(
            set(pin) == {"bytes", "sha256"}
            and path.stat().st_size == pin["bytes"]
            and digest(path) == pin["sha256"],
            "QUALIFIED_EVIDENCE_DRIFT",
        )


def _qualified(candidate, case_id, path):
    descriptor_path, inputs = descriptor(candidate)
    require(case_id in ACCEPTED, "QUALIFIED_SORTING_ONLY")
    selected = inputs["cases"][case_id]
    require(
        Path(path).resolve() == Path(selected["spec"]["output"]).resolve()
        and digest(path) == ACCEPTED[case_id],
        "EXACT_ACCEPTED_QUALIFIED_OBJECT",
    )
    verify_files(inputs)
    cache_key = (case_id, digest(descriptor_path), digest(Path(__file__)))
    if cache_key in _REPLAY_CACHE:
        result = _REPLAY_CACHE[cache_key]
        require(result == _witness(case_id, read(path), descriptor_path), "EXACT_CACHED_WITNESS")
        return json.loads(canonical(result))
    environment = {
        key: val
        for key, val in os.environ.items()
        if key.upper()
        in {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "HOME",
            "USERPROFILE",
            "USERNAME",
            "USER",
            "LOGNAME",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LOCALAPPDATA",
            "APPDATA",
            "UV_CACHE_DIR",
        }
    }
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_OFFLINE="1",
        CLEARML_OFFLINE_MODE="1",
        UV_CACHE_DIR=str(Path(os.environ["TMPDIR"]) / "qualified-bridge-uv"),
        TORCHINDUCTOR_CACHE_DIR=str(Path(os.environ["TMPDIR"]) / "qualified-bridge-inductor"),
    )
    command = [
        "uv",
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        sys.executable,
        "python",
        "-B",
        str(Path(__file__).resolve()),
        "--replay",
        str(descriptor_path),
        "--case",
        case_id,
    ]
    child = subprocess.run(command, env=environment, capture_output=True, timeout=300, check=False)
    require(child.returncode == 0, "QUALIFIED_NATIVE_COMPOSITE_REPLAY_FAILED")
    result = json.loads(child.stdout)
    require(result == _witness(case_id, read(path), descriptor_path), "EXACT_QUALIFIED_WITNESS")
    verify_files(inputs)
    _REPLAY_CACHE[cache_key] = json.loads(canonical(result))
    return result


def _witness(case_id, value, descriptor_path):
    require(
        digest(Path(read(descriptor_path)["cases"][case_id]["spec"]["output"]))
        == ACCEPTED[case_id],
        "QUALIFIED_ACCEPTANCE_PIN",
    )
    require(value["later_pilots_authorized"] is False, "PRESERVE_QUALIFIED_AUTHORITY")
    return {
        "schema": SCHEMA,
        "kind": SCOPES[case_id],
        "case_id": case_id,
        "acceptance_sha256": ACCEPTED[case_id],
        "bridge_inputs_sha256": digest(descriptor_path),
        "whole_object_sha256": hashlib.sha256(canonical(value).encode()).hexdigest(),
        "native_sha256": hashlib.sha256(canonical(value["native"]).encode()).hexdigest(),
        "qualification": value["qualification"],
        "original_provider_errors": value["original_provider_errors"],
        "later_pilots_authorized": False,
    }


def binding(candidate):
    """Replay both accepted tails; this is not a new launch or review permission."""
    path, value = descriptor(candidate)
    return {
        "schema": "SEED202_QUALIFIED_LAUNCH_BINDING_V1",
        "inputs_sha256": digest(path),
        "witnesses": {
            case_id: _qualified(candidate, case_id, selected["spec"]["output"])
            for case_id, selected in value["cases"].items()
        },
    }


def replay_predecessor(plan, path, case_id):
    """Explicit dispatcher: never reinterpret a failed/new MQAR as qualified."""
    if case_id in ACCEPTED:
        require("qualified_launch_bridge" in plan, "QUALIFIED_BRIDGE_REQUIRED")
        witness = _qualified(plan["candidate_root"], case_id, path)
        require(
            witness == plan["qualified_launch_bridge"]["witnesses"][case_id],
            "QUALIFIED_PLAN_BINDING",
        )
        return witness
    from seed202_pilot.state import replay_acceptance

    value = replay_acceptance(Path(path), case_id)
    require(
        value["status"] == "ACCEPTED_SEED202_FULL" and value["case_id"] == case_id,
        "ORDINARY_MQAR_REQUIRED",
    )
    return {
        "schema": SCHEMA,
        "kind": ORDINARY,
        "case_id": case_id,
        "acceptance_sha256": digest(path),
    }


def expected_evidence(plan):
    """Sealer consumes exact tagged case/scope evidence, not synthetic science dicts."""
    if "qualified_launch_bridge" not in plan:
        return [
            {
                "case_id": row["case_id"],
                "acceptance_sha256": row["sha256"],
                "replay_scope": ORDINARY,
            }
            for row in plan["predecessors"]
        ]
    classic = plan["qualified_launch_bridge"]["witnesses"]["sorting-classic-seed202"]
    return [
        {
            "case_id": classic["case_id"],
            "acceptance_sha256": classic["acceptance_sha256"],
            "replay_scope": classic["kind"],
        },
        *[
            {
                "case_id": row["case_id"],
                "acceptance_sha256": row["sha256"],
                "replay_scope": SCOPES.get(row["case_id"], ORDINARY),
            }
            for row in plan["predecessors"]
        ],
    ]


def proof_rows(plan):
    """Fresh mandatory replay for every qualified and ordinary predecessor."""
    if "qualified_launch_bridge" not in plan:
        from seed202_pilot.state import replay_acceptance

        for row in plan["predecessors"]:
            replay_acceptance(Path(row["path"]), row["case_id"])
        return expected_evidence(plan)
    _p, inputs = descriptor(plan["candidate_root"])
    replay_predecessor(
        plan,
        inputs["cases"]["sorting-classic-seed202"]["spec"]["output"],
        "sorting-classic-seed202",
    )
    for row in plan["predecessors"]:
        require(digest(row["path"]) == row["sha256"], "PREDECESSOR_ACCEPTANCE_PIN")
        replay_predecessor(plan, row["path"], row["case_id"])
    return expected_evidence(plan)


def next_qualified_case(candidate, classic, acceptances):
    """Extend only the host transition; each newly accepted MQAR stays ordinary."""
    from seed202_pilot.cli import _candidate
    from seed202_pilot.contract import cases
    from seed202_pilot.transition import verify_qualified_predecessor

    _candidate(Path(candidate))
    verify_qualified_predecessor(Path(classic))
    order = cases()
    require(type(acceptances) is list and 0 < len(acceptances) < len(order), "QUALIFIED_PREFIX")
    plan = {"candidate_root": str(candidate), "qualified_launch_bridge": binding(candidate)}
    for path, case in zip(acceptances, order, strict=False):
        replay_predecessor(plan, path, case.case_id)
    return order[len(acceptances)]


def _archive_root(archive, root, expected):
    import zipfile

    require(digest(archive) == expected, "QUALIFIED_ARCHIVE_PIN")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        require(len(names) == len(set(names)), "QUALIFIED_UNIQUE_MEMBERS")
        require(
            set(names)
            == {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()},
            "QUALIFIED_IMPORTROOT_INVENTORY",
        )
        for name in names:
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and chr(92) not in name,
                "QUALIFIED_SAFE_MEMBER",
            )
            require(
                source.read(name) == (Path(root) / name).read_bytes(), "QUALIFIED_IMPORTROOT_BYTES"
            )


def _run_replay(descriptor_path, case_id):
    """Fresh offline child: genuine native/FULL/metrics and whole-composite replay."""
    import contextlib
    import logging
    import socket
    import tempfile

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU_HOST_ONLY")
    logging.disable(logging.CRITICAL)

    def prohibited(*_args, **_kwargs):
        raise PermissionError("STRUCTURAL_REPLAY_ONLY")

    socket.socket.connect = prohibited
    socket.create_connection = prohibited
    _descriptor_path, value = descriptor(Path(descriptor_path).parents[2])
    selected = value["cases"][case_id]
    require(
        digest(selected["spec"]["output"]) == ACCEPTED[case_id], "EXACT_ACCEPTED_QUALIFIED_OBJECT"
    )
    _archive_root(selected["archive"], selected["root"], ARCHIVES[case_id])
    science = Path(value["science"])
    _archive_root(
        value["science_archive"],
        science,
        "71872615b0cfbe2406cd08762cc02c390c88a70b6bb5f4bb38d6571830c8a3bb",
    )
    release = Path(selected["spec"]["release"])
    sys.path[:0] = [
        str(release / "src"),
        str(Path(selected["root"]) / "inherited-two-stage-host"),
        str(science / "operator"),
        str(science / "src"),
        str(science / "tools"),
        str(science / "operator/inherited"),
    ]
    from tail_package import bind_wheel, imported_sources

    with tempfile.TemporaryDirectory(
        prefix="qualified-native-", dir=os.environ["TMPDIR"]
    ) as scratch:
        wheel = Path(scratch) / "wheel"
        pins = bind_wheel(science / "compare_attention-0.1.0-py3-none-any.whl", wheel)
        import torch

        torch.nn.Module.__call__ = prohibited
        torch.Tensor.backward = prohibited
        torch.autograd.backward = prohibited
        torch.optim.Optimizer.__init__ = prohibited
        torch.set_rng_state = prohibited
        torch.cuda.set_rng_state = prohibited
        torch.cuda.set_rng_state_all = prohibited
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            if case_id == "sorting-deltanet-seed202":
                from tail_transition import replay

                result = replay(selected["spec"])
            else:
                from tail_composite import composite
                from tail_host_science import paths

                spec = selected["spec"]
                paths(
                    Path(spec["input"]),
                    Path(selected["root"]) / "dataset/recovery-input.zip.bin",
                    read(spec["plan"])["input_sha256"],
                )
                # The accepted Classic validator constructs a model under fork_rng.
                # Reuse its independently accepted immutable native proof instead of
                # restoring any RNG in a HOST launch. Hash every original dependency
                # and replay the entire genuine composite with that exact proof.
                native = read(spec["native"])
                require(native == read(spec["output"])["native"], "CLASSIC_ACCEPTED_NATIVE_PROOF")
                for name, sha256 in native["raw_bindings"].items():
                    require(
                        digest(Path(spec["input"]) / "original" / name) == sha256,
                        "CLASSIC_NATIVE_RAW_BINDING",
                    )
                provider = {
                    "ref" if key == "target" else key: val
                    for key, val in read(spec["provider"]).items()
                }
                result = composite(
                    release,
                    spec["result"],
                    native,
                    read(spec["services"]),
                    provider,
                    read(spec["review"]),
                    read(spec["plan"]),
                    spec["data"],
                    spec["output"],
                )
            require(
                canonical(result) == canonical(read(selected["spec"]["output"])),
                "WHOLE_QUALIFIED_OBJECT_REQUIRED",
            )
            imported_sources(wheel, pins)
    return _witness(case_id, result, descriptor_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--case", choices=ACCEPTED, required=True)
    args = parser.parse_args()
    try:
        print(canonical(_run_replay(args.replay, args.case)))
    except Exception as error:  # Do not expose raw provider or authentication errors.
        import traceback

        frames = [
            {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
            for frame in traceback.extract_tb(error.__traceback__)
        ]
        print(
            canonical(
                {
                    "status": "REPLAY_REJECTED",
                    "exception_class": type(error).__name__,
                    "frames": frames,
                }
            )
        )
        raise SystemExit(1) from None
