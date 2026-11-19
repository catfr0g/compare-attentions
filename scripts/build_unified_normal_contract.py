"""Create-only engineering predecessor/source contract packet; never launch authority."""

import argparse
import hashlib
import shutil
from pathlib import Path

from unified_normal_contract import Policy, canonical, digest, read, require
from unified_normal_source import PROFILE, RUNTIME_RECORD, SOURCE, WHEEL


def pin(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def save(path, value):
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build(inputs, output):
    """Snapshot declared source/replay bytes without duplicating scientific arrays."""
    require(
        set(inputs)
        == {
            "sorting_candidate",
            "mqar_spec",
            "mqar_science",
            "mqar_transition_review",
            "normal_package",
            "normal_case",
            "runtime_record",
        },
        "UNIFIED_BUILD_INPUTS",
    )
    output = Path(output).resolve()
    require(not output.exists(), "UNIFIED_CREATE_ONLY")
    old = Path(inputs["sorting_candidate"])
    old_descriptor = old / "payload/qualified-launch/bridge.json"
    sorting = read(old_descriptor)
    normal = Path(inputs["normal_package"])
    require(digest(normal / "normal-source-identity.json") == SOURCE, "UNIFIED_REVIEWED_NORMAL")
    runtime_path = Path(inputs["runtime_record"])
    require(digest(runtime_path) == RUNTIME_RECORD, "UNIFIED_AUTHENTIC_RUNTIME_REFERENCE")
    runtime_fields = ["completed", 0, "acceptance", "timing", "metadata", "execution_runtime"]
    runtime = read(runtime_path)
    for field in runtime_fields:
        runtime = runtime[field]
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in ("unified_normal_contract", "unified_normal_replay", "unified_normal_source"):
        shutil.copyfile(Path(__file__).with_name(name + ".py"), source / (name + ".py"))
    files = dict(sorting["files"])
    roots = set(sorting["roots"])

    def include(path):
        path = Path(path).resolve()
        require(path.is_file() and not path.is_symlink() and not path.is_junction(), "UNIFIED_FILE")
        require(path.suffix.lower() not in {".env", ".pem", ".key"}, "UNIFIED_NO_PRIVATE_BYTES")
        files[str(path)] = pin(path)

    def tree(root):
        root = Path(root).resolve()
        require(root.is_dir() and not root.is_symlink() and not root.is_junction(), "UNIFIED_ROOT")
        roots.add(str(root))
        for path in root.rglob("*"):
            require(not path.is_symlink() and not path.is_junction(), "UNIFIED_LINK")
            if path.is_file():
                include(path)

    for path in (
        old_descriptor,
        old / "payload/source/src/seed202_pilot/qualified_bridge.py",
        inputs["mqar_spec"],
        inputs["mqar_transition_review"],
        runtime_path,
    ):
        include(path)
    tree(normal)
    tree(source)
    spec = read(inputs["mqar_spec"])
    # The complete input includes genuine FULL/native proof, not restored model state.
    for name in ("input", "data", "result"):
        tree(spec[name])
    tree(Path(spec["result"]).parent / "result-services")
    tree(Path(spec["release"]).parent)
    tree(inputs["mqar_science"])
    for name in ("native", "services", "provider", "review", "plan", "output"):
        include(spec[name])
    policies = {
        "sorting-classic-seed202": Policy.SORTING_CLASSIC,
        "sorting-deltanet-seed202": Policy.SORTING_DELTA,
    }
    predecessors = []
    for case, policy in policies.items():
        accepted = sorting["cases"][case]["spec"]["output"]
        predecessors.append(
            {
                "case_id": case,
                "policy": policy.value,
                "path": accepted,
                "sha256": digest(accepted),
                "replay": {
                    "driver": str(old / "payload/source/src/seed202_pilot/qualified_bridge.py"),
                    "descriptor": str(old_descriptor),
                },
            }
        )
    predecessors.append(
        {
            "case_id": "mqar-k64-classic-seed202",
            "policy": Policy.MQAR_CONTINUATION.value,
            "path": spec["output"],
            "sha256": digest(spec["output"]),
            "replay": {
                "spec": spec,
                "science": inputs["mqar_science"],
                "transition_review": inputs["mqar_transition_review"],
            },
        }
    )
    case = read(inputs["normal_case"])["case"]
    case["runtime_sha256"] = hashlib.sha256(canonical(runtime).encode()).hexdigest()
    request = {
        "schema": "UNIFIED_NORMAL_REQUEST_V1",
        "run": case,
        "predecessors": predecessors,
        "normal": {
            "profile": PROFILE,
            "package": str(normal),
            "source_identity_sha256": SOURCE,
            "wheel_sha256": WHEEL,
            "runtime_record": str(runtime_path),
            "runtime_record_sha256": RUNTIME_RECORD,
            "runtime_path": runtime_fields,
            "runtime_sha256": case["runtime_sha256"],
        },
        "science": sorting["science"],
        "roots": sorted(roots),
        "files": files,
    }
    save(output / "payload/unified/request.json", request)
    save(
        output / "STATUS.json",
        {
            "status": "ENGINEERING_CONTRACT_PACKET_NOT_COMPLETE_OPERATOR",
            "launch_authority": False,
            "cloud_mutations": 0,
            "independent_approval": False,
            "request_sha256": digest(output / "payload/unified/request.json"),
            "remaining": [
                "public planner integration",
                "source renderer and authority binder",
                "prepare/LIVE/seal/SDK integration",
                "NORMAL task/service/next integration",
                "source-bound regression and complete review freeze",
            ],
        },
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build(read(args.inputs), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
