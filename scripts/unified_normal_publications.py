"""Immutable private-HF byte observations shared by NORMAL HOST and replay."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def read(path):
    return json.loads(Path(path).read_bytes())


def publication_readback(root, case, values, snapshot):
    """Read every declared object; persist a source-closed immutable observation."""
    from normal_training_plan import digest, require
    from quality_services import HFTransport
    from seed202_pilot.storage import create_json, inventory

    snapshot = Path(snapshot)
    existing = snapshot.exists()
    if not existing:
        snapshot.mkdir(parents=True, exist_ok=False)
    root = Path(root)
    envelope = read(root.parent / (root.name + "-publication.json"))
    require(
        envelope["schema"] == "normal-training-publication-v1" and envelope["case"] == case,
        "UNIFIED_HOST_PUBLICATION_CASE",
    )
    periodic = read(root / "periodic-readback.json")["publications"]
    require(
        [row["step"] for row in periodic] == case["checkpoint_steps"]
        and len({row["revision"] for row in periodic}) == len(periodic),
        "UNIFIED_HOST_PERIODIC_GRID",
    )
    final = envelope["final"]
    transport = HFTransport(values)
    receipts = []
    for row in [*periodic, final]:
        require(row["state"] == "verified", "UNIFIED_HOST_PUBLICATION_STATUS")
        observed = transport.read(row, row["revision"])
        require(
            observed["private"] is True
            and observed["revision"] == row["revision"]
            and set(observed["files"]) == set(row["files"]),
            "UNIFIED_HOST_HF_LISTING",
        )
        for name, raw in observed["files"].items():
            require(
                type(raw) is bytes
                and len(raw) == row["files"][name]["bytes"]
                and hashlib.sha256(raw).hexdigest() == row["files"][name]["sha256"],
                "UNIFIED_HOST_HF_BYTES",
            )
            member = PurePosixPath(str(row["revision"])) / name
            require(
                not member.is_absolute()
                and ".." not in member.parts
                and "\\" not in str(member)
                and ":" not in str(member),
                "UNIFIED_HOST_SNAPSHOT_PATH",
            )
            target = snapshot / str(member)
            if not existing:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            require(target.read_bytes() == raw, "UNIFIED_HOST_SNAPSHOT_BYTES")
            if row is final:
                require(
                    digest(root / name) == row["files"][name]["sha256"],
                    "UNIFIED_HOST_HF_FINAL_LOCAL",
                )
        receipts.append({"revision": row["revision"], "files": row["files"]})
    require(
        set(final["files"])
        == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()},
        "UNIFIED_HOST_FULL_FINAL_INVENTORY",
    )
    files = {
        str(PurePosixPath(row["revision"]) / name): pin
        for row in [*periodic, final]
        for name, pin in row["files"].items()
    }
    require(
        {k: v for k, v in inventory(snapshot).items() if k != "snapshot.json"} == files,
        "UNIFIED_HOST_SNAPSHOT_INVENTORY",
    )
    manifest = {"case": case, "publications": receipts, "files": files}
    if existing:
        require(read(snapshot / "snapshot.json") == manifest, "UNIFIED_HOST_SNAPSHOT_MANIFEST")
    else:
        create_json(snapshot / "snapshot.json", manifest)
    return {
        "publications": receipts,
        "fresh_sdk_readback": True,
        "remote_mutations": 0,
        "snapshot": str(snapshot.resolve()),
        "files": files,
    }
