"""Bind existing extracted import roots to exact authenticated archives; no array copies."""

import json
import zipfile

from mqar_final_tail_host_contract import exact, pin
from mqar_final_tail_native import require
from mqar_final_tail_package import check


def paths(root, archive, expected_sha256):
    from seed202_pilot.prepared_files import safe_root

    root = safe_root(root)
    require(pin(archive)["sha256"] == expected_sha256, "TAIL_IMPORT_ARCHIVE")
    with zipfile.ZipFile(archive) as source:
        raw = source.read("manifest.json")
        manifest = json.loads(raw)
        names = source.namelist()
        require(
            len(names) == len(set(names))
            and set(names) == set(manifest["files"]) | {"manifest.json"},
            "TAIL_IMPORT_ZIP_SET",
        )
    require((root / "manifest.json").read_bytes() == raw, "TAIL_IMPORT_MANIFEST")
    require(exact(check(root), manifest), "TAIL_IMPORT_SOURCE_BYTES")
    return [str(root / n) for n in [*manifest["files"], "manifest.json"]], manifest["files"]
