"""Exact immutable mounted-root inventories and safe authenticated extraction."""

import json
import zipfile
from pathlib import Path, PurePosixPath

from mqar_final_tail_native import digest, require


def inventory(root):
    root = Path(root)
    result = {}
    require(not root.is_symlink() and not root.is_junction(), "ROOT_LINK")
    for path in root.rglob("*"):
        require(not path.is_symlink() and not path.is_junction(), "MEMBER_LINK")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
    return result


def check(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    actual = inventory(root)
    actual.pop("manifest.json")
    require(actual == manifest["files"], "EXACT_MANIFEST_INVENTORY")
    return manifest


def extract(archive, output, expected):
    require(digest(archive) == expected, "ARCHIVE_SHA")
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "FRESH_EXTRACTION")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        require(len(names) == len(set(names)), "DUPLICATE_ARCHIVE_MEMBER")
        for name in names:
            path = PurePosixPath(name)
            info = source.getinfo(name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name
                and not info.is_dir()
                and not name.endswith((".pyc", ".pyo"))
                and ((info.external_attr >> 16) & 0o170000) != 0o120000,
                "SAFE_ARCHIVE_MEMBER",
            )
        output.mkdir(parents=True, exist_ok=False)
        source.extractall(output)
    return output


def bind_wheel(wheel, output):
    import sys

    expected = "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
    require(
        not any(
            name == "compare_attention" or name.startswith("compare_attention.")
            for name in sys.modules
        ),
        "FRESH_SCIENTIFIC_IMPORT",
    )
    extract(wheel, output, expected)
    pins = inventory(output)
    sys.path.insert(0, str(output))
    return pins


def imported_sources(root, pins):
    import sys

    root = Path(root).resolve()
    require(inventory(root) == pins, "EXTRACTED_WHEEL_INVENTORY")
    observed = {}
    for name, module in list(sys.modules.items()):
        if name == "compare_attention" or name.startswith("compare_attention."):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), "FOREIGN_SCIENTIFIC_IMPORT")
            relative = path.relative_to(root).as_posix()
            require(
                relative in pins and digest(path) == pins[relative]["sha256"],
                "IMPORTED_SOURCE_BYTES",
            )
            observed[name] = pins[relative]["sha256"]
    return observed
