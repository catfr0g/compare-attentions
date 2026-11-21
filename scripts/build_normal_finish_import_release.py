"""Derive a future-only import-root correction from the authenticated trained c7 release."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from completed_normal_publication import pin, require
from unified_normal_profiles import REMAINING_FOUR, authenticate_package


def build_release(parent, repository, output):
    """Only selected-child setup changes; every scientific/helper byte is inherited."""
    parent, repository, output = map(Path, (parent, repository, output))
    identity = authenticate_package(parent, REMAINING_FOUR)
    require(not output.exists(), "NORMAL_IMPORT_RELEASE_CREATE_ONLY")
    shutil.copytree(parent, output)
    shutil.copyfile(repository / "scripts/normal_training_launch.py", output / "setup.py")
    updated = {
        **identity,
        "finish_import_parent_identity_sha256": pin(parent / "normal-source-identity.json")[
            "sha256"
        ],
        "finish_import_policy": "AUTHENTICATED_INHERITED_SOURCE_ROOT_V1",
        "files": {
            p.relative_to(output).as_posix(): pin(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "normal-source-identity.json"
        },
    }
    (output / "normal-source-identity.json").write_bytes(
        json.dumps(updated, sort_keys=True, indent=2).encode() + b"\n"
    )
    require(
        {name for name, value in updated["files"].items() if value != identity["files"][name]}
        == {"setup.py"},
        "NORMAL_IMPORT_RELEASE_ONLY_SETUP",
    )
    return output


def archive_release(package, archive):
    """Deterministic create-only member archive, independent of filesystem clocks."""
    package, archive = map(Path, (package, archive))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as saved:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                member = "normal-package/" + path.relative_to(package).as_posix()
                info = zipfile.ZipInfo(member, (2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                saved.writestr(info, path.read_bytes())
    return pin(archive)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("parent", "repository", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    package = build_release(args.parent, args.repository, args.output / "normal-package")
    archive_release(package, args.output / "normal-package.zip.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
