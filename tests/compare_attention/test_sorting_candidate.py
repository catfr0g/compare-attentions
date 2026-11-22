"""Read-only acceptance of the actual Sorting frozen archive; no extraction/model execution."""

import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "local/experiments/sorting-kaggle-v1/candidate-v1.zip"


def test_full_archive_inventory_hashes_safe_names_and_live_source_closure():
    assert CANDIDATE.is_file(), "the real frozen candidate must exist"
    with zipfile.ZipFile(CANDIDATE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["task"] == "sorting" and not manifest["cloud_launch_approved"]
        assert set(names) == set(manifest["files"]) | {"manifest.json"}
        for name, spec in manifest["files"].items():
            assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            assert "\\" not in name and ".env" not in PurePosixPath(name).parts
            assert "__pycache__" not in name and not name.endswith((".pt", ".pyc"))
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]
            assert archive.getinfo(name).file_size == spec["bytes"]
            if name.startswith("source/"):
                local = ROOT / name.removeprefix("source/")
                assert local.read_bytes() == archive.read(name)
        for path in (ROOT / "src").rglob("*.py"):
            assert "source/" + path.relative_to(ROOT).as_posix() in manifest["files"]
        for path in (ROOT / "tests").rglob("*.py"):
            assert "source/" + path.relative_to(ROOT).as_posix() in manifest["files"]


def test_actual_wheel_record_every_source_and_unchanged_generic_dependencies():
    with zipfile.ZipFile(CANDIDATE) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        data = archive.read("compare_attention-0.1.0-py3-none-any.whl")
        assert hashlib.sha256(data).hexdigest() == manifest["source_wheel_sha256"]
        with zipfile.ZipFile(io.BytesIO(data)) as wheel:
            record = next(n for n in wheel.namelist() if n.endswith(".dist-info/RECORD"))
            entries = list(csv.reader(io.StringIO(wheel.read(record).decode())))
            assert {r[0] for r in entries} == set(wheel.namelist())
            for name, digest, size in entries:
                if name == record:
                    assert digest == size == ""
                    continue
                raw = wheel.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
                assert digest == "sha256=" + expected.decode() and int(size) == len(raw)
                if name.endswith(".py"):
                    assert raw == archive.read("source/src/" + name)
        original = (
            ROOT / "local/releases/variable-tracking-operator-v1/variable-tracking-operator-v1.zip"
        )
        with original.open("rb") as stream:
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest()
                == manifest["original_operator_sha256"]
            )
        with zipfile.ZipFile(original) as old:
            assert len(manifest["inherited_generic_helpers"]) == 11
            for name in manifest["inherited_generic_helpers"]:
                assert archive.read(name) == old.read(name)
