"""Stdlib-only package integrity and extraction. Never imports training code."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

TOOLS = (
    "colab_package.py",
    "colab_runner.py",
    "colab_verify.py",
    "colab_services.py",
    "colab_checkpoint_backup.py",
    "colab_checkpoint_protocol.py",
    "colab_checkpoint_store.py",
)
SOURCE_SHA256 = "1beba22ca8f9897e0b7367e07f4a016559946f1183fc8413deff2490c6e85ef1"
RELEASE_SHA256 = "7feabbfc925971c7c537a506d0caba8c1a3ce903506e5ca37a367921402747a9"
RECEIPT_SHA256 = "e0be512aceb80fac19812e663cbb5e89d45dc1d6d97de138f2ace25b95bc1a16"
HELPER_SHA256 = "fa184277a3d0f82ec5b75d8ab0d965f04a0aa42de74f52ada7d3eed262dbd657"
FAMILIES = ("classic", "deltanet")
CONFIGS = {name: f"experiments/fp16_k64_small/full/{name}/seed_42.yaml" for name in FAMILIES}
WHEEL = "compare_attention-0.1.0-py3-none-any.whl"
RELEASE_FILES = {WHEEL, "requirements.txt", "release-manifest.json", *CONFIGS.values()}
DATA_FILES = {
    "meta.json",
    *(f"{split}_{kind}.npy" for split in ("train", "val", "test") for kind in ("inputs", "labels")),
}


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def blob_record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def record(path):
    return {"bytes": Path(path).stat().st_size, "sha256": checksum(path)}


def safe_name(name):
    parts = PurePosixPath(name).parts
    if (
        not parts
        or name != PurePosixPath(name).as_posix()
        or "\\" in name
        or any(p in (".", "..") or ":" in p for p in parts)
        or PurePosixPath(name).is_absolute()
    ):
        raise ValueError("unsafe archive path: " + name)
    return name


def zip_members(archive):
    names = set()
    for item in archive.infolist():
        safe_name(item.orig_filename)
        safe_name(item.filename)
        mode = item.external_attr >> 16
        if item.filename in names or item.is_dir() or stat.S_ISLNK(mode):
            raise ValueError("duplicate/directory/symlink archive member")
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise ValueError("non-regular archive member")
        names.add(item.filename)
    return names


def verify_directory(root):  # pylint: disable=too-many-branches
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported package manifest")
    if (
        manifest.get("storage") != "hf"
        or manifest.get("periodic_full_checkpoints") is not True
        or manifest.get("tracking") != "clearml"
        or manifest.get("hf_upload") is not True
    ):
        raise ValueError("required HF periodic recovery storage policy missing")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink package input")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("unexpected or missing package files")
    for name, details in manifest["files"].items():
        safe_name(name)
        if record(root / name) != details:
            raise ValueError("package integrity mismatch: " + name)
    payload = (
        {"release/" + name for name in RELEASE_FILES}
        | {"data/" + name for name in DATA_FILES}
        | {"tools/" + name for name in TOOLS}
        | {"support/controls_verifier.py"}
    )
    if set(manifest["files"]) != payload:
        raise ValueError("unexpected or missing frozen payload/tools")
    if (
        manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("original_release_sha256") != RELEASE_SHA256
        or checksum(root / "release/release-manifest.json") != RECEIPT_SHA256
        or checksum(root / "support/controls_verifier.py") != HELPER_SHA256
    ):
        raise ValueError("frozen source/release/helper binding mismatch")
    configs_sha = hashlib.sha256(
        json.dumps(manifest.get("configs"), sort_keys=True).encode()
    ).hexdigest()
    data = {
        "spec": manifest.get("dataset_spec"),
        "files": {name: manifest["files"]["data/" + name] for name in DATA_FILES},
    }
    data_sha = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if (
        configs_sha != "eb444ef05a46966073bb855b3d155338738592e58400b6c7fa827197037fe947"
        or data_sha != manifest.get("data_fingerprint")
        or manifest.get("default_family") != "deltanet"
        or manifest.get("expected_updates") != {"classic": 43117, "deltanet": 47265}
    ):
        raise ValueError("frozen scientific/data manifest binding mismatch")
    receipt = json.loads((root / "release/release-manifest.json").read_text(encoding="utf-8"))
    if set(receipt) != RELEASE_FILES - {"release-manifest.json"}:
        raise ValueError("frozen release receipt differs")
    for name, details in receipt.items():
        if manifest["files"]["release/" + name] != details:
            raise ValueError("frozen release integrity mismatch: " + name)
    return manifest


def extract_verified(archive, expected_sha256, target):
    archive, target = Path(archive), Path(target)
    if checksum(archive) != expected_sha256:
        raise ValueError("archive SHA256 mismatch")
    if target.exists() or target.is_symlink():
        raise FileExistsError("extraction target must be new")
    with zipfile.ZipFile(archive) as source:
        zip_members(source)
        target.mkdir(parents=True)
        try:
            for item in source.infolist():
                path = target / item.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as incoming, path.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            manifest = verify_directory(target)
            for path in target.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)
            return manifest
        except Exception:
            shutil.rmtree(target)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = extract_verified(args.archive, args.sha256, args.output_dir)
    print(json.dumps({"verified_files": len(manifest["files"]), "training_started": False}))


if __name__ == "__main__":
    main()
