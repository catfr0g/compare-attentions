"""Full native checkpoint publication without changing the frozen trainer."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from colab_package import checksum


def receipt_path(result):
    result = Path(result)
    return result.parent / (result.name + ".hf-checkpoint-receipt.json")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def identity(package, result):
    package, result = Path(package), Path(result)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", result.name):
        raise ValueError("Unsafe recovery run name")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "run": result.name,
        "package_manifest_sha256": checksum(package / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "data_fingerprint": manifest["data_fingerprint"],
        "data_path": str((package / "data").resolve()),
    }


def private_head(api, repo, token):
    info = api.model_info(repo, revision="main", token=token)
    if info.private is not True or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Private recovery repository HEAD required")
    return info.sha


def remote_bytes(download, repo, token, revision, name):
    return Path(
        download(
            repo_id=repo,
            filename=name,
            repo_type="model",
            revision=revision,
            token=token,
            force_download=True,
        )
    ).read_bytes()


def commit(api, repo, token, head, files):
    from huggingface_hub import CommitOperationAdd  # pylint: disable=import-outside-toplevel

    info = api.create_commit(
        repo_id=repo,
        repo_type="model",
        token=token,
        parent_commit=head,
        commit_message="Full recovery publication",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=value)
            for name, value in files.items()
        ],
    )
    if not re.fullmatch(r"[0-9a-f]{40}", info.oid):
        raise ValueError("Recovery commit revision missing")
    return info.oid


def write_receipt(result, value, *, fresh=False):
    path = receipt_path(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
    else:
        pending = path.with_suffix(".json.tmp")
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)


def recovery_check(package, result, *, api, download, repo, token):
    """Reserve a new immutable run identity and prove pinned read/write before training."""
    value = identity(package, result)
    head = private_head(api, repo, token)
    prefix = "recoveries/" + value["run"] + "/"
    if receipt_path(result).exists() or any(
        name.startswith(prefix)
        for name in api.list_repo_files(repo, revision=head, repo_type="model", token=token)
    ):
        raise ValueError("Recovery prefix collision; use a fresh result identity")
    name, data = prefix + "run.json", encoded(value)
    revision = commit(api, repo, token, head, {name: data})
    if remote_bytes(download, repo, token, revision, name) != data:
        raise ValueError("Recovery manifest readback mismatch")
    write_receipt(result, {"verified": True, "revision": revision, "identity": value}, fresh=True)


def validate_pointer(row):
    if (
        not isinstance(row, dict)
        or set(row) != {"step", "path", "sha256"}
        or type(row["step"]) is not int
        or row["step"] < 0
        or row["path"] != f"step_{row['step']:08d}.pt"
        or not isinstance(row["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    ):
        raise ValueError("Invalid native checkpoint pointer")
    return row


def local_snapshot(result):
    folder = Path(result) / "checkpoints"
    if folder.is_symlink() or (folder / "latest.json").is_symlink():
        raise ValueError("Symlink checkpoint")
    row = validate_pointer(json.loads((folder / "latest.json").read_text(encoding="utf-8")))
    path = folder / row["path"]
    if path.is_symlink() or checksum(path) != row["sha256"]:
        raise ValueError("Native checkpoint checksum mismatch")
    return path, row


def publish_checkpoint(package, result, *, api, download, repo, token):
    """Commit full opaque native bytes and pointer together; no retry after uncertainty."""
    value = identity(package, result)
    receipt = json.loads(receipt_path(result).read_text(encoding="utf-8"))
    if receipt.get("identity") != value or receipt.get("verified") is not True:
        raise ValueError("Recovery startup proof missing or changed")
    path, row = local_snapshot(result)
    prefix = "recoveries/" + value["run"] + "/"
    head = private_head(api, repo, token)
    if remote_bytes(download, repo, token, head, prefix + "run.json") != encoded(value):
        raise ValueError("Remote run identity mismatch")
    if prefix + row["path"] in api.list_repo_files(
        repo, revision=head, repo_type="model", token=token
    ):
        raise ValueError("Recovery snapshot is immutable; reconcile before retry")
    revision = commit(
        api,
        repo,
        token,
        head,
        {prefix + row["path"]: str(path), prefix + "latest.json": encoded(row)},
    )
    if remote_bytes(download, repo, token, revision, prefix + "latest.json") != encoded(row):
        raise ValueError("Recovery pointer readback mismatch")
    records = api.get_paths_info(
        repo_id=repo,
        paths=[prefix + row["path"]],
        revision=revision,
        repo_type="model",
        token=token,
    )
    if len(records) != 1 or records[0].path != prefix + row["path"]:
        raise ValueError("Remote checkpoint metadata missing")
    lfs = records[0].lfs
    digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if digest is None:
        digest = hashlib.sha256(
            remote_bytes(download, repo, token, revision, prefix + row["path"])
        ).hexdigest()
    if digest != row["sha256"] or local_snapshot(result)[1] != row:
        raise ValueError("Remote/local checkpoint hash mismatch")
    write_receipt(
        result,
        {
            "verified": True,
            "revision": revision,
            "identity": value,
            "step": row["step"],
            "sha256": row["sha256"],
        },
    )


def fetch_checkpoint(package, run, revision, target, fresh_result, *, api, download, repo, token):
    """Fetch explicit pinned manifests and full bytes; trainer retains strict resume authority."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Explicit pinned recovery revision required")
    target, fresh_result = Path(target), Path(fresh_result)
    if fresh_result.exists() and (not fresh_result.is_dir() or any(fresh_result.iterdir())):
        raise ValueError("Resume requires a fresh result directory")
    if target.exists() or target.is_symlink():
        raise ValueError("Recovery download target must be new")
    expected = identity(package, Path(run))
    prefix = "recoveries/" + expected["run"] + "/"
    private_head(api, repo, token)
    value = json.loads(remote_bytes(download, repo, token, revision, prefix + "run.json"))
    if value != expected:
        raise ValueError("Recovery package/source/data path mismatch")
    row = validate_pointer(
        json.loads(remote_bytes(download, repo, token, revision, prefix + "latest.json"))
    )
    data = remote_bytes(download, repo, token, revision, prefix + row["path"])
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Recovery download checksum mismatch")
    target.mkdir(parents=True)
    (target / row["path"]).write_bytes(data)
    (target / "latest.json").write_bytes(encoded(row))
    (target / "run.json").write_bytes(encoded(value))
    return target / row["path"]


@contextlib.contextmanager
def checkpoint_backups(env_file, package_dir, result_dir, *, tracker_class=None, publish=None):
    """Synchronous callback runs inside the native excluded checkpoint boundary."""
    if tracker_class is None:
        from compare_attention.train.backup import BackupTracker  # pylint: disable=import-outside-toplevel

        tracker_class = BackupTracker
    original = tracker_class.save_checkpoint
    if publish is None:
        from colab_runner import clean_environment  # pylint: disable=import-outside-toplevel

        def publish():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("colab_services.py")),
                    "checkpoint",
                    "--env-file",
                    str(env_file),
                    "--package-dir",
                    str(package_dir),
                    "--result-dir",
                    str(result_dir),
                ],
                env=clean_environment(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def save(tracker, payload, step, keep_last):
        original(tracker, payload, step, keep_last)
        publish()

    tracker_class.save_checkpoint = save
    try:
        yield
    finally:
        tracker_class.save_checkpoint = original
