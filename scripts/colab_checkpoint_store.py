"""HF commit/readback protocol; lazy SDK imports and bounded temporary cache."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
from pathlib import Path

from colab_checkpoint_protocol import atomic_json, checksum, encoded, identity, receipt_path


def private_head(api, repo, token, *, create=False):
    """Reject public targets; create missing targets only as private model repos."""
    from huggingface_hub.errors import RepositoryNotFoundError  # pylint: disable=import-outside-toplevel

    try:
        info = api.model_info(repo, revision="main", token=token)
    except RepositoryNotFoundError as error:
        if not create or error.response.status_code != 404:
            raise
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=False, token=token)
        info = api.model_info(repo, revision="main", token=token)
    if info.private is not True or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Private checkpoint repository HEAD required")
    return info.sha


def remote_file(download, repo, token, revision, name, directory):
    """Use an ephemeral local directory, never an unbounded global Hub cache."""
    path = download(
        repo_id=repo,
        filename=name,
        repo_type="model",
        revision=revision,
        token=token,
        force_download=True,
        local_dir=directory,
    )
    return Path(path)


def commit(api, repo, token, head, files, deletes=()):
    """One optimistic commit atomically updates native bytes, pointer and ownership."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete  # pylint: disable=import-outside-toplevel

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=value)
        for name, value in files.items()
    ]
    operations.extend(CommitOperationDelete(path_in_repo=name, is_folder=False) for name in deletes)
    result = api.create_commit(
        repo_id=repo,
        repo_type="model",
        token=token,
        parent_commit=head,
        revision="main",
        operations=operations,
        commit_message="Verified full native checkpoint",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", result.oid):
        raise ValueError("Pinned checkpoint commit missing")
    return result.oid


def probe_store(package, result, *, api, download, repo, token):
    """Reserve a unique run prefix with a nonce; no output-directory pollution."""
    value = identity(package, result)
    prefix = f"runs/{value['run']}/checkpoints/"
    if Path(result).exists() or Path(result).is_symlink() or receipt_path(result).exists():
        raise ValueError("Checkpoint probe requires a new result sink and receipt")
    head = private_head(api, repo, token, create=True)
    if any(
        name.startswith(f"runs/{value['run']}/")
        for name in api.list_repo_files(repo, revision=head, repo_type="model", token=token)
    ):
        raise ValueError("Checkpoint run collision; select a fresh result identity")
    metadata = {"identity": value, "nonce": secrets.token_hex(16), "owned": []}
    uncertain = {
        "verified": False,
        "state": "probe_unknown",
        "identity": value,
        "metadata": metadata,
    }
    atomic_json(receipt_path(result), uncertain)
    with tempfile.TemporaryDirectory(prefix="hf-checkpoint-readback-") as directory:
        try:
            revision = commit(api, repo, token, head, {prefix + "backup.json": encoded(metadata)})
        except Exception:  # pylint: disable=broad-exception-caught
            try:
                reconciled = private_head(api, repo, token)
                if remote_file(
                    download, repo, token, reconciled, prefix + "backup.json", directory
                ).read_bytes() == encoded(metadata):
                    atomic_json(
                        receipt_path(result), {**uncertain, "reconciled_revision": reconciled}
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise RuntimeError("Ambiguous checkpoint probe; reconciliation required") from None
        if (
            remote_file(
                download, repo, token, revision, prefix + "backup.json", directory
            ).read_bytes()
            != encoded(metadata)
            or private_head(api, repo, token) != revision
        ):
            raise ValueError("Checkpoint probe readback or HEAD mismatch")
    atomic_json(
        receipt_path(result),
        {"verified": True, "revision": revision, "identity": value, "metadata": metadata},
    )


# Exact integer semantics reject bool, matching the frozen native pointer.
def pointer(row):
    """Accept only canonical native checkpoint names, hashes and integer steps."""
    if (  # pylint: disable=too-many-boolean-expressions
        not isinstance(row, dict)
        or set(row) != {"step", "path", "sha256"}
        or type(row["step"]) is not int  # pylint: disable=unidiomatic-typecheck
        or row["step"] < 0
        or row["path"] != f"step_{row['step']:08d}.pt"
        or not isinstance(row["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    ):
        raise ValueError("Invalid native checkpoint pointer")
    return row


def local_snapshot(result):
    """Reject symlinks/traversal and corrupted bytes before invoking the network."""
    result = Path(result)
    folder = result / "checkpoints"
    if result.is_symlink() or folder.is_symlink() or (folder / "latest.json").is_symlink():
        raise ValueError("Symlink checkpoint input")
    row = pointer(json.loads((folder / "latest.json").read_text(encoding="utf-8")))
    path = folder / row["path"]
    if path.is_symlink() or not path.is_file() or checksum(path) != row["sha256"]:
        raise ValueError("Native checkpoint checksum mismatch")
    return path, row


# Preserve the ordered transaction and unknown-publication barrier in one scope.
# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def publish_store(package, result, *, api, download, repo, token):
    """Synchronous upload and full pinned-byte hash proof before training resumes."""
    value = identity(package, result)
    path, row = local_snapshot(result)
    receipt = json.loads(receipt_path(result).read_text(encoding="utf-8"))
    if receipt.get("verified") is not True or receipt.get("identity") != value:
        raise ValueError("Verified checkpoint startup proof required")
    previous = receipt["metadata"]
    prefix = f"runs/{value['run']}/checkpoints/"
    head = private_head(api, repo, token)
    with tempfile.TemporaryDirectory(prefix="hf-checkpoint-transfer-") as directory:
        if remote_file(
            download, repo, token, head, prefix + "backup.json", directory
        ).read_bytes() != encoded(previous):
            raise ValueError("Remote checkpoint ownership changed")
        owned = [pointer(item) for item in previous["owned"]]
        if (
            previous.get("identity") != value
            or len(owned) > 2
            or sorted({item["step"] for item in owned}) != [item["step"] for item in owned]
            or not re.fullmatch(r"[0-9a-f]{32}", previous["nonce"])
        ):
            raise ValueError("Invalid checkpoint ownership proof")
        if owned and remote_file(
            download, repo, token, head, prefix + "latest.json", directory
        ).read_bytes() != encoded(owned[-1]):
            raise ValueError("Remote checkpoint pointer changed")
        if owned and row["step"] <= max(item["step"] for item in owned):
            if row != owned[-1] or receipt.get("step") != row["step"]:
                raise ValueError("Checkpoint step is not increasing")
            for name, expected in (("latest.json", encoded(row)), (row["path"], None)):
                remote = remote_file(download, repo, token, head, prefix + name, directory)
                if (
                    checksum(remote) != row["sha256"]
                    if expected is None
                    else remote.read_bytes() != expected
                ):
                    raise ValueError("Idempotent checkpoint readback mismatch")
            return
        listing = set(api.list_repo_files(repo, revision=head, repo_type="model", token=token))
        if prefix + row["path"] in listing:
            raise ValueError("Refusing unowned checkpoint overwrite")
        staged = Path(directory) / "snapshot.pt"
        shutil.copyfile(path, staged)
        if checksum(staged) != row["sha256"]:
            raise ValueError("Staged native checkpoint checksum mismatch")
        metadata = {**previous, "owned": [*owned, row][-2:]}
        deletes = [prefix + item["path"] for item in owned if item not in metadata["owned"]]
        for stale in (item for item in owned if item not in metadata["owned"]):
            downloaded = remote_file(download, repo, token, head, prefix + stale["path"], directory)
            if checksum(downloaded) != stale["sha256"]:
                raise ValueError("Refusing to prune changed remote checkpoint bytes")
            downloaded.unlink()
        files = {
            prefix + row["path"]: str(staged),
            prefix + "latest.json": encoded(row),
            prefix + "backup.json": encoded(metadata),
        }
        uncertain = {**receipt, "verified": False, "state": "publication_unknown", "candidate": row}
        atomic_json(receipt_path(result), uncertain)
        try:
            revision = commit(api, repo, token, head, files, deletes)
        except Exception:  # pylint: disable=broad-exception-caught
            # A request error can follow a committed write. Read only; never upload again.
            try:
                reconciled = private_head(api, repo, token)
                if (
                    remote_file(
                        download, repo, token, reconciled, prefix + "latest.json", directory
                    ).read_bytes()
                    == encoded(row)
                    and remote_file(
                        download, repo, token, reconciled, prefix + "backup.json", directory
                    ).read_bytes()
                    == encoded(metadata)
                    and checksum(
                        remote_file(
                            download, repo, token, reconciled, prefix + row["path"], directory
                        )
                    )
                    == row["sha256"]
                ):
                    atomic_json(
                        receipt_path(result), {**uncertain, "reconciled_revision": reconciled}
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise RuntimeError("Ambiguous checkpoint commit; reconciliation required") from None
        final_listing = set(
            api.list_repo_files(repo, revision=revision, repo_type="model", token=token)
        )
        if any(name in final_listing for name in deletes) or not set(files) <= final_listing:
            raise ValueError("Checkpoint retention/listing readback mismatch")
        for name, content in files.items():
            remote = remote_file(download, repo, token, revision, name, directory)
            if name == prefix + row["path"]:
                valid = checksum(remote) == row["sha256"]
            else:
                valid = remote.read_bytes() == content
            if not valid:
                raise ValueError("Pinned checkpoint readback mismatch")
        if local_snapshot(result)[1] != row or private_head(api, repo, token) != revision:
            raise ValueError("Local checkpoint or remote HEAD changed during publication")
        atomic_json(Path(result) / "checkpoints/backup.json", metadata)
        atomic_json(
            receipt_path(result),
            {
                "verified": True,
                "revision": revision,
                "identity": value,
                "metadata": metadata,
                "step": row["step"],
                "sha256": row["sha256"],
            },
        )


# All downloaded objects share one bounded cache and publication lifetime.
# pylint: disable-next=too-many-locals
def restore_store(package, run, revision, target, *, api, download, repo, token):
    """Validate source/data/pointer and SHA before publishing a new restore folder."""
    target = Path(target)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or target.exists()
        or target.is_symlink()
        or str(run) != Path(run).name
        or target.resolve().is_relative_to(Path(package).resolve())
    ):
        raise ValueError("Pinned revision and new restore directory required")
    expected = identity(package, Path(run))
    info = api.model_info(repo, revision=revision, token=token)
    if info.private is not True or info.sha != revision:
        raise ValueError("Private pinned checkpoint repository required")
    prefix = f"runs/{run}/checkpoints/"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-restore-", dir=target.parent) as directory:
        folder = Path(directory)
        metadata_bytes = remote_file(
            download, repo, token, revision, prefix + "backup.json", folder
        ).read_bytes()
        metadata = json.loads(metadata_bytes)
        row_bytes = remote_file(
            download, repo, token, revision, prefix + "latest.json", folder
        ).read_bytes()
        row = pointer(json.loads(row_bytes))
        if metadata["identity"] != expected or row not in metadata["owned"]:
            raise ValueError("Restore frozen source/data/package mismatch")
        snapshot = remote_file(download, repo, token, revision, prefix + row["path"], folder)
        if checksum(snapshot) != row["sha256"]:
            raise ValueError("Restore pinned snapshot checksum mismatch")
        staged = folder / "verified"
        staged.mkdir()
        shutil.copyfile(snapshot, staged / row["path"])
        if checksum(staged / row["path"]) != row["sha256"]:
            raise ValueError("Restore staged snapshot checksum mismatch")
        (staged / "latest.json").write_bytes(row_bytes)
        (staged / "backup.json").write_bytes(metadata_bytes)
        # mkdir is exclusive; never replace a target created concurrently.
        target.mkdir()
        try:
            for item in staged.iterdir():
                item.replace(target / item.name)
        except BaseException:
            shutil.rmtree(target)
            raise
    return target / row["path"]
