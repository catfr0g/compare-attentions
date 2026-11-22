"""Isolated full checkpoint transport: synthetic accounts, no network."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def core():
    assert importlib.util.find_spec("colab_checkpoint_backup"), "periodic HF publisher missing"
    return importlib.import_module("colab_checkpoint_backup")


# Explicit fake protocol state makes remote failure modes independently testable.
# pylint: disable-next=too-many-instance-attributes
class Store:
    """Revisioned fake implementing actual HF SDK commit operation objects."""

    def __init__(self, root):
        self.root = root
        self.sha = "a" * 40
        self.files = {}
        self.revisions = {self.sha: {}}
        self.commits = []
        self.private = True
        self.corrupt = None
        self.fail_commit = False
        self.downloads = []

    def model_info(self, *_args, **kwargs):
        revision = kwargs.get("revision", "main")
        return SimpleNamespace(
            private=self.private, sha=self.sha if revision == "main" else revision
        )

    def list_repo_files(self, *_args, **kwargs):
        return list(self.revisions[kwargs["revision"]])

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.sha
        assert kwargs["repo_type"] == "model" and kwargs["revision"] == "main"
        self.commits.append(kwargs)
        for op in kwargs["operations"]:
            if type(op).__name__ == "CommitOperationDelete":
                del self.files[op.path_in_repo]
            else:
                value = op.path_or_fileobj
                self.files[op.path_in_repo] = (
                    value if isinstance(value, bytes) else Path(value).read_bytes()
                )
        self.sha = f"{len(self.commits):040x}"
        self.revisions[self.sha] = dict(self.files)
        if self.fail_commit:
            raise RuntimeError("synthetic-token private-account.example")
        return SimpleNamespace(oid=self.sha)

    def download(self, **kwargs):
        assert kwargs["force_download"] and kwargs["repo_type"] == "model"
        assert kwargs["revision"] in self.revisions
        self.downloads.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        value = self.revisions[kwargs["revision"]][kwargs["filename"]]
        path.write_bytes(b"corrupt" if self.corrupt == kwargs["filename"] else value)
        return str(path)


@pytest.fixture(name="setup")
def checkpoint_fixture(tmp_path):
    module = core()
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps({"source_sha256": "b" * 64, "data_fingerprint": "c" * 64})
    )
    result = tmp_path / "run-new"
    store = Store(tmp_path)
    options = {
        "api": store,
        "download": store.download,
        "repo": "synthetic/repo",
        "token": "synthetic-token",
    }
    return module, package, result, store, options


def snapshot(module, result, step):
    folder = result / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step_{step:08d}.pt"
    path.write_bytes(f"opaque full optimizer/scaler/RNG state {step}".encode())
    row = {"step": step, "path": path.name, "sha256": module.checksum(path)}
    (folder / "latest.json").write_text(json.dumps(row))
    return path, row


def test_probe_atomic_publish_and_pinned_bytes(setup):
    module, package, result, store, options = setup
    module.probe(package, result, **options)
    assert not result.exists()
    assert module.receipt_path(result).is_file()
    path, row = snapshot(module, result, 125)
    module.publish(package, result, **options)
    prefix = f"runs/{result.name}/checkpoints/"
    assert {op.path_in_repo for op in store.commits[-1]["operations"]} == {
        prefix + path.name,
        prefix + "latest.json",
        prefix + "backup.json",
    }
    assert json.loads(store.files[prefix + "latest.json"]) == row
    proof = json.loads(module.receipt_path(result).read_text())
    assert proof["verified"] is True and proof["revision"] == store.sha
    assert proof["step"] == 125 and proof["sha256"] == row["sha256"]
    assert (result / "checkpoints/backup.json").read_bytes() == store.files[prefix + "backup.json"]
    assert any(call["filename"] == prefix + path.name for call in store.downloads)
    assert all(not Path(call["local_dir"]).exists() for call in store.downloads)
    assert "synthetic" not in module.receipt_path(result).read_text()


def test_retention_two_authenticated_owned_only_and_idempotency(setup):
    module, package, result, store, options = setup
    module.probe(package, result, **options)
    prefix = f"runs/{result.name}/checkpoints/"
    store.files[prefix + "unowned.pt"] = b"never delete"
    store.files["other-run/checkpoints/step_00000125.pt"] = b"never delete"
    store.revisions[store.sha] = dict(store.files)
    for step in (125, 250, 375):
        snapshot(module, result, step)
        module.publish(package, result, **options)
    assert prefix + "step_00000125.pt" not in store.files
    assert prefix + "step_00000250.pt" in store.files
    assert prefix + "step_00000375.pt" in store.files
    assert store.files[prefix + "unowned.pt"] == b"never delete"
    assert store.files["other-run/checkpoints/step_00000125.pt"] == b"never delete"
    count = len(store.commits)
    module.publish(package, result, **options)
    assert len(store.commits) == count
    assert json.loads(module.receipt_path(result).read_text())["verified"] is True


@pytest.mark.parametrize(
    "failure", ["public", "snapshot", "pointer", "ownership", "readback", "ambiguous"]
)
def test_fail_closed_no_false_receipt_or_secret_output(setup, failure, capsys):
    module, package, result, store, options = setup
    module.probe(package, result, **options)
    path, row = snapshot(module, result, 125)
    prefix = f"runs/{result.name}/checkpoints/"
    if failure == "public":
        store.private = False
    elif failure == "snapshot":
        path.write_bytes(b"corrupted")
    elif failure == "pointer":
        row["path"] = "../escaped.pt"
        (path.parent / "latest.json").write_text(json.dumps(row))
    elif failure == "ownership":
        store.files[prefix + "backup.json"] = b"changed"
        store.revisions[store.sha] = dict(store.files)
    elif failure == "readback":
        store.corrupt = prefix + path.name
    elif failure == "ambiguous":
        store.fail_commit = True
    with pytest.raises(RuntimeError, match="Checkpoint transport failed") as error:
        module.publish(package, result, **options)
    assert "synthetic" not in str(error.value) + str(capsys.readouterr())
    proof = json.loads(module.receipt_path(result).read_text())
    assert proof.get("step") != 125 and path.exists()
    before = len(store.commits)
    if failure in ("readback", "ambiguous"):
        assert proof["verified"] is False and proof["state"] == "publication_unknown"
        with pytest.raises(RuntimeError):
            module.publish(package, result, **options)
        assert len(store.commits) == before
        if failure == "ambiguous":
            assert proof["reconciled_revision"] == store.sha


@pytest.mark.parametrize(
    "fault", [None, "corrupt", "source", "existing", "revision", "package-target", "stage-copy"]
)
def test_restore_pinned_opaque_snapshot_new_folder_only(setup, tmp_path, fault, monkeypatch):
    module, package, result, store, options = setup
    module.probe(package, result, **options)
    original, row = snapshot(module, result, 125)
    module.publish(package, result, **options)
    target = tmp_path / "restored"
    revision = store.sha
    if fault == "corrupt":
        store.corrupt = f"runs/{result.name}/checkpoints/" + row["path"]
    if fault == "source":
        (package / "manifest.json").write_text(
            json.dumps({"source_sha256": "d" * 64, "data_fingerprint": "c" * 64})
        )
    if fault == "existing":
        target.mkdir()
        (target / "keep").write_bytes(b"untouched")
    if fault == "revision":
        revision = "main"
    if fault == "package-target":
        target = package / "restore-should-not-pollute"
    if fault == "stage-copy":
        backend = importlib.import_module("colab_checkpoint_store")
        monkeypatch.setattr(
            backend.shutil, "copyfile", lambda _src, dst: Path(dst).write_bytes(b"bad-stage")
        )
    if fault:
        with pytest.raises(RuntimeError):
            module.restore(package, result.name, revision, target, **options)
        if fault != "existing":
            assert not target.exists()
        else:
            assert list(target.iterdir()) == [target / "keep"]
    else:
        restored = module.restore(package, result.name, revision, target, **options)
        assert restored.read_bytes() == original.read_bytes()
        assert module.checksum(restored) == row["sha256"]
        assert json.loads((target / "latest.json").read_text()) == row
        with pytest.raises(RuntimeError):
            module.restore(package, result.name, revision, target, **options)


@pytest.mark.parametrize(
    "fault",
    [
        "probe-public",
        "probe-readback",
        "duplicate-probe",
        "unowned",
        "remote-pointer",
        "local-symlink",
    ],
)
def test_namespace_privacy_and_pointer_guards(setup, fault, monkeypatch):
    module, package, result, store, options = setup
    prefix = f"runs/{result.name}/checkpoints/"
    if fault.startswith("probe-"):
        store.private = fault != "probe-public"
        if fault == "probe-readback":
            store.corrupt = prefix + "backup.json"
        with pytest.raises(RuntimeError):
            module.probe(package, result, **options)
        assert not result.exists()
        if module.receipt_path(result).exists():
            assert json.loads(module.receipt_path(result).read_text())["verified"] is False
        return
    module.probe(package, result, **options)
    if fault == "duplicate-probe":
        before = len(store.commits)
        with pytest.raises(RuntimeError):
            module.probe(package, result, **options)
        assert len(store.commits) == before
        return
    path, _ = snapshot(module, result, 125)
    if fault == "unowned":
        store.files[prefix + path.name] = b"unowned"
        store.revisions[store.sha] = dict(store.files)
    elif fault == "remote-pointer":
        module.publish(package, result, **options)
        store.files[prefix + "latest.json"] = b"tampered"
        store.revisions[store.sha] = dict(store.files)
        snapshot(module, result, 250)
    else:
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    before = len(store.commits)
    with pytest.raises(RuntimeError):
        module.publish(package, result, **options)
    assert len(store.commits) == before


@pytest.mark.parametrize("fault", ["missing-repo", "advanced-head", "staged-copy"])
# pylint: disable-next=too-many-locals
def test_private_creation_and_transfer_races(setup, fault, monkeypatch):
    module, package, result, store, options = setup
    backend = importlib.import_module("colab_checkpoint_store")
    if fault == "missing-repo":
        existing_info = store.model_info
        first = [True]
        created = []

        def missing(*args, **kwargs):
            if first[0]:
                first[0] = False

                response = httpx2.Response(
                    404,
                    request=httpx2.Request(
                        "GET", "https://huggingface.co/api/models/synthetic/repo"
                    ),
                )
                raise RepositoryNotFoundError("synthetic-private-error", response=response)
            return existing_info(*args, **kwargs)

        store.model_info = missing
        store.create_repo = lambda **kwargs: created.append(kwargs)
        module.probe(package, result, **options)
        assert created == [
            {
                "repo_id": options["repo"],
                "repo_type": "model",
                "private": True,
                "exist_ok": False,
                "token": options["token"],
            }
        ]
        return
    module.probe(package, result, **options)
    path, _ = snapshot(module, result, 125)
    if fault == "staged-copy":
        original = backend.shutil.copyfile

        def damaged(src, dst):
            original(src, dst)
            Path(dst).write_bytes(b"corrupted-stage")

        monkeypatch.setattr(backend.shutil, "copyfile", damaged)
    else:
        original = store.create_commit

        def advanced(**kwargs):
            info = original(**kwargs)
            store.sha = "f" * 40
            store.revisions[store.sha] = dict(store.files)
            return info

        store.create_commit = advanced
    with pytest.raises(RuntimeError):
        module.publish(package, result, **options)
    assert path.exists()
    assert json.loads(module.receipt_path(result).read_text()).get("step") != 125


def test_retention_never_deletes_changed_owned_bytes(setup):
    module, package, result, store, options = setup
    module.probe(package, result, **options)
    for step in (125, 250):
        snapshot(module, result, step)
        module.publish(package, result, **options)
    stale = f"runs/{result.name}/checkpoints/step_00000125.pt"
    store.files[stale] = b"modified-by-someone-else"
    store.revisions[store.sha] = dict(store.files)
    snapshot(module, result, 375)
    before = len(store.commits)
    with pytest.raises(RuntimeError):
        module.publish(package, result, **options)
    assert len(store.commits) == before
    assert store.files[stale] == b"modified-by-someone-else"


def test_probe_unknown_write_has_durable_readonly_barrier(setup):
    module, package, result, store, options = setup
    store.fail_commit = True
    with pytest.raises(RuntimeError):
        module.probe(package, result, **options)
    assert not result.exists()
    proof = json.loads(module.receipt_path(result).read_text())
    assert proof["verified"] is False and proof["state"] == "probe_unknown"
    assert proof["reconciled_revision"] == store.sha
    count = len(store.commits)
    with pytest.raises(RuntimeError):
        module.probe(package, result, **options)
    assert len(store.commits) == count


def test_probe_rejects_existing_sink_before_network(setup):
    module, package, result, store, options = setup
    result.mkdir()
    calls = []
    original = store.model_info
    store.model_info = lambda *a, **k: (calls.append(True), original(*a, **k))[1]
    with pytest.raises(RuntimeError):
        module.probe(package, result, **options)
    assert not calls and not store.commits
