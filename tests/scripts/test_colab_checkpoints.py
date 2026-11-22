"""Offline full-snapshot recovery contracts; never real credentials or network."""

# Service imports remain local to preserve import-order acceptance checks.
# pylint: disable=import-outside-toplevel

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class Store:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.private = True
        self.sha = "a" * 40
        self.commits = []
        self.corrupt = False

    def model_info(self, *_args, **_kwargs):
        return SimpleNamespace(private=self.private, sha=self.sha)

    def list_repo_files(self, *_args, **_kwargs):
        return list(self.files)

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.sha
        self.commits.append(kwargs)
        for op in kwargs["operations"]:
            value = op.path_or_fileobj
            self.files[op.path_in_repo] = (
                value if isinstance(value, bytes) else Path(value).read_bytes()
            )
        self.sha = f"{len(self.commits):040x}"
        return SimpleNamespace(oid=self.sha)

    def download(self, **kwargs):
        assert kwargs["revision"] == self.sha and kwargs["force_download"]
        path = self.root / "download.bin"
        path.write_bytes(b"bad" if self.corrupt else self.files[kwargs["filename"]])
        return path

    def get_paths_info(self, **kwargs):
        assert kwargs["revision"] == self.sha
        return [
            SimpleNamespace(
                path=name, lfs=SimpleNamespace(sha256=hashlib.sha256(self.files[name]).hexdigest())
            )
            for name in kwargs["paths"]
        ]


def package(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"source_sha256": "b" * 64, "data_fingerprint": "c" * 64})
    )
    result = tmp_path / "run-unique"
    return root, result


def test_recovery_check_private_atomic_manifest_and_pinned_proof(tmp_path):
    module = core()
    root, result = package(tmp_path)
    store = Store(tmp_path)
    module.recovery_check(
        root,
        result,
        api=store,
        download=store.download,
        repo="synthetic/repo",
        token="synthetic-token",
    )
    assert len(store.commits) == 1
    manifest = json.loads(store.files["recoveries/run-unique/run.json"])
    assert manifest["data_path"] == str((root / "data").resolve())
    assert "synthetic" not in json.dumps(manifest)
    receipt = module.receipt_path(result)
    assert json.loads(receipt.read_text())["revision"] == store.sha
    with pytest.raises(ValueError, match="collision"):
        module.recovery_check(
            root,
            result,
            api=store,
            download=store.download,
            repo="synthetic/repo",
            token="synthetic-token",
        )


@pytest.mark.parametrize("failure", ["public", "readback"])
def test_startup_failure_has_no_receipt(tmp_path, failure):
    module = core()
    root, result = package(tmp_path)
    store = Store(tmp_path)
    store.private = failure != "public"
    store.corrupt = failure == "readback"
    with pytest.raises(ValueError):
        module.recovery_check(
            root,
            result,
            api=store,
            download=store.download,
            repo="synthetic/repo",
            token="synthetic-token",
        )
    assert not module.receipt_path(result).exists()
    if failure == "public":
        assert not store.commits


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def core():
    assert importlib.util.find_spec("colab_checkpoints"), "HF recovery helper missing"
    return importlib.import_module("colab_checkpoints")


def test_recovery_service_isolated_hf_and_controlled_errors(tmp_path, monkeypatch, capsys):  # pylint: disable=too-many-locals
    import builtins

    import colab_checkpoint_backup as backup
    import colab_services as service
    import huggingface_hub

    root, result = package(tmp_path)
    store = Store(tmp_path)
    values = {key: "synthetic-secret" for key in service.KEYS}
    values["HF_REPO"] = "synthetic/repo"
    monkeypatch.setattr(service, "load_credentials", lambda _: values)
    monkeypatch.setattr(service.os, "environ", {"HF_HUB_OFFLINE": "1"})
    imports = []
    original_import = builtins.__import__

    def check_environment():
        assert "HF_HUB_OFFLINE" not in service.os.environ
        assert not any(key.startswith("CLEARML_") for key in service.os.environ)
        assert service.os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert service.os.environ["HF_TOKEN"] == values["HF_TOKEN"]
        assert Path(service.os.environ["HF_HOME"]).is_dir()

    def import_online(name, *args, **kwargs):
        if name == "huggingface_hub":
            check_environment()
            imports.append(name)
        return original_import(name, *args, **kwargs)

    def factory(*, token):
        assert token == values["HF_TOKEN"]
        check_environment()
        return store

    monkeypatch.setattr(builtins, "__import__", import_online)
    monkeypatch.setitem(huggingface_hub.__dict__, "HfApi", factory)
    monkeypatch.setitem(huggingface_hub.__dict__, "hf_hub_download", store.download)
    arguments = ["--env-file", str(tmp_path / "private.env"), "--package-dir", str(root)]
    result_arguments = [*arguments, "--result-dir", str(result)]
    assert backup.main(["probe", *result_arguments]) == 0
    assert backup.receipt_path(result).exists() and not result.exists()
    path, row = native_snapshot(backup, result)
    assert backup.main(["publish", *result_arguments]) == 0
    assert json.loads(backup.receipt_path(result).read_text())["step"] == row["step"]
    target = tmp_path / "downloaded"
    assert (
        backup.main(
            [
                "restore",
                *arguments,
                "--run",
                result.name,
                "--revision",
                store.sha,
                "--target",
                str(target),
            ]
        )
        == 0
    )
    assert (target / path.name).read_bytes() == path.read_bytes()
    assert len(imports) >= 3
    store.private = False
    with pytest.raises(RuntimeError, match="Checkpoint transport failed") as error:
        backup.service_phase(
            tmp_path / "private.env",
            root,
            result,
            "publish",
            api_factory=factory,
            download=store.download,
        )
    assert "synthetic" not in str(error.value)
    assert backup.main(["publish", *result_arguments]) == 1
    output = capsys.readouterr()
    assert "synthetic" not in output.out + output.err
    assert "local snapshots retained; reconcile before retry" in output.err


def native_snapshot(module, result, step=125):
    folder = result / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step_{step:08d}.pt"
    path.write_bytes(b"opaque complete native optimizer/scaler/RNG payload")
    row = {"step": step, "path": path.name, "sha256": module.checksum(path)}
    (folder / "latest.json").write_text(json.dumps(row))
    return path, row


@pytest.mark.parametrize("failure", [None, "metadata", "local", "ambiguous"])
def test_checkpoint_atomic_full_snapshot_pointer_and_verified_receipt(tmp_path, failure):
    module = core()
    root, result = package(tmp_path)
    store = Store(tmp_path)
    kwargs = {"api": store, "download": store.download, "repo": "synthetic/repo", "token": "secret"}
    module.recovery_check(root, result, **kwargs)
    path, row = native_snapshot(module, result)
    if failure == "metadata":
        store.get_paths_info = lambda **_: [SimpleNamespace(path="wrong", lfs=None)]
    if failure == "local":
        path.write_bytes(b"changed")
    if failure == "ambiguous":

        def fail(**_):
            raise RuntimeError("secret")

        store.create_commit = fail
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            module.publish_checkpoint(root, result, **kwargs)
        assert "step" not in json.loads(module.receipt_path(result).read_text())
    else:
        module.publish_checkpoint(root, result, **kwargs)
        operations = store.commits[-1]["operations"]
        assert {op.path_in_repo for op in operations} == {
            "recoveries/run-unique/step_00000125.pt",
            "recoveries/run-unique/latest.json",
        }
        assert json.loads(store.files["recoveries/run-unique/latest.json"]) == row
        report = json.loads(module.receipt_path(result).read_text())
        assert report["sha256"] == row["sha256"] and report["step"] == 125
        assert report["revision"] == store.sha
        with pytest.raises(ValueError, match="immutable"):
            module.publish_checkpoint(root, result, **kwargs)


@pytest.mark.parametrize("bad", ["../escape.pt", "step_00000001.pt", "/abs.pt"])
def test_untrusted_native_pointer_refused(tmp_path, bad):
    module = core()
    _, result = package(tmp_path)
    _, row = native_snapshot(module, result)
    row["path"] = bad
    (result / "checkpoints/latest.json").write_text(json.dumps(row))
    with pytest.raises(ValueError):
        module.local_snapshot(result)


@pytest.mark.parametrize("corrupt", [False, True])
def test_fetch_pinned_snapshot_sha_and_data_path_before_strict_resume(tmp_path, corrupt):
    module = core()
    root, result = package(tmp_path)
    store = Store(tmp_path)
    kwargs = {"api": store, "download": store.download, "repo": "synthetic/repo", "token": "secret"}
    module.recovery_check(root, result, **kwargs)
    _, row = native_snapshot(module, result)
    module.publish_checkpoint(root, result, **kwargs)
    target = tmp_path / "retrieved"
    fresh = tmp_path / "new-result"
    if corrupt:
        store.files["recoveries/run-unique/step_00000125.pt"] = b"wrong bytes"
        with pytest.raises(ValueError, match="checksum"):
            module.fetch_checkpoint(root, "run-unique", store.sha, target, fresh, **kwargs)
        assert not target.exists()
    else:
        path = module.fetch_checkpoint(root, "run-unique", store.sha, target, fresh, **kwargs)
        assert module.checksum(path) == row["sha256"]
        assert json.loads((target / "latest.json").read_text()) == row
        with pytest.raises(ValueError, match="pinned"):
            module.fetch_checkpoint(root, "run-unique", "main", tmp_path / "other", fresh, **kwargs)
        fresh.mkdir()
        (fresh / "metrics.jsonl").write_text("old")
        with pytest.raises(ValueError, match="fresh"):
            module.fetch_checkpoint(
                root, "run-unique", store.sha, tmp_path / "other", fresh, **kwargs
            )


def test_scoped_callback_publishes_only_after_native_save_and_restores(tmp_path):
    module = core()
    calls = []

    class Tracker:  # pylint: disable=too-few-public-methods
        def save_checkpoint(self, payload, step, keep_last):
            calls.append(("native", payload, step, keep_last))

    original = Tracker.save_checkpoint
    with module.checkpoint_backups(
        "private.env",
        tmp_path,
        tmp_path / "result",
        tracker_class=Tracker,
        publish=lambda: calls.append("remote"),
    ):
        tracker = Tracker()
        for step in (125, 250, 267):
            tracker.save_checkpoint({"step": step}, step, 2)
    assert [call if isinstance(call, str) else call[0] for call in calls] == [
        "native",
        "remote",
        "native",
        "remote",
        "native",
        "remote",
    ]
    assert Tracker.save_checkpoint is original
    with pytest.raises(RuntimeError):
        with module.checkpoint_backups(
            "private.env",
            tmp_path,
            tmp_path / "result",
            tracker_class=Tracker,
            publish=lambda: (_ for _ in ()).throw(RuntimeError("stop")),
        ):
            tracker.save_checkpoint({}, 375, 2)
    assert Tracker.save_checkpoint is original
    assert calls[-1][0] == "native"
