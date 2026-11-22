"""Portable package integrity, with no training dependencies."""
# Test-first module imports and pytest fixtures are intentionally local.
# pylint: disable=import-outside-toplevel,redefined-outer-name,exec-used

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module():
    spec = importlib.util.spec_from_file_location("colab_package", SCRIPTS / "colab_package.py")
    if spec is None or spec.loader is None or not (SCRIPTS / "colab_package.py").exists():
        pytest.fail("portable package verifier is missing")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_safe_extract_roundtrip_and_hash_before_execution(tmp_path, frozen_fixture):
    import export_colab

    core = module()
    output = export_colab.export(frozen_fixture, tmp_path / "portable")
    archive = output / "colab-package.zip"
    target = tmp_path / "out"
    core.extract_verified(archive, core.checksum(archive), target)
    assert core.verify_directory(target)["schema_version"] == 3
    assert (target / "data/meta.json").stat().st_mode & 0o200 == 0
    (target / "data/meta.json").chmod(0o600)
    (target / "data/meta.json").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        core.verify_directory(target)
    with pytest.raises(ValueError, match="SHA256"):
        core.extract_verified(archive, "0" * 64, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b", "./a"])
def test_unsafe_archive_paths(tmp_path, name):
    core = module()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        item = zipfile.ZipInfo("placeholder")
        item.filename = name  # ZipInfo constructor normalizes backslashes on Windows.
        stream.writestr(item, "not code")
    with pytest.raises(ValueError, match="unsafe"):
        core.extract_verified(archive, core.checksum(archive), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["symlink", "duplicate"])
def test_special_archive_members(tmp_path, kind):
    core = module()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        item = zipfile.ZipInfo("member")
        if kind == "symlink":
            item.external_attr = 0o120777 << 16
        stream.writestr(item, "target")
        if kind == "duplicate":
            with pytest.warns(UserWarning):
                stream.writestr("member", "again")
    with pytest.raises(ValueError, match="duplicate/directory/symlink"):
        core.extract_verified(archive, core.checksum(archive), tmp_path / "out")


@pytest.fixture
def frozen_fixture(tmp_path):
    """Real frozen wheel/configs, tiny opaque data; no GPU or array regeneration."""
    import shutil

    core = module()
    frozen = Path(
        "E:/HermesArtifacts/compare-attention/fp16-k64-small/"
        "272c2da7a9ef619711e4e92a1a98400ac0ed64b08f481d13752093d450820975"
    )
    state = json.loads((frozen / "series.json").read_text())
    root = tmp_path / "frozen"
    (root / "release").mkdir(parents=True)
    shutil.copyfile(frozen / "release/release.bundle", root / "release/release.bundle")
    (root / "data").mkdir()
    for name in core.DATA_FILES:
        path = root / "data" / name
        path.write_bytes(b"tiny fixture " + name.encode())
        state["identity"]["data"]["files"][name] = core.record(path)
        state["artifacts"]["data/" + name] = core.record(path)
    (root / "data/.env").write_text("SECRET_SENTINEL=never-export")
    series = root / "series.json"
    import hashlib

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    data = state["identity"]["data"]
    data["fingerprint"] = digest({key: data[key] for key in ("spec", "files")})
    identity = state["identity"]
    identity["fingerprint"] = digest({key: identity[key] for key in ("source", "configs", "data")})
    series.write_text(json.dumps(state))
    return series


def test_export_frozen_allowlist_and_notebook_bootstrap(frozen_fixture, tmp_path):
    import ast

    import export_colab

    core = module()
    output = tmp_path / "portable"
    export_colab.export(frozen_fixture, output)
    archive = output / "colab-package.zip"
    with zipfile.ZipFile(archive) as stream:
        names = set(stream.namelist())
        assert not any(".env" in name or "credential" in name for name in names)
        assert {"release/" + n for n in core.RELEASE_FILES} <= names
        assert {"data/" + n for n in core.DATA_FILES} <= names
        assert "tools/colab_services.py" in names
        manifest = json.loads(stream.read("manifest.json"))
        assert manifest["schema_version"] == 3
        assert manifest["storage"] == "hf"
        assert manifest["periodic_full_checkpoints"] is True
        assert "tools/colab_checkpoint_backup.py" in names
        assert not any("colab_api" in name for name in names)
        assert manifest["tracking"] == "clearml" and manifest["hf_upload"] is True
        assert manifest["expected_updates"]["deltanet"] == 47265
        assert "never-export" not in stream.read("manifest.json").decode()
    template = (output / "colab.env.example").read_text()
    import colab_services

    assert set(template.splitlines()) == {key + "=" for key in colab_services.KEYS}
    notebook = json.loads((output / "launch_colab.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert core.checksum(archive) in code
    assert "pip install" not in code
    assert "userdata" not in code
    assert "ENV_FILE = Path('/content/private/kaggle-env.env')" in code
    assert "ARCHIVE_PATH = Path('/content/colab-package.zip')" in code
    assert (
        "RESULT_DIR = Path('/content/compare-attention-results/deltanet-full-seed42-run01')" in code
    )
    assert "google.colab" not in code and "drive.mount" not in code
    assert "'--env-file', str(ENV_FILE)" in code
    core.extract_verified(archive, core.checksum(archive), tmp_path / "unpacked")
    with pytest.raises(FileExistsError):
        export_colab.export(frozen_fixture, output)
    state = json.loads(frozen_fixture.read_text())
    (frozen_fixture.parent / "data/train_inputs.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        export_colab.export(frozen_fixture, tmp_path / "bad")
    assert state["identity"]["configs"]


def test_export_rejects_identity_mutation_before_output(frozen_fixture, tmp_path):
    import export_colab

    state = json.loads(frozen_fixture.read_text())
    state["identity"]["data"]["fingerprint"] = "0" * 64
    frozen_fixture.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="fingerprint"):
        export_colab.export(frozen_fixture, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_release_cannot_be_replaced_by_recomputed_receipt(frozen_fixture, tmp_path):
    import export_colab

    core = module()
    state = json.loads(frozen_fixture.read_text())
    bundle = frozen_fixture.parent / "release/release.bundle"
    with zipfile.ZipFile(bundle) as stream:
        blobs = {name: stream.read(name) for name in stream.namelist()}
    blobs["requirements.txt"] = b"torch==99.0\n"
    receipt = json.loads(blobs["release-manifest.json"])
    receipt["requirements.txt"] = core.blob_record(blobs["requirements.txt"])
    blobs["release-manifest.json"] = json.dumps(receipt).encode()
    with zipfile.ZipFile(bundle, "w") as stream:
        for name, data in blobs.items():
            stream.writestr(name, data)
    state["archive_sha256"] = core.checksum(bundle)
    state["artifacts"]["release/release.bundle"] = core.record(bundle)
    frozen_fixture.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="pinned release"):
        export_colab.export(frozen_fixture, tmp_path / "bad")


def test_dataset_spec_is_fixed_not_a_new_factor(frozen_fixture):
    import export_colab

    state = json.loads(frozen_fixture.read_text())
    state["identity"]["data"]["spec"]["num_kv_pairs"] = 8
    with pytest.raises(ValueError, match="K64 dataset"):
        export_colab.scientific_configs(state)


def test_real_cli_export_extract_inspect(frozen_fixture, tmp_path):
    import subprocess

    core = module()
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "export_colab.py"),
            "--series",
            str(frozen_fixture),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["training_started"] is False
    archive = output / "colab-package.zip"
    unpacked = tmp_path / "unpacked"
    extracted = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "colab_package.py"),
            "--archive",
            str(archive),
            "--sha256",
            core.checksum(archive),
            "--output-dir",
            str(unpacked),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        json.loads(extracted.stdout)["verified_files"]
        == len(core.RELEASE_FILES) + len(core.DATA_FILES) + len(core.TOOLS) + 1
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(unpacked / "tools/colab_runner.py"),
            "--package-dir",
            str(unpacked),
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        json.loads(completed.stdout)["verified_files"]
        == len(core.RELEASE_FILES) + len(core.DATA_FILES) + len(core.TOOLS) + 1
    )
    import colab_runner
    import colab_services

    private = tmp_path / "synthetic-private.env"
    values = {key: "SYNTHETIC_ONLY_VALUE" for key in colab_services.KEYS}
    values.update({key: "https://example.invalid" for key in values if key.endswith("HOST")})
    values["HF_REPO"] = "owner/private-model"
    private.write_text("\n".join(f'{key}="{value}"' for key, value in values.items()))
    checked = subprocess.run(
        [
            sys.executable,
            str(unpacked / "tools/colab_services.py"),
            "check",
            "--env-file",
            str(private),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=colab_runner.clean_environment(),
    )
    assert "SYNTHETIC_ONLY_VALUE" not in checked.stdout + checked.stderr
    assert all(key in checked.stdout for key in colab_services.KEYS)
    assert not list(unpacked.rglob("__pycache__"))


@pytest.mark.parametrize("gpu", [None, "NVIDIA GeForce RTX 5070, 12.0", "Tesla T4, 7.5"])
def test_notebook_first_cell_precedes_oauth_and_installs(monkeypatch, gpu):
    import subprocess

    import export_colab

    monkeypatch.setattr(sys, "version_info", (3, 13, 15))

    def query(*_a, **_k):
        if gpu is None:
            raise FileNotFoundError("no GPU")
        return gpu

    monkeypatch.setattr(subprocess, "check_output", query)
    first = export_colab.notebook("a" * 64)["cells"][1]
    code = "".join(first["source"])
    assert "drive" not in code and "pip" not in code and "userdata" not in code
    if gpu == "Tesla T4, 7.5":
        exec(compile(code, "<first-colab-cell>", "exec"), {})
    else:
        with pytest.raises(RuntimeError, match="T4"):
            exec(compile(code, "<first-colab-cell>", "exec"), {})


@pytest.mark.parametrize("attack", ["tamper", "traversal", "symlink", "duplicate"])
def test_notebook_bootstrap_rejects_before_extracted_code(monkeypatch, tmp_path, attack):  # pylint: disable=too-many-locals
    from types import ModuleType, SimpleNamespace

    import export_colab

    monkeypatch.setattr(sys, "version_info", (3, 13, 16))
    import subprocess

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "Tesla T4, 7.5")
    core = module()
    archive = tmp_path / "incoming.zip"
    work = tmp_path / "must-not-exist"
    with zipfile.ZipFile(archive, "w") as stream:
        item = zipfile.ZipInfo("../escape" if attack == "traversal" else "member")
        if attack == "symlink":
            item.external_attr = 0o120777 << 16
        stream.writestr(item, "never execute")
        if attack == "duplicate":
            with pytest.warns(UserWarning):
                stream.writestr("member", "again")
    google = ModuleType("google")
    google.__path__ = []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda _: None)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    notebook = export_colab.notebook("0" * 64 if attack == "tamper" else core.checksum(archive))
    code = "".join(notebook["cells"][2]["source"])
    code = code.replace("'/content/colab-package.zip'", repr(str(archive)))
    code = code.replace("'/content/compare-attention-package'", repr(str(work)))
    with pytest.raises(ValueError):
        exec(compile(code, "<bootstrap-cell>", "exec"), {})
    assert not work.exists()


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "storage",
        "periodic",
        "tracking",
        "upload",
        "tool",
        "source",
        "release",
        "receipt",
        "config",
        "data",
    ],
)
def test_verified_package_rejects_policy_or_payload_downgrades(
    frozen_fixture, tmp_path, monkeypatch, change
):
    import export_colab

    core = module()
    output = export_colab.export(frozen_fixture, tmp_path / "portable")
    archive = output / "colab-package.zip"
    target = tmp_path / "package"
    core.extract_verified(archive, core.checksum(archive), target)
    path = target / "manifest.json"
    path.chmod(0o600)
    manifest = json.loads(path.read_text())
    if change == "schema":
        manifest["schema_version"] = 2
    elif change == "storage":
        manifest["storage"] = "drive"
    elif change == "periodic":
        manifest["periodic_full_checkpoints"] = False
    elif change == "tracking":
        manifest["tracking"] = "none"
    elif change == "upload":
        manifest["hf_upload"] = False
    elif change == "tool":
        name = "tools/colab_checkpoint_backup.py"
        del manifest["files"][name]
        (target / name).chmod(0o600)
        (target / name).unlink()
    elif change in ("source", "release"):
        manifest["source_sha256" if change == "source" else "original_release_sha256"] = "0" * 64
    elif change == "config":
        manifest["configs"]["deltanet"]["schedule"]["batch_size"] = 1
    elif change == "data":
        manifest["data_fingerprint"] = "0" * 64
    else:
        name = "release/requirements.txt"
        (target / name).chmod(0o600)
        (target / name).write_bytes(b"torch==99.0\n")
        manifest["files"][name] = core.record(target / name)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        core.verify_directory(target)
    import colab_runner

    monkeypatch.setattr(colab_runner, "preflight", lambda: None)
    monkeypatch.setattr(colab_runner, "execute", lambda *_a, **_k: pytest.fail("child attempted"))
    with pytest.raises(ValueError):
        colab_runner.run(target, tmp_path / "venv", tmp_path / "result")
