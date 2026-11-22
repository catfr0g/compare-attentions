"""Export the existing frozen K64 release/data, never live source or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from colab_package import (
    CONFIGS,
    DATA_FILES,
    RELEASE_FILES,
    TOOLS,
    blob_record,
    checksum,
    record,
    zip_members,
)
from colab_runner import PREFLIGHT
from colab_services import KEYS

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "local/experiments/fp16-controls/run_series.py"
HELPER_SHA256 = "fa184277a3d0f82ec5b75d8ab0d965f04a0aa42de74f52ada7d3eed262dbd657"
RELEASE_SHA256 = "7feabbfc925971c7c537a506d0caba8c1a3ce903506e5ca37a367921402747a9"
SOURCE_SHA256 = "1beba22ca8f9897e0b7367e07f4a016559946f1183fc8413deff2490c6e85ef1"
REFERENCE = {
    "provider": "kaggle",
    "python": "3.13.16",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "cudnn": 91900,
    "triton": "3.6.0",
    "gpu": "Tesla T4",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or record(path) != expected:
        raise ValueError("frozen integrity mismatch: " + path.name)


def scientific_configs(state):
    configs = {}
    spec = {
        "vocab_size": 8192,
        "input_seq_len": 512,
        "num_kv_pairs": 64,
        "power_a": 0.01,
        "n_train": 100000,
        "n_val": 1000,
        "n_test": 1000,
        "seed": 1337,
    }
    if state["identity"]["data"]["spec"] != spec:
        raise ValueError("Only the frozen K64 dataset specification is supported")
    for family, name in CONFIGS.items():
        config = state["identity"]["configs"][name]
        kv = {key: value for key, value in spec.items() if key not in ("vocab_size", "seed")}
        kv["dataset_seed"] = spec["seed"]
        if config["kv"] != kv:
            raise ValueError("Experiment differs from the frozen K64 dataset")
        schedule = config["schedule"]
        required = {
            "train_flops": 10**16,
            "axis": "train_flops",
            "batch_size": 32,
            "seed": 42,
            "dtype": "float16",
            "compile": True,
            "val_every_steps": 25,
            "log_every_steps": 25,
            "checkpoint_every_steps": 125,
            "checkpoint_keep_last": 2,
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.001,
            "early_stopping_min_steps": 2000,
            "amp_max_scale": 128.0,
        }
        model = {
            "vocab_size": 8192,
            "block_size": 512,
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "dropout": 0.0,
            "bias": True,
            "attention": family,
        }
        optim = {
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "lr_schedule": "constant",
            "warmup_fraction": 0.05,
            "min_lr_ratio": 0.1,
        }
        if schedule != required or config["model"] != model or config["optim"] != optim:
            raise ValueError("Only the frozen full K64 seed42 scientific contract is supported")
        if config.get("resume_from") is not None or config.get("hf_repo") is not None:
            raise ValueError("frozen config contains unexpected resume/upload settings")
        configs[family] = {"model": model, "schedule": schedule, "optim": optim}
    return configs


def bootstrap_code(
    sha,
    *,
    archive_path="/content/colab-package.zip",
    work_dir="/content/compare-attention-package",
    venv="/content/compare-attention-venv",
    result_dir="/content/compare-attention-results/deltanet-full-seed42-run01",
    env_file="/content/private/kaggle-env.env",
):
    """Pure source: preflight, then SHA-bound extraction, without credentials/OAuth."""
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("A literal lowercase SHA256 is required")
    bootstrap = """import hashlib, json, shutil, stat, sys, zipfile
from pathlib import Path, PurePosixPath
# Upload the ZIP via Colab Files separately from the private env file.
ARCHIVE_PATH = Path('/content/colab-package.zip')
WORK_DIR = Path('/content/compare-attention-package')
VENV = Path('/content/compare-attention-venv')
RESULT_DIR = Path('/content/compare-attention-results/deltanet-full-seed42-run01')
ENV_FILE = Path('/content/private/kaggle-env.env')
FAMILY = 'deltanet'
RESUME_FROM = None  # Optional explicit previous Colab checkpoint; always a NEW RESULT_DIR.
EXPECTED_SHA256 = '__SHA__'
with ARCHIVE_PATH.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != EXPECTED_SHA256:
    raise ValueError('Archive SHA256 mismatch: refusing all extracted code')
if WORK_DIR.exists() or WORK_DIR.is_symlink():
    raise FileExistsError('WORK_DIR must be new; never overwrite verified inputs')
with zipfile.ZipFile(ARCHIVE_PATH) as archive:
    names = set()
    for item in archive.infolist():
        name = item.filename
        if item.orig_filename != name:
            raise ValueError('Noncanonical ZIP member')
        path = PurePosixPath(name)
        mode = item.external_attr >> 16
        if (not path.parts or path.is_absolute() or path.as_posix() != name
                or '\\\\' in name or any(p in ('.', '..') or ':' in p for p in path.parts)
                or name in names or item.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
            raise ValueError('Unsafe ZIP member: ' + name)
        names.add(name)
    WORK_DIR.mkdir(parents=True)
    try:
        for item in archive.infolist():
            path = WORK_DIR / item.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    except Exception:
        shutil.rmtree(WORK_DIR)
        raise
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORK_DIR / 'tools'))
from colab_package import verify_directory
manifest = verify_directory(WORK_DIR)
for path in WORK_DIR.rglob('*'):
    if path.is_file():
        path.chmod(0o444)
print('Verified package; source:', manifest['source_sha256'])
""".replace("__SHA__", sha)
    paths = {
        "ARCHIVE_PATH": archive_path,
        "WORK_DIR": work_dir,
        "VENV": venv,
        "RESULT_DIR": result_dir,
        "ENV_FILE": env_file,
    }
    bootstrap = re.sub(
        r"^(ARCHIVE_PATH|WORK_DIR|VENV|RESULT_DIR|ENV_FILE) = Path\(.*\)$",
        lambda match: f"{match[1]} = Path({str(paths[match[1]])!r})",
        bootstrap,
        flags=re.MULTILINE,
    )
    return PREFLIGHT + bootstrap


def notebook(archive_sha):
    """Trusted notebook embeds the archive SHA before any extracted code executes."""
    bootstrap = bootstrap_code(archive_sha)
    setup_cell = """import subprocess
RUNNER = WORK_DIR / 'tools/colab_runner.py'
base = [sys.executable, '-B', str(RUNNER), '--package-dir', str(WORK_DIR), '--venv', str(VENV)]
subprocess.run(base + ['--inspect'], check=True)
# Only this cell installs into a NEW isolated venv. It does not train.
subprocess.run(base + ['--setup'], check=True)
"""
    run_cell = """# Explicit full run; no short job, warm-up update, reconnect or quota bypass.
command = base + ['--run', '--family', FAMILY, '--result-dir', str(RESULT_DIR),
                  '--env-file', str(ENV_FILE)]
if RESUME_FROM is not None:
    command += ['--resume-from', str(RESUME_FROM)]
subprocess.run(command, check=True)
print('Completed gates, public test eval and verified HF upload:', RESULT_DIR)
"""
    markdown = (
        "# Colab T4: полный K64, seed42\n\n"
        "Выберите GPU T4 в настройках среды. GPU и время сессии не гарантированы. "
        "Выполняйте ячейки по порядку. Первая проверяет T4 и Python до packages и credentials.\n\n"
        "Загрузите ZIP через панель Files в /content/colab-package.zip. "
        "Задайте ARCHIVE_PATH и новый RESULT_DIR. "
        "Colab дополняет Kaggle, не заменяет его. По умолчанию свежий DeltaNet, ClearML и HF ON. "
        "Отдельно загрузите приватный kaggle-env.env вне ZIP, package и RESULT_DIR; "
        "задайте ENV_FILE. "
        "Не вставляйте значения ключей в notebook. Colab Secrets не нужны. "
        "Не используйте Run all до проверки путей. Данные/venv локальны; "
        "полные checkpoints каждые125 updates и финальный подтверждаются в приватном HF.  "
        "Python 3.13.x разрешён; runtime не объявляется "
        "совпавшим с Kaggle. Никаких обещаний побитовой инициализации.\n\n"
        "После обрыва извлеките тот же ZIP в тот же WORK_DIR и установите venv заново. "
        "Resume: явно скачанный проверенный HF snapshot/latest.json и НОВЫЙ RESULT_DIR. "
        "Строгая проверка trainer может отказать при несовместимости. "
        "Не подставляйте Kaggle/SMALL веса. Никаких anti-idle/reconnect скриптов."
    )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(True)}]
    for code in (PREFLIGHT, bootstrap, setup_cell, run_cell):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(True),
            }
        )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
    }


def frozen_inputs(series):
    state = json.loads(series.read_text(encoding="utf-8"))
    scientific_configs(state)
    identity = state["identity"]
    data = identity["data"]
    if data["fingerprint"] != digest({key: data[key] for key in ("spec", "files")}):
        raise ValueError("data fingerprint mismatch")
    if identity["fingerprint"] != digest(
        {key: identity[key] for key in ("source", "configs", "data")}
    ):
        raise ValueError("series fingerprint mismatch")
    bundle = series.parent / "release/release.bundle"
    checked(bundle, state["artifacts"]["release/release.bundle"])
    if checksum(bundle) != RELEASE_SHA256 or digest(identity["source"]) != SOURCE_SHA256:
        raise ValueError("pinned release/source differs; no live rebuild is permitted")
    if checksum(bundle) != state["archive_sha256"]:
        raise ValueError("frozen archive integrity mismatch")
    checked(HELPER, {"bytes": HELPER.stat().st_size, "sha256": HELPER_SHA256})
    if set(identity["data"]["files"]) != DATA_FILES:
        raise ValueError("expected exactly seven canonical data files")
    for name, details in identity["data"]["files"].items():
        checked(series.parent / "data" / name, details)
        if state["artifacts"]["data/" + name] != details:
            raise ValueError("frozen data manifests disagree")
    with zipfile.ZipFile(bundle) as frozen:
        if zip_members(frozen) != RELEASE_FILES:
            raise ValueError("unexpected/secret frozen release file")
        release_manifest = json.loads(frozen.read("release-manifest.json"))
        if set(release_manifest) != RELEASE_FILES - {"release-manifest.json"}:
            raise ValueError("unexpected frozen release manifest")
        release = {name: frozen.read(name) for name in RELEASE_FILES}
    for name, details in release_manifest.items():
        if blob_record(release[name]) != details:
            raise ValueError("frozen release integrity mismatch: " + name)
    return state, release


def export(series, output_dir):
    series, output_dir = Path(series), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("export output-dir must be new")
    state, release = frozen_inputs(series)
    identity = state["identity"]
    tools = {"tools/" + name: (ROOT / "scripts" / name).read_bytes() for name in TOOLS}
    tools["support/controls_verifier.py"] = HELPER.read_bytes()
    blobs = {**{"release/" + name: data for name, data in release.items()}, **tools}
    files = {name: blob_record(data) for name, data in blobs.items()}
    files.update({"data/" + name: value for name, value in identity["data"]["files"].items()})
    manifest = {
        "schema_version": 3,
        "storage": "hf",
        "periodic_full_checkpoints": True,
        "source_sha256": digest(identity["source"]),
        "series_fingerprint": identity["fingerprint"],
        "original_release_sha256": RELEASE_SHA256,
        "data_fingerprint": identity["data"]["fingerprint"],
        "dataset_spec": identity["data"]["spec"],
        "configs": scientific_configs(state),
        "default_family": "deltanet",
        "files": files,
        "kaggle_reference": REFERENCE,
        "matched_runtime_claimed": False,
        "tracking": "clearml",
        "hf_upload": True,
        "expected_updates": {"classic": 43117, "deltanet": 47265},
    }
    return write_package(series, output_dir, blobs, manifest)


def write_package(series, output_dir, blobs, manifest):
    files = manifest["files"]
    partial = output_dir.with_name(output_dir.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        archive = partial / "colab-package.zip"
        with zipfile.ZipFile(
            archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as target:
            for name, data in sorted(blobs.items()):
                target.writestr(name, data)
            for name in sorted(DATA_FILES):
                target.write(series.parent / "data" / name, "data/" + name)
            target.writestr("manifest.json", json.dumps(manifest, indent=2))
        # Verify copied bytes, not merely their pre-copy hashes; bounded memory for large arrays.
        with zipfile.ZipFile(archive) as copied:
            for name, details in files.items():
                with copied.open(name) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if (
                    actual != details["sha256"]
                    or copied.getinfo(name).file_size != details["bytes"]
                ):
                    raise ValueError("export integrity mismatch: " + name)
        sha = checksum(archive)
        (partial / "launch_colab.ipynb").write_text(
            json.dumps(notebook(sha), indent=2), encoding="utf-8"
        )
        (partial / "colab.env.example").write_text(
            "".join(key + "=\n" for key in KEYS), encoding="utf-8"
        )
        (partial / "export-report.json").write_text(
            json.dumps(
                {
                    "archive_sha256": sha,
                    "archive_bytes": archive.stat().st_size,
                    "files": len(files) + 1,
                    "source_sha256": manifest["source_sha256"],
                    "training_started": False,
                },
                indent=2,
            )
        )
        partial.rename(output_dir)
    except Exception:
        shutil.rmtree(partial)
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = export(args.series, args.output_dir)
    print((output / "export-report.json").read_text())


if __name__ == "__main__":
    main()
