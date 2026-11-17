"""Проверяемые immutable-артефакты Kaggle; только явный список файлов без secrets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_attention.cli.train import load_experiment
from compare_attention.data.store import load_config


def digest(payload: Any) -> str:
    """Стабильный идентификатор JSON."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def checksum(path: Path) -> str:
    """Потоковый SHA256 без загрузки массива целиком."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_manifest(root: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Размер и digest каждого разрешённого файла."""
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": checksum(path)}
        for path in sorted(paths)
    }


def verify_files(root: Path, files: dict[str, dict[str, Any]]) -> None:
    """Не доверять имени или metadata: проверить все bytes и запретить выход из root."""
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"missing/unsafe input: {name}")
        if path.stat().st_size != expected["bytes"] or checksum(path) != expected["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")


def data_manifest(root: Path) -> dict[str, Any]:
    """Полная metadata и шесть массивов, shape/dtype плюс checksums."""
    config = load_config(root)
    paths = [root / "meta.json"]
    for split in ("train", "val", "test"):
        for kind in ("inputs", "labels"):
            path = root / f"{split}_{kind}.npy"
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            valid = array.shape == (getattr(config, f"n_{split}"), config.input_seq_len)
            valid = valid and array.dtype == np.int64
            array._mmap.close()  # pylint: disable=protected-access
            if not valid:
                raise ValueError(f"invalid shape/dtype: {path.name}")
            paths.append(path)
    payload = {"spec": config.model_dump(), "files": file_manifest(root, paths)}
    return {**payload, "fingerprint": digest(payload)}


def snapshot(root: Path, configs: list[Path], data: Path) -> dict[str, Any]:
    """Fingerprint рабочего содержимого, lock, configs и полных подготовленных данных."""
    root = root.resolve()
    parsed = {}
    for path in configs:
        name = path.resolve().relative_to(root).as_posix()
        if name in parsed:
            raise ValueError(f"duplicate config: {name}")
        parsed[name] = load_experiment(path).model_dump(mode="json")
    if not parsed:
        raise ValueError("at least one config is required")
    prepared = data_manifest(data)
    for name, experiment in parsed.items():
        spec = dict(experiment["kv"])
        spec["seed"] = spec.pop("dataset_seed")
        spec["vocab_size"] = experiment["model"]["vocab_size"]
        if spec != prepared["spec"]:
            raise ValueError(f"prepared KV specification differs: {name}")
    paths = list((root / "src/compare_attention").rglob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "uv.lock", "README.md")]
    payload = {"source": file_manifest(root, paths), "configs": parsed, "data": prepared}
    return {**payload, "fingerprint": digest(payload)}
