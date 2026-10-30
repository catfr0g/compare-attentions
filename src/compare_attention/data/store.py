"""Запись и чтение подготовленного kv_retrieval."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from compare_attention.data.config import (
    DATASET_NAME,
    FORMAT_VERSION,
    IGNORE_INDEX,
    META_NAME,
    SOURCE,
    KVRetrievalConfig,
)
from compare_attention.data.records import KVRetrievalDataset, PrepareResult, TokenSplit
from compare_attention.data.sampling import generate_dataset

logger = logging.getLogger(__name__)


def _meta_dict(config: KVRetrievalConfig) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "name": DATASET_NAME,
        "source": SOURCE,
        "ignore_index": IGNORE_INDEX,
        "dtype": "int64",
    }
    meta.update(config.model_dump())
    return meta


def _read_meta(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / META_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_meta(meta: dict[str, Any]) -> KVRetrievalConfig:
    return KVRetrievalConfig.model_validate(meta)


def _same_params(meta: dict[str, Any], config: KVRetrievalConfig) -> bool:
    header_ok = (
        meta.get("format_version") == FORMAT_VERSION
        and meta.get("name") == DATASET_NAME
        and meta.get("ignore_index") == IGNORE_INDEX
    )
    if not header_ok:
        return False
    try:
        return _config_from_meta(meta) == config
    except (KeyError, ValueError):
        return False


def _array_matches(path: Path, shape: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    array = np.load(path, mmap_mode="r")
    matches = array.shape == shape and array.dtype == np.int64
    mapped = getattr(array, "_mmap", None)
    if mapped is not None:
        mapped.close()
    return matches


def _split_ready(output_dir: Path, split: str, shape: tuple[int, int]) -> bool:
    inputs_ready = _array_matches(output_dir / f"{split}_inputs.npy", shape)
    labels_ready = _array_matches(output_dir / f"{split}_labels.npy", shape)
    return inputs_ready and labels_ready


def _is_complete(output_dir: Path, config: KVRetrievalConfig) -> bool:
    meta = _read_meta(output_dir)
    if meta is None or not _same_params(meta, config):
        return False
    counts = {"train": config.n_train, "val": config.n_val, "test": config.n_test}
    for split, count in counts.items():
        if not _split_ready(output_dir, split, (count, config.input_seq_len)):
            return False
    return True


def _publish(staging: Path, output_dir: Path, dataset: KVRetrievalDataset) -> None:
    arrays = {
        "train_inputs.npy": dataset.train.inputs,
        "train_labels.npy": dataset.train.labels,
        "val_inputs.npy": dataset.validation.inputs,
        "val_labels.npy": dataset.validation.labels,
        "test_inputs.npy": dataset.test.inputs,
        "test_labels.npy": dataset.test.labels,
    }
    for name, array in arrays.items():
        np.save(staging / name, array)
    for name in arrays:
        (staging / name).replace(output_dir / name)
    payload = json.dumps(_meta_dict(dataset.config), indent=2)
    meta_path = staging / META_NAME
    meta_path.write_text(f"{payload}\n", encoding="utf-8", newline="\n")
    meta_path.replace(output_dir / META_NAME)


def _write_dataset(output_dir: Path, dataset: KVRetrievalDataset) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        _publish(staging, output_dir, dataset)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)


def _status(config: KVRetrievalConfig) -> str:
    return (
        f"preparing {DATASET_NAME}: "
        f"train={config.n_train} val={config.n_val} test={config.n_test} "
        f"seq_len={config.input_seq_len} "
        f"pairs={config.num_kv_pairs} vocab={config.vocab_size} seed={config.seed}"
    )


def prepare_kv_retrieval(
    config: KVRetrievalConfig,
    output_dir: Path | str,
    force: bool = False,
) -> PrepareResult:
    """Записать train/val/test в ``output_dir``."""
    destination = Path(output_dir)
    existing = _read_meta(destination)
    if not force and _is_complete(destination, config):
        logger.info("%s already prepared: %s", DATASET_NAME, destination)
        return PrepareResult(output_dir=destination, written=False)
    if existing is not None and not _same_params(existing, config) and not force:
        raise FileExistsError(
            f"{destination} already contains {DATASET_NAME} with different parameters. "
            "Pass --force to overwrite.",
        )
    logger.info(_status(config))
    _write_dataset(destination, generate_dataset(config))
    logger.info("%s written: %s", DATASET_NAME, destination)
    return PrepareResult(output_dir=destination, written=True)


def load_config(output_dir: Path | str) -> KVRetrievalConfig:
    """Прочитать параметры из ``meta.json``."""
    destination = Path(output_dir)
    meta = _read_meta(destination)
    if meta is None:
        raise FileNotFoundError(destination / META_NAME)
    parsed = _config_from_meta(meta)
    if not _same_params(meta, parsed):
        meta_path = destination / META_NAME
        raise ValueError(f"{meta_path} is not a {DATASET_NAME} dataset")
    return parsed


def load_split(output_dir: Path | str, split: str, *, mmap: bool = False) -> TokenSplit:
    """Загрузить ``train``, ``val`` или ``test``. ``mmap=True`` не читает массивы целиком."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    destination = Path(output_dir)
    mode = "r" if mmap else None
    inputs = np.load(destination / f"{split}_inputs.npy", mmap_mode=mode)
    labels = np.load(destination / f"{split}_labels.npy", mmap_mode=mode)
    return TokenSplit(inputs=inputs, labels=labels)
