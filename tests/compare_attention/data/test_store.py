# pylint: disable=redefined-outer-name
"""Запись kv_retrieval на диск."""

import json
from pathlib import Path

import numpy as np
import pytest

from compare_attention.data.config import (
    DATASET_NAME,
    FORMAT_VERSION,
    IGNORE_INDEX,
    SOURCE,
    KVRetrievalConfig,
)
from compare_attention.data.store import load_config, load_split, prepare_kv_retrieval
from tests.compare_attention.data.factories import config as _config

_SAVED = {"n_train": 3, "n_val": 2, "seed": 5}
_SPLITS = (("train", 3), ("val", 2), ("test", 2))
type _Prepared = tuple[KVRetrievalConfig, Path]


@pytest.fixture
def prepared(tmp_path: Path) -> _Prepared:
    saved = _config(**_SAVED)
    output = tmp_path / "kv_retrieval"
    prepare_kv_retrieval(saved, output)
    return saved, output


def test_prepare_marks_dataset_as_written(tmp_path: Path) -> None:
    saved = _config(**_SAVED)
    output = tmp_path / "kv_retrieval"

    result = prepare_kv_retrieval(saved, output)

    assert result.written


def test_same_config_is_not_written_again(prepared: _Prepared) -> None:
    saved, output = prepared

    result = prepare_kv_retrieval(saved, output)

    assert not result.written


def test_skipped_prepare_keeps_train_file(prepared: _Prepared) -> None:
    _, output = prepared
    modified_at = (output / "train_inputs.npy").stat().st_mtime_ns

    prepare_kv_retrieval(_config(**_SAVED), output)

    assert (output / "train_inputs.npy").stat().st_mtime_ns == modified_at


def test_meta_matches_saved_config(prepared: _Prepared) -> None:
    saved, output = prepared

    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))

    assert meta == {
        "format_version": FORMAT_VERSION,
        "name": DATASET_NAME,
        "source": SOURCE,
        "ignore_index": IGNORE_INDEX,
        "dtype": "int64",
        **saved.model_dump(),
    }


def test_load_config_returns_saved_config(prepared: _Prepared) -> None:
    saved, output = prepared

    loaded = load_config(output)

    assert loaded == saved


@pytest.mark.parametrize(("split", "rows"), _SPLITS)
@pytest.mark.parametrize("field", ("inputs", "labels"))
def test_saved_array_has_one_row_per_example(
    prepared: _Prepared,
    split: str,
    rows: int,
    field: str,
) -> None:
    saved, output = prepared

    loaded = load_split(output, split)

    assert getattr(loaded, field).shape == (rows, saved.input_seq_len)


def test_saved_inputs_are_int64(prepared: _Prepared) -> None:
    _, output = prepared

    loaded = load_split(output, "train")

    assert loaded.inputs.dtype == np.int64


def test_different_config_requires_force(prepared: _Prepared) -> None:
    _, output = prepared

    with pytest.raises(FileExistsError, match="--force"):
        prepare_kv_retrieval(_config(n_train=4), output)


def test_force_reports_that_dataset_was_replaced(prepared: _Prepared) -> None:
    _, output = prepared

    result = prepare_kv_retrieval(_config(n_train=4), output, force=True)

    assert result.written


def test_forced_config_is_what_load_returns(prepared: _Prepared) -> None:
    _, output = prepared
    replacement = _config(n_train=4)
    prepare_kv_retrieval(replacement, output, force=True)

    loaded = load_config(output)

    assert loaded == replacement


def test_unknown_split_is_rejected(prepared: _Prepared) -> None:
    _, output = prepared

    with pytest.raises(ValueError, match="train"):
        load_split(output, "holdout")
