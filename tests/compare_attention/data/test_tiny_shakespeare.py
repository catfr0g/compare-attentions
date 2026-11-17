"""Character LM preparation preserves raw offsets and same-position targets."""

import hashlib
import importlib
import json

import numpy as np
import pytest

from compare_attention.data.store import load_config, load_split


def preparation():
    return importlib.import_module(
        "compare_attention.data.tiny_shakespeare"
    ).prepare_tiny_shakespeare


def test_prepare_has_disjoint_splits_and_exact_next_character_targets(tmp_path):
    # Full raw-offset/array oracle keeps each boundary visible together.
    # pylint: disable=too-many-locals
    raw = ("ab\nc" * 50).encode()
    source = tmp_path / "input.txt"
    source.write_bytes(raw)
    output = tmp_path / "prepared"
    preparation()(source, output, context=8, expected_sha256=hashlib.sha256(raw).hexdigest())
    config = load_config(output)
    assert config.name == "tiny_shakespeare_char"
    assert not hasattr(config, "num_kv_pairs")
    meta = json.loads((output / "meta.json").read_text())
    assert meta["alphabet"] == sorted(set(raw.decode()))
    assert meta["source_sha256"] == hashlib.sha256(raw).hexdigest()
    bounds = [(0, 180), (180, 190), (190, 200)]
    for split, (start, end) in zip(("train", "val", "test"), bounds, strict=True):
        arrays = load_split(output, split)
        assert arrays.inputs.dtype == arrays.labels.dtype == np.int64
        assert arrays.inputs.shape == arrays.labels.shape
        labels = arrays.labels[arrays.labels != -1]
        decoded = "".join(meta["alphabet"][i] for i in labels)
        assert decoded == raw.decode()[start + 1 : end]
        assert len(labels) == end - start - 1
        for row, (x, y) in enumerate(zip(arrays.inputs, arrays.labels, strict=True)):
            valid = y != -1
            origin = start + row * 8
            assert (
                "".join(meta["alphabet"][i] for i in x[valid])
                == raw.decode()[origin : origin + valid.sum()]
            )
        assert meta["splits"][split]["raw_range"] == [start, end]
    with pytest.raises(FileExistsError):
        preparation()(source, output, context=8, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_source_hash_rejected_before_write(tmp_path):
    source = tmp_path / "input.txt"
    source.write_bytes(b"abc" * 100)
    with pytest.raises(ValueError, match="SHA256"):
        preparation()(source, tmp_path / "out", expected_sha256="0" * 64)
    assert not (tmp_path / "out").exists()


def test_heldout_alphabet_requires_training_coverage(tmp_path):
    source = tmp_path / "input.txt"
    raw = b"a" * 95 + b"z" * 5
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="alphabet"):
        preparation()(source, tmp_path / "out", expected_sha256=hashlib.sha256(raw).hexdigest())
    assert not (tmp_path / "out").exists()


def test_character_loader_rejects_changed_arrays(tmp_path):
    source = tmp_path / "input.txt"
    raw = b"ab\nc" * 50
    source.write_bytes(raw)
    output = tmp_path / "data"
    preparation()(source, output, context=8, expected_sha256=hashlib.sha256(raw).hexdigest())
    labels = np.load(output / "test_labels.npy", allow_pickle=False)
    labels[0, 0] = (labels[0, 0] + 1) % 4
    np.save(output / "test_labels.npy", labels, allow_pickle=False)
    with pytest.raises(ValueError, match="integrity"):
        load_config(output)
