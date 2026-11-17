"""Selective Count literal upstream semantics and immutable fixed pools."""

# Defer imports so the recorded RED cycle fails on a missing adapter assertion.
# pylint: disable=import-outside-toplevel

import importlib.util

import numpy as np
import pytest


def test_output_class_range_and_bad_metadata(tmp_path):
    from compare_attention.data.selective_count import SelectiveCountConfig, prepare_selective_count

    config = prepare_selective_count(tmp_path / "count", n_train=32, n_val=2, n_test=3)
    assert config.output_class_range == [33, 545]
    assert config.input_symbol_range == [1, 32]
    assert config.vocab_size == 546
    for field, value in (
        ("class_low", 32),
        ("max_count", 511),
        ("metric_normalizer", 16),
        ("vocab_size", 545),
    ):
        with pytest.raises(ValueError):
            SelectiveCountConfig.model_validate({**config.model_dump(), field: value})


def test_single_answer_query_included():
    assert importlib.util.find_spec("compare_attention.data.selective_count") is not None
    from compare_attention.data.selective_count import encode_labels

    x = np.array([[1, 2, 1, 2, 2], [2, 1, 1, 1, 1]], dtype=np.int64)
    expected = np.array([[-1, -1, -1, -1, 36], [-1, -1, -1, -1, 37]])
    np.testing.assert_array_equal(encode_labels(x, n_symbols=32), expected)


def test_prepare_and_corruption(tmp_path):
    from compare_attention.data.selective_count import prepare_selective_count, validate_arrays
    from compare_attention.data.store import load_config, load_split

    path = tmp_path / "count"
    config = prepare_selective_count(path, context=16, n_train=35, n_val=5, n_test=7)
    assert config.vocab_size == 50
    assert config.class_low == 33
    assert config.metric_normalizer == 1.5
    assert load_config(path) == config
    for split in ("train", "val", "test"):
        arrays = load_split(path, split)
        validate_arrays(arrays.inputs, arrays.labels, config)
        assert np.all(arrays.labels[:, :-1] == -1)
    with pytest.raises(FileExistsError):
        prepare_selective_count(path)
    arrays = load_split(path, "test")
    arrays.labels[0, -2] = arrays.labels[0, -1]
    with pytest.raises(ValueError, match="last"):
        validate_arrays(arrays.inputs, arrays.labels, config)
    arrays.labels[0, -2] = -1
    arrays.labels[0, -1] = 33
    with pytest.raises(ValueError, match="oracle"):
        validate_arrays(arrays.inputs, arrays.labels, config)
    with (path / "test_labels.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context": 0},
        {"context": True},
        {"n_symbols": 1},
        {"n_train": 0},
        {"n_val": 0},
        {"n_test": 0},
        {"seed": -1},
    ],
)
def test_invalid_generation_parameters_no_write(tmp_path, kwargs):
    from compare_attention.data.selective_count import prepare_selective_count

    with pytest.raises(ValueError):
        prepare_selective_count(tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad").exists()


def test_independent_deterministic_splits(tmp_path):
    from compare_attention.data.selective_count import prepare_selective_count

    first = prepare_selective_count(tmp_path / "a", context=16, n_train=32, n_val=3, n_test=5)
    second = prepare_selective_count(tmp_path / "b", context=16, n_train=32, n_val=3, n_test=5)
    assert first == second
    assert len(set(first.split_seeds.values())) == 3
