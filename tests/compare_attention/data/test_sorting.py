"""Sorting fixed-pool source parity, independent oracle and typed metadata."""

import importlib.util


def test_sorting_source_collator_oracle_and_store(tmp_path):
    assert importlib.util.find_spec("compare_attention.data.sorting") is not None
    import numpy as np

    from compare_attention.data.sorting import prepare_sorting, verify_integrity
    from compare_attention.data.store import load_config

    path = tmp_path / "sorting"
    config = prepare_sorting(path, n_train=4, n_val=2, n_test=3)
    assert (config.input_seq_len, config.prompt_length, config.answer_length) == (128, 40, 40)
    assert config.vocab_size == 8
    verify_integrity(path, config)
    assert load_config(path) == config
    for split, count in (("train", 4), ("val", 2), ("test", 3)):
        x, y = (np.load(path / f"{split}_{k}.npy") for k in ("inputs", "labels"))
        assert x.shape == y.shape == (count, 128)
        assert np.all(y[:, :39] == -1) and np.all(y[:, 79:] == -1)
        assert np.all(y[:, 78] == 3) and np.all(x[:, 79:] == 0)
        assert np.all((y != -1).sum(1) == 40)


def test_sorting_strict_metadata_and_oracle_corruption(tmp_path):
    import numpy as np
    import pytest

    from compare_attention.data.sorting import SortingConfig, prepare_sorting, validate_arrays

    path = tmp_path / "data"
    c = prepare_sorting(path, n_train=2, n_val=1, n_test=3)
    raw = c.model_dump()
    for key, value in (
        ("n_train", True),
        ("n_items", 8.0),
        ("duplicates", 1),
        ("descending", True),
        ("input_seq_len", 79),
        ("source_commit", "wrong"),
    ):
        with pytest.raises(ValueError):
            SortingConfig.model_validate({**raw, key: value})
    for key in ("source_sha256", "split_seeds", "sha256"):
        with pytest.raises(ValueError):
            SortingConfig.model_validate({**raw, key: {}})
    x, y = (np.load(path / f"test_{k}.npy") for k in ("inputs", "labels"))
    for kind, position, value in (
        ("inputs", 127, 1),
        ("labels", 0, 4),
        ("labels", 78, 0),
        ("inputs", 40, 0),
    ):
        bad_x, bad_y = x.copy(), y.copy()
        (bad_x if kind == "inputs" else bad_y)[0, position] = value
        with pytest.raises(ValueError):
            validate_arrays(bad_x, bad_y, c)


def test_sorting_duplicate_multiplicity_oracle(tmp_path):
    import numpy as np

    from compare_attention.data.sorting import encode_items, oracle, parse_items, prepare_sorting

    c = prepare_sorting(tmp_path / "data", n_train=1, n_val=1, n_test=1)
    items = [(5, 4, 4, 4)] * 3 + [(4, 4, 4, 4)] * 5
    answer, parsed = oracle(encode_items(items, terminal=1), c)
    assert parsed == items
    assert parse_items(answer, c, answer=True) == sorted(items)
    assert np.count_nonzero(answer == 5) == 3
