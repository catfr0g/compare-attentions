"""Pinned permutation task: same-position right products and full vocabulary."""

# pylint: disable=import-outside-toplevel
import itertools

import numpy as np
import pytest


def test_prepare_matches_native_rng_and_noncommuting_oracle(tmp_path):
    # Independent point-action oracle and raw RNG evidence are deliberately kept together.
    # pylint: disable=too-many-locals
    from compare_attention.data.permutation import prepare_permutation, validate_arrays
    from compare_attention.data.store import load_config, load_split

    config = prepare_permutation(tmp_path / "s5", group="s5", n_train=32, n_val=3, n_test=5)
    assert config.vocab_size == 121
    assert config.input_seq_len == 32
    assert config == load_config(tmp_path / "s5")
    arrays = load_split(tmp_path / "s5", "train")
    rng = np.random.default_rng(config.split_seeds["train"])
    perms = list(itertools.permutations(range(5)))
    inverse = {p: i + 1 for i, p in enumerate(perms)}
    for x, y in zip(arrays.inputs, arrays.labels, strict=True):
        assert np.array_equal(x, 1 + rng.choice(np.arange(120), size=32, replace=True))
        for end in range(1, 33):
            # Independent point action: apply newest/rightmost move FIRST.
            result = []
            for point in range(5):
                for token in reversed(x[:end]):
                    point = perms[token - 1][point]
                result.append(point)
            assert y[end - 1] == inverse[tuple(result)]
    assert np.any(arrays.inputs[:, 1:] != arrays.labels[:, 1:])
    bad = arrays.labels.copy()
    bad[0, 1] = 0
    with pytest.raises(ValueError, match="oracle"):
        validate_arrays(arrays.inputs, bad, config)
    with pytest.raises(FileExistsError):
        prepare_permutation(tmp_path / "s5", group="s5")


def test_c5_sparse_classes_and_corruption(tmp_path):
    from compare_attention.data.permutation import prepare_permutation, validate_arrays
    from compare_attention.data.store import load_split

    config = prepare_permutation(
        tmp_path / "c5", group="c5", context=4, n_train=32, n_val=3, n_test=5
    )
    arrays = load_split(tmp_path / "c5", "train")
    assert config.allowed_token_ids == [1, 34, 65, 91, 97]
    assert np.isin(arrays.inputs, config.allowed_token_ids).all()
    assert np.isin(arrays.labels, config.allowed_token_ids).all()
    for token in (0, 2, 121):
        x = arrays.inputs.copy()
        x[0, 0] = token
        with pytest.raises(ValueError, match="factors"):
            validate_arrays(x, arrays.labels, config)
    for mutation in ("shift", "mask", "outside"):
        y = arrays.labels.copy()
        if mutation == "shift":
            y = np.roll(y, 1, axis=1)
        else:
            y[0, 1] = -1 if mutation == "mask" else 2
        with pytest.raises(ValueError, match="oracle"):
            validate_arrays(arrays.inputs, y, config)
    with pytest.raises(ValueError, match="distinct"):
        prepare_permutation(tmp_path / "impossible", group="c5", context=1, n_train=32)


def test_noncommuting_order_and_first_position():
    from compare_attention.data.permutation import (
        ALL_PERMUTATIONS,
        PERMUTATION_INDEX,
        encode_labels,
    )

    a, b = (1, 0, 2, 3, 4), (0, 2, 1, 3, 4)
    x = np.array([[1 + PERMUTATION_INDEX[a], 1 + PERMUTATION_INDEX[b]]])
    labels = encode_labels(x)
    assert labels[0, 0] == x[0, 0]
    assert tuple(ALL_PERMUTATIONS[labels[0, 1] - 1]) == tuple(a[b[j]] for j in range(5))
    assert labels[0, 1] != encode_labels(x[:, ::-1])[0, 1]
