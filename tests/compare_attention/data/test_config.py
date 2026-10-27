"""Параметры kv_retrieval."""

import pytest

from tests.compare_attention.data.factories import config as _config


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"input_seq_len": 15}, "even"),
        ({"vocab_size": 16}, "vocab_size"),
        ({"input_seq_len": 16, "num_kv_pairs": 5, "vocab_size": 64}, r"4 \* num_kv_pairs"),
        ({"power_a": 0.0}, "power_a"),
        ({"n_train": 0}, "n_train"),
        ({"n_val": 0}, "n_val"),
        ({"n_test": 0}, "n_test"),
        ({"seed": -1}, "seed"),
        ({"num_kv_pairs": 0}, "num_kv_pairs"),
    ],
)
def test_invalid_config_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)
