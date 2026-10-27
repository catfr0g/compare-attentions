"""Параметры GPT-2."""

import pytest

from tests.compare_attention.models.factories import gpt2_config


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"block_size": 0}, "block_size"),
        ({"n_layer": 0}, "n_layer"),
        ({"n_head": 0}, "n_head"),
        ({"n_embd": 0}, "n_embd"),
        ({"dropout": -0.1}, "dropout"),
        ({"attention": "mamba"}, "attention"),
        ({"n_embd": 7, "n_head": 2}, "n_head"),
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gpt2_config(**overrides)
