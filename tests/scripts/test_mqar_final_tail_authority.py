"""Exact mounted authority belongs to the new nonce and final publication."""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_authority_is_mandatory_and_closed():
    assert importlib.util.find_spec("mqar_final_tail_authority") is not None, (
        "mounted authority join missing"
    )
    from mqar_final_tail_authority import verify

    with pytest.raises(ValueError):
        verify({}, Path("unused"), "0" * 32, "unused", Path("unused"))
