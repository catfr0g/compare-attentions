"""Canonical qualified recovery cannot be a permissive ordinary-training shortcut."""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_whole_canonical_acceptance_replay_rejects_extra_fields(tmp_path):
    assert importlib.util.find_spec("mqar_final_tail_composite") is not None, (
        "whole MQAR composite missing"
    )
    from mqar_final_tail_composite import persist

    path = tmp_path / "engineering.json"
    value = {"schema": "ENGINEERING_ONLY", "later_pilots_authorized": False, "native": {"a": 1}}
    assert persist(path, value) == value
    assert persist(path, value) == value
    with pytest.raises(ValueError):
        persist(path, {**value, "unexpected_acceptance_key": True})


def test_no_qualified_review_no_acceptance(tmp_path):
    assert importlib.util.find_spec("mqar_final_tail_composite") is not None, (
        "whole MQAR composite missing"
    )
    from mqar_final_tail_composite import require_review

    with pytest.raises(ValueError):
        require_review({}, "0" * 64)
