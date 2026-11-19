"""Next-case qualifier is narrow; never grants launch authority."""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_next_requires_whole_replay_and_independent_transition_review():
    assert importlib.util.find_spec("mqar_final_tail_next") is not None, (
        "qualified MQAR next consumer missing"
    )
    from mqar_final_tail_next import review_transition

    with pytest.raises(ValueError):
        review_transition({}, "0" * 64, "1" * 64)


def test_existing_ordinary_mqar_acceptance_not_globally_weakened(tmp_path):
    assert importlib.util.find_spec("mqar_final_tail_next") is not None, (
        "qualified MQAR next consumer missing"
    )
    from mqar_final_tail_next import ordinary_is_unchanged
    from mqar_final_tail_test_layout import CANDIDATE

    assert ordinary_is_unchanged(CANDIDATE)
