"""Finite remaining-six source selection preserves both accepted historical profiles."""

from pathlib import Path

import pytest

PROFILE = "REMAINING_SIX_ELASTIC_FIXED_V1"
SOURCE = "115df61b5390f324fd06fc81bb091d04a0bce74dfa8587be5aa560b01ed27e2e"


@pytest.fixture(autouse=True)
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))


def test_remaining_six_registers_existing_fixed_source():
    from normal_remaining_scope import CASE_IDS
    from unified_normal_profiles import FRESH_RESTART, REMAINING, is_remaining_profile, selected

    assert is_remaining_profile(PROFILE)
    assert selected(PROFILE) == selected(FRESH_RESTART)
    assert selected(PROFILE)["source"] == SOURCE
    assert selected(REMAINING)["source"] != SOURCE
    assert len(CASE_IDS[2:]) == 6


@pytest.mark.parametrize("case_id", ["sorting-classic-seed303", "sorting-deltanet-seed303"])
def test_remaining_six_rejects_completed_cases_before_authorization(case_id):
    from unified_normal_generation import intent_value

    with pytest.raises(ValueError, match="REMAINING_SIX_CASE_SCOPE"):
        intent_value(
            {
                "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
                "nonce": "123456789abcdef0123456789abcdef0",
                "case_id": case_id,
                "source_identity_sha256": SOURCE,
                "authorization": {},
            },
            PROFILE,
        )


def test_remaining_six_cannot_reuse_old_plan_pins():
    from normal_remaining_scope import CASE_IDS
    from unified_normal_profiles import REMAINING
    from unified_normal_remaining import PLAN_PINS, SIX_PLAN_PINS, require_plan_pin

    assert tuple(SIX_PLAN_PINS) == CASE_IDS[2:]
    for case_id in CASE_IDS[2:]:
        require_plan_pin(case_id, SIX_PLAN_PINS[case_id], PROFILE)
        with pytest.raises(ValueError, match="REMAINING_SIX_FROZEN_PLAN"):
            require_plan_pin(case_id, PLAN_PINS[case_id], PROFILE)
    for case_id, pin in PLAN_PINS.items():
        require_plan_pin(case_id, pin, REMAINING)


def test_remaining_six_rejects_wrong_source_before_authorization():
    from unified_normal_generation import intent_value

    with pytest.raises(ValueError, match="^UNIFIED_GENERATION_SCOPE$"):
        intent_value(
            {
                "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
                "nonce": "123456789abcdef0123456789abcdef0",
                "case_id": "mqar-k64-classic-seed303",
                "source_identity_sha256": "0" * 64,
                "authorization": {"path": "MUST_NOT_READ", "sha256": "0" * 64},
            },
            PROFILE,
        )


def test_remaining_six_retains_exact_prefix_order():
    from normal_remaining_scope import CASE_IDS, require_prefix

    assert require_prefix(CASE_IDS[2], list(CASE_IDS[:2])) == 2
    with pytest.raises(ValueError, match="^REMAINING_ACCEPTED_PREFIX$"):
        require_prefix(CASE_IDS[2], list(reversed(CASE_IDS[:2])))
