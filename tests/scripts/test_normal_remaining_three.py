"""Finite release membership and qualified-prefix policy controls."""

import os
import subprocess
from pathlib import Path

import pytest
from normal_remaining_scope import CASE_IDS
from unified_normal_contract import Policy

from tests.scripts.test_normal_finish_import_roots import PACKAGE


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="authenticated deployed c7 package unavailable")
def test_future_release_only_changes_setup_and_identity(tmp_path):
    from build_normal_finish_import_release import build_release

    root = build_release(PACKAGE, Path(__file__).parents[2], tmp_path / "release")
    changed = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.read_bytes() != (PACKAGE / p.relative_to(root)).read_bytes()
    }
    assert changed == {"setup.py", "normal-source-identity.json"}
    with pytest.raises(ValueError, match="CREATE_ONLY"):
        build_release(PACKAGE, Path(__file__).parents[2], root)


def test_public_operator_distributes_sorting404_replay_imports(tmp_path):
    from unified_normal_contract import read
    from unified_normal_operator import build

    selected = os.environ.get("UNIFIED_HOST_COPY_INPUTS")
    if not selected:
        pytest.skip("genuine BOUND remaining-three builder inputs not selected")
    inputs = Path(selected)
    candidate = build(
        read(inputs), Path(os.environ["UNIFIED_HOST_COPY_ARCHIVE"]), tmp_path / "candidate"
    )
    host = candidate / "payload/source/src"
    # Isolated interpreter: neither repository scripts nor PYTHONPATH can hide
    # a missing emitted dependency, including the helper's transitive imports.
    code = (
        "import pathlib,sys;sys.path.insert(0,sys.argv[1]);"
        "import completed_sorting404_publication as helper;"
        "import completed_normal_publication as native;"
        "import unified_normal_profiles as profiles;"
        "root=pathlib.Path(sys.argv[1]).resolve();"
        "assert all(pathlib.Path(m.__file__).resolve().parent == root "
        "for m in (helper,native,profiles));"
        "assert callable(helper.replay_completed)"
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "--project",
            str(Path(__file__).resolve().parents[2]),
            "python",
            "-I",
            "-B",
            "-c",
            code,
            str(host),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_remaining_three_requires_two_truthful_qualified_predecessors():
    from unified_normal_profiles import REMAINING_THREE
    from unified_normal_remaining import prefix_policies

    rows = [
        {
            "case_id": case,
            "policy": (
                Policy.COMPLETED_NORMAL
                if i == 3
                else Policy.COMPLETED_SORTING404
                if i == 4
                else Policy.ORDINARY
            ),
        }
        for i, case in enumerate(CASE_IDS[:5])
    ]
    prefix_policies(rows, REMAINING_THREE)
    rows[4]["policy"] = Policy.ORDINARY
    with pytest.raises(ValueError, match="QUALIFIED_CASE4"):
        prefix_policies(rows, REMAINING_THREE)


@pytest.mark.parametrize("case", CASE_IDS[:5])
def test_new_profile_rejects_all_previous_cases(case):
    from unified_normal_generation import intent_value
    from unified_normal_profiles import REMAINING_THREE, selected

    with pytest.raises(ValueError, match="REMAINING_THREE_CASE_SCOPE"):
        intent_value(
            {
                "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
                "case_id": case,
                "nonce": "a" * 32,
                "source_identity_sha256": selected(REMAINING_THREE)["source"],
                "authorization": {},
            },
            REMAINING_THREE,
        )


def test_new_profile_plan_pins_do_not_relabel_previous_four():
    from unified_normal_profiles import REMAINING_THREE
    from unified_normal_remaining import FOUR_PLAN_PINS, THREE_PLAN_PINS, require_plan_pin

    assert set(THREE_PLAN_PINS) == set(CASE_IDS[5:])
    for case, pin in THREE_PLAN_PINS.items():
        require_plan_pin(case, pin, REMAINING_THREE)
        with pytest.raises(ValueError, match="REMAINING_THREE_FROZEN_PLAN"):
            require_plan_pin(case, FOUR_PLAN_PINS[case], REMAINING_THREE)


def test_historical_mqar303_whole_qualified_receipt_still_replays():
    from completed_normal_publication import digest, read, replay_completed

    publisher = Path(__file__).parents[2] / "scripts/completed_normal_publication.py"
    assert digest(publisher) == "3edf6c8492ce1a40adaf054377c517202bad9127fdc4be8f77b4638dbe3d0f8c"
    closure = PACKAGE.parents[1] / "completed-mqar-delta303-final"
    spec = {
        "subject": str(closure / "publication-subject.json"),
        "review": str(closure / "parent-technical-review.json"),
        "journal": str(closure / "hf-final.jsonl"),
    }
    receipt = read(closure / "qualified-acceptance.json")
    assert replay_completed(spec, receipt) == receipt
