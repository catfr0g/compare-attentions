"""Pinned Kaggle SDK persists absolute paths as one filename component."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_long_sdk_upload_state_component_rejected_before_folder_or_credentials(tmp_path):
    from mqar_final_tail_fixture_plan import fixture_plan
    from mqar_final_tail_host_contract import bind
    from mqar_final_tail_host_prepare import prepare_local
    from mqar_final_tail_test_layout import ART, DATA

    prior = fixture_plan(tmp_path)
    journal = ART / "ENGINEERING-long-path" / ("x" * 150) / "journals"
    plan = bind(prior["candidate_root"], journal, prior["decision_path"], DATA)
    with pytest.raises(ValueError, match="SDK_UPLOAD_STATE_FILENAME_LIMIT"):
        prepare_local(plan)
