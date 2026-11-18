"""Explicit MQAR continuation dispatch; original host guards remain intact."""

import importlib
import sys

import pytest
from test_seed202_mqar_builder import HISTORY, WORK


def test_scoped_host_mode_and_action_sequence(tmp_path):
    from build_seed202_mqar_continuation import stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    old_modules = {k: v for k, v in sys.modules.items() if k.startswith("seed202_pilot")}
    for key in old_modules:
        del sys.modules[key]
    sys.path.insert(0, str(source))
    try:
        mode = importlib.import_module("seed202_pilot.fresh_mode")
        scoped = importlib.import_module("seed202_pilot.mqar_host_resume")
        deployment = {
            "mode": "MQAR_CLASSIC_STEP980_V1",
            "case_id": "mqar-k64-classic-seed202",
            "nonce": "4a2060304e6b47229b64b0647fc138d1",
            "dataset_target": "zimike/sorting-mqar-seed202-4a2060304e6b4722/1",
            "resume_input_target": "zimike/seed202-resume-input-4a2060304e6b4722/1",
            "kernel_target": "zimike/seed202-pilot-4a2060304e6b4722",
        }
        assert mode.is_two_stage(deployment)
        assert mode.actions({"deployment": deployment}) == ("publish", "publish-resume", "submit")
        assert scoped.scoped_targets(deployment)["resume"].endswith("4a2060304e6b4722")
        with pytest.raises(ValueError):
            scoped.scoped_targets({**deployment, "mode": "FRESH_TWO_STAGE"})
        legacy = importlib.import_module("seed202_pilot.host_resume")
        with pytest.raises(ValueError, match="FINITE_CONTINUATION_SCOPE"):
            legacy.scoped_targets(deployment)
    finally:
        sys.path.remove(str(source))
        for key in list(sys.modules):
            if key.startswith("seed202_pilot"):
                del sys.modules[key]
        sys.modules.update(old_modules)
