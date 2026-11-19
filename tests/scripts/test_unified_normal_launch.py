"""Genuine accepted predecessor and saved-source launch regressions (no auth)."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path("E:/HermesArtifacts/compare-attention/unified-normal-launch-contract-v1")
MQAR = Path("E:/HermesArtifacts/compare-attention/mq01")
NORMAL_SOURCE = "89b7e3ac5b42716e24d0d2a1994d13c32845e0c93a668f284e3cb341fe301cb8"
NORMAL_WHEEL = "a1e1db146ac0fef3583edbc6bee1d41ed1e83dffe85702c09667d34c96047e90"


@pytest.fixture
def candidate(monkeypatch):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    root = Path(os.environ.get("UNIFIED_NORMAL_CANDIDATE", str(ROOT / "c")))
    monkeypatch.syspath_prepend(str(root / "payload/source/src"))
    return root


def test_public_make_plan_genuine_qualified_mqar(candidate, tmp_path):
    from seed202_pilot.cli import _candidate
    from seed202_pilot.seal import singleton_plan

    _candidate(candidate)
    descriptor = json.loads((candidate / "payload/qualified-launch/bridge.json").read_bytes())
    sorting = descriptor["cases"]["sorting-deltanet-seed202"]["spec"]["output"]
    plan = singleton_plan(
        "mqar-k64-deltanet-seed202",
        candidate,
        tmp_path / "unsubmitted-journal",
        [Path(sorting), MQAR / "qualified-composite.json"],
    )
    assert plan["case_id"] == "mqar-k64-deltanet-seed202"
    assert not (tmp_path / "unsubmitted-journal").exists()
    assert plan["predecessors"][-1]["sha256"] == (
        "9a75cfe268660608e835582acf74f1a4438ccc1281e3143b07c6c84a3cd347ed"
    )


def test_public_next_replays_genuine_prefix_and_normal_source(candidate, capsys):
    from seed202_pilot.cli import main
    from unified_normal_contract import descriptor

    _path, request = descriptor(candidate)
    rows = request["predecessors"]
    assert (
        main(
            [
                "next",
                "--candidate-dir",
                str(candidate),
                "--qualified-predecessor",
                rows[0]["path"],
                "--acceptance",
                rows[1]["path"],
                "--acceptance",
                rows[2]["path"],
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value == {"status": "NEXT_CASE_VERIFIED", "case_id": request["run"]["id"]}


def test_next_source_validation_is_mandatory(candidate, monkeypatch):
    import unified_normal_source
    from unified_normal_contract import descriptor, next_qualified_case

    _path, request = descriptor(candidate)
    rows = request["predecessors"]

    def rejected(*_args):
        raise ValueError("UNIFIED_NEXT_SOURCE_REQUIRED")

    monkeypatch.setattr(unified_normal_source, "operator_profile", rejected)
    with pytest.raises(ValueError, match="UNIFIED_NEXT_SOURCE_REQUIRED"):
        next_qualified_case(candidate, rows[0]["path"], [r["path"] for r in rows[1:]])


def test_public_source_binding_selects_repaired_normal(candidate):
    from seed202_pilot.fresh_mode import binding

    plan = json.loads((candidate / "PLAN-UNAPPROVED.json").read_bytes())
    deployment = next(
        row for row in plan["deployments"] if row["case_id"] == "mqar-k64-deltanet-seed202"
    )
    bound = binding(candidate, deployment)
    assert bound["normal_plan_sha256"] == deployment["normal_plan_sha256"]
    assert bound["normal_runtime_sha256"] != "0" * 64
    assert bound.get("normal_source_identity_sha256") == NORMAL_SOURCE
    assert bound.get("normal_wheel_sha256") == NORMAL_WHEEL
    source = (candidate / "payload/deploy" / deployment["case_id"] / "kernel/kernel.py").read_text()
    assert '"setup.py"' in source and '"worker"' in source and '"finish"' in source
    assert "seed202_pilot.cloud_runner" not in source
