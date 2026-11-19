"""Real Kaggle SDK HTTP encoding; synthetic authority never grants real permission."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_two_stage_saved_host_publication_all_mounted_bytes(tmp_path, monkeypatch):
    assert importlib.util.find_spec("mqar_final_tail_host_cli") is not None, (
        "complete MQAR tail-specific public host bridge is missing"
    )
    from mqar_final_tail_test_support import World

    world = World(tmp_path, monkeypatch)
    for action in ("publish", "submit"):
        receipt, approval = world.prepare(action)
        assert world.command(action, action, receipt=receipt, approval=approval) == 0, world.errors
    published = world.datasets[world.plan["dataset_target"][:-2]]
    assert set(published) == set(world.plan["dataset_files"])
    assert "execution-decision.json" in published
    assert {r["numeric_id"] for r in world.plan["predecessors"]} == {145778849, 145850896}
    assert [method for method, _ in world.writes] == ["CreateDataset", "SaveKernel"]
    assert "StartBlobUpload" in world.serialized
    assert world.plan["optimizer_updates_permitted"] == 0
