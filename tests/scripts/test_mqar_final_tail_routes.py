"""Whole public tail host route; synthetic service/provider state, no cloud mutation."""

from pathlib import Path

import pytest
from mqar_final_tail_test_support import World


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_real_sdk_presend_serializes_bearer_and_dto(world):
    from mqar_final_tail_host_contract import SCIENCE

    world.provider.dataset(SCIENCE)
    assert world.serialized == ["GetDataset"]


def test_saved_cli_two_stage_seal_sdk_and_fresh_readback(world):
    for action in ("publish", "submit"):
        receipt, approval = world.prepare(action)
        assert world.command(action, action, receipt=receipt, approval=approval) == 0, world.errors
        journal = Path(world.plan["journals"][action])
        assert (journal / "acknowledged.json").is_file()
        assert world.command(action, action, receipt=receipt, approval=approval) == 1
    assert [m for m, _ in world.writes] == ["CreateDataset", "SaveKernel"]
    requested = [r.dataset_data_sources for _, m, r in world.requests if m == "SaveKernel"]
    assert requested == [world.plan["requested_versioned_inputs"]]
    methods = {m for _, m, _ in world.requests}
    assert {
        "DownloadDataset",
        "StartBlobUpload",
        "CreateDataset",
        "SaveKernel",
        "GetKernelSessionStatus",
        "GetAcceleratorQuotaStatistics",
    } <= methods
    assert {
        r.file_name
        for _, m, r in world.requests
        if m == "DownloadDataset" and "ca-kv" in r.dataset_slug
    } == set(world.plan["data_files"])


@pytest.mark.parametrize("lost", ["publish", "submit"])
def test_unknown_ack_get_only_no_repeat_and_same_authority(world, lost):
    for action in ("publish", "submit"):
        receipt, approval = world.prepare(action)
        world.lost_ack = lost
        code = world.command(action, action, receipt=receipt, approval=approval)
        if action == lost:
            assert code == 1
            before = list(world.writes)
            assert world.command(action, action, receipt=receipt, approval=approval) == 1
            assert (
                world.command("prepare", action, output=world.root / "forbidden-receipt.json") == 1
            )
            assert world.command("reconcile", action) == 0
            assert world.writes == before
            break
        assert code == 0


def test_fsync_failure_inhibits_before_mutation(world):
    receipt, approval = world.prepare("publish")
    world.fail_fsync = True
    assert world.command("publish", receipt=receipt, approval=approval) == 1
    assert world.writes == []


def test_proof_fsync_crash_leaves_intent_and_blocks_all_writes_before_auth(world, monkeypatch):
    import os

    receipt, approval = world.prepare("publish")
    target = Path(world.plan["journals"]["publish"]) / "fresh-under-lock.json"
    real = os.fsync
    fired = []

    def fsync(fd):
        if target.exists() and not fired:
            a, b = os.fstat(fd), target.stat()
            if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
                fired.append(True)
                raise OSError("SYNTHETIC_DURABILITY_CRASH")
        return real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    assert world.command("publish", receipt=receipt, approval=approval) == 1
    assert fired and world.writes == []
    assert (target.parent / "intent.json").is_file()
    import seed202_pilot.provider as provider

    auth = []
    monkeypatch.setattr(provider, "inherited_provider_class", lambda *_: auth.append(True))
    assert world.command("prepare", "submit", output=world.root / "escape.json") == 1
    assert world.command("publish", receipt=receipt, approval=approval) == 1
    assert auth == [] and world.writes == []
