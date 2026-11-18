"""Public parser -> genuine FULL -> unchanged native preflight, both layouts."""

import contextlib
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

DATA = Path(
    "E:/HermesArtifacts/compare-attention/fp16-scale128/0c6e680b88ee66c10846fbe577111c0819fe5fd9927ff591a3367528f0078cd3/data"
)
ROOT = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/mqar-classic-resume-repair-v1/mount-closure-development-01"
)


def populate(mount, slug, layout):
    physical = (mount / "datasets/zimike" if layout == "owner" else mount) / slug
    physical.mkdir(parents=True)
    for name in ("meta.json", "data-manifest.json"):
        shutil.copyfile(DATA / name, physical / name)
    return physical


@pytest.mark.parametrize("layout", ["flat", "owner"])
def test_genuine_worker_mount_full_and_native_preflight(
    tmp_path, monkeypatch, layout, record_property
):
    import torch

    quality_freeze = SimpleNamespace(
        WHEEL="compare_attention-0.1.0-py3-none-any.whl",
        WHEEL_SHA="512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3",
        verify_package=lambda _root: None,
    )
    monkeypatch.setitem(sys.modules, "quality_freeze", quality_freeze)
    import quality_inherit
    from seed202_mqar_mount_fixture import metadata_full
    from seed202_pilot import (
        cloud_contract,
        cloud_native,
        continuation,
        mqar_authority,
        mqar_package_guard,
        mqar_restore_gate,
        mqar_resume_entry,
        mqar_worker_paths,
    )
    from seed202_pilot.contract import MQAR_DATA

    from compare_attention.models import gpt2
    from compare_attention.train import checkpoint, distributed, distributed_checkpoint, loop

    mount = tmp_path / "input"
    physical = populate(mount, MQAR_DATA["source_ref"].split("/")[1], layout)
    original = ROOT / "genuine-resume/original"
    checkpoint_path = original / "out/checkpoints/step_00000980.pt"
    payload = checkpoint.load_checkpoint(checkpoint_path)
    before = copy.deepcopy(payload["config"])
    events = []
    genuine_data_check = cloud_contract._data

    def verified(spec, path):
        assert path == physical
        # Read-only filesystem transport of genuine bytes, no array copies/links.
        genuine_data_check(spec, DATA)

    monkeypatch.setattr(cloud_contract, "_data", verified)
    monkeypatch.setattr(mqar_worker_paths, "MOUNT_ROOT", mount)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("HOST_SCIENTIFIC_EXECUTION_PROHIBITED")

    for obj, name in (
        (gpt2, "GPT2"),
        (loop, "GPT2"),
        (torch.optim, "AdamW"),
        (torch.random, "fork_rng"),
        (checkpoint, "restore_state"),
        (checkpoint, "restore_rng"),
        (torch, "set_rng_state"),
    ):
        monkeypatch.setattr(obj, name, forbidden)
    actual_full = metadata_full(cloud_native.validate_full)

    def full(value, case, **kwargs):
        proof = actual_full(value, case, **kwargs)
        assert proof["updates"] == 980
        events.append("complete-exact-cloud-full-metadata")
        return proof

    monkeypatch.setattr(cloud_native, "validate_full", full)
    monkeypatch.setattr(cloud_native, "qualify_execution", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for key, value in (("RANK", "1"), ("LOCAL_RANK", "1"), ("WORLD_SIZE", "2")):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mqar_authority, "validate_args", lambda _args: {"files": {}})
    monkeypatch.setattr(mqar_package_guard, "install", lambda _science: None)
    runtime = SimpleNamespace(guard=lambda: None, runtime_attestation=lambda: {})
    monkeypatch.setattr(
        quality_inherit,
        "leaf",
        lambda name: (
            runtime if name == "runtime" else SimpleNamespace(extract_wheel=lambda *_args: {})
        ),
    )
    monkeypatch.setattr(quality_freeze, "verify_package", lambda _root: None)
    monkeypatch.setattr(continuation, "install_diagnostic", lambda _root: None)

    @contextlib.contextmanager
    def observe(value, _directory, rank):
        assert rank == 1 and value["config"] == before
        events.append("observe-restore-boundary-no-restore")
        yield

    monkeypatch.setattr(mqar_restore_gate, "observe_restore", observe)
    native_preflight = distributed_checkpoint.preflight

    def preflight(path, config):
        value = native_preflight(path, config)
        assert value["config"] == before
        events.append("unchanged-native-strict-preflight")
        return value

    monkeypatch.setattr(distributed_checkpoint, "preflight", preflight)

    class Run:
        world_size = 2
        device = SimpleNamespace(type="cuda")

        def batch(self, _batch):
            return 32

        def gather(self, value):
            return [value, value]

        def require(self, condition, _message):
            assert condition

        def call(self, function, **_kwargs):
            return function()

    @contextlib.contextmanager
    def distributed_metadata(_enabled):
        run = Run()
        token = distributed._ACTIVE.set(run)
        try:
            yield run
        finally:
            distributed._ACTIVE.reset(token)

    monkeypatch.setattr(distributed, "distributed_run", distributed_metadata)
    actual_split = loop.load_split

    def split(path, selected, **kwargs):
        assert Path(path) == physical
        events.append("physical-split-" + selected)
        return actual_split(DATA, selected, mmap=True, **kwargs)

    monkeypatch.setattr(loop, "load_split", split)

    class ReachedBoundary(Exception):
        pass

    def seed_boundary(_seed):
        events.append("before-seed-model-optimizer-rng-execution")
        raise ReachedBoundary

    monkeypatch.setattr(loop, "_seed", seed_boundary)
    # Isolate only bind_wheel's freshness view; actual imports remain frozen.
    scope = {
        **mqar_resume_entry.main.__globals__,
        "sys": SimpleNamespace(path=sys.path, modules={}),
    }
    entry = FunctionType(
        mqar_resume_entry.main.__code__, scope, "main", mqar_resume_entry.main.__defaults__
    )
    candidate = Path(os.environ["MQAR_TEST_CANDIDATE"])
    values = {
        "release": candidate / "payload",
        "science": tmp_path / "science",
        "mqar": physical,
        "private": tmp_path / "UNREAD_PRIVATE",
        "output": tmp_path / "out",
        "case": "mqar-k64-classic-seed202",
        "nonce": "0" * 32,
        "resume-input": original,
        "phase": "worker",
        "authority-root": tmp_path / "authority",
        "evidence-sha256": "0" * 64,
        "authority-sha256": "0" * 64,
    }
    argv = [item for key, value in values.items() for item in ("--" + key, str(value))]
    previous = sys.path.copy()
    try:
        with pytest.raises(ReachedBoundary):
            entry(argv)
    finally:
        sys.path[:] = previous
    assert events == [
        "complete-exact-cloud-full-metadata",
        "observe-restore-boundary-no-restore",
        "unchanged-native-strict-preflight",
        "physical-split-train",
        "physical-split-val",
        "before-seed-model-optimizer-rng-execution",
    ]
    assert payload["config"] == before
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == mqar_worker_paths.FULL_SHA
    record_property(
        "mount_preflight",
        json.dumps(
            {
                "layout": layout,
                "events": events,
                "checkpoint_sha256": mqar_worker_paths.FULL_SHA,
                "actual_restore": False,
                "data_bytes": "GENUINE_HASH_VERIFIED_FILESYSTEM_TRANSPORT",
            }
        ),
    )
