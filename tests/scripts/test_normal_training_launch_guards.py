"""Saved setup safety boundaries: CPU host observer versus unmasked target worker."""

import importlib.util
import json
import shutil
import sys

import build_normal_training_lifecycle as builder
import pytest
from normal_training_plan import digest

from tests.scripts.test_normal_training_closure import PARENT, REPO


def fixture(tmp_path):
    package = builder.build_package(PARENT, REPO, tmp_path / "normal")
    plan = tmp_path / "unapproved-production-metadata.json"
    plan.write_text(
        json.dumps(
            {
                "engineering_only": False,
                "source_identity_sha256": digest(package / "normal-source-identity.json"),
            }
        )
    )
    spec = importlib.util.spec_from_file_location("saved_normal_setup_guard", package / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = ["--uv", shutil.which("uv"), "--python", sys.executable, "--", "--plan", str(plan)]
    return module, args


def test_saved_target_setup_rejects_host_mask_without_spawning(tmp_path, monkeypatch):
    module, args = fixture(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU host mask was stripped before a target spawn")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="NORMAL_TARGET_MASK_FORBIDDEN"):
        module.main([*args, "--phase", "worker"])


def test_saved_production_host_observer_always_masks_cuda(tmp_path, monkeypatch):
    module, args = fixture(tmp_path)
    captured = {}

    class Child:
        def __init__(self, _command, **kwargs):
            captured.update(kwargs["env"])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def wait(self, **_):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    assert module.main([*args, "--phase", "accept"]) == 0
    assert captured.get("CUDA_VISIBLE_DEVICES") == "-1"
