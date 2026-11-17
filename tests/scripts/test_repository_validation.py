"""Portable developer gates must not mutate authenticated scientific inputs."""

import hashlib
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    "platform,cuda,compiler,available,expected",
    [
        ("win32", False, "cl", None, True),
        ("win32", False, "clang-cl", None, True),
        ("win32", False, "cl", "C:/toolchain/cl.exe", False),
        ("win32", True, "cl", None, False),
        ("linux", False, "cl", None, False),
    ],
)
def test_compiler_prerequisite_only_applies_to_unavailable_windows_cpu(
    monkeypatch, platform, cuda, compiler, available, expected
):
    from tests import conftest  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(conftest.shutil, "which", lambda name: available)
    assert conftest.missing_windows_cpu_compiler(platform, cuda, compiler) is expected


ROOT = Path(__file__).resolve().parents[2]


def test_hooks_are_offline_uv_checks_without_source_mutation():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    assert all(repo["repo"] == "local" for repo in config["repos"])
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    entries = [hook["entry"] for hook in hooks]
    assert all(entry.startswith("uv run --offline --no-sync ") for entry in entries)
    assert any("ruff check" in entry and "--fix" not in entry for entry in entries)
    assert any("ruff format --check" in entry for entry in entries)
    assert not any("--fix" in entry for entry in entries)


@pytest.mark.parametrize("task", ["sorting", "tracking"])
def test_ruff_exempt_snapshots_keep_all_declared_upstream_pins(task):
    import importlib  # pylint: disable=import-outside-toplevel

    module = importlib.import_module("compare_attention.data." + task)
    assert module.__file__ is not None
    source = Path(module.__file__).with_name(task + "_source")
    assert module.SOURCE_COMMIT == "30922c4100992a245a23103a790ad737ea8ede4c"
    for name, expected in module.SOURCE_HASHES.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected


def test_installed_hook_template_requires_uv_and_cpu_before_python():
    hook = (ROOT / ".githooks/pre-commit").read_text()
    invocation = hook.index("exec uv run --offline --no-sync python -B -m pre_commit")
    assert hook.index("export CUDA_VISIBLE_DEVICES=-1") < invocation
    assert hook.index("export PYTHONDONTWRITEBYTECODE=1") < invocation
    assert "--hook-type=pre-commit" in hook


@pytest.mark.parametrize(
    "relative",
    [
        "tests/compare_attention/cli/test_evaluate.py",
        "tests/compare_attention/models/test_lazy_backend.py",
        "tests/compare_attention/train/test_cuda_subprocess.py",
    ],
)
def test_python_subprocess_tests_use_explicit_offline_uv(relative):
    source = (ROOT / relative).read_text()
    for required in ('"uv"', '"--offline"', '"--no-sync"', '"-B"'):
        assert required in source
    assert "[sys.executable," not in source
    assert "\n        sys.executable,\n" not in source
