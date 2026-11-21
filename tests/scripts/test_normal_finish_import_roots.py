"""Exercise saved setup's selected UV child at the actual task_quality import."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from normal_training_plan import digest

PACKAGE = Path(
    "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/"
    "tracking-nonblocking-repair24/source-final/normal-package"
)
PROBE = """import importlib.abc
import json
import sys
from pathlib import Path

package = Path(sys.argv[1]).resolve()
class Reached(BaseException):
    pass
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "seed202_pilot.metrics":
            import seed202_pilot.contract as contract
            origin = Path(contract.__file__).resolve()
            assert origin == package / "inherited-source/seed202_pilot/contract.py"
            print(json.dumps({
                "boundary": "REAL_TASK_QUALITY_CONTRACT_IMPORTED", "origin": str(origin)
            }))
            raise Reached
        if fullname.split(".")[0] in {"torch", "clearml", "huggingface_hub", "compare_attention"}:
            raise AssertionError("FORBIDDEN_SCIENTIFIC_OR_SDK_IMPORT:" + fullname)
sys.meta_path.insert(0, Boundary())
from unified_normal_host import task_quality
try:
    task_quality(package, ".", {"engineering_only": False}, ".")
except Reached:
    pass
"""


def run_selected_child(tmp_path, monkeypatch, setup):
    """Keep actual saved setup, wheel extraction, UV argv and child environment."""
    spec = importlib.util.spec_from_file_location("finish_import_setup", setup)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = setup.parent
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "engineering_only": True,
                "source_identity_sha256": digest(package / "normal-source-identity.json"),
            }
        )
    )
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    original = subprocess.Popen
    captured = {}

    class ObservedChild:
        def __init__(self, process):
            self.process = process

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.process.__exit__(*args)

        def wait(self, timeout):
            stdout, stderr = self.process.communicate(timeout=timeout)
            captured.update(returncode=self.process.returncode, stdout=stdout, stderr=stderr)
            return self.process.returncode

    def child(argv, **kwargs):
        # Only select the import sentinel; never enter the scientific worker.
        assert argv[-1] == "finish"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        roots = kwargs["env"]["PYTHONPATH"].split(os.pathsep)
        assert str(Path(__file__).parents[2] / "scripts") not in roots
        command = argv[: argv.index(str(probe)) + 1] + [str(package)]
        process = original(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
        )
        return ObservedChild(process)

    monkeypatch.setattr(module.subprocess, "Popen", child)
    assert (
        module.main(
            [
                "--uv",
                shutil.which("uv"),
                "--python",
                sys.executable,
                "--engineering-bootstrap",
                str(probe),
                "--",
                "--plan",
                str(plan),
                "--phase",
                "finish",
            ]
        )
        == captured["returncode"]
    )
    return captured


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="authenticated deployed c7 package unavailable")
def test_actual_emitted_finish_roots_reach_first_transitive_contract(tmp_path, monkeypatch):
    import hashlib

    fixed = tmp_path / "fixed"
    shutil.copytree(PACKAGE, fixed)
    maintained = Path(__file__).parents[2] / "scripts/normal_training_launch.py"
    shutil.copyfile(maintained, fixed / "setup.py")
    identity_path = fixed / "normal-source-identity.json"
    identity = json.loads(identity_path.read_bytes())
    identity["files"]["setup.py"] = {
        "bytes": maintained.stat().st_size,
        "sha256": hashlib.sha256(maintained.read_bytes()).hexdigest(),
    }
    identity_path.write_text(json.dumps(identity))
    result = run_selected_child(tmp_path, monkeypatch, fixed / "setup.py")
    assert result["returncode"] == 0, result["stderr"]
    assert "REAL_TASK_QUALITY_CONTRACT_IMPORTED" in result["stdout"]


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="authenticated deployed c7 package unavailable")
def test_historical_deployed_roots_reproduce_missing_dependency(tmp_path, monkeypatch):
    result = run_selected_child(tmp_path, monkeypatch, PACKAGE / "setup.py")
    assert result["returncode"] != 0
    assert "ModuleNotFoundError: No module named 'seed202_pilot'" in result["stderr"]
    assert "task_quality" in result["stderr"]


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="authenticated deployed c7 package unavailable")
def test_genuine_future_builder_emits_repaired_selected_child(tmp_path, monkeypatch):
    from build_normal_finish_import_release import build_release

    fixed = build_release(PACKAGE, Path(__file__).parents[2], tmp_path / "fixed")
    result = run_selected_child(tmp_path, monkeypatch, fixed / "setup.py")
    assert result["returncode"] == 0, result["stderr"]
    assert "REAL_TASK_QUALITY_CONTRACT_IMPORTED" in result["stdout"]
