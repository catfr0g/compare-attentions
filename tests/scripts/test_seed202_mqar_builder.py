"""Authenticated continuation parser staging; no scientific execution."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORK = Path("C:/Users/Michael/AppData/Local/hermes/cache/scratch/mqar202-repair-work-v1")
HISTORY = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/"
    "periodic-resume-repair-v1/candidate-05-final03/sorting-mqar-seed202-pilot-v1.zip"
)


def test_stage_reuses_authenticated_public_resume_parser(tmp_path):
    from build_seed202_mqar_continuation import stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    for name in ("mqar_resume_runner", "mqar_resume_entry", "mqar_resume_observer"):
        raw = (source / "seed202_pilot" / (name + ".py")).read_text(encoding="utf-8")
        ast.parse(raw)
        assert "if __name__" in raw
    runner = (source / "seed202_pilot/mqar_resume_runner.py").read_text(encoding="utf-8")
    assert '"--resume-input"' in runner
    assert '"seed202_pilot.mqar_resume_observer"' in runner
    entry = (source / "seed202_pilot/mqar_resume_entry.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume-input", type=Path, required=True)' in entry
    assert "seed202_pilot.mqar_resume_case" in entry
    assert (source / "seed202_pilot/cloud_runner.py").read_bytes() == (
        WORK / "parent/source/src/seed202_pilot/cloud_runner.py"
    ).read_bytes()
    worker = (source / "seed202_pilot/mqar_resume_case.py").read_text(encoding="utf-8")
    assert "resume_payload = load_resume(args.resume_input, case)" in worker
    assert "with observe_restore(resume_payload, root, rank):" in worker
    assert "resume_from=args.resume_input / CHECKPOINT" in worker
    assert 'normal_return_envelope(case["id"], result, service)' in worker
    assert "TrainingServiceTracker(" in worker
    assert 'install_diagnostic(root / "diagnostic")' in worker


@pytest.mark.parametrize("entry", ["mqar_resume_runner", "mqar_resume_entry"])
def test_public_parser_in_selected_uv_child_reaches_only_metadata_boundary(tmp_path, entry):
    from build_seed202_mqar_continuation import stage_source

    source = stage_source(WORK / "parent", HISTORY, tmp_path / "source")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "quality_inherit.py").write_text(
        "def leaf(name):\n"
        " class Boundary:\n"
        "  def guard(self):\n"
        "   print('ENGINEERING_METADATA_BOUNDARY_CPU_ONLY', flush=True)\n"
        "   raise SystemExit(73)\n"
        " return Boundary()\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(shim), str(source)]),
        "TMPDIR": str(tmp_path),
    }
    command = [
        shutil.which("uv"),
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        sys.executable,
        "python",
        "-B",
        "-m",
        "seed202_pilot." + entry,
    ]
    for name in ("release", "science", "mqar", "private", "output", "resume-input"):
        command += ["--" + name, str(tmp_path)]
    from test_seed202_mqar_composite import _engineering_authority

    authority = _engineering_authority(tmp_path)
    command += [
        "--case",
        "mqar-k64-classic-seed202",
        "--nonce",
        "4a" * 16,
        "--authority-root",
        str(tmp_path / "mounted-authority"),
        "--evidence-sha256",
        authority["evidence_sha256"],
        "--authority-sha256",
        authority["authority_sha256"],
    ]
    if entry.endswith("entry"):
        command += ["--phase", "worker"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 73, result.stderr
    assert result.stdout.strip() == "ENGINEERING_METADATA_BOUNDARY_CPU_ONLY"
    wrong = command.copy()
    wrong[wrong.index("mqar-k64-classic-seed202")] = "sorting-classic-seed202"
    rejected = subprocess.run(wrong, env=env, capture_output=True, text=True, check=False)
    assert rejected.returncode != 73
    assert "ENGINEERING_METADATA_BOUNDARY_CPU_ONLY" not in rejected.stdout
