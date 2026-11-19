"""Fresh UV process isolates online SDK constants from offline parent imports."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from mqar_final_tail_test_layout import CANDIDATE as OWN
from mqar_final_tail_test_layout import SCIENCE


def test_real_hf_sdk_unknown_ack_is_get_only(tmp_path):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="0",
        CLEARML_OFFLINE_MODE="0",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "scripts")
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
        str(Path(__file__).resolve().parents[2] / "scripts/mqar_final_tail_hf_unknown_child.py"),
        str(OWN / "payload/src"),
        str(SCIENCE),
        str(tmp_path),
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_SDK_HTTP_UNKNOWN_ACK_GET_ONLY_ENGINEERING_FIXTURE" in result.stdout
