"""Fresh service absence child; mode and suppression precede SDK authentication/imports."""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mqar_final_tail_native import require


def read_absence(nonce, values):
    from clearml import Task
    from huggingface_hub import HfApi

    require(
        not Task.get_tasks(
            project_name="compare-attention",
            task_name="^mqar202-final10780-tail-recovery-" + nonce + "$",
        ),
        "TAIL_CLEARML_EXISTS",
    )
    api = HfApi(token=values["HF_TOKEN"])
    info = api.repo_info(values["HF_REPO"], repo_type="model", token=values["HF_TOKEN"])
    require(info.private is True, "TAIL_HF_PRIVATE")
    files = api.list_repo_files(
        values["HF_REPO"], repo_type="model", revision=info.sha, token=values["HF_TOKEN"]
    )
    require(
        not any(n.startswith("mqar202-final10780-tail-recovery/" + nonce + "/") for n in files),
        "TAIL_HF_EXISTS",
    )
    return {"status": "ABSENT_FRESH_TAIL_SERVICES", "nonce": nonce}


def child(private, science, nonce):
    logging.disable(logging.CRITICAL)
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        os.environ.update(
            CLEARML_OFFLINE_MODE="0", HF_HUB_OFFLINE="0", HF_HUB_DISABLE_TELEMETRY="1"
        )
        sys.path[:0] = [
            str(Path(science) / p) for p in ("operator", "src", "operator/inherited", "tools")
        ]
        from quality_inherit import leaf

        service = leaf("services")
        values = service.credentials(private, rank=0)
        with service.service_env(values, "track"):
            result = read_absence(nonce, values)
    print(json.dumps(result, sort_keys=True))


def absence(private, science, nonce):
    # Preserve OS essentials; fresh child imports no native parser or tracker first.
    keep = {
        "SYSTEMROOT",
        "PATH",
        "HOME",
        "USERPROFILE",
        "USERNAME",
        "USER",
        "LOGNAME",
        "TEMP",
        "TMP",
        "TMPDIR",
        "UV_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(
        CLEARML_CONFIG_FILE=str(Path(os.environ["TMPDIR"]) / "tail-no-host-config.conf"),
        CLEARML_CACHE_DIR=str(Path(os.environ["TMPDIR"]) / "tail-service-cache"),
        PYTHONDONTWRITEBYTECODE="1",
        CLEARML_OFFLINE_MODE="0",
        HF_HUB_OFFLINE="0",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
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
        str(Path(__file__).resolve()),
        str(private),
        str(science),
        nonce,
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=100, check=False
    )
    require(result.returncode == 0, "TAIL_SERVICE_READBACK_CHILD_FAILED")
    value = json.loads(result.stdout)
    require(
        value == {"status": "ABSENT_FRESH_TAIL_SERVICES", "nonce": nonce},
        "TAIL_SERVICE_READBACK_SCHEMA",
    )
    return value


if __name__ == "__main__":
    try:
        child(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    except Exception:
        raise SystemExit(1) from None
