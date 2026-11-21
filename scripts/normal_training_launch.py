"""Create-only wheel extraction and selected-UV NORMAL child; no installs or cloud APIs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from normal_driver_environment import validated_driver_environment
from normal_launch_diagnostics import NormalArgumentParser, normal_cli


def main(argv=None):
    parser = NormalArgumentParser(allow_abbrev=False)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--engineering-bootstrap", type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    tokens = args.worker_args
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    root = Path(__file__).resolve().parent
    identity_path = root / "normal-source-identity.json"
    identity = json.loads(identity_path.read_bytes())
    for name, pin in identity["files"].items():
        part = PurePosixPath(name)
        if part.is_absolute() or ".." in part.parts or "\\" in name or ":" in name:
            raise ValueError("NORMAL_SETUP_SOURCE_PATH")
        path = root / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.is_symlink() or path.stat().st_size != pin["bytes"] or actual != pin["sha256"]:
            raise ValueError("NORMAL_SETUP_SOURCE_BYTES")
    plan_path = Path(tokens[tokens.index("--plan") + 1])
    plan = json.loads(plan_path.read_bytes())
    if plan["source_identity_sha256"] != hashlib.sha256(identity_path.read_bytes()).hexdigest():
        raise ValueError("NORMAL_SETUP_NEW_SOURCE_BINDING")
    engineering = plan["engineering_only"] is True
    phase = tokens[tokens.index("--phase") + 1]
    host_observer = phase == "accept"
    if not engineering and not host_observer and "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("NORMAL_TARGET_MASK_FORBIDDEN")
    if args.engineering_bootstrap is not None and not engineering:
        raise ValueError("NORMAL_NO_PRODUCTION_BOOTSTRAP")
    if engineering and (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError("NORMAL_CPU_ENV_BEFORE_CHILD")
    if not args.uv.is_file() or not args.python.is_file():
        raise ValueError("NORMAL_EXPLICIT_SELECTED_UV_PYTHON")
    scratch = Path(os.environ["TMPDIR"])
    wheel_root = scratch / ("normal-wheel-" + uuid.uuid4().hex)
    wheel_root.mkdir(exist_ok=False)
    wheel = root / "compare_attention-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as saved:
        names = saved.namelist()
        if len(names) != len(set(names)):
            raise ValueError("NORMAL_WHEEL_DUPLICATE")
        for name in names:
            part = PurePosixPath(name)
            if part.is_absolute() or ".." in part.parts or "\\" in name or ":" in name:
                raise ValueError("NORMAL_WHEEL_UNSAFE")
        saved.extractall(wheel_root)
    allowed = {
        "NORMAL_DIAGNOSTIC_ROOT",
        "NORMAL_DIAGNOSTIC_PHASE",
        "NORMAL_DIAGNOSTIC_SOURCE",
        "NORMAL_DIAGNOSTIC_CASE",
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "PATHEXT",
        "COMSPEC",
        "OS",
        "USERNAME",
        "USER",
        "LOGNAME",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "UV_CACHE_DIR",
        "TORCH_HOME",
        "CLEARML_CONFIG_FILE",
        "CLEARML_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(validated_driver_environment(root, os.environ))
    roots = [
        wheel_root,
        root,
        root / "src",
        root / "inherited-source",
        root / "inherited/operator",
        root / "inherited/operator/inherited",
        root / "inherited/operator/inherited/tools",
        root / "inherited/tools",
    ]
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, roots)),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_DOTENV_DISABLED="1",
        HF_HUB_OFFLINE="0",
        CLEARML_OFFLINE_MODE="0",
        CLEARML_LOG_TASK_TO_BACKEND="0",
        CLEARML_CONFIG_FILE=os.devnull,
        CLEARML_DISABLE_VAULT_SUPPORT="1",
        CLEARML_TASK_NO_REUSE="1",
    )
    cache_root = scratch / ("normal-caches-" + uuid.uuid4().hex)
    for key in (
        "HF_HOME",
        "CLEARML_CACHE_DIR",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "UV_CACHE_DIR",
    ):
        env[key] = str(cache_root / key.lower())
    env.update(TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch))
    if engineering or host_observer:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    base = [
        str(args.uv),
        "run",
        "--offline",
        "--no-project",
        "--no-sync",
        "--python",
        str(args.python),
        "python",
        "-B",
    ]
    script = args.engineering_bootstrap or root / "worker.py"
    child = base + [str(script), *tokens]
    if tokens[tokens.index("--phase") + 1] == "worker":
        rendezvous = ["--standalone"]
        if os.name == "nt":
            env.update(
                USE_LIBUV="0",
                TORCH_DISABLE_SHARE_RDZV_TCP_STORE="1",
                GLOO_SOCKET_IFNAME="Loopback Pseudo-Interface 1",
            )
            rendezvous = [
                "--rdzv-backend=c10d",
                "--rdzv-conf=store_type=file",
                "--rdzv-endpoint=" + str(scratch / uuid.uuid4().hex),
                "--local-addr=127.0.0.1",
            ]
        child = base + [
            str(root / "normal_torchrun_entry.py"),
            "--nnodes=1",
            "--nproc-per-node=2",
            *rendezvous,
            "--no-python",
            *child,
        ]
    try:
        with subprocess.Popen(child, env=env, cwd=root) as process:
            try:
                return process.wait(timeout=120 if engineering else 28800)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                import psutil

                for descendant in psutil.Process(process.pid).children(recursive=True):
                    descendant.kill()
                process.kill()
                process.wait(timeout=15)
                raise RuntimeError("NORMAL_OWNED_CHILD_TIMEOUT_STOP") from None
    finally:
        # Remove only this invocation's private extraction, never a shared cache.
        shutil.rmtree(wheel_root)


if __name__ == "__main__":
    raise SystemExit(normal_cli(main, "setup"))
