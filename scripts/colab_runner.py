"""Explicit, isolated Colab T4 execution; inspection is stdlib-only and offline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from colab_package import CONFIGS, WHEEL, checksum, verify_directory  # noqa: E402  # pylint: disable=wrong-import-position

DRIVE_ROOT = Path("/content/drive/MyDrive")
LOCAL_ROOT = Path("/content")

PREFLIGHT = """import sys, subprocess
if sys.version_info[:2] != (3, 13):
    raise RuntimeError('Use existing Python 3.13.x; do not replace the Colab kernel')
try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap',
                                  '--format=csv,noheader'], text=True).strip()
except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError('No T4 GPU: select a Colab T4 GPU runtime') from error
if gpu.splitlines() not in (['Tesla T4, 7.5'], ['T4, 7.5']):
    raise RuntimeError('Requires exactly one T4 (7.5); received: ' + gpu)
print('T4 preflight passed; Python ' + sys.version.split()[0])
"""
PROBE = """import json, sys, tempfile, os
from pathlib import Path
import torch, triton, dotenv
assert sys.version_info[:2] == (3, 13), 'Python 3.13 required'
assert sys.prefix != sys.base_prefix, 'Isolated venv required'
assert torch.__version__ == '2.11.0+cu130', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert triton.__version__ == '3.6.0', triton.__version__
assert torch.cuda.is_available(), 'CUDA unavailable: check Colab cu130 driver compatibility'
assert torch.cuda.device_count() == 1, 'Exactly one T4 required'
assert torch.cuda.get_device_name(0) in ('Tesla T4', 'T4'), 'T4 required'
assert torch.cuda.get_device_capability(0) == (7, 5), 'SM75 required'
x = torch.ones((16, 16), device='cuda', dtype=torch.float16)
assert (x @ x).float().sum().item() == 4096, 'CUDA FP16 probe failed'
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / '.env'
    path.write_text('COLAB_DOTENV_SENTINEL=forbidden')
    dotenv.load_dotenv(path, override=True)
assert 'COLAB_DOTENV_SENTINEL' not in os.environ, 'dotenv disabling unsupported'
print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__,
 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
 'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0),
 'provider': 'google-colab', 'matched_to_kaggle': False}))
"""


def clean_environment():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("HF_", "HUGGING", "CLEARML_", "KAGGLE_", "PIP_"))
        and key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONINSPECT")
    }
    env.update(
        PYTHON_DOTENV_DISABLED="1",
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def preflight():
    # Also embedded verbatim as the notebook's first executable cell.
    exec(compile(PREFLIGHT, "<colab-preflight>", "exec"), {})  # pylint: disable=exec-used


def venv_python(venv):
    return Path(venv) / "bin/python"


def installation_plan(package, venv, python):
    pip = [str(python), "-m", "pip", "--python", str(venv_python(venv)), "install"]
    return [
        [str(python), "-m", "venv", str(venv), "--without-pip"],
        pip + ["torch==2.11.0+cu130", "--index-url", "https://download.pytorch.org/whl/cu130"],
        pip + ["triton==3.6.0", "--no-deps"],
        pip + ["--no-deps", "-r", str(package / "release/requirements.txt")],
        pip + ["--no-deps", str(package / "release" / WHEEL)],
    ]


def persistent_result(path, package, resume=None, *, venv=None, env_file=None, storage="hf"):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(LOCAL_ROOT.resolve()):
        raise ValueError("result-dir must be absolute under /content")
    if storage not in ("hf", "drive"):
        raise ValueError("unsupported storage backend")
    if storage == "drive":
        if not path.resolve().is_relative_to(DRIVE_ROOT.resolve()):
            raise ValueError("Drive storage requires result-dir under MyDrive")
        if not os.path.ismount(str(DRIVE_ROOT.parent)):
            raise ValueError("Explicit Google Drive output requested but not mounted")
    elif path.resolve().is_relative_to(DRIVE_ROOT.parent.resolve()):
        raise ValueError("Drive paths require explicit --storage drive")
    if path.exists() or path.is_symlink():
        raise FileExistsError("result-dir must be new, including for resume")
    forbidden = [package.resolve(), LOCAL_ROOT.resolve() / "compare-attention-private"]
    if venv is not None:
        forbidden.append(Path(venv).resolve())
    if env_file is not None:
        private = Path(env_file).resolve()
        forbidden.append(private if private.parent == LOCAL_ROOT.resolve() else private.parent)
    if any(
        path.resolve().is_relative_to(root) or root.is_relative_to(path.resolve())
        for root in forbidden
    ):
        raise ValueError("result-dir must be outside package, venv and private directories")
    if resume is not None:
        resume = Path(resume)
        if not resume.is_absolute() or not resume.is_file():
            raise ValueError("resume-from must be an explicit absolute existing checkpoint")
    return path


def train_command(package, venv, result, family="deltanet", resume=None):
    command = [
        str(venv_python(venv)),
        "-m",
        "compare_attention.cli.train",
        "--config",
        str(package / "release" / CONFIGS[family]),
        "--data",
        str(package / "data"),
        "--result-dir",
        str(result),
        "--tracker",
        "clearml",
    ]
    if resume:
        command += ["--resume-from", str(resume)]
    return command


def execute(command, env, cwd, *, capture=False):
    return subprocess.run(command, check=True, env=env, cwd=cwd, text=True, capture_output=capture)


def setup(package, venv):
    preflight()
    verify_directory(package)
    venv = Path(venv)
    if venv.exists():
        raise FileExistsError("venv must be new; select another local directory")
    if (
        not venv.resolve().is_relative_to(LOCAL_ROOT)
        or venv.resolve().is_relative_to(Path("/content/drive"))
        or venv.resolve().is_relative_to(package.resolve())
    ):
        raise ValueError("venv must be local under /content, outside package and Drive")
    env = clean_environment()
    for command in installation_plan(package, venv, sys.executable):
        execute(command, env, package)
    try:
        result = execute([str(venv_python(venv)), "-c", PROBE], env, package, capture=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Pinned cu130 CUDA probe failed; no fallback. Check Colab driver.\n"
            + (error.stderr or "")
        ) from error
    (venv / "colab-runtime.json").write_text(result.stdout, encoding="utf-8")
    (venv / "colab-package.sha256").write_text(checksum(package / "manifest.json"))
    print(result.stdout)


def private_env_path(env_file, package, result, *, venv=None):
    """The file path, not any credential value, may cross a subprocess boundary."""
    if env_file is None or not Path(env_file).is_file():
        raise ValueError("--env-file requires an existing separate private file")
    path = Path(env_file).resolve()
    excluded = [package.resolve(), result.resolve()]
    if venv is not None:
        excluded.append(Path(venv).resolve())
    if any(path.is_relative_to(root) for root in excluded):
        raise ValueError("--env-file must be outside package, venv and result directories")
    return path


def service_command(package, venv, phase, env_file, arguments=()):
    return [
        str(venv_python(venv)),
        str(package / "tools/colab_services.py"),
        phase,
        "--env-file",
        str(env_file),
        *arguments,
    ]


def run(package, venv, result, family="deltanet", resume=None, *, env_file=None, storage="hf"):
    preflight()
    manifest = verify_directory(package)
    result = persistent_result(
        result, package, resume, venv=venv, env_file=env_file, storage=storage
    )
    env_file = private_env_path(env_file, package, result, venv=venv)
    if (Path(venv) / "colab-package.sha256").read_text() != checksum(package / "manifest.json"):
        raise ValueError("venv was not set up for this verified package")
    env = clean_environment()
    runtime = execute([str(venv_python(venv)), "-c", PROBE], env, package, capture=True)
    runtime = json.loads(runtime.stdout)
    execute(service_command(package, venv, "check", env_file), env, package)
    execute(
        [
            str(venv_python(venv)),
            str(package / "tools/colab_checkpoint_backup.py"),
            "probe",
            "--env-file",
            str(env_file),
            "--result-dir",
            str(result),
            "--package-dir",
            str(package),
        ],
        env,
        package,
    )
    result.parent.mkdir(parents=True, exist_ok=True)
    command = train_command(package, venv, result, family, resume)
    execute(
        service_command(
            package,
            venv,
            "train",
            env_file,
            ["--package-dir", str(package), "--hf-checkpoints", "--", *command[3:]],
        ),
        env,
        package,
    )
    execute(
        [
            str(venv_python(venv)),
            str(package / "tools/colab_verify.py"),
            str(package),
            str(result),
            family,
            runtime["gpu"],
        ],
        env,
        package,
    )
    execute(
        [
            str(venv_python(venv)),
            "-m",
            "compare_attention.cli.evaluate",
            "--checkpoint",
            str(result / "model.pt"),
            "--data",
            str(package / "data"),
            "--split",
            "test",
            "--batch-size",
            "32",
            "--device",
            "cuda",
            "--dtype",
            "float16",
            "--output",
            str(result / "test-evaluation.json"),
        ],
        env,
        package,
    )
    verify_directory(package)
    provenance = {
        "provider": "google-colab",
        "actual_runtime": runtime,
        "kaggle_reference": manifest["kaggle_reference"],
        "matched_runtime": False,
        "bitwise_initialization_claimed": False,
        "family": family,
        "fresh": resume is None,
        "tracking": "clearml",
        "storage": "hf",
        "local_result_backend": storage,
        "checkpoint_backup_enabled": True,
        "hf_upload_requested": True,
        "result_sha256": {
            path.relative_to(result).as_posix(): checksum(path)
            for path in result.rglob("*")
            if path.is_file()
        },
        "package_manifest_sha256": checksum(package / "manifest.json"),
    }
    with (result / "colab-provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    upload_only(package, venv, result, env_file)


def upload_only(package, venv, result, env_file):
    """Retry only a completed upload; never train or replace a previous result."""
    verify_directory(package)
    env_file = private_env_path(env_file, package, result, venv=venv)
    execute(
        service_command(
            package,
            venv,
            "upload",
            env_file,
            ["--result-dir", str(result), "--package-dir", str(package)],
        ),
        clean_environment(),
        package,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--inspect", "--verify-only", action="store_true", dest="inspect")
    actions.add_argument("--setup", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--upload-only", action="store_true")
    actions.add_argument("--check-credentials", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--storage", choices=("hf", "drive"), default="hf")
    parser.add_argument("--venv", type=Path, default=Path("/content/compare-attention-venv"))
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--family", choices=tuple(CONFIGS), default="deltanet")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    package = args.package_dir.resolve()
    if args.setup:
        setup(package, args.venv)
    elif args.run or args.upload_only or args.check_credentials:
        if args.check_credentials:
            verify_directory(package)
            env_file = private_env_path(
                args.env_file, package, args.result_dir or package, venv=args.venv
            )
            execute(
                service_command(package, args.venv, "check", env_file), clean_environment(), package
            )
            return
        if args.result_dir is None:
            parser.error("--run/--upload-only requires --result-dir")
        if args.upload_only:
            upload_only(package, args.venv, args.result_dir, args.env_file)
        else:
            run(
                package,
                args.venv,
                args.result_dir,
                args.family,
                args.resume_from,
                env_file=args.env_file,
                storage=args.storage,
            )
    else:
        manifest = verify_directory(package)
        print(
            json.dumps(
                {
                    "verified_files": len(manifest["files"]),
                    "dry_run": True,
                    "family": args.family,
                    "training_started": False,
                    "source_sha256": manifest["source_sha256"],
                    "data_fingerprint": manifest["data_fingerprint"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
