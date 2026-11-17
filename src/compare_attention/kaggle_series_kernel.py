"""Generated isolated kernel for immutable staged series (no compiler-cache claims)."""

from __future__ import annotations

import json
from typing import Any

from compare_attention.kaggle_kernel import SECRETS
from compare_attention.kaggle_run import Kernel

# Exact runtime identities, not fuzzy marketing-name matching. Unknown variants fail closed.
_REQUIRED_GPUS = {
    "NvidiaTeslaT4": {"name": "Tesla T4", "compute_capability": [7, 5]},
    "NvidiaL4": {"name": "NVIDIA L4", "compute_capability": [8, 9]},
    "NvidiaTeslaA100": {"name": "NVIDIA A100-SXM4-40GB", "compute_capability": [8, 0]},
}

_TEMPLATE = '''"""Immutable series kernel; generated, never edits input data."""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

_IDENTITY = __IDENTITY__
_RELEASE = __RELEASE__
_ARCHIVE_SHA = __ARCHIVE_SHA__
_DATA = __DATA__
_DATA_SPEC = __DATA_SPEC__
_CONFIG = __CONFIG__
_DTYPE = __DTYPE__
_COMPILE = __COMPILE__
_REQUIRED_GPU = __REQUIRED_GPU__
_WHEEL = __WHEEL__
_CREDENTIALS = __CREDENTIALS__
_SECRETS = __SECRETS__
_WORK = Path("/kaggle/working")
_TIMINGS = []


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _find(name, dataset):
    slug = dataset.split("/")[-1]
    candidates = [Path(p) for p in glob.glob("/kaggle/input/**/" + name, recursive=True)
                  if slug in Path(p).parts]
    if len(candidates) != 1:
        raise RuntimeError("expected exactly one " + name + " for " + dataset)
    return candidates[0]


def _verify(root, files):
    for name, record in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError("missing/unsafe input: " + name)
        if path.stat().st_size != record["bytes"] or _hash(path) != record["sha256"]:
            raise RuntimeError("checksum mismatch: " + name)


def _phase(name, action):
    start = time.perf_counter()
    try:
        return action()
    finally:
        row = dict(phase=name, seconds=time.perf_counter() - start)
        _TIMINGS.append(row)
        (_WORK / "phase-timings.json").write_text(json.dumps(_TIMINGS))
        print(json.dumps(row), flush=True)


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_preflight():
    # Only hardware/version evidence; no secrets, FLA imports or install side effects.
    report = dict(torch=None, cuda=None, cuda_available=False, device_count=0,
                  gpu=None, triton=None, dtype=_DTYPE, compile=_COMPILE,
                  native_bf16_supported=False, passed=False, failure=None)
    try:
        import torch
        report.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                      cuda_available=bool(torch.cuda.is_available()),
                      device_count=torch.cuda.device_count(), triton=_package_version("triton"))
        if report["cuda_available"] and report["device_count"] > 0:
            index = torch.cuda.current_device()
            report["gpu"] = dict(name=torch.cuda.get_device_name(index), index=index,
                                 compute_capability=list(torch.cuda.get_device_capability(index)))
            report["native_bf16_supported"] = bool(
                torch.cuda.is_bf16_supported(including_emulation=False))
        if not report["cuda"] or not report["cuda_available"] or report["gpu"] is None:
            raise RuntimeError("CUDA GPU required")
        if any(report["gpu"][key] != value for key, value in _REQUIRED_GPU.items()):
            raise RuntimeError("delivered GPU differs from requested accelerator")
        if not report["triton"]:
            raise RuntimeError("Triton distribution required; refusing blind installation")
        if _COMPILE and _DTYPE == "bfloat16" and not report["native_bf16_supported"]:
            raise RuntimeError("compiled BF16 requires native device support")
        report["passed"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__
    finally:
        (_WORK / "runtime-preflight.json").write_text(json.dumps(report))
        print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("runtime preflight failed: " + report["failure"])


def _credentials():
    if _CREDENTIALS:
        env = _find("kaggle-env.env", _CREDENTIALS)
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() in _SECRETS and value.strip():
                os.environ.setdefault(key.strip(), value.strip())
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for key in _SECRETS:
            try:
                value = client.get_secret(key)
                if value:
                    os.environ.setdefault(key, value)
            except Exception:
                pass
    except Exception:
        pass


def _inputs():
    archive = _find("release.bundle", _RELEASE)
    if _hash(archive) != _ARCHIVE_SHA:
        raise RuntimeError("release archive checksum mismatch")
    root = Path("/kaggle/temp/compare-attention-release")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (root / name).resolve().is_relative_to(root.resolve()):
                raise RuntimeError("unsafe archive member")
        bundle.extractall(root)
    _verify(root, json.loads((root / "release-manifest.json").read_text()))
    data = _find("data-manifest.json", _DATA).parent
    actual = json.loads((data / "data-manifest.json").read_text())
    if actual != _DATA_SPEC:
        raise RuntimeError("prepared KV manifest differs")
    _verify(data, _DATA_SPEC["files"])
    return root, data


def _train(root, data):
    # Runtime mount paths vary across Kaggle versions; input checks precede this rewrite.
    import yaml
    config = yaml.safe_load((root / _CONFIG).read_text())
    config["data"] = str(data)
    runtime = Path("/kaggle/temp/compare-attention-runtime.yaml")
    runtime.write_text(yaml.safe_dump(config))
    subprocess.check_call([sys.executable, "-m", "compare_attention.cli.train",
                           "--config=" + str(runtime),
                           "--result-dir=" + str(_WORK / _IDENTITY["slug"])])
    result = _WORK / _IDENTITY["slug"]
    files = {}
    for path in result.rglob("*"):
        if path.is_file():
            files[path.relative_to(_WORK).as_posix()] = dict(bytes=path.stat().st_size,
                                                            sha256=_hash(path))
    (_WORK / "result-manifest.json").write_text(json.dumps(dict(identity=_IDENTITY, files=files)))


def main():
    _phase("runtime_preflight", _runtime_preflight)
    _credentials()
    root, data = _phase("verify_inputs", _inputs)
    _phase("install_dependencies", lambda: subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore",
         "-r", str(root / "requirements.txt")]))
    _phase("install_wheel", lambda: subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "--root-user-action=ignore", str(root / _WHEEL)]))
    _phase("train", lambda: _train(root, data))


main()
'''


def render_series_kernel(
    state: dict[str, Any],
    name: str,
    credentials: str | None,
    *,
    accelerator: str = "NvidiaTeslaT4",
) -> Kernel:
    """Exact immutable inputs; credentials are an independent private dataset reference."""
    if accelerator not in _REQUIRED_GPUS:
        raise ValueError(f"unsupported runtime accelerator: {accelerator}")
    entry = state["runs"][name]
    identity = {
        "series": state["identity"]["fingerprint"],
        "config": name,
        "id": f"{state['owner']}/{entry['slug']}",
        "slug": entry["slug"],
    }
    values = {
        "IDENTITY": identity,
        "RELEASE": state["release_id"],
        "ARCHIVE_SHA": state["archive_sha256"],
        "DATA": state["data_id"],
        "DATA_SPEC": state["identity"]["data"],
        "CONFIG": name,
        "DTYPE": state["identity"]["configs"][name]["schedule"]["dtype"],
        "COMPILE": state["identity"]["configs"][name]["schedule"]["compile"],
        "REQUIRED_GPU": _REQUIRED_GPUS[accelerator],
        "WHEEL": state["wheel"],
        "CREDENTIALS": credentials,
        "SECRETS": SECRETS,
    }
    code = _TEMPLATE
    for key, value in values.items():
        code = code.replace(f"__{key}__", repr(value))
    sources = [state["release_id"], state["data_id"]]
    if credentials:
        sources.append(credentials)
    metadata = {
        "id": identity["id"],
        "title": entry["slug"],
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "dataset_sources": sources,
        "enable_internet": True,
        "enable_gpu": True,
    }
    # Check JSON serializability before any cloud operation.
    json.dumps(metadata)
    return Kernel(code=code, metadata=metadata)
