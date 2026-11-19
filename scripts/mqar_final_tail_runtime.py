"""Authenticated deployment-v3 hardware/mount/error functions, not its training main."""

import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

from inherited_kernel import (  # noqa: F401
    driver_directory,
    extract_archive,
    hardware,
    inspect_hardware,
    resolve_mount,
)
from inherited_kernel import public_error as inherited_public_error
from mqar_final_tail_native import require

INHERITED_SHA = "3675f6a9f914194bbcfdeeea60d6994480205c1d22d88615429825eb48d7ac8e"


def public_error(error, phase):
    """Extend inherited finite error metadata with strictly filtered worker versions."""
    result = inherited_public_error(error, phase)
    raw = getattr(error, "runtime_qualification", None)
    if (
        type(error) is ValueError
        and error.args == ("PINNED_TORCH_RUNTIME_REQUIRED",)
        and type(raw) is dict
    ):
        runtime = {"phase": "prepared_inference_worker"}
        for key in ("torch", "cuda", "python"):
            value = raw.get(key)
            runtime[key] = (
                value
                if type(value) is str
                and re.fullmatch(
                    r"[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?(?:\+cu[0-9]{2,4})?", value
                )
                else None
            )
        value = raw.get("cudnn")
        runtime["cudnn"] = value if type(value) is int and 0 <= value <= 999999 else None
        value = raw.get("implementation")
        runtime["implementation"] = value if value in ("CPython", "PyPy") else None
        value = raw.get("executable")
        runtime["executable"] = value if value in ("python", "python3", "python.exe") else None
        result["runtime_qualification"] = runtime
    return result


def guard():
    """Actual metadata/driver guard before private input, model or install."""
    path = Path(__file__).with_name("inherited_kernel.py")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == INHERITED_SHA, "INHERITED_KERNEL_PIN")
    driver = driver_directory()
    topology = inspect_hardware(driver)
    import torch

    if not (
        platform.system() == "Linux"
        and platform.python_version() == "3.13.16"
        and str(torch.__version__) == "2.11.0+cu130"
        and torch.version.cuda == "13.0"
        and torch.backends.cudnn.version() == 91900
    ):
        error = ValueError("PINNED_TORCH_RUNTIME_REQUIRED")
        error.runtime_qualification = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        }
        destination = os.environ.get("C64_RECOVERY_RUNTIME_DIAGNOSTIC")
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(public_error(error, "worker"), stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        raise error
    require(
        not any(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")),
        "INFERENCE_NOT_DISTRIBUTED",
    )
    return topology
