"""Authenticated auxiliary namespace adapter; scientific functions stay unchanged."""

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

WHEEL = "compare_attention-0.1.0-py3-none-any.whl"
WHEEL_SHA = "512cb5b073e47deca36d2bfb41f2d183e0711b8817b9a0aa18dbd2e07591b2d3"
FREEZE_SHA = "6c22c64d0cb4c0a186bc615faff5f4d8b0ca6074669c8e4dc7dfc33ea5f40745"
MANIFEST_SHA = "f3029cc05e7534abadf78b0b87e70ef577995bd5eaedf19e31c1a31d6b956cb3"


def install(science):
    """Bind exact compatibility names without executing the old freeze CLI imports."""
    from quality_recovery import require
    from seed202_pilot import cloud_contract, cloud_native

    raw = (Path(science) / "operator/quality_freeze.py").read_bytes()
    require(
        hashlib.sha256((Path(science) / "manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA,
        "AUTHENTIC_SCIENCE_MANIFEST",
    )
    require(hashlib.sha256(raw).hexdigest() == FREEZE_SHA, "AUTHENTIC_SCIENCE_FREEZE_SOURCE")
    node = next(
        node
        for node in ast.parse(raw).body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_package"
    )
    scope = {"Path": Path, "json": json, "digest": cloud_contract.digest, "require": require}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "quality_freeze.py", "exec"), scope)
    previous = {
        name: sys.modules.get(name)
        for name in ("quality_freeze", "quality_contract", "quality_native")
    }
    views = {
        "quality_freeze": {
            "WHEEL": WHEEL,
            "WHEEL_SHA": WHEEL_SHA,
            "verify_package": scope["verify_package"],
        },
        "quality_contract": {
            name: getattr(cloud_contract, name)
            for name in ("canonical", "derive_case", "digest", "verify_history")
        },
        "quality_native": {
            name: getattr(cloud_native, name)
            for name in ("SHARED", "validate_full", "verify_shared")
        },
    }
    for name, values in views.items():
        view = types.ModuleType(name)
        view.__dict__.update(values)
        view.__file__ = str(Path(__file__).resolve())
        sys.modules[name] = view
    return previous


def restore(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
