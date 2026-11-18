"""Exact continuation data identity, separate from read-only provider access."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import FunctionType

from seed202_pilot.contract import MQAR_DATA

SOURCE_SHA = "19ff980d566691e4fde21130f3fff90400e8773011206b86a09cb21e5661ff6a"
FULL_SHA = "e2289ced84240f1f81dfcaa254741457a7863ae9a40db6ab68f9cc021e904542"
CASE = "mqar-k64-classic-seed202"
MOUNT_ROOT = Path("/kaggle/input")


def original_identity(original):
    """Derive the finite owner identity from the hash-pinned original saved main."""
    from seed202_pilot.storage import digest

    original = Path(original)
    source = original.parent / "original-kernel.py"
    if (
        digest(source) != SOURCE_SHA
        or digest(original / "out/checkpoints/step_00000980.pt") != FULL_SHA
    ):
        raise ValueError("WORKER_ORIGINAL_IDENTITY_BYTES")
    tree = ast.parse(source.read_bytes())
    declarations = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "BINDING"
    ]
    if len(declarations) != 1:
        raise ValueError("WORKER_ORIGINAL_BINDING")
    binding = declarations[0]
    if binding["case_id"] != CASE or binding["mqar_source_ref"] != MQAR_DATA["source_ref"]:
        raise ValueError("WORKER_ORIGINAL_DATA_REF")
    owner, slug, version = binding["mqar_source_ref"].split("/")
    if version != "1" or owner != "zimike":
        raise ValueError("WORKER_ORIGINAL_DATA_REF")
    return "/kaggle/input/datasets/" + owner + "/" + slug


def bind_case(config_path, physical, original, *, mount_root=None):
    """Parse verified physical bytes; map only the explicitly declared data field."""
    from seed202_pilot.cloud_contract import derive_case

    root = MOUNT_ROOT if mount_root is None else Path(mount_root)
    root = root.resolve(strict=True)
    identity = original_identity(original)
    owner, slug, _version = MQAR_DATA["source_ref"].split("/")
    candidates = (root / "datasets" / owner / slug, root / slug)
    found = [path for path in candidates if (path / "meta.json").exists()]
    physical = Path(physical)
    if len(found) != 1 or physical != found[0]:
        raise ValueError("EXACT_WORKER_DATA_MOUNT")
    if any(path.is_symlink() or path.is_junction() for path in (physical, *physical.parents)):
        raise ValueError("UNSAFE_WORKER_DATA_MOUNT")
    case = derive_case(config_path, physical)
    if case["id"] != CASE:
        raise ValueError("EXACT_WORKER_DATA_CASE")
    # derive_case verifies every immutable data byte and the exact approved YAML.
    result = copy.deepcopy(case)
    result["config"]["data"] = identity
    result["worker_data_binding"] = {
        "field": "config.data",
        "identity": identity,
        "physical": str(physical),
        "original_source_sha256": SOURCE_SHA,
        "original_full_sha256": FULL_SHA,
    }
    return result


def training_config(case):
    """Revalidate physical data using the real typed constructor, then bind identity."""
    from compare_attention.train.config import TrainConfig

    binding = case["worker_data_binding"]
    if (
        binding["field"] != "config.data"
        or binding["original_source_sha256"] != SOURCE_SHA
        or binding["original_full_sha256"] != FULL_SHA
        or case["config"]["data"] != binding["identity"]
    ):
        raise ValueError("WORKER_DATA_BINDING")
    raw = copy.deepcopy(case["config"])
    raw["data"] = binding["physical"]
    parsed = TrainConfig.model_validate(raw)
    identity = binding["identity"]

    class IdentityPath(type(Path())):
        """Identity only, never a hidden filesystem redirection."""

        def __str__(self):
            return identity

    result = parsed.model_copy(update={"data": IdentityPath(identity)})
    if result.model_dump(mode="json") != case["config"]:
        raise ValueError("WORKER_DATA_CONFIG_DRIFT")
    return result


def bound_train(case):
    """Keep native code objects; bind only split-file access in a private namespace."""
    from compare_attention.train import loop

    binding = case["worker_data_binding"]
    identity, physical = binding["identity"], Path(binding["physical"])
    original_load = loop.load_split

    def load_split(path, split, **kwargs):
        if str(Path(path)) != str(Path(identity)):
            raise ValueError("WORKER_SPLIT_IDENTITY")
        return original_load(physical, split, **kwargs)

    scope = dict(vars(loop))
    scope["load_split"] = load_split
    for name in ("_load", "_train", "train"):
        function = vars(loop)[name]
        selected = FunctionType(function.__code__, scope, function.__name__, function.__defaults__)
        selected.__kwdefaults__ = function.__kwdefaults__
        scope[name] = selected
    return scope["train"]
